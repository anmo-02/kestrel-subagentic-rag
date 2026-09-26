"""
Phase 3-4: The shared subagent builder.

A subagent is deliberately narrow: it sees only (a) its own system
instructions and (b) the task message the supervisor hands it -- never chat
history, never other subagents' findings. It has exactly one tool (search
over its own Chroma collection) and always finishes with a SubagentResult.

We implement this as an explicit two-step pipeline rather than a generic
open-ended agent loop:

  1. GATHER -- the model may call the search tool up to MAX_SEARCH_ROUNDS
     times (e.g. to rephrase a query that didn't return anything useful),
     then must stop.
  2. STRUCTURE -- one call to with_structured_output(SubagentResult), given
     the task and every chunk actually retrieved. Because this call sees
     only real retrieved chunks, a finding cannot exist in the output
     without a real section/page/version behind it.

Any exception anywhere in this pipeline is caught inside Subagent.run() and
turned into a SubagentResult with covered="no" and a gap explaining the
failure -- so one subagent failing never crashes the graph (Phase 6 requires
this: "if a subagent fails ... record a gap ... and continue with the rest").
"""

from __future__ import annotations

import time
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from ingest.build_collections import get_client
from subagents.schemas import SubagentResult

MAX_SEARCH_ROUNDS = 3
TOP_K = 4


def make_search_tool(collection_name: str, embeddings, persist_directory: str) -> BaseTool:
    """
    Build a search tool scoped to exactly one Chroma collection.

    We explicitly build our own PersistentClient (via ingest.build_collections
    .get_client, the same one ingestion uses) and hand it to Chroma via
    `client=`, rather than letting Chroma build one internally from
    `persist_directory=`. Chroma's internal path keys its client cache by a
    fixed "ephemeral" identifier per-process, so constructing several
    Chroma() instances that point at different persist_directory values
    within one process (as our own test suite does, and as any long-running
    server handling multiple persist directories would) raises "instance
    already exists with different settings". Passing an explicit client
    sidesteps that entirely.
    """
    client = get_client(persist_directory)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    @tool
    def search_policy(query: str) -> List[dict]:
        """Search this department's policy clauses. Returns up to 4 clauses
        with their section number, page, and full text. Call again with
        different wording if the first search doesn't return what you need."""
        docs = vectorstore.similarity_search(query, k=TOP_K)
        return [
            {
                "text": d.page_content,
                "section_number": d.metadata.get("section_number"),
                "section_heading": d.metadata.get("section_heading"),
                "page": d.metadata.get("page"),
                "document_id": d.metadata.get("document_id"),
                "document_title": d.metadata.get("document_title"),
                "version": d.metadata.get("version"),
                "effective_date": d.metadata.get("effective_date"),
            }
            for d in docs
        ]

    return search_policy


def _fallback_result(department: str, message: str) -> SubagentResult:
    return SubagentResult(department=department, covered="no", findings=[], gaps=[message])


def _empty_token_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _accumulate_tokens(total: dict, message) -> None:
    usage = getattr(message, "usage_metadata", None) if message is not None else None
    if not usage:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] += usage.get(key, 0) or 0


class Subagent:
    """One department, one collection, one tool, one fixed output schema."""

    def __init__(
        self,
        department: str,
        instructions: str,
        search_tool: BaseTool,
        tool_model,
        structuring_model,
    ):
        self.department = department
        self.instructions = instructions
        self.search_tool = search_tool
        self.tool_model = tool_model.bind_tools([search_tool])
        self.structuring_model = structuring_model.with_structured_output(SubagentResult, include_raw=True)

    def run(self, task_message: str, timeout_seconds: float = 30.0) -> dict:
        """
        Returns a dict: department, result (SubagentResult), queries run,
        chunks_seen, elapsed_seconds, error (None on success).
        A caller-side timeout is enforced by the LangGraph node (Phase 6),
        not here -- this method just guarantees it never raises.
        """
        start = time.time()
        try:
            result, queries, chunks_seen, tokens = self._run_inner(task_message)
            return {
                "department": self.department,
                "result": result,
                "queries": queries,
                "chunks_seen": chunks_seen,
                "elapsed_seconds": time.time() - start,
                "tokens_used": tokens,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 -- a subagent must never crash the graph
            return {
                "department": self.department,
                "result": _fallback_result(
                    self.department,
                    f"The {self.department} subagent could not be checked ({exc.__class__.__name__}: {exc}).",
                ),
                "queries": [],
                "chunks_seen": [],
                "elapsed_seconds": time.time() - start,
                "tokens_used": _empty_token_usage(),
                "error": str(exc),
            }

    def _run_inner(self, task_message: str) -> Tuple[SubagentResult, List[str], List[dict], dict]:
        messages = [SystemMessage(content=self.instructions), HumanMessage(content=task_message)]
        queries: List[str] = []
        chunks_seen: List[dict] = []
        tokens = _empty_token_usage()

        for _ in range(MAX_SEARCH_ROUNDS):
            ai_msg = self.tool_model.invoke(messages)
            _accumulate_tokens(tokens, ai_msg)
            messages.append(ai_msg)
            if not getattr(ai_msg, "tool_calls", None):
                break
            for tool_call in ai_msg.tool_calls:
                queries.append(tool_call["args"].get("query", ""))
                tool_result = self.search_tool.invoke(tool_call["args"])
                chunks_seen.extend(tool_result)
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

        structuring_prompt = (
            f"{self.instructions}\n\n"
            f"--- TASK ---\n{task_message}\n\n"
            f"--- SEARCH QUERIES RUN ---\n{queries}\n\n"
            f"--- CLAUSES RETRIEVED (use ONLY these; never invent a section number) ---\n{chunks_seen}\n\n"
            f"Produce the final structured result now."
        )
        structured = self.structuring_model.invoke(structuring_prompt)
        _accumulate_tokens(tokens, structured.get("raw"))
        result: SubagentResult = structured["parsed"]
        return result, queries, chunks_seen, tokens


def default_chat_model(model_name: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)
