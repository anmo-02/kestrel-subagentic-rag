"""
Phase 10: the single-agent, single-collection baseline -- the "pilot" design
from the assignment's Section 2 -- rebuilt for a fair side-by-side
comparison against the subagentic design.

Same PDF splitting (ingest/clean.py) and the same embeddings model. The only
thing that differs is architecture: one Chroma collection holding every
chunk from all three documents, and one GPT-4o agent with one search tool,
instead of three narrow subagents behind a planning/synthesis supervisor.
This is deliberately the design the assignment's pilot describes failing
with: mixed-up rules, missing parts on multi-department questions, no
conflict awareness, and no forced "not covered" when nothing is found.
"""

from __future__ import annotations

import time
from typing import List

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ingest.build_collections import get_client
from ingest.clean import clean_all
from models import EmployeeProfile
from supervisor.schemas import FinalAnswer

BASELINE_COLLECTION = "baseline_all_policy"
TOP_K = 4
MAX_SEARCH_ROUNDS = 3

BASELINE_INSTRUCTIONS = """You are Kestrel Systems' employee help-desk assistant. You have ONE search \
tool over ALL company policy (HR, IT, and Finance/travel combined in one collection). Search for \
whatever the question needs, using as many searches as it takes, then answer the employee directly. \
Cite the section number for every rule, e.g. [KSPL-HR-POL-004, section 4.4]. If nothing relevant is \
found, say the question is not covered and suggest the HR Business Partner or IT service desk as \
appropriate. If you notice conflicting numbers across documents, say so."""


def build_baseline_collection(persist_directory: str, embeddings) -> int:
    """Same chunks as the three real collections, upserted into one combined collection."""
    client = get_client(persist_directory)
    collection = client.get_or_create_collection(name=BASELINE_COLLECTION)
    all_chunks = clean_all()
    flat = [c for chunks in all_chunks.values() for c in chunks]
    ids = [c.chunk_id for c in flat]
    documents = [c.text for c in flat]
    metadatas = [c.metadata() for c in flat]
    vectors = embeddings.embed_documents(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    return len(flat)


def _make_search_tool(embeddings, persist_directory: str):
    client = get_client(persist_directory)
    vectorstore = Chroma(client=client, collection_name=BASELINE_COLLECTION, embedding_function=embeddings)

    @tool
    def search_all_policy(query: str) -> List[dict]:
        """Search ALL Kestrel policy documents (HR, IT, Finance combined). Returns up to 4 clauses."""
        docs = vectorstore.similarity_search(query, k=TOP_K)
        return [
            {
                "text": d.page_content,
                "section_number": d.metadata.get("section_number"),
                "page": d.metadata.get("page"),
                "document_id": d.metadata.get("document_id"),
                "department": d.metadata.get("department"),
                "version": d.metadata.get("version"),
                "effective_date": d.metadata.get("effective_date"),
            }
            for d in docs
        ]

    return search_all_policy


def _accumulate(tokens: dict, message) -> None:
    usage = getattr(message, "usage_metadata", None) if message is not None else None
    if not usage:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        tokens[key] += usage.get(key, 0) or 0


class BaselineAgent:
    """One agent, one tool, one combined collection -- no department separation."""

    def __init__(self, model, embeddings, persist_directory: str):
        self.search_tool = _make_search_tool(embeddings, persist_directory)
        self.tool_model = model.bind_tools([self.search_tool])
        self.structuring_model = model.with_structured_output(FinalAnswer, include_raw=True)

    def run(self, question: str, profile: EmployeeProfile) -> dict:
        start = time.time()
        messages = [
            SystemMessage(content=BASELINE_INSTRUCTIONS),
            HumanMessage(content=f"{profile.as_context_line()}\n\nQuestion: {question}"),
        ]
        chunks_seen: List[dict] = []
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for _ in range(MAX_SEARCH_ROUNDS):
            ai_msg = self.tool_model.invoke(messages)
            _accumulate(tokens, ai_msg)
            messages.append(ai_msg)
            if not getattr(ai_msg, "tool_calls", None):
                break
            for tool_call in ai_msg.tool_calls:
                result = self.search_tool.invoke(tool_call["args"])
                chunks_seen.extend(result)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        prompt = (
            f"{BASELINE_INSTRUCTIONS}\n\nQuestion: {question}\n{profile.as_context_line()}\n\n"
            f"Clauses retrieved:\n{chunks_seen}\n\nAnswer now."
        )
        structured = self.structuring_model.invoke(prompt)
        _accumulate(tokens, structured.get("raw"))
        answer: FinalAnswer = structured["parsed"]
        answer.departments_consulted = sorted({c["department"] for c in chunks_seen if c.get("department")})

        return {
            "final_answer": answer,
            "chunks_seen": chunks_seen,
            "elapsed_seconds": time.time() - start,
            "tokens": tokens,
        }
