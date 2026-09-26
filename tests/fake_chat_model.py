"""
A scripted, heuristic fake chat model used ONLY to test the LangGraph
workflow's WIRING -- routing, parallel fan-out, the merge reducer, timeout
handling, empty-plan short-circuit, structured-output plumbing -- without an
OpenAI key or network access.

It does not reason. For bind_tools() it issues exactly one tool call using
the incoming task text as the query, then stops. For with_structured_output()
it extracts a plausible Pydantic instance from the prompt text with simple
regex/keyword heuristics. This proves the graph has no wiring bug; it says
nothing about real answer quality, which is what eval/ (run against the real
models) is for.
"""

from __future__ import annotations

import re
from typing import Any, List, Type

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from subagents.schemas import Finding, SubagentResult
from supervisor.schemas import ConflictNote, DepartmentTask, FinalAnswer, Plan


class _BoundToolFake:
    def __init__(self, parent: "FakeChatModel", tools):
        self.parent = parent
        self.tools = tools

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        return self.parent._tool_step(messages, self.tools)


class _StructuredFake:
    def __init__(self, parent: "FakeChatModel", schema: Type[BaseModel], include_raw: bool):
        self.parent = parent
        self.schema = schema
        self.include_raw = include_raw

    def invoke(self, prompt: Any):
        text = self.parent._as_text(prompt)
        parsed = self.parent._build_structured(self.schema, text)
        if not self.include_raw:
            return parsed
        raw = AIMessage(content="", usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70})
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class FakeChatModel:
    def bind_tools(self, tools):
        return _BoundToolFake(self, tools)

    def with_structured_output(self, schema, include_raw: bool = False):
        return _StructuredFake(self, schema, include_raw)

    # -- tool-calling step: search once, then stop --
    def _tool_step(self, messages: List[BaseMessage], tools) -> AIMessage:
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        human = next((m for m in reversed(messages) if m.__class__.__name__ == "HumanMessage"), None)
        query_text = (human.content if human else "policy")[:60]
        if ai_count == 0:
            tool = tools[0]
            return AIMessage(
                content="",
                tool_calls=[{"name": tool.name, "args": {"query": query_text}, "id": "call_1"}],
                usage_metadata={"input_tokens": 80, "output_tokens": 15, "total_tokens": 95},
            )
        return AIMessage(content="done", usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45})

    @staticmethod
    def _as_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return "\n".join(getattr(m, "content", str(m)) for m in prompt)
        return str(prompt)

    def _build_structured(self, schema: Type[BaseModel], text: str) -> BaseModel:
        if schema is Plan:
            return self._build_plan(text)
        if schema is SubagentResult:
            return self._build_subagent_result(text)
        if schema is FinalAnswer:
            return self._build_final_answer(text)
        raise NotImplementedError(f"FakeChatModel has no heuristic for {schema}")

    # --- Plan: keyword-route the question to departments ---
    def _build_plan(self, text: str) -> Plan:
        question_line = next(
            (l for l in text.splitlines() if l.strip().lower().startswith("employee question:")), "Employee question: (unknown)"
        )
        lower = question_line.lower()  # match only the actual question, not the instructions describing each department
        depts = []
        if any(k in lower for k in ["leave", "notice period", "hybrid", "relocat", "resign", "onboard", "grievance"]):
            depts.append("hr")
        if any(k in lower for k in ["vpn", "laptop", "password", "ticket", "chatgpt", "device", "access card", "overseas access"]):
            depts.append("it")
        if any(k in lower for k in ["expense", "hotel", "meal", "reimburse", "allowance", "settlement", "advance", "corporate card", "relocat"]):
            depts.append("finance")
        if "share price" in lower or "stock" in lower:
            depts = []

        tasks = [DepartmentTask(department=d, task_message=f"{question_line}\n\nFind everything relevant.") for d in depts]
        return Plan(departments=tasks, reason=f"Keyword match -> {depts or 'none'}")

    # --- SubagentResult: turn whatever chunks were retrieved into findings ---
    def _build_subagent_result(self, text: str) -> SubagentResult:
        if "KSPL-HR" in text:
            department = "hr"
        elif "KSPL-IT" in text:
            department = "it"
        elif "KSPL-FIN" in text:
            department = "finance"
        else:
            department = "hr"

        chunk_dicts = re.findall(r"\{[^{}]*'section_number':[^{}]*\}", text)

        def grab(raw: str, key: str) -> str:
            m = re.search(rf"'{key}':\s*'([^']*)'", raw) or re.search(rf"'{key}':\s*(\d+)", raw)
            return m.group(1) if m else ""

        findings = []
        for raw in chunk_dicts[:3]:
            section = grab(raw, "section_number")
            if not section:
                continue
            findings.append(
                Finding(
                    rule=(grab(raw, "text") or "See retrieved clause.")[:200],
                    section=section,
                    page=int(grab(raw, "page") or 1),
                    document_id=grab(raw, "document_id") or "UNKNOWN",
                    version=grab(raw, "version") or "0.0",
                    effective_date=grab(raw, "effective_date") or "2026-01-01",
                )
            )
        covered = "yes" if findings else "no"
        gaps = [] if findings else ["No relevant clause found (fake test model)."]
        return SubagentResult(department=department, covered=covered, findings=findings, gaps=gaps)

    # --- FinalAnswer: stitch findings together, flag any overlap candidate ---
    def _build_final_answer(self, text: str) -> FinalAnswer:
        overlap_section_match = re.search(r"CANDIDATE OVERLAPS TO CHECK FOR REAL CONFLICTS:\n(.*?)\n\n", text, re.DOTALL)
        overlap_text = overlap_section_match.group(1) if overlap_section_match else "[]"
        has_overlap = overlap_text.strip() not in ("[]", "")

        findings_match = re.search(r"FINDINGS \(one per rule.*?\):\n(.*?)\n\n", text, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1)
        else:
            # Different prompt shape (e.g. the Phase 10 baseline agent, which
            # has no "FINDINGS (one per rule...)" header) -- fall back to
            # scanning the whole text for any retrieved-chunk dicts, same as
            # _build_subagent_result does.
            chunk_dicts = re.findall(r"\{[^{}]*'section_number':[^{}]*\}", text)
            findings_text = "[" + ", ".join(chunk_dicts[:3]) + "]" if chunk_dicts else "[]"
        no_findings_at_all = findings_text.strip() == "[]"

        conflicts = []
        if has_overlap:
            conflicts.append(
                ConflictNote(
                    topic="temporary accommodation nights",
                    documents=["KSPL-HR-POL-004 sec 4.4", "KSPL-FIN-POL-003 sec 6.4"],
                    values=["15 nights", "10 nights"],
                    applies="10 nights (Finance 6.4 explicitly replaces the earlier limit)",
                    reason="Rule 5a: Finance 6.4 explicitly states it replaces earlier limits.",
                )
            )

        if no_findings_at_all:
            answer = "Not covered by any department's policy (fake test model)."
        else:
            answer = "Findings:\n" + findings_text[:800]

        return FinalAnswer(answer_markdown=answer, conflicts=conflicts, gaps=[], departments_consulted=[])
