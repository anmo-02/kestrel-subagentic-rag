"""
Phase 7: combine every subagent's findings into one cited answer, catching
and resolving conflicts between documents.

Conflict detection is two-layered on purpose:
  1. A cheap, deterministic keyword-overlap pass (find_topic_overlaps) flags
     CANDIDATE overlaps between findings from different departments. This
     exists because leaving conflict-spotting entirely to the model's
     judgement is exactly how a real conflict could get missed -- the
     pilot's whole failure mode was silently mixing rules that looked
     similar. A candidate list is cheap insurance.
  2. The synthesis model (GPT-4o) makes the actual call: is this really the
     same topic, do the values really differ, and if so which one applies,
     using the conflict-resolution order given in its instructions.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from models import EmployeeProfile
from subagents.schemas import SubagentResult
from supervisor.schemas import FinalAnswer
from supervisor.skills import skills_for_question

SYNTHESIS_INSTRUCTIONS = """You are the supervisor of Kestrel Systems' employee help-desk assistant, \
writing the final answer to an employee from the findings your HR, IT and Finance subagents \
already gathered. You did not read the policies yourself -- you only know what is in FINDINGS \
below.

Hard rules:
1. CITE EVERY RULE. Every rule in your answer ends with a citation like \
[KSPL-FIN-POL-003, section 6.2]. No exceptions.
2. USE ONLY FINDINGS. No number, name, deadline, or rule may appear in your answer unless a \
finding below states it. Never fill a gap from general knowledge.
3. SHOW CALCULATIONS AS GIVEN. If a finding already shows a calculation (e.g. "75% of INR 4,000 \
= INR 3,000"), carry it through exactly -- do not redo the arithmetic yourself, and do not state \
only the final number.
4. DETECT CONFLICTS. If two findings from different departments cover the same topic (e.g. both \
give a number of nights for temporary accommodation during relocation) but disagree, this is a \
conflict. You will be given a list of CANDIDATE OVERLAPS to check -- confirm or dismiss each one.
5. RESOLVE CONFLICTS IN THIS ORDER:
   a) A clause that explicitly says it replaces or supersedes an earlier limit wins.
   b) Otherwise, the Finance policy prevails on financial benefits (Finance policy Section 1.3).
   c) Otherwise, the finding with the newer effective_date wins.
   d) If still unclear, present both values in the answer and tell the employee to confirm with \
the HR Business Partner.
   For every conflict you resolve, add an entry to `conflicts` naming both documents/sections, \
both values, which one applies, and which rule above you used.
6. REPORT GAPS. Anything no subagent could answer goes in `gaps`. Where it helps, point to a \
contact: the HR Business Partner (general HR topics not in the handbook, per HR Handbook \
Section 7.4), hr-help@kestrel.example (grievances), or the IT service desk at it.kestrel.example \
(IT topics).
7. If NO department was consulted (no findings at all were provided), say plainly that this is \
outside what the assistant covers, state in one line what it CAN help with (HR, IT, and Finance \
policy questions), and do not invent any content.
"""

_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "in", "on",
    "at", "per", "with", "as", "up", "be", "by", "may", "must", "will", "any",
    "days", "day", "within", "least",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def find_topic_overlaps(flat_findings: List[dict], min_shared_keywords: int = 2) -> List[dict]:
    """
    flat_findings: list of dicts with at least 'department' and 'rule'.
    Returns candidate pairs from DIFFERENT departments whose rule text shares
    enough keywords to plausibly be about the same topic.
    """
    overlaps = []
    for a, b in combinations(flat_findings, 2):
        if a["department"] == b["department"]:
            continue
        shared = _keywords(a["rule"]) & _keywords(b["rule"])
        if len(shared) >= min_shared_keywords:
            overlaps.append({"a": a, "b": b, "shared_keywords": sorted(shared)})
    return overlaps


def _flatten_findings(subagent_results: Dict[str, SubagentResult]) -> List[dict]:
    flat = []
    for dept, result in subagent_results.items():
        for f in result.findings:
            flat.append(
                {
                    "department": dept,
                    "rule": f.rule,
                    "section": f.section,
                    "page": f.page,
                    "document_id": f.document_id,
                    "version": f.version,
                    "effective_date": f.effective_date,
                }
            )
    return flat


def synthesize(
    model,
    question: str,
    profile: EmployeeProfile,
    subagent_results: Dict[str, SubagentResult],
    plan_reason: Optional[str] = None,
    return_usage: bool = False,
):
    flat_findings = _flatten_findings(subagent_results)
    all_gaps = [g for r in subagent_results.values() for g in r.gaps]
    overlaps = find_topic_overlaps(flat_findings)

    if not subagent_results:
        # Empty plan -- go straight to the out-of-scope reply, no LLM call needed.
        answer = FinalAnswer(
            answer_markdown=(
                "That's outside what I can help with -- I can only answer questions about "
                "Kestrel's HR, IT, and Finance/travel policies. Try asking about leave, VPN or "
                "device issues, expense limits, or relocation, for example."
            ),
            conflicts=[],
            gaps=[],
            departments_consulted=[],
        )
        return (answer, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}) if return_usage else answer

    prompt = (
        f"EMPLOYEE QUESTION: {question}\n\n"
        f"{profile.as_context_line()}\n\n"
        f"PLAN REASON: {plan_reason or 'n/a'}\n\n"
        f"FINDINGS (one per rule, already retrieved -- do not add to these):\n{flat_findings}\n\n"
        f"GAPS ALREADY REPORTED BY SUBAGENTS:\n{all_gaps}\n\n"
        f"CANDIDATE OVERLAPS TO CHECK FOR REAL CONFLICTS:\n{overlaps}\n\n"
        f"Write the final answer now."
    )

    structuring_model = model.with_structured_output(FinalAnswer, include_raw=True)
    system_content = SYNTHESIS_INSTRUCTIONS
    for skill_text in skills_for_question(question):
        system_content += f"\n\n--- ADDITIONAL FORMATTING SKILL (apply on top of the rules above) ---\n{skill_text}"
    messages = [SystemMessage(content=system_content), HumanMessage(content=prompt)]
    structured = structuring_model.invoke(messages)
    answer: FinalAnswer = structured["parsed"]
    answer.departments_consulted = list(subagent_results.keys())
    if not return_usage:
        return answer
    usage = getattr(structured.get("raw"), "usage_metadata", None) or {}
    return answer, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
