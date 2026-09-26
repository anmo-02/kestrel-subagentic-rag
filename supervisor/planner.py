"""
Phase 5: the supervisor's planning step.

Decides which departments (zero to three) a question needs, and writes a
complete, standalone task message for each. Two safety nets sit on top of
the LLM's own plan, because leaving these entirely to the model's discretion
is exactly how the pilot's "missing parts" failure happened:

  1. Every task message is deterministically prefixed with the employee's
     full profile, regardless of whether the model remembered to restate it.
  2. If the question names a city, and a Finance task exists, we
     deterministically append whether that city is metro/non-metro (when we
     can tell), rather than trusting the model to notice.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from models import EmployeeProfile, METRO_CITIES, is_metro
from supervisor.schemas import Plan

PLANNER_INSTRUCTIONS = """You are the supervisor of Kestrel Systems' employee help-desk assistant.

Three specialist subagents are available, each reading ONLY its own document:

- hr: HR Policy Handbook -- working hours, hybrid/remote work, leave, internal transfer and
  relocation (leave + process, NOT the money), onboarding, resignation and exit, grievances.
- it: IT Services & Information Security Policy -- laptops, accounts, VPN and remote access, the
  service desk and ticket priorities, equipment during transfers, lost devices, exit, data and AI
  tool rules.
- finance: Travel, Expense & Relocation Policy -- expense claims, travel class, daily hotel/meal
  limits by grade and city, approvals, relocation MONEY (allowance, travel, accommodation,
  goods shifting, recovery on early resignation), travel advances, full and final settlement,
  corporate cards.

Rules for planning:
- A question can need one, two, or all three departments. Relocation and resignation/exit
  questions typically need more than one, because the topic is split across documents by design
  (e.g. relocation leave is HR, relocation money is Finance, equipment is IT).
- Every task_message must stand completely on its own: the subagent that receives it will see
  NOTHING else -- not this conversation, not the other departments' task messages, not their
  findings. If a detail matters, put it in the task_message.
- If the question does not relate to any of the three documents (e.g. it's a general knowledge
  question, small talk, or about something no Kestrel policy covers, such as company financial
  performance), return an EMPTY departments list. Do not force a question into a department just
  to have something to say.
- Do not merge two departments' worth of instructions into one task_message; give each department
  its own, focused on only what that department's document could answer.
"""


def _build_messages(question: str, profile: EmployeeProfile, chat_history: Optional[List[BaseMessage]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = [SystemMessage(content=PLANNER_INSTRUCTIONS)]
    if chat_history:
        # Only the last few turns -- enough for follow-up questions ("and what about IT?")
        messages.extend(chat_history[-6:])
    context = f"{profile.as_context_line()}\n\nEmployee question: {question}"
    messages.append(HumanMessage(content=context))
    return messages


_CITY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(METRO_CITIES, key=len, reverse=True)) + r"|Nashik|Goa|Dubai|Nagpur|Surat|Jaipur|Lucknow|Indore|Kochi)\b",
    re.IGNORECASE,
)


def _find_mentioned_city(question: str) -> Optional[str]:
    m = _CITY_PATTERN.search(question)
    return m.group(1) if m else None


def _apply_safety_nets(plan: Plan, question: str, profile: EmployeeProfile) -> Plan:
    city = _find_mentioned_city(question)
    metro_note = ""
    if city:
        metro = is_metro(city)
        if metro is True:
            metro_note = f"\n\nNote: {city} is a METRO city under Section 4.1."
        elif metro is False:
            metro_note = f"\n\nNote: {city} is a NON-METRO city under Section 4.1 (75% of the metro hotel limit applies)."

    for task in plan.departments:
        prefix = profile.as_context_line() + "\n\n"
        suffix = metro_note if task.department == "finance" else ""
        if profile.as_context_line() not in task.task_message:
            task.task_message = prefix + task.task_message
        if suffix and suffix.strip() not in task.task_message:
            task.task_message = task.task_message + suffix
    return plan


def plan_question(
    model,
    question: str,
    profile: EmployeeProfile,
    chat_history: Optional[List[BaseMessage]] = None,
    return_usage: bool = False,
):
    structuring_model = model.with_structured_output(Plan, include_raw=True)
    messages = _build_messages(question, profile, chat_history)
    structured = structuring_model.invoke(messages)
    plan: Plan = structured["parsed"]
    plan = _apply_safety_nets(plan, question, profile)
    if not return_usage:
        return plan
    usage = getattr(structured.get("raw"), "usage_metadata", None) or {}
    return plan, {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
