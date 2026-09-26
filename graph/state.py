from __future__ import annotations

from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage

from models import EmployeeProfile
from supervisor.schemas import FinalAnswer, Plan


def merge_subagent_runs(a: Optional[Dict[str, dict]], b: Optional[Dict[str, dict]]) -> Dict[str, dict]:
    """
    Reducer for the parallel Send fan-out in Phase 6: each subagent node
    returns {department: run_dict} and this merges them together instead of
    the default "last write wins" behaviour, which would silently drop every
    department but the one that happened to finish last.
    """
    merged: Dict[str, dict] = dict(a or {})
    merged.update(b or {})
    return merged


class GraphState(TypedDict, total=False):
    question: str
    profile: EmployeeProfile
    chat_history: List[BaseMessage]
    plan: Optional[Plan]
    plan_tokens: dict
    subagent_runs: Annotated[Dict[str, dict], merge_subagent_runs]
    final_answer: Optional[FinalAnswer]
    synthesis_tokens: dict
