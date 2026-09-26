"""
Phase 6: connect planning -> parallel subagents -> synthesis with LangGraph.

Key mechanics:
  * `route_after_planning` returns a list of `Send` objects, one per
    department in the plan -- this is what makes every chosen subagent run
    concurrently instead of one after another. An empty plan routes straight
    to synthesis, which returns the out-of-scope reply without calling any
    subagent (Phase 6's "empty plan: skip the subagents" requirement).
  * `subagent_runs` uses the `merge_subagent_runs` reducer (graph/state.py)
    so results from parallel branches accumulate instead of overwriting each
    other.
  * Each subagent call is wrapped in a real thread-level timeout
    (SUBAGENT_TIMEOUT_SECONDS). If it doesn't return in time, the graph
    stops waiting and records a gap for that department -- it does not wait
    for the slow call or crash the run.
"""

from __future__ import annotations

import concurrent.futures
from typing import Dict, Optional, Set

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.state import GraphState
from subagents.finance import build_finance_subagent
from subagents.hr import build_hr_subagent
from subagents.it import build_it_subagent
from subagents.schemas import SubagentResult
from supervisor.planner import plan_question
from supervisor.schemas import Plan
from supervisor.synthesis import synthesize

SUBAGENT_TIMEOUT_SECONDS = 30.0

_BUILDERS = {
    "hr": build_hr_subagent,
    "it": build_it_subagent,
    "finance": build_finance_subagent,
}


def _disabled_run_result(department: str) -> dict:
    return {
        "department": department,
        "result": SubagentResult(
            department=department,
            covered="no",
            findings=[],
            gaps=[f"The {department} subagent is currently disabled and could not be checked."],
        ),
        "queries": [],
        "chunks_seen": [],
        "elapsed_seconds": 0.0,
        "tokens_used": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "error": "disabled",
    }


def _timeout_run_result(department: str) -> dict:
    return {
        "department": department,
        "result": SubagentResult(
            department=department,
            covered="no",
            findings=[],
            gaps=[f"The {department} subagent did not respond within {SUBAGENT_TIMEOUT_SECONDS:.0f}s and was skipped."],
        ),
        "queries": [],
        "chunks_seen": [],
        "elapsed_seconds": SUBAGENT_TIMEOUT_SECONDS,
        "tokens_used": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "error": "timeout",
    }


def build_graph(
    planning_model,
    synthesis_model,
    embeddings,
    persist_directory: str,
    subagent_model_name: str = "gpt-4o-mini",
    subagent_model=None,
    disabled_departments: Optional[Set[str]] = None,
):
    """
    Returns a compiled LangGraph app.

    Models and embeddings are injected (rather than constructed inside) so
    the test suite can pass fakes and exercise the real graph topology --
    fan-out, reducer, timeout, empty-plan short-circuit -- with no OpenAI
    key or network access. `subagent_model`, if given, overrides
    `subagent_model_name` for all three subagents (used by tests to inject
    a fake; production code leaves it as None and gets real gpt-4o-mini).
    """
    disabled_departments = disabled_departments or set()
    subagents = {
        dept: builder(embeddings, persist_directory, subagent_model_name, model=subagent_model)
        for dept, builder in _BUILDERS.items()
        if dept not in disabled_departments
    }

    def planning_node(state: GraphState) -> dict:
        plan, tokens = plan_question(
            planning_model, state["question"], state["profile"], state.get("chat_history"), return_usage=True
        )
        return {"plan": plan, "plan_tokens": tokens}

    def route_after_planning(state: GraphState):
        plan: Plan = state["plan"]
        if not plan.departments:
            return "synthesis"
        return [
            Send("run_subagent", {"department": task.department, "task_message": task.task_message})
            for task in plan.departments
        ]

    def run_subagent_node(payload: dict) -> dict:
        department = payload["department"]
        task_message = payload["task_message"]

        if department in disabled_departments:
            return {"subagent_runs": {department: _disabled_run_result(department)}}

        subagent = subagents[department]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(subagent.run, task_message)
        try:
            run_result = future.result(timeout=SUBAGENT_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            run_result = _timeout_run_result(department)
        finally:
            # Don't block graph progress waiting for an abandoned call to
            # finish in the background -- Python can't hard-kill a thread,
            # but we stop waiting on it, which is what "continue with the
            # rest" requires.
            executor.shutdown(wait=False)

        return {"subagent_runs": {department: run_result}}

    def synthesis_node(state: GraphState) -> dict:
        subagent_runs: Dict[str, dict] = state.get("subagent_runs", {})
        subagent_results: Dict[str, SubagentResult] = {dept: run["result"] for dept, run in subagent_runs.items()}
        plan: Optional[Plan] = state.get("plan")
        final, tokens = synthesize(
            synthesis_model,
            state["question"],
            state["profile"],
            subagent_results,
            plan_reason=plan.reason if plan else None,
            return_usage=True,
        )
        return {"final_answer": final, "synthesis_tokens": tokens}

    graph = StateGraph(GraphState)
    graph.add_node("planning", planning_node)
    graph.add_node("run_subagent", run_subagent_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "planning")
    graph.add_conditional_edges("planning", route_after_planning, ["run_subagent", "synthesis"])
    graph.add_edge("run_subagent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()
