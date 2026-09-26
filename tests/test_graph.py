"""
End-to-end tests of the compiled LangGraph app (Phase 6), using:
  - real Chroma collections built from the REAL policy chunks (Phase 1/2),
  - FakeEmbeddings (deterministic, no network) for the vector store,
  - FakeChatModel (heuristic, no network) for planning/subagents/synthesis.

These prove the graph's WIRING is correct -- routing, parallel fan-out, the
merge reducer, the empty-plan short-circuit, a disabled department, and a
real thread-level timeout -- independent of real model/embedding quality,
which is what eval/ (run with real OpenAI credentials) measures instead.
"""

import time

import pytest

import graph.workflow as workflow_module
from graph.workflow import build_graph
from ingest.build_collections import build_all_collections
from models import EmployeeProfile
from tests.fake_chat_model import FakeChatModel
from tests.fake_embeddings import FakeEmbeddings


@pytest.fixture()
def persist_dir(tmp_path):
    d = str(tmp_path / "chroma_graph_test")
    build_all_collections(persist_directory=d, embeddings=FakeEmbeddings())
    return d


def _make_app(persist_dir, disabled_departments=None, subagent_model=None):
    return build_graph(
        planning_model=FakeChatModel(),
        synthesis_model=FakeChatModel(),
        embeddings=FakeEmbeddings(),
        persist_directory=persist_dir,
        subagent_model=subagent_model or FakeChatModel(),
        disabled_departments=disabled_departments,
    )


def test_multi_department_question_fans_out_and_merges(persist_dir):
    app = _make_app(persist_dir)
    profile = EmployeeProfile(name="Asha", grade="G5", office="Pune", joining_date="2023-01-10")
    result = app.invoke(
        {
            "question": "I'm relocating from Pune to Mumbai next month. What happens to my laptop, access card, relocation allowance and leave?",
            "profile": profile,
            "chat_history": [],
        }
    )
    assert set(result["subagent_runs"].keys()) == {"hr", "it", "finance"}
    for dept, run in result["subagent_runs"].items():
        assert run["error"] is None
        assert run["department"] == dept
    assert result["final_answer"] is not None


def test_empty_plan_short_circuits_with_no_subagents_called(persist_dir):
    app = _make_app(persist_dir)
    profile = EmployeeProfile(name="Meera", grade="G3", office="Pune")
    result = app.invoke(
        {"question": "What is Kestrel's share price today?", "profile": profile, "chat_history": []}
    )
    assert result.get("subagent_runs", {}) == {}
    assert "outside" in result["final_answer"].answer_markdown.lower()


def test_disabled_department_produces_a_gap_not_a_crash(persist_dir):
    app = _make_app(persist_dir, disabled_departments={"it"})
    profile = EmployeeProfile(name="Dev", grade="G4", office="Mumbai")
    result = app.invoke(
        {
            "question": "I'm relocating and need my laptop and access card sorted, plus my relocation allowance.",
            "profile": profile,
            "chat_history": [],
        }
    )
    assert result["subagent_runs"]["it"]["error"] == "disabled"
    assert "disabled" in result["subagent_runs"]["it"]["result"].gaps[0].lower()
    # the other two departments were unaffected
    assert result["subagent_runs"]["hr"]["error"] is None
    assert result["subagent_runs"]["finance"]["error"] is None


class _SlowForHR(FakeChatModel):
    """Sleeps only when acting as the HR subagent, to test the real timeout path."""

    def _tool_step(self, messages, tools):
        system = messages[0].content if messages else ""
        if "HR policy specialist" in system:
            time.sleep(1.5)
        return super()._tool_step(messages, tools)


def test_slow_subagent_times_out_without_hanging_the_graph(persist_dir, monkeypatch):
    monkeypatch.setattr(workflow_module, "SUBAGENT_TIMEOUT_SECONDS", 0.3)
    app = _make_app(persist_dir, subagent_model=_SlowForHR())
    profile = EmployeeProfile(name="Priya", grade="G2", office="Bengaluru")

    started = time.time()
    result = app.invoke(
        {
            "question": "I'm relocating and need my laptop, access card, and relocation allowance sorted, and want to know my leave entitlement.",
            "profile": profile,
            "chat_history": [],
        }
    )
    elapsed = time.time() - started

    assert elapsed < 1.5, "the graph waited for the slow HR call instead of timing it out"
    assert result["subagent_runs"]["hr"]["error"] == "timeout"
    assert "did not respond within" in result["subagent_runs"]["hr"]["result"].gaps[0]
    # IT and Finance were not the slow one and should have completed normally
    assert result["subagent_runs"]["it"]["error"] is None
    assert result["subagent_runs"]["finance"]["error"] is None
    assert result["final_answer"] is not None
