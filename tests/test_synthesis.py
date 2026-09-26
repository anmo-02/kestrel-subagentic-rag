from models import EmployeeProfile
from subagents.schemas import Finding, SubagentResult
from supervisor.synthesis import synthesize
from tests.fake_chat_model import FakeChatModel


def _hr_result_with_accommodation_finding() -> SubagentResult:
    return SubagentResult(
        department="hr",
        covered="yes",
        findings=[
            Finding(
                rule="On arrival at the new location, an employee may stay in a company-arranged guest house or hotel for up to 15 nights.",
                section="4.4",
                page=1,
                document_id="KSPL-HR-POL-004",
                version="4.1",
                effective_date="2026-04-01",
            )
        ],
        gaps=[],
    )


def _finance_result_with_accommodation_and_allowance() -> SubagentResult:
    return SubagentResult(
        department="finance",
        covered="yes",
        findings=[
            Finding(
                rule="Temporary accommodation during relocation is up to 10 nights at the grade hotel limit; this replaces any earlier limit, effective 1 July 2026.",
                section="6.4",
                page=2,
                document_id="KSPL-FIN-POL-003",
                version="3.0",
                effective_date="2026-07-01",
            ),
            Finding(
                rule="Relocation allowance for G5 is INR 75,000, paid with the first salary at the new location.",
                section="6.2",
                page=2,
                document_id="KSPL-FIN-POL-003",
                version="3.0",
                effective_date="2026-07-01",
            ),
        ],
        gaps=[],
    )


def test_conflict_is_flagged_when_hr_and_finance_disagree():
    profile = EmployeeProfile(name="Asha", grade="G5", office="Pune")
    subagent_results = {
        "hr": _hr_result_with_accommodation_finding(),
        "finance": _finance_result_with_accommodation_and_allowance(),
    }
    answer = synthesize(FakeChatModel(), "How many nights of temporary accommodation do I get?", profile, subagent_results)
    assert answer.conflicts, "expected the HR-4.4 / Finance-6.4 conflict to be flagged"
    conflict = answer.conflicts[0]
    assert "KSPL-HR-POL-004" in " ".join(conflict.documents)
    assert "KSPL-FIN-POL-003" in " ".join(conflict.documents)
    assert set(answer.departments_consulted) == {"hr", "finance"}


def test_no_conflict_when_only_one_department_answers():
    profile = EmployeeProfile(name="Ravi", grade="G2", office="Bengaluru")
    subagent_results = {
        "it": SubagentResult(
            department="it",
            covered="yes",
            findings=[
                Finding(
                    rule="Report a lost device within 24 hours by raising a P1 ticket.",
                    section="7.1",
                    page=1,
                    document_id="KSPL-IT-POL-002",
                    version="2.3",
                    effective_date="2026-01-01",
                )
            ],
            gaps=[],
        )
    }
    answer = synthesize(FakeChatModel(), "I lost my laptop, what do I do?", profile, subagent_results)
    assert answer.conflicts == []
    assert answer.departments_consulted == ["it"]


def test_empty_subagent_results_short_circuits_to_out_of_scope():
    profile = EmployeeProfile(name="Meera", grade="G3", office="Pune")
    answer = synthesize(FakeChatModel(), "What is Kestrel's share price today?", profile, {})
    assert "outside" in answer.answer_markdown.lower()
    assert answer.departments_consulted == []
    assert answer.conflicts == []
