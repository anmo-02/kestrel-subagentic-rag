import csv

from eval.generate_comparison_report import _summary, write_combined_sheet, write_comparison_report

_FIELDS_SUB = [
    "id", "question", "expected_departments", "chosen_departments", "routing_correct",
    "keypoint_coverage", "hit_keypoints", "missed_keypoints", "citations_grounded",
    "expect_conflict", "conflict_flagged", "conflict_handling_correct", "time_seconds",
    "total_tokens", "answer", "manual_override",
]

_FIELDS_BASE = [
    "id", "question", "expected_departments", "chosen_departments (inferred from retrieved chunks)",
    "routing_correct", "keypoint_coverage", "hit_keypoints", "missed_keypoints",
    "citations_grounded", "expect_conflict", "conflict_flagged", "conflict_handling_correct",
    "time_seconds", "total_tokens", "answer", "manual_override",
]


def _row(fields, **overrides):
    base = {f: "" for f in fields}
    base.update(overrides)
    return base


def _subagent_rows():
    return [
        _row(
            _FIELDS_SUB, id="1", question="Q1", expected_departments="hr", chosen_departments="hr",
            routing_correct="True", keypoint_coverage="1.0", citations_grounded="True",
            expect_conflict="False", conflict_flagged="False", conflict_handling_correct="True",
            time_seconds="1.0", total_tokens="100", answer="ans1",
        ),
        _row(
            _FIELDS_SUB, id="14", question="How many nights...", expected_departments="finance,hr",
            chosen_departments="finance,hr", routing_correct="True", keypoint_coverage="1.0",
            citations_grounded="True", expect_conflict="True", conflict_flagged="True",
            conflict_handling_correct="True", time_seconds="2.0", total_tokens="300", answer="ans14",
        ),
    ]


def _baseline_rows():
    return [
        _row(
            _FIELDS_BASE, id="1", question="Q1", expected_departments="hr",
            **{"chosen_departments (inferred from retrieved chunks)": "hr,it"},
            routing_correct="False", keypoint_coverage="0.5", citations_grounded="False",
            expect_conflict="False", conflict_flagged="False", conflict_handling_correct="True",
            time_seconds="0.8", total_tokens="80", answer="base_ans1",
        ),
        _row(
            _FIELDS_BASE, id="14", question="How many nights...", expected_departments="finance,hr",
            **{"chosen_departments (inferred from retrieved chunks)": "finance"},
            routing_correct="False", keypoint_coverage="0.5", citations_grounded="False",
            expect_conflict="True", conflict_flagged="False", conflict_handling_correct="False",
            time_seconds="1.5", total_tokens="250", answer="base_ans14",
        ),
    ]


def test_summary_computes_correct_aggregates():
    summary = _summary(_subagent_rows())
    assert summary["questions"] == 2
    assert summary["routing_accuracy"] == 1.0
    assert summary["avg_tokens"] == 200.0


def test_combined_sheet_has_design_column_and_all_rows(tmp_path):
    out_path = str(tmp_path / "combined.csv")
    write_combined_sheet(_subagent_rows(), _baseline_rows(), out_path)

    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4
    designs = {r["design"] for r in rows}
    assert designs == {"subagent", "baseline"}
    # the baseline's differently-named department column is normalized to match the subagent's
    baseline_row = next(r for r in rows if r["design"] == "baseline" and r["id"] == "1")
    assert baseline_row["chosen_departments"] == "hr,it"


def test_report_flags_fake_mode_with_a_warning(tmp_path):
    out_path = str(tmp_path / "report.md")
    write_comparison_report(_subagent_rows(), _baseline_rows(), out_path, mode_label="fake")
    content = open(out_path, encoding="utf-8").read()
    assert "not real model runs" in content
    assert "Do not submit this as-is" in content


def test_report_omits_warning_in_real_mode_and_surfaces_the_conflict_question(tmp_path):
    out_path = str(tmp_path / "report.md")
    write_comparison_report(_subagent_rows(), _baseline_rows(), out_path, mode_label="real")
    content = open(out_path, encoding="utf-8").read()
    assert "not real model runs" not in content
    # Q14 is the expect_conflict=True question and should be surfaced as a differing example,
    # since the baseline gets it wrong (conflict_handling_correct=False) and the subagent gets it right.
    assert "Q14." in content
