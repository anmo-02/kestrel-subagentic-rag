"""
Generates the two Phase 10 submission deliverables from real eval runs:

  1. eval/results/combined_evaluation_sheet_<ts>.csv -- one sheet, all 16
     questions, BOTH designs (a "design" column tells them apart), per the
     assignment's Section 9 ("Evaluation sheet: All 16 questions for both
     designs, with the metrics from Phase 9").
  2. eval/results/comparison_report_<ts>.md -- the one-page comparison
     report ("Subagent design vs single-agent baseline, with your
     conclusion"), with the summary table and two example questions filled
     in automatically from the real numbers, plus a conclusion draft you
     should read and edit -- it's a fair starting point, not a substitute
     for your own judgement.

Usage:
    python -m ingest.build_collections
    python -m eval.run_eval --mode real
    python -m eval.run_baseline_eval --mode real
    python -m eval.generate_comparison_report

By default it picks the MOST RECENT subagent_real_*.csv and baseline_real_*.csv
in eval/results/. Pass --subagent-csv / --baseline-csv to point at specific
files instead (e.g. to compare an older run).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _latest(pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} in {RESULTS_DIR}/ -- run eval/run_eval.py and "
            f"eval/run_baseline_eval.py first."
        )
    return matches[-1]


def _read_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _summary(rows: list) -> dict:
    n = len(rows)
    return {
        "questions": n,
        "routing_accuracy": round(sum(_to_bool(r["routing_correct"]) for r in rows) / n, 3),
        "avg_keypoint_coverage": round(sum(float(r["keypoint_coverage"]) for r in rows) / n, 3),
        "citation_grounding_rate": round(sum(_to_bool(r["citations_grounded"]) for r in rows) / n, 3),
        "conflict_handling_accuracy": round(sum(_to_bool(r["conflict_handling_correct"]) for r in rows) / n, 3),
        "avg_time_seconds": round(sum(float(r["time_seconds"]) for r in rows) / n, 2),
        "avg_tokens": round(sum(float(r["total_tokens"]) for r in rows) / n, 1),
        "total_tokens": round(sum(float(r["total_tokens"]) for r in rows), 0),
    }


def write_combined_sheet(subagent_rows: list, baseline_rows: list, out_path: str) -> None:
    fieldnames = ["design"] + list(subagent_rows[0].keys())
    baseline_dept_key = "chosen_departments (inferred from retrieved chunks)"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in subagent_rows:
            writer.writerow({"design": "subagent", **r})
        for r in baseline_rows:
            row = dict(r)
            row["chosen_departments"] = row.pop(baseline_dept_key, "")
            writer.writerow({"design": "baseline", **row})


def _pick_example_questions(subagent_rows: list, baseline_rows: list) -> list:
    by_id_sub = {r["id"]: r for r in subagent_rows}
    by_id_base = {r["id"]: r for r in baseline_rows}
    candidates = []
    for qid in by_id_sub:
        s, b = by_id_sub[qid], by_id_base.get(qid)
        if not b:
            continue
        diff_score = (
            (_to_bool(s["routing_correct"]) != _to_bool(b["routing_correct"]))
            + (_to_bool(s["conflict_handling_correct"]) != _to_bool(b["conflict_handling_correct"]))
            + abs(float(s["keypoint_coverage"]) - float(b["keypoint_coverage"]))
        )
        # The accommodation-conflict questions are the headline case the
        # assignment is built around -- surface them even on a small diff.
        if s.get("expect_conflict", "").strip().lower() == "true":
            diff_score += 0.5
        candidates.append((diff_score, qid, s, b))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[:2]


def write_comparison_report(subagent_rows: list, baseline_rows: list, out_path: str, mode_label: str = "real") -> None:
    sub_summary = _summary(subagent_rows)
    base_summary = _summary(baseline_rows)
    examples = _pick_example_questions(subagent_rows, baseline_rows)

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    lines = []
    lines.append("# Comparison Report: Subagentic RAG vs. Single-Agent Baseline\n")
    if mode_label != "real":
        lines.append(
            f"> ⚠️ **Generated from `--mode {mode_label}` data, not real model runs.** The numbers below "
            f"come from scripted/deterministic test doubles (see `tests/fake_chat_model.py`) used to "
            f"verify this harness works, not from GPT-4o or GPT-4o-mini. **Do not submit this as-is** -- "
            f"regenerate with `--mode-label real` after running `eval/run_eval.py --mode real` and "
            f"`eval/run_baseline_eval.py --mode real`.\n"
        )
    lines.append(
        "Both designs were run against the same 16-question acceptance test set, the same "
        "clause-level chunks, and the same embedding model (`text-embedding-3-small`) -- the only "
        "difference is architecture: three narrow subagents behind a planning/synthesis supervisor, "
        "versus one GPT-4o agent with one search tool over a single combined collection.\n"
    )

    lines.append("## Summary\n")
    lines.append("| Metric | Subagent design | Single-agent baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| Routing accuracy | {fmt_pct(sub_summary['routing_accuracy'])} | {fmt_pct(base_summary['routing_accuracy'])} |")
    lines.append(f"| Avg. key-point coverage | {fmt_pct(sub_summary['avg_keypoint_coverage'])} | {fmt_pct(base_summary['avg_keypoint_coverage'])} |")
    lines.append(f"| Citation grounding rate | {fmt_pct(sub_summary['citation_grounding_rate'])} | {fmt_pct(base_summary['citation_grounding_rate'])} |")
    lines.append(f"| Conflict-handling accuracy | {fmt_pct(sub_summary['conflict_handling_accuracy'])} | {fmt_pct(base_summary['conflict_handling_accuracy'])} |")
    lines.append(f"| Avg. time / question | {sub_summary['avg_time_seconds']}s | {base_summary['avg_time_seconds']}s |")
    lines.append(f"| Avg. tokens / question | {sub_summary['avg_tokens']} | {base_summary['avg_tokens']} |")
    lines.append(f"| Total tokens (16 questions) | {int(sub_summary['total_tokens'])} | {int(base_summary['total_tokens'])} |")
    lines.append("")

    lines.append("## Where the two designs differ most\n")
    for _, qid, s, b in examples:
        lines.append(f"**Q{qid}. {s['question']}**\n")
        lines.append(f"- Subagent: departments=`{s['expected_departments']}` chosen=`{s['chosen_departments']}`, "
                      f"routing {'correct' if _to_bool(s['routing_correct']) else 'WRONG'}, "
                      f"coverage {fmt_pct(float(s['keypoint_coverage']))}, "
                      f"conflict handling {'correct' if _to_bool(s['conflict_handling_correct']) else 'WRONG'}.")
        base_chosen = b.get("chosen_departments (inferred from retrieved chunks)", b.get("chosen_departments", "?"))
        lines.append(f"- Baseline: chosen (inferred)=`{base_chosen}`, "
                      f"routing {'correct' if _to_bool(b['routing_correct']) else 'WRONG'}, "
                      f"coverage {fmt_pct(float(b['keypoint_coverage']))}, "
                      f"conflict handling {'correct' if _to_bool(b['conflict_handling_correct']) else 'WRONG'}.")
        lines.append("")

    lines.append("## Conclusion (draft -- read and edit before submitting)\n")
    verdict_bits = []
    if sub_summary["routing_accuracy"] > base_summary["routing_accuracy"]:
        verdict_bits.append("routes multi-department questions more reliably")
    if sub_summary["conflict_handling_accuracy"] > base_summary["conflict_handling_accuracy"]:
        verdict_bits.append("catches the cross-document conflict more reliably")
    if sub_summary["citation_grounding_rate"] > base_summary["citation_grounding_rate"]:
        verdict_bits.append("grounds its citations more reliably")
    cost_note = (
        f"at {sub_summary['avg_tokens']} vs. {base_summary['avg_tokens']} average tokens/question "
        f"({'more' if sub_summary['avg_tokens'] > base_summary['avg_tokens'] else 'fewer'} than the baseline)"
    )
    if verdict_bits:
        lines.append(
            f"The subagent design {', and '.join(verdict_bits)} than the single-agent baseline on this "
            f"test set, {cost_note}. This matches the session's expectation: subagents cost more on simple "
            f"single-department questions but win on the multi-domain ones Kestrel's document ownership "
            f"structure actually produces (relocation and resignation questions split across HR/IT/Finance "
            f"by design)."
        )
    else:
        lines.append(
            f"On this run, the baseline was competitive with the subagent design on the automated metrics "
            f"above, {cost_note}. **Read the two example questions and the full per-question CSV before "
            f"accepting this** -- the automated keypoint/citation checks are a starting point, not a "
            f"substitute for actually reading a handful of answers side by side, especially on Q12/Q14 "
            f"(the accommodation-nights conflict), which is the case the subagent design's conflict "
            f"resolution exists for."
        )
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subagent-csv", default=None)
    parser.add_argument("--baseline-csv", default=None)
    parser.add_argument("--mode-label", default="real", help="Used only to pick the right filename pattern if --*-csv aren't given.")
    args = parser.parse_args()

    subagent_csv = args.subagent_csv or _latest(f"subagent_{args.mode_label}_*.csv")
    baseline_csv = args.baseline_csv or _latest(f"baseline_{args.mode_label}_*.csv")

    subagent_rows = _read_rows(subagent_csv)
    baseline_rows = _read_rows(baseline_csv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_suffix = "" if args.mode_label == "real" else f"_{args.mode_label}"
    combined_path = os.path.join(RESULTS_DIR, f"combined_evaluation_sheet{mode_suffix}_{timestamp}.csv")
    report_path = os.path.join(RESULTS_DIR, f"comparison_report{mode_suffix}_{timestamp}.md")

    write_combined_sheet(subagent_rows, baseline_rows, combined_path)
    write_comparison_report(subagent_rows, baseline_rows, report_path, mode_label=args.mode_label)

    print(f"Subagent results:  {subagent_csv}")
    print(f"Baseline results:  {baseline_csv}")
    print(f"Combined sheet:    {combined_path}")
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    main()
