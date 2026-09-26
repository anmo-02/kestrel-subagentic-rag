"""
Phase 9: run the 16-question acceptance test set through the subagent
design and produce an evaluation sheet (CSV) plus summary metrics.

Two modes:
  --mode real  (default) -- runs against real gpt-4o / gpt-4o-mini /
      text-embedding-3-small. Needs OPENAI_API_KEY in .env and a populated
      chroma_db/ (run `python -m ingest.build_collections` first). This is
      the mode that produces meaningful accuracy numbers for grading.
  --mode fake  -- runs the exact same harness against the deterministic
      FakeChatModel/FakeEmbeddings test doubles (tests/). This proves the
      HARNESS ITSELF is correct (routing comparison, citation-grounding
      check, conflict check, CSV writing) with no API key or network --
      it does NOT produce meaningful accuracy numbers, since the fake
      components don't reason. The CSV is labelled accordingly either way.

What's automated vs. left for you to judge:
  - departments chosen, time, and tokens are recorded exactly.
  - "citations grounded" is checked automatically: every [DOC-ID, section N]
    citation in the answer must correspond to an actual finding a subagent
    reported (this catches hallucinated citations; it can't catch a MISSING
    citation on a true rule, since we don't do independent fact extraction).
  - "conflict flagged" is checked automatically against expect_conflict.
  - "key points present" is a best-effort automatic substring check (the
    assignment itself says wording can differ) -- it's a starting point,
    not the final word. Review the `answer` and `missed_keypoints` columns
    yourself and use the `manual_override` column to correct any case where
    the substring check was too strict about phrasing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.test_questions import TEST_QUESTIONS, TestQuestion  # noqa: E402
from graph.workflow import build_graph  # noqa: E402
from ingest.build_collections import PERSIST_DIR, build_all_collections  # noqa: E402

CITATION_RE = re.compile(r"\[([A-Z0-9\-]+),\s*section\s*([\d.]+)", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9%.]+", " ", text.lower())


def keypoint_hits(answer_markdown: str, must_contain: list) -> tuple:
    norm_answer = _normalize(answer_markdown)
    hits, misses = [], []
    for point in must_contain:
        if _normalize(point) in norm_answer:
            hits.append(point)
        else:
            misses.append(point)
    return hits, misses


def citations_are_grounded(answer_markdown: str, subagent_runs: dict) -> bool:
    valid_pairs = set()
    for run in subagent_runs.values():
        for f in run["result"].findings:
            valid_pairs.add((f.document_id.upper(), f.section))
    cited = {(doc.upper(), sec) for doc, sec in CITATION_RE.findall(answer_markdown)}
    if not valid_pairs:
        return True  # nothing to cite (out-of-scope / fully uncovered question)
    if not cited:
        return False
    return cited.issubset(valid_pairs)


def total_tokens_for(result: dict) -> int:
    total = result.get("plan_tokens", {}).get("total_tokens", 0)
    total += result.get("synthesis_tokens", {}).get("total_tokens", 0)
    for run in result.get("subagent_runs", {}).values():
        total += run["tokens_used"]["total_tokens"]
    return total


def run_single(app, tq: TestQuestion) -> dict:
    start = time.time()
    result = app.invoke({"question": tq.question, "profile": tq.profile, "chat_history": []})
    elapsed = time.time() - start

    final = result["final_answer"]
    plan = result.get("plan")
    subagent_runs = result.get("subagent_runs", {})

    chosen_departments = sorted(plan.departments and [t.department for t in plan.departments] or [])
    expected_departments = sorted(tq.expected_departments)
    routing_correct = chosen_departments == expected_departments

    hits, misses = keypoint_hits(final.answer_markdown, tq.must_contain)
    coverage = len(hits) / len(tq.must_contain) if tq.must_contain else 1.0

    grounded = citations_are_grounded(final.answer_markdown, subagent_runs)
    conflict_flagged = bool(final.conflicts)
    conflict_ok = (conflict_flagged == tq.expect_conflict)

    return {
        "id": tq.id,
        "question": tq.question,
        "expected_departments": ",".join(expected_departments),
        "chosen_departments": ",".join(chosen_departments),
        "routing_correct": routing_correct,
        "keypoint_coverage": round(coverage, 2),
        "hit_keypoints": "; ".join(hits),
        "missed_keypoints": "; ".join(misses),
        "citations_grounded": grounded,
        "expect_conflict": tq.expect_conflict,
        "conflict_flagged": conflict_flagged,
        "conflict_handling_correct": conflict_ok,
        "time_seconds": round(elapsed, 2),
        "total_tokens": total_tokens_for(result),
        "answer": final.answer_markdown,
        "manual_override": "",  # blank -- fill in "pass"/"fail" if you disagree with the automatic keypoint check
    }


def summarize(rows: list) -> dict:
    n = len(rows)
    return {
        "questions": n,
        "routing_accuracy": round(sum(r["routing_correct"] for r in rows) / n, 3),
        "avg_keypoint_coverage": round(sum(r["keypoint_coverage"] for r in rows) / n, 3),
        "citation_grounding_rate": round(sum(r["citations_grounded"] for r in rows) / n, 3),
        "conflict_handling_accuracy": round(sum(r["conflict_handling_correct"] for r in rows) / n, 3),
        "avg_time_seconds": round(sum(r["time_seconds"] for r in rows) / n, 2),
        "avg_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
        "total_tokens": sum(r["total_tokens"] for r in rows),
    }


def _build_app(mode: str):
    if mode == "fake":
        from tests.fake_chat_model import FakeChatModel
        from tests.fake_embeddings import FakeEmbeddings

        persist_dir = os.path.join(os.path.dirname(__file__), "..", ".eval_fake_chroma")
        build_all_collections(persist_directory=persist_dir, embeddings=FakeEmbeddings())
        return build_graph(
            planning_model=FakeChatModel(),
            synthesis_model=FakeChatModel(),
            embeddings=FakeEmbeddings(),
            persist_directory=persist_dir,
            subagent_model=FakeChatModel(),
        )

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not os.path.exists(PERSIST_DIR):
        build_all_collections()
    return build_graph(
        planning_model=ChatOpenAI(model="gpt-4o", temperature=0),
        synthesis_model=ChatOpenAI(model="gpt-4o", temperature=0),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=PERSIST_DIR,
        subagent_model_name="gpt-4o-mini",
    )


def main():
    parser = argparse.ArgumentParser(description="Run the 16-question acceptance test set.")
    parser.add_argument("--mode", choices=["real", "fake"], default="real")
    parser.add_argument("--out", default=None, help="Output CSV path (default: eval/results/subagent_<mode>_<timestamp>.csv)")
    args = parser.parse_args()

    app = _build_app(args.mode)

    rows = [run_single(app, tq) for tq in TEST_QUESTIONS]
    summary = summarize(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join(out_dir, f"subagent_{args.mode}_{timestamp}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{args.mode}] Wrote {len(rows)} rows to {out_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if args.mode == "fake":
        print("\nNOTE: --mode fake exercises the harness only (no real reasoning). "
              "Run --mode real with an OpenAI key for meaningful accuracy numbers.")


if __name__ == "__main__":
    main()
