"""
Phase 10: run the same 16-question acceptance test set through the
single-agent baseline (eval/baseline.py) and produce a directly comparable
evaluation sheet to eval/run_eval.py's subagent-design output.

Same two modes as run_eval.py: --mode real (needs an OpenAI key) or
--mode fake (harness smoke test only, see run_eval.py's module docstring
for why the fake numbers aren't meaningful).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.baseline import BaselineAgent, build_baseline_collection  # noqa: E402
from eval.run_eval import CITATION_RE, keypoint_hits  # noqa: E402
from eval.test_questions import TEST_QUESTIONS, TestQuestion  # noqa: E402
from ingest.build_collections import PERSIST_DIR  # noqa: E402


def citations_grounded_against_chunks(answer_markdown: str, chunks_seen: list) -> bool:
    valid_pairs = {(c["document_id"].upper(), c["section_number"]) for c in chunks_seen if c.get("document_id")}
    cited = {(doc.upper(), sec) for doc, sec in CITATION_RE.findall(answer_markdown)}
    if not valid_pairs:
        return True
    if not cited:
        return False
    return cited.issubset(valid_pairs)


def run_single(agent: BaselineAgent, tq: TestQuestion) -> dict:
    start = time.time()
    run = agent.run(tq.question, tq.profile)
    elapsed = time.time() - start

    final = run["final_answer"]
    chosen_departments = sorted(final.departments_consulted)
    expected_departments = sorted(tq.expected_departments)
    routing_correct = chosen_departments == expected_departments

    hits, misses = keypoint_hits(final.answer_markdown, tq.must_contain)
    coverage = len(hits) / len(tq.must_contain) if tq.must_contain else 1.0

    grounded = citations_grounded_against_chunks(final.answer_markdown, run["chunks_seen"])
    conflict_flagged = bool(final.conflicts)
    conflict_ok = conflict_flagged == tq.expect_conflict

    return {
        "id": tq.id,
        "question": tq.question,
        "expected_departments": ",".join(expected_departments),
        "chosen_departments (inferred from retrieved chunks)": ",".join(chosen_departments),
        "routing_correct": routing_correct,
        "keypoint_coverage": round(coverage, 2),
        "hit_keypoints": "; ".join(hits),
        "missed_keypoints": "; ".join(misses),
        "citations_grounded": grounded,
        "expect_conflict": tq.expect_conflict,
        "conflict_flagged": conflict_flagged,
        "conflict_handling_correct": conflict_ok,
        "time_seconds": round(elapsed, 2),
        "total_tokens": run["tokens"]["total_tokens"],
        "answer": final.answer_markdown,
        "manual_override": "",
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


def _build_agent(mode: str) -> BaselineAgent:
    if mode == "fake":
        from tests.fake_chat_model import FakeChatModel
        from tests.fake_embeddings import FakeEmbeddings

        persist_dir = os.path.join(os.path.dirname(__file__), "..", ".eval_fake_chroma")
        embeddings = FakeEmbeddings()
        build_baseline_collection(persist_dir, embeddings)
        return BaselineAgent(FakeChatModel(), embeddings, persist_dir)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists(PERSIST_DIR):
        build_baseline_collection(PERSIST_DIR, embeddings)
    return BaselineAgent(ChatOpenAI(model="gpt-4o", temperature=0), embeddings, PERSIST_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run the 16-question acceptance test set against the single-agent baseline.")
    parser.add_argument("--mode", choices=["real", "fake"], default="real")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    agent = _build_agent(args.mode)
    rows = [run_single(agent, tq) for tq in TEST_QUESTIONS]
    summary = summarize(rows)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join(out_dir, f"baseline_{args.mode}_{timestamp}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{args.mode}] Wrote {len(rows)} rows to {out_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
