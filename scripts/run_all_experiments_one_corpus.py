#!/usr/bin/env python3
"""
Run every registered RAG experiment on one PDF with a fixed eval limit.

Baseline chunk family shares one ETL + gold + index; chunk-preset experiments
each re-run ETL (rechunks cached MD) + gold + index, then eval.

Usage (repo root):
  python3 scripts/run_all_experiments_one_corpus.py --pdf Nvidia10k2026.pdf --corpus-id nvidia10k2026 --eval-limit 10

Env set per subprocess: RAG_CORPUS_JSONL, RAG_GOLD_CSV, RAG_CHROMA_SUBDIR, RAG_CORPUS_ID
Results: outputs/logs/{corpus_id}_{experiment_id}_results.csv
Log: outputs/logs/{corpus_id}_full_experiments_run.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Experiments that only change top_k / prompt / cite (same corpus as baseline ETL).
SAME_CHUNK_EXPERIMENTS = [
    "baseline",
    "k3",
    "k8",
    "k10",
    "baseline_cot",
    "baseline_strict_cite",
    "k8_cot",
]

# Chunking presets: each needs its own ETL + gold + index before eval.
CHUNK_VARIANT_EXPERIMENTS = ["fine_chunks", "coarse_chunks", "deep_headers_fine"]


def _env(corpus_id: str) -> dict:
    e = os.environ.copy()
    e["RAG_CORPUS_JSONL"] = f"data/processed/{corpus_id}/corpus.jsonl"
    e["RAG_GOLD_CSV"] = f"data/synthetic/gold_{corpus_id}.csv"
    e["RAG_CHROMA_SUBDIR"] = f"chroma_{corpus_id}"
    e["RAG_CORPUS_ID"] = corpus_id
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pdf",
        required=True,
        help="Filename under data/raw/ (e.g. Nvidia10k2026.pdf)",
    )
    ap.add_argument(
        "--corpus-id",
        required=True,
        help="Slug for paths and log prefixes (e.g. nvidia10k2026)",
    )
    ap.add_argument("--eval-limit", type=int, default=10)
    ap.add_argument(
        "--gold-sample-size",
        type=int,
        default=50,
        help="Chunks to attempt for gold (need enough rows after critique for eval-limit).",
    )
    args = ap.parse_args()

    corpus_id = args.corpus_id
    log_path = ROOT / "outputs" / "logs" / f"{corpus_id}_full_experiments_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def run_pipeline(argv: list[str]) -> int:
        cmd = [sys.executable, "src/run_pipeline.py", *argv]
        line = f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n"
        print(line, end="", flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line)
        r = subprocess.run(cmd, cwd=str(ROOT), env=_env(corpus_id))
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"exit={r.returncode}\n")
        return r.returncode

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n========== NEW RUN {datetime.now().isoformat()} ==========\n")
        lf.write(f"pdf={args.pdf} corpus_id={corpus_id} eval_limit={args.eval_limit}\n")

    # --- Wave 1: baseline ETL + gold + Chroma; no eval ---
    code = run_pipeline(
        [
            "--pdf",
            args.pdf,
            "--experiment",
            "baseline",
            "--skip-eval",
            "--gold-sample-size",
            str(args.gold_sample_size),
        ]
    )
    if code != 0:
        return code

    # --- Same-chunk experiments: eval only ---
    for exp in SAME_CHUNK_EXPERIMENTS:
        code = run_pipeline(
            [
                "--skip-etl",
                "--skip-gold",
                "--skip-rebuild-index",
                "--experiment",
                exp,
                "--eval-limit",
                str(args.eval_limit),
            ]
        )
        if code != 0:
            return code

    # --- Chunk-preset experiments: ETL + gold + index, then eval ---
    for exp in CHUNK_VARIANT_EXPERIMENTS:
        code = run_pipeline(
            [
                "--pdf",
                args.pdf,
                "--experiment",
                exp,
                "--skip-eval",
                "--gold-sample-size",
                str(args.gold_sample_size),
            ]
        )
        if code != 0:
            return code
        code = run_pipeline(
            [
                "--skip-etl",
                "--skip-gold",
                "--skip-rebuild-index",
                "--experiment",
                exp,
                "--eval-limit",
                str(args.eval_limit),
            ]
        )
        if code != 0:
            return code

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"========== COMPLETE {datetime.now().isoformat()} ==========\n")

    print(
        f"\nDone. Per-experiment CSVs: outputs/logs/{corpus_id}_<experiment>_results.csv\n"
        f"Run log: {log_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
