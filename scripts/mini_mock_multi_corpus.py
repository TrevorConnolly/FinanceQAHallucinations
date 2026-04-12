#!/usr/bin/env python3
"""
Run a small end-to-end trial across multiple raw PDFs (per-corpus paths via env).

For each corpus: ETL + gold + index (baseline chunk preset), then eval a handful of experiments.

Env vars (set in subprocess):
  RAG_CORPUS_JSONL, RAG_GOLD_CSV, RAG_CHROMA_SUBDIR, RAG_CORPUS_ID

Usage (from repo root):
  python3 scripts/mini_mock_multi_corpus.py
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# pdf filename in data/raw/, slug for outputs (no spaces)
CORPORA: list[tuple[str, str, str]] = [
    ("nvidia10k2026", "Nvidia10k2026.pdf", "10k_sec"),
    ("jpm10k2025", "JPM10k2025.pdf", "10k_sec"),
    ("keytruda_label", "Keytruda_Label.pdf", "fda_label"),
    ("humira_label", "Humira_Label.pdf", "fda_label"),
]

# Subset of experiments (mock): baseline + a few knobs
EXPERIMENTS = ["baseline", "k8", "baseline_cot", "baseline_strict_cite"]

GOLD_SAMPLE = 10
EVAL_LIMIT = 2


def _env_for_corpus(corpus_id: str) -> dict:
    e = os.environ.copy()
    e["RAG_CORPUS_JSONL"] = f"data/processed/{corpus_id}/corpus.jsonl"
    e["RAG_GOLD_CSV"] = f"data/synthetic/gold_{corpus_id}.csv"
    e["RAG_CHROMA_SUBDIR"] = f"chroma_{corpus_id}"
    e["RAG_CORPUS_ID"] = corpus_id
    return e


def _run(args: list[str], env: dict) -> int:
    print("\n>>>", " ".join(args), flush=True)
    r = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
    )
    return r.returncode


def main() -> int:
    os.chdir(ROOT)
    for corpus_id, pdf, genre in CORPORA:
        print(f"\n{'=' * 60}\nCORPUS {corpus_id} ({genre})  pdf={pdf}\n{'=' * 60}")
        env = _env_for_corpus(corpus_id)

        # ETL + gold + Chroma index; no eval yet (baseline preset only drives chunking)
        code = _run(
            [
                "src/run_pipeline.py",
                "--pdf",
                pdf,
                "--experiment",
                "baseline",
                "--skip-eval",
                "--gold-sample-size",
                str(GOLD_SAMPLE),
            ],
            env,
        )
        if code != 0:
            print(f"ERROR: pipeline phase 1 failed for {corpus_id}", file=sys.stderr)
            return code

        for exp in EXPERIMENTS:
            env2 = _env_for_corpus(corpus_id)
            code = _run(
                [
                    "src/run_pipeline.py",
                    "--skip-etl",
                    "--skip-gold",
                    "--experiment",
                    exp,
                    "--eval-limit",
                    str(EVAL_LIMIT),
                ],
                env2,
            )
            if code != 0:
                print(f"ERROR: eval failed {corpus_id} {exp}", file=sys.stderr)
                return code

    print("\nAll mini-mock steps finished. Run: python3 scripts/aggregate_interaction_view.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
