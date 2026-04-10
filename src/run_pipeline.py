#!/usr/bin/env python3
"""Orchestrate ETL, synthetic gold generation, vector index rebuild, and RAG evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_paths(root: str) -> None:
    os.chdir(root)
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _reset_chroma() -> None:
    from vector_store import CHROMA_PATH

    if os.path.isdir(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed vector index at {CHROMA_PATH}")


def _rebuild_index() -> None:
    from vector_store import FinancialRetriever

    retriever = FinancialRetriever()
    retriever.build_index()


def main() -> None:
    root = _project_root()
    _ensure_paths(root)

    from experiment_config import get_experiment, list_experiment_ids

    parser = argparse.ArgumentParser(
        description="Full pipeline: raw PDF → corpus → gold Q&A → RAG + evaluation.",
        epilog=(
            "Registered --experiment values: "
            + ", ".join(list_experiment_ids())
            + ". Chunking presets apply when ETL runs (not with --skip-etl unless corpus matches)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="PDF filename under data/raw/ (default: etl_pipeline.FILENAME)",
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Use existing data/processed/corpus.jsonl",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Use existing data/synthetic/gold_dataset.csv",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--gold-sample-size",
        type=int,
        default=None,
        help="Override SAMPLE_SIZE for gold generation",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Delete Chroma persist dir and re-ingest (e.g. corpus changed but skip-etl)",
    )
    parser.add_argument(
        "--skip-rebuild-index",
        action="store_true",
        help="After ETL, keep existing Chroma (unsafe if chunk UUIDs changed)",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate every row in the gold CSV",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Evaluate first N rows (overrides evaluator.TEST_LIMIT)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help="Registered RAG preset (chunking/top_k/prompt/strict-cite). See experiment_config.EXPERIMENTS.",
    )
    args = parser.parse_args()

    if args.rebuild_index and args.skip_rebuild_index:
        parser.error("--rebuild-index and --skip-rebuild-index are mutually exclusive")
    if args.eval_all and args.eval_limit is not None:
        parser.error("--eval-all and --eval-limit are mutually exclusive")

    try:
        experiment = get_experiment(args.experiment)
    except KeyError as e:
        parser.error(str(e))

    etl_ran = False
    if not args.skip_etl:
        from etl_pipeline import FILENAME as DEFAULT_PDF, run_etl

        pdf = args.pdf or DEFAULT_PDF
        if not run_etl(
            pdf_filename=pdf,
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            header_levels=experiment.header_levels(),
        ):
            sys.exit(1)
        etl_ran = True

    need_rebuild = bool(args.rebuild_index) or (
        etl_ran and not args.skip_rebuild_index
    )
    if need_rebuild:
        _reset_chroma()
        _rebuild_index()

    if not args.skip_gold:
        from generate_gold_dataset import run_gold_generation

        run_gold_generation(sample_size=args.gold_sample_size)

    if not args.skip_eval:
        from evaluator import run_evaluation_pipeline

        if args.eval_all:
            run_evaluation_pipeline(
                evaluate_all=True,
                experiment=experiment,
            )
        elif args.eval_limit is not None:
            run_evaluation_pipeline(
                test_limit=args.eval_limit,
                experiment=experiment,
            )
        else:
            run_evaluation_pipeline(experiment=experiment)

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
