#!/usr/bin/env python3
"""
Aggregate multi-corpus experiment results into CSVs + markdown tables.

Tier A: same chunking as baseline — fair Δ vs baseline across experiments.
Tier B: chunk presets — separate tables (different gold per preset).

Outputs under outputs/analysis/:
  tier_a_long.csv
  tier_a_delta_vs_baseline.csv
  tier_a_cross_corpus.csv
  tier_b_long.csv
  tier_b_by_corpus.csv
  summary_tables.md
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
OUT = ROOT / "outputs" / "analysis"

CORPORA = ["nvidia10k2026", "jpm10k2025", "humira_label", "keytruda_label"]

GENRE = {
    "nvidia10k2026": "finance_10k",
    "jpm10k2025": "finance_10k",
    "humira_label": "fda_label",
    "keytruda_label": "fda_label",
}

INDUSTRY = {
    "nvidia10k2026": "Finance (10-K)",
    "jpm10k2025": "Finance (10-K)",
    "humira_label": "Healthcare (FDA label)",
    "keytruda_label": "Healthcare (FDA label)",
}

TIER_A = [
    "baseline",
    "k3",
    "k8",
    "k10",
    "baseline_cot",
    "baseline_strict_cite",
    "k8_cot",
]

TIER_B = ["fine_chunks", "coarse_chunks", "deep_headers_fine"]

METRICS = [
    "recall_at_5",
    "citation_hallucination",
    "gold_chunk_cited",
    "context_hallucination",
    "claim_level_hallucination_rate",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def _read_results(corpus_id: str, experiment_id: str) -> pd.DataFrame | None:
    p = LOGS / f"{corpus_id}_{experiment_id}_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _row_means(df: pd.DataFrame, corpus_id: str, experiment_id: str) -> dict:
    out = {
        "corpus_id": corpus_id,
        "genre": GENRE.get(corpus_id, ""),
        "industry_label": INDUSTRY.get(corpus_id, ""),
        "experiment": experiment_id,
        "n_rows": len(df),
    }
    for m in METRICS:
        if m not in df.columns:
            out[m] = float("nan")
            out[m + "_n_nan"] = len(df)
        else:
            s = df[m]
            out[m] = s.mean(skipna=True)
            out[m + "_n_nan"] = int(s.isna().sum())
    return out


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        if isinstance(x, float):
            return float_fmt.format(x)
        return str(x)

    try:
        return df.to_markdown(floatfmt=float_fmt.replace("{:", "").replace("}", ""))
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Tier A long ---
    tier_a_rows = []
    for c in CORPORA:
        for e in TIER_A:
            df = _read_results(c, e)
            if df is None:
                continue
            tier_a_rows.append(_row_means(df, c, e))
    tier_a_long = pd.DataFrame(tier_a_rows)
    tier_a_long.to_csv(OUT / "tier_a_long.csv", index=False)

    tier_a_industry = (
        tier_a_long.groupby(["genre", "experiment"], as_index=False)[METRICS].mean()
        if not tier_a_long.empty
        else pd.DataFrame()
    )
    tier_a_industry.to_csv(OUT / "tier_a_pooled_by_industry.csv", index=False)

    # --- Tier A delta vs baseline (per corpus) ---
    delta_rows = []
    base = tier_a_long[tier_a_long["experiment"] == "baseline"].set_index("corpus_id")
    for c in tier_a_long["corpus_id"].unique():
        if c not in base.index:
            continue
        b = base.loc[c]
        for e in TIER_A:
            if e == "baseline":
                continue
            sub = tier_a_long[
                (tier_a_long["corpus_id"] == c) & (tier_a_long["experiment"] == e)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            dr = {
                "corpus_id": c,
                "genre": GENRE.get(c, ""),
                "industry_label": INDUSTRY.get(c, ""),
                "experiment": e,
            }
            for m in METRICS:
                key = f"delta_{m}"
                bv, ev = b.get(m), r.get(m)
                if pd.isna(bv) or pd.isna(ev):
                    dr[key] = float("nan")
                else:
                    dr[key] = float(ev) - float(bv)
            delta_rows.append(dr)
    tier_a_delta = pd.DataFrame(delta_rows)
    tier_a_delta.to_csv(OUT / "tier_a_delta_vs_baseline.csv", index=False)

    # --- Cross-corpus: for each experiment (not baseline), summarize each delta metric ---
    cross_rows = []
    if not tier_a_delta.empty:
        for e in sorted(tier_a_delta["experiment"].unique()):
            sub = tier_a_delta[tier_a_delta["experiment"] == e]
            row = {"experiment": e}
            for m in METRICS:
                col = f"delta_{m}"
                if col not in sub.columns:
                    continue
                s = sub[col].dropna()
                if s.empty:
                    row[f"{m}_mean_delta"] = float("nan")
                    row[f"{m}_min_delta"] = float("nan")
                    row[f"{m}_max_delta"] = float("nan")
                else:
                    row[f"{m}_mean_delta"] = s.mean()
                    row[f"{m}_min_delta"] = s.min()
                    row[f"{m}_max_delta"] = s.max()
            cross_rows.append(row)
    tier_a_cross = pd.DataFrame(cross_rows)
    tier_a_cross.to_csv(OUT / "tier_a_cross_corpus.csv", index=False)

    # --- Tier B long (separate; no delta vs tier A baseline) ---
    tier_b_rows = []
    for c in CORPORA:
        for e in TIER_B:
            df = _read_results(c, e)
            if df is None:
                continue
            tier_b_rows.append(_row_means(df, c, e))
    tier_b_long = pd.DataFrame(tier_b_rows)
    tier_b_long.to_csv(OUT / "tier_b_long.csv", index=False)

    # Tier B: mean per (corpus, experiment) already in long; add pooled by genre
    tier_b_by_corpus = tier_b_long.copy()
    tier_b_by_corpus.to_csv(OUT / "tier_b_by_corpus.csv", index=False)

    tier_b_pooled = (
        tier_b_long.groupby(["genre", "experiment"], as_index=False)[METRICS]
        .mean()
        if not tier_b_long.empty
        else pd.DataFrame()
    )
    tier_b_pooled.to_csv(OUT / "tier_b_pooled_by_genre.csv", index=False)

    # --- Markdown ---
    lines: list[str] = []
    lines.append("# Experiment analysis tables")
    lines.append("")
    lines.append(
        "Generated by `scripts/generate_analysis_tables.py`. "
        "**Tier A** = same chunking/gold as baseline (fair Δ vs baseline). "
        "**Tier B** = chunk presets with **separate gold** per preset — compare within Tier B, not to Tier A means."
    )
    lines.append("")

    lines.append("## Tier A — pooled by industry (2 corpora each)")
    lines.append("")
    lines.append(
        "**finance_10k** = Nvidia + JPM; **fda_label** = Humira + Keytruda. "
        "Use this slide for “finance vs healthcare” at a glance."
    )
    lines.append("")
    if not tier_a_industry.empty:
        for m in ["context_hallucination", "answer_correctness", "recall_at_5"]:
            piv = tier_a_industry.pivot_table(
                index="genre",
                columns="experiment",
                values=m,
                aggfunc="mean",
            )
            lines.append(f"### {m} (Tier A, pooled)")
            lines.append("")
            lines.append(_md_table(piv.round(4)))
            lines.append("")

    lines.append("## Tier A — mean metrics by corpus and experiment")
    lines.append("")
    if not tier_a_long.empty:
        pv = tier_a_long.pivot_table(
            index=["corpus_id", "genre"],
            columns="experiment",
            values="context_hallucination",
            aggfunc="mean",
        )
        lines.append("### context_hallucination (lower is better)")
        lines.append("")
        lines.append(_md_table(pv.round(4)))
        lines.append("")

        pv2 = tier_a_long.pivot_table(
            index=["corpus_id", "genre"],
            columns="experiment",
            values="answer_correctness",
            aggfunc="mean",
        )
        lines.append("### answer_correctness (higher is better)")
        lines.append("")
        lines.append(_md_table(pv2.round(4)))
        lines.append("")

    lines.append("## Tier A — Δ vs baseline (per corpus)")
    lines.append("")
    lines.append(
        "For hallucination metrics, **negative** Δ is improvement. "
        "For `answer_correctness` / `recall_at_5` / `faithfulness`, **positive** Δ is improvement."
    )
    lines.append("")
    if not tier_a_delta.empty:
        for m in ["context_hallucination", "answer_correctness", "recall_at_5"]:
            col = f"delta_{m}"
            if col not in tier_a_delta.columns:
                continue
            piv = tier_a_delta.pivot_table(
                index="corpus_id",
                columns="experiment",
                values=col,
                aggfunc="mean",
            )
            lines.append(f"### Δ {m}")
            lines.append("")
            lines.append(_md_table(piv.round(4)))
            lines.append("")

    lines.append("## Tier A — cross-corpus summary (mean of Δ across 4 corpora)")
    lines.append("")
    if not tier_a_cross.empty:
        slim_cols = [c for c in tier_a_cross.columns if "mean_delta" in c]
        slim = tier_a_cross[["experiment"] + slim_cols]
        lines.append(_md_table(slim.round(4)))
        lines.append("")

        # Heuristic rankings (directional)
        lines.append("### Directional ranking notes (Tier A only)")
        lines.append("")
        if "context_hallucination_mean_delta" in tier_a_cross.columns:
            best_h = tier_a_cross.loc[
                tier_a_cross["context_hallucination_mean_delta"].idxmin()
            ]
            worst_h = tier_a_cross.loc[
                tier_a_cross["context_hallucination_mean_delta"].idxmax()
            ]
            lines.append(
                f"- **Largest average reduction in context_hallucination vs baseline:** `{best_h['experiment']}` "
                f"(mean Δ = {best_h['context_hallucination_mean_delta']:.4f})"
            )
            lines.append(
                f"- **Largest average increase in context_hallucination vs baseline:** `{worst_h['experiment']}` "
                f"(mean Δ = {worst_h['context_hallucination_mean_delta']:.4f})"
            )
        if "answer_correctness_mean_delta" in tier_a_cross.columns:
            best_c = tier_a_cross.loc[
                tier_a_cross["answer_correctness_mean_delta"].idxmax()
            ]
            worst_c = tier_a_cross.loc[
                tier_a_cross["answer_correctness_mean_delta"].idxmin()
            ]
            lines.append(
                f"- **Largest average gain in answer_correctness vs baseline:** `{best_c['experiment']}` "
                f"(mean Δ = {best_c['answer_correctness_mean_delta']:.4f})"
            )
            lines.append(
                f"- **Largest average drop in answer_correctness vs baseline:** `{worst_c['experiment']}` "
                f"(mean Δ = {worst_c['answer_correctness_mean_delta']:.4f})"
            )
        lines.append("")

    lines.append("## Tier B — separate chunking experiments (do not compare Δ to Tier A)")
    lines.append("")
    if not tier_b_long.empty:
        for m in ["context_hallucination", "answer_correctness"]:
            piv = tier_b_long.pivot_table(
                index=["corpus_id", "genre"],
                columns="experiment",
                values=m,
                aggfunc="mean",
            )
            lines.append(f"### {m} (Tier B)")
            lines.append("")
            lines.append(_md_table(piv.round(4)))
            lines.append("")

        lines.append("### Tier B — pooled mean by genre (2 corpora per genre)")
        lines.append("")
        if not tier_b_pooled.empty:
            lines.append(_md_table(tier_b_pooled.round(4)))
            lines.append("")

    lines.append("## Files written")
    lines.append("")
    lines.append("- `tier_a_long.csv` — one row per (corpus × experiment), metric means.")
    lines.append("- `tier_a_pooled_by_industry.csv` — Tier A means pooled by `genre` (finance vs FDA labels).")
    lines.append("- `tier_a_delta_vs_baseline.csv` — Δ vs baseline per corpus.")
    lines.append("- `tier_a_cross_corpus.csv` — mean/min/max of Δ across corpora per experiment.")
    lines.append("- `tier_b_long.csv`, `tier_b_by_corpus.csv`, `tier_b_pooled_by_genre.csv` — Tier B only.")
    lines.append("")

    lines.append("## Using these files for presentation graphics")
    lines.append("")
    lines.append(
        "- **CSV → charts:** Import any of the CSVs into **Excel, Google Sheets, or Numbers** and insert "
        "**bar / clustered bar / heatmap** charts. `tier_a_pooled_by_industry.csv` and "
        "`tier_a_cross_corpus.csv` are the quickest for slides."
    )
    lines.append(
        "- **Python:** `pandas.read_csv` + **matplotlib** or **plotly** for Δ vs baseline **forest plots** "
        "(one row per corpus per experiment)."
    )
    lines.append(
        "- **This markdown:** Renders in GitHub, VS Code, or paste tables into **Slides/Keynote** as tables, "
        "or use as a script for figures."
    )
    lines.append(
        "- **Tier A vs B:** Use **separate figures** — do not put Tier A Δ-next-to-baseline on the same axis as Tier B raw means."
    )
    lines.append("")

    (OUT / "summary_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote CSVs and {OUT / 'summary_tables.md'}")


if __name__ == "__main__":
    main()
