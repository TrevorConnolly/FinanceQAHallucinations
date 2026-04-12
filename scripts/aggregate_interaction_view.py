#!/usr/bin/env python3
"""
Build interaction-style tables from per-corpus experiment result CSVs.

Reads outputs/logs/{corpus_id}_{experiment}_results.csv and writes:
  outputs/logs/mock_interaction_long.csv
  outputs/logs/mock_interaction_delta_vs_baseline.csv
  outputs/logs/mock_interaction_summary.md

Usage (from repo root):
  python3 scripts/aggregate_interaction_view.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"

CORPUS_IDS = ["nvidia10k2026", "jpm10k2025", "keytruda_label", "humira_label"]
EXPERIMENTS = ["baseline", "k8", "baseline_cot", "baseline_strict_cite"]

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

GENRE = {
    "nvidia10k2026": "10k_sec",
    "jpm10k2025": "10k_sec",
    "keytruda_label": "fda_label",
    "humira_label": "fda_label",
}


def main() -> None:
    rows = []
    for cid in CORPUS_IDS:
        for exp in EXPERIMENTS:
            p = LOGS / f"{cid}_{exp}_results.csv"
            if not p.exists():
                print(f"skip missing: {p.name}")
                continue
            df = pd.read_csv(p)
            row = {
                "corpus_id": cid,
                "genre": GENRE.get(cid, ""),
                "experiment": exp,
                "n_rows": len(df),
            }
            for m in METRICS:
                if m not in df.columns:
                    row[m] = float("nan")
                else:
                    row[m] = df[m].mean(skipna=True)
            rows.append(row)

    long_df = pd.DataFrame(rows)
    long_path = LOGS / "mock_interaction_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path}")

    # Delta vs baseline (per corpus, per metric)
    delta_rows = []
    base = long_df[long_df["experiment"] == "baseline"].set_index("corpus_id")
    for cid in long_df["corpus_id"].unique():
        if cid not in base.index:
            continue
        b = base.loc[cid]
        for exp in EXPERIMENTS:
            if exp == "baseline":
                continue
            sub = long_df[(long_df["corpus_id"] == cid) & (long_df["experiment"] == exp)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            dr = {"corpus_id": cid, "genre": GENRE.get(cid, ""), "experiment": exp}
            for m in METRICS:
                if m not in b.index or m not in r:
                    dr[f"delta_{m}"] = float("nan")
                else:
                    bv, ev = b[m], r[m]
                    dr[f"delta_{m}"] = (
                        float(ev) - float(bv)
                        if pd.notna(ev) and pd.notna(bv)
                        else float("nan")
                    )
            delta_rows.append(dr)

    ddf: pd.DataFrame | None = None
    if delta_rows:
        ddf = pd.DataFrame(delta_rows)
        dpath = LOGS / "mock_interaction_delta_vs_baseline.csv"
        ddf.to_csv(dpath, index=False)
        print(f"Wrote {dpath}")

    if long_df.empty:
        print("No result rows; run mini_mock_multi_corpus.py first.")
        return

    def _pivot(metric: str) -> pd.DataFrame:
        return long_df.pivot_table(
            index="corpus_id",
            columns="experiment",
            values=metric,
            aggfunc="mean",
        )

    def _md_table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown()
        except ImportError:
            return "```\n" + df.to_string() + "\n```"

    key_metrics = [
        ("claim_level_hallucination_rate", "Lower is better (unsupported atomic claims / total)."),
        ("context_hallucination", "Lower is better (any unsupported factual content in passages)."),
        ("faithfulness", "Higher is better (RAGAS)."),
        ("citation_hallucination", "Lower is better (citation issues)."),
    ]

    md_lines = [
        "# Mock trial interaction view",
        "",
        "Means per (corpus_id × experiment). Compare columns across experiments; rows are corpora.",
        "",
    ]
    for metric, note in key_metrics:
        if metric not in long_df.columns:
            continue
        pv = _pivot(metric)
        md_lines.extend(
            [
                f"## {metric}",
                "",
                note,
                "",
                _md_table(pv),
                "",
            ]
        )

    # Delta vs baseline: mean by genre (10-K vs label) for selected metrics
    if ddf is not None and not ddf.empty:
        for metric in [
            "delta_claim_level_hallucination_rate",
            "delta_context_hallucination",
            "delta_faithfulness",
            "delta_citation_hallucination",
        ]:
            ddf2 = ddf.copy()
            if metric not in ddf2.columns:
                continue
            ddf2["_g"] = ddf2["genre"]
            sub = ddf2.dropna(subset=[metric])
            if sub.empty:
                continue
            by_genre = sub.groupby(["experiment", "_g"], as_index=False)[metric].mean()
            piv = by_genre.pivot(index="experiment", columns="_g", values=metric)
            md_lines.extend(
                [
                    f"## Δ vs baseline (mean): {metric}",
                    "",
                    "Positive delta on `faithfulness` is good; on hallucination/citation metrics, negative is good.",
                    "",
                    _md_table(piv),
                    "",
                ]
            )

    md_lines.extend(
        [
            "## Genre legend",
            "- `10k_sec`: SEC-style 10-K corpora",
            "- `fda_label`: FDA drug label PDFs",
            "",
        ]
    )
    mpath = LOGS / "mock_interaction_summary.md"
    mpath.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
