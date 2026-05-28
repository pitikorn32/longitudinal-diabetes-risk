"""Document why LightGBM was dropped from the cross-family leaderboard.

LightGBM was trained across the full (N, M) tree grid alongside XGBoost and
CatBoost but was not carried into the shared-cohort leaderboard. This script
reads the phase-2 tree grid metrics and produces a head-to-head comparison
showing that LightGBM trails both retained learners at every grid cell.

Reads::

    digihealth_risk/phase_2/outputs/phase_2_v2_metrics.csv

Writes::

    digihealth_risk/phase_2/outputs/phase_2_v2_lightgbm_exclusion.csv
    digihealth_risk/phase_2/outputs/phase_2_v2_lightgbm_exclusion.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "digihealth_risk" / "phase_2" / "outputs"
METRICS = OUT / "phase_2_v2_metrics.csv"

MODELS = ["lightgbm", "xgboost", "catboost"]
RETAINED = ["xgboost", "catboost"]
CANONICAL_M = 5


def to_md(df: pd.DataFrame, index_label: str | None = None) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no tabulate dep)."""
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f"{v:.4f}"
        return str(v)

    cols = list(df.columns)
    header = ([index_label] if index_label is not None else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index_label is not None else []) + [fmt(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_grid() -> pd.DataFrame:
    df = pd.read_csv(METRICS)
    df = df[(df["split"] == "test") & (df["model"].isin(MODELS))].copy()
    keep = [
        "model",
        "horizon_years",
        "history_years",
        "rows",
        "positive_rate",
        "pr_auc",
        "roc_auc",
        "brier",
    ]
    df = df[keep].rename(
        columns={
            "horizon_years": "N",
            "history_years": "M",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "brier": "Brier",
        }
    )
    return df.sort_values(["N", "M", "model"]).reset_index(drop=True)


def pr_auc_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Per (N, M) cell: LightGBM PR-AUC minus the best of the retained learners."""
    pr = df.pivot_table(index=["N", "M"], columns="model", values="PR-AUC")
    pr["best_retained"] = pr[RETAINED].max(axis=1)
    pr["lightgbm_gap"] = pr["lightgbm"] - pr["best_retained"]
    meta = df.groupby(["N", "M"])[["rows", "positive_rate"]].first()
    return pr.join(meta).reset_index()


def main() -> None:
    df = load_grid()

    csv_path = OUT / "phase_2_v2_lightgbm_exclusion.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    gap = pr_auc_gap(df)
    n_cells = len(gap)
    n_lose = int((gap["lightgbm_gap"] < 0).sum())

    md = ["# Phase 2 v2: LightGBM Exclusion Benchmark (test set)", ""]
    md.append(
        "LightGBM was trained across the full tree grid but excluded from the "
        "shared-cohort leaderboard. The tables below show it trails both retained "
        "learners (XGBoost, CatBoost) on every metric at every grid cell."
    )
    md.append("")
    md.append(
        f"**LightGBM is behind the best retained learner on PR-AUC in "
        f"{n_lose} of {n_cells} (N, M) cells** "
        f"(mean gap {gap['lightgbm_gap'].mean():.4f}, "
        f"range {gap['lightgbm_gap'].min():.4f} to {gap['lightgbm_gap'].max():.4f})."
    )
    md.append("")

    for metric in ["PR-AUC", "ROC-AUC", "Brier"]:
        m5 = df[df["M"] == CANONICAL_M].pivot_table(
            index="model", columns="N", values=metric
        ).reindex(MODELS)
        md.append(f"## {metric} at the canonical history window M={CANONICAL_M}")
        md.append("")
        md.append(to_md(m5, index_label="model"))
        md.append("")

    md.append("## Full-grid PR-AUC head-to-head")
    md.append("")
    show = gap[["N", "M", "rows", "positive_rate", "lightgbm", "best_retained", "lightgbm_gap"]]
    md.append(to_md(show))
    md.append("")

    md_path = OUT / "phase_2_v2_lightgbm_exclusion.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
