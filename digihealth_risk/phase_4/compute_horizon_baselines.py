"""Compute the random-classifier PR-AUC baseline per prediction horizon.

The Phase 4 shared-cohort leaderboard ranks models by PR-AUC within each
prediction horizon $N$. Because PR-AUC has no universal random-classifier
baseline (its random baseline equals the positive-class prevalence), absolute
PR-AUC values are not directly comparable across horizons. This script computes
the per-horizon baseline and the lift of each horizon's rank-1 model over it
so the leaderboard can be read with the correct reference point.

Inputs:
    digihealth_risk/phase_4/outputs/phase_4_2_v2_cross_family_ranking.csv

Outputs:
    digihealth_risk/phase_4/outputs/phase_4_horizon_baselines.csv

Run from the repository root:
    python digihealth_risk/phase_4/compute_horizon_baselines.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_2_v2_cross_family_ranking.csv"
OUTPUT = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_horizon_baselines.csv"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cross-family ranking missing: {INPUT}. "
            "Run Phase 4 (cross_family_comparison.py) before this script."
        )

    ranking = pd.read_csv(INPUT)
    required = {"horizon_years", "rows", "positives", "positive_rate", "pr_auc", "roc_auc"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Cross-family ranking is missing required columns: {sorted(missing)}")

    # Every row inside a horizon shares the same shared-cohort row/positive count;
    # take the first row per horizon for the baseline statistics.
    cohort = (
        ranking.groupby("horizon_years")
        .agg(
            shared_cohort_rows=("rows", "first"),
            positives=("positives", "first"),
            positive_rate=("positive_rate", "first"),
        )
        .reset_index()
        .rename(columns={"horizon_years": "N"})
    )

    # Rank-1 PR-AUC per horizon (the leaderboard winner).
    rank1 = (
        ranking.sort_values(["horizon_years", "pr_auc"], ascending=[True, False])
        .groupby("horizon_years")
        .first()
        .reset_index()
        [["horizon_years", "approach", "model_family", "history_years", "calibration_method", "pr_auc", "roc_auc"]]
        .rename(columns={
            "horizon_years": "N",
            "approach": "rank1_approach",
            "model_family": "rank1_family",
            "history_years": "rank1_M",
            "calibration_method": "rank1_calibration",
            "pr_auc": "rank1_pr_auc",
            "roc_auc": "rank1_roc_auc",
        })
    )

    out = cohort.merge(rank1, on="N", how="left")
    out["random_pr_auc"] = out["positive_rate"]  # by definition for the random classifier
    out["random_roc_auc"] = 0.5
    out["pr_auc_lift_over_random"] = out["rank1_pr_auc"] / out["random_pr_auc"]

    column_order = [
        "N",
        "shared_cohort_rows",
        "positives",
        "positive_rate",
        "random_pr_auc",
        "random_roc_auc",
        "rank1_approach",
        "rank1_family",
        "rank1_M",
        "rank1_calibration",
        "rank1_pr_auc",
        "rank1_roc_auc",
        "pr_auc_lift_over_random",
    ]
    out = out[column_order]
    out.to_csv(OUTPUT, index=False)

    # Pretty-print for the console / log
    print(f"Wrote {OUTPUT}\n")
    display = out.copy()
    display["positive_rate"] = display["positive_rate"].map(lambda x: f"{x*100:.2f}%")
    display["random_pr_auc"] = display["random_pr_auc"].map(lambda x: f"{x:.4f}")
    display["rank1_pr_auc"] = display["rank1_pr_auc"].map(lambda x: f"{x:.4f}")
    display["rank1_roc_auc"] = display["rank1_roc_auc"].map(lambda x: f"{x:.4f}")
    display["pr_auc_lift_over_random"] = display["pr_auc_lift_over_random"].map(lambda x: f"{x:.2f}x")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
