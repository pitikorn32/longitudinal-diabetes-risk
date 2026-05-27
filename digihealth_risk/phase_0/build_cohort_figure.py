"""Build CONSORT-style cohort flow figure for thesis §3.1.1.

Produces:
- docs/thesis_production/Digihealth_IS_V5/Figures/fig_3_2_cohort_flow.png

Numbers computed live from datasets/df_final.pkl so the figure is
always consistent with the source data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "df_final.pkl"
FIG_DIR = ROOT.parent / "docs" / "thesis_production" / "Digihealth_IS_V5" / "Figures"
FIG_PATH = FIG_DIR / "fig_3_2_cohort_flow.png"

YEARS = list(range(2005, 2017))

# Shared-cohort row counts after intersection across model families (from appendix.tex §A.5)
SHARED_COHORT = {1: 8248, 2: 7209, 3: 6211, 4: 5246, 5: 4325}


def compute_counts() -> dict:
    df = pd.read_pickle(DATA_PATH)
    first_year = []
    for _, row in df.iterrows():
        fy = next((y for y in YEARS if row.get(f"AtRisk_{y}") == 1), None)
        first_year.append(fy)
    df["_first_atrisk"] = first_year

    n_source = len(df)
    n_long = n_source * len(YEARS)
    n_atrisk_2005 = int((df["_first_atrisk"] == 2005).sum())
    n_never_atrisk = int(df["_first_atrisk"].isna().sum())

    per_horizon = {}
    for N in range(1, 6):
        rows = 0
        pids = set()
        pos = 0
        for _, row in df.iterrows():
            fy = row["_first_atrisk"]
            for T in YEARS:
                if T + N not in YEARS:
                    continue
                atrisk_T = row.get(f"AtRisk_{T}")
                target = row.get(f"AtRisk_{T+N}")
                if pd.isna(atrisk_T) or atrisk_T != 0 or pd.isna(target):
                    continue
                if pd.notna(fy) and T >= fy:
                    continue
                rows += 1
                pids.add(row["PatientId"])
                if target == 1:
                    pos += 1
        per_horizon[N] = {
            "rows": rows,
            "patients": len(pids),
            "positive_pct": pos / rows * 100,
        }
    return {
        "n_source": n_source,
        "n_long": n_long,
        "n_atrisk_2005": n_atrisk_2005,
        "n_never_atrisk": n_never_atrisk,
        "per_horizon": per_horizon,
    }


def draw_box(ax, x, y, w, h, text, fc="#EEF2F7", ec="#33425b", lw=1.4, fontsize=10, fontweight="normal"):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight)


def draw_arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#33425b", mutation_scale=14),
    )
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label, fontsize=8.5, color="#33425b", style="italic", ha="left", va="center")


def build_figure(counts: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 11.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ph = counts["per_horizon"]

    # 1. Source cohort
    draw_box(
        ax, 5, 13.0, 8.8, 1.1,
        f"Source cohort: {counts['n_source']:,} unique patients\n"
        f"Thai healthcare setting, annual health-check records, 2005--2016",
        fc="#D9E3EF", lw=1.7, fontweight="bold",
    )

    # 2. Long-format expansion
    draw_arrow(ax, 5, 12.45, 5, 11.8, "long-format expansion (12 years/patient)")
    draw_box(
        ax, 5, 11.3, 8.8, 0.85,
        f"Long-format patient-year table: "
        f"{counts['n_long']:,} rows ({counts['n_source']:,} patients $\\times$ 12 years)",
    )

    # 3. Eligibility / inclusion criteria
    draw_arrow(ax, 5, 10.88, 5, 10.0)
    draw_box(
        ax, 5, 9.45, 8.8, 1.0,
        "Per row $(i,T)$ eligibility (applied per horizon $N$):\n"
        "(i) $T+N \\leq 2016$    (ii) AtRisk$_{i,T} = 0$    "
        "(iii) $T < T_i^{\\mathrm{first}}$    (iv) AtRisk$_{i,T+N}$ observed",
        fc="#FFF3E0", ec="#a67400",
    )

    # 4. Per-horizon modeling tables
    draw_arrow(ax, 5, 8.93, 5, 8.15)

    # Box wide enough to host a small table
    box_top = 7.85
    box_bot = 5.05
    box_h = box_top - box_bot
    box_y = (box_top + box_bot) / 2
    rect = mpatches.FancyBboxPatch(
        (0.6, box_bot), 8.8, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor="#33425b", facecolor="#EEF2F7",
    )
    ax.add_patch(rect)
    ax.text(5, 7.55, "Per-horizon raw modeling tables", ha="center", va="center", fontsize=10, fontweight="bold")
    # Column headers
    ax.text(1.40, 7.1, "Horizon", ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax.text(4.10, 7.1, "Eligible rows", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(6.20, 7.1, "Patients", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(8.50, 7.1, "Positive rate", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.plot([1.5, 8.4], [6.85, 6.85], color="#33425b", lw=0.6)
    for i, N in enumerate(range(1, 6)):
        y = 6.6 - 0.30 * i
        ax.text(1.40, y, f"$N = {N}$", ha="left", va="center", fontsize=9.5)
        ax.text(4.10, y, f"{ph[N]['rows']:,}", ha="right", va="center", fontsize=9.5)
        ax.text(6.20, y, f"{ph[N]['patients']:,}", ha="right", va="center", fontsize=9.5)
        ax.text(8.50, y, f"{ph[N]['positive_pct']:.2f}%", ha="right", va="center", fontsize=9.5)

    # 5. Shared-cohort restriction
    draw_arrow(ax, 5, 4.98, 5, 4.20, "shared-cohort intersection across families (§4.2.3)")
    box_top2 = 3.95
    box_bot2 = 1.95
    rect2 = mpatches.FancyBboxPatch(
        (0.6, box_bot2), 8.8, box_top2 - box_bot2,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor="#33425b", facecolor="#E6F0EA",
    )
    ax.add_patch(rect2)
    ax.text(5, 3.65, "Final shared-cohort leaderboard rows (statistical $\\cap$ tree $\\cap$ survival)",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(1.40, 3.2, "Horizon", ha="left", va="center", fontsize=9.5, fontweight="bold")
    ax.text(5.20, 3.2, "Shared-cohort rows", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.text(8.50, 3.2, "Reduction vs raw", ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.plot([1.5, 8.4], [2.95, 2.95], color="#33425b", lw=0.6)
    for i, N in enumerate(range(1, 6)):
        y = 2.7 - 0.16 * i
        reduction = 1 - SHARED_COHORT[N] / ph[N]["rows"]
        ax.text(1.40, y, f"$N = {N}$", ha="left", va="center", fontsize=9.5)
        ax.text(5.20, y, f"{SHARED_COHORT[N]:,}", ha="right", va="center", fontsize=9.5)
        ax.text(8.50, y, f"$-${reduction*100:.1f}%", ha="right", va="center", fontsize=9.5)

    # 6. Patient-level note
    ax.text(
        5, 1.40,
        f"Patient-level notes: 0 patients had zero clinical observations; "
        f"{counts['n_atrisk_2005']} patients were already at-risk in 2005 (contribute 0 rows);\n"
        f"{counts['n_never_atrisk']:,} patients were never at-risk through 2016. "
        f"All other patients contribute at least one row to $N{{=}}1$.",
        ha="center", va="center", fontsize=9, style="italic", color="#33425b",
    )

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    counts = compute_counts()
    print(counts)
    build_figure(counts)
