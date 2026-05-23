# Phase 7 — Year-features Ablation

## What this ablation does

Retrains every model in phases 2, 4, and 5 with three calendar-time features
removed from the training set:

- `Year`
- `Year_centered`
- `Year_centered_sq`

All other features — including patient-relative time signals (`Age`,
`years_since_last_fbs`, `has_fbs_this_year`, `is_missing_last_year`) and
rolling history-window slopes — are retained.

The goal is to quantify how much the calendar-time features contribute to
in-sample predictive performance, and thereby inform whether they should be
dropped from the deployed model for construct validity (so the same patient
state scored in 2026 vs 2027 produces the same risk).

## Why this matters

The thesis baseline includes `Year_centered_sq` because Ljung-Box found mild
non-linear drift in population-level risk across the 2005–2016 training
window (p = 0.03). That justification holds *within* the training window,
but raises two problems at deployment:

1. **Construct validity**: two patients with identical features should not
   receive different risks merely because they are scored a year apart.
2. **OOD extrapolation**: when deployed in 2026, `Year_centered_sq` is far
   above any value seen in training (440+ vs a training max of 121).

Phase 7 produces the head-to-head numbers needed to decide whether the
in-sample gain is worth those deployment costs.

## How it works

`year_ablation_utils.py` monkey-patches the canonical `LEAKAGE_OR_METADATA_COLUMNS`
set and `engineer_features()` function in `digihealth_risk.phase_2.train_tree_models`.

After the patch, every downstream training entry point excludes the three
Year features. No phase 2/4/5 scripts are forked or copied.

## Layout

| File | Purpose |
|---|---|
| `year_ablation_utils.py` | Monkey-patch helper. Idempotent. |
| `train_trees_no_year.py` | Phase 2 uncalibrated tree grid (5 models × 5 horizons × 3 history). |
| `calibrate_trees_no_year.py` | Phase 4 calibrated tree grid (xgboost + catboost × 30 configs × 3 calibrators). |
| `train_monotonic_no_year.py` | Phase 5 monotonic families (xgboost, catboost, lightgbm, ebm, logistic) × 5 horizons, M = 5. |
| `compare_with_baseline.py` | Joins ablation outputs against baseline metrics and writes the report. |
| `run_all.sh` | Orchestrator. |

## Run

```bash
# Full ablation + comparison report
bash digihealth_risk/phase_7/run_all.sh

# Only re-generate the comparison report against existing outputs
bash digihealth_risk/phase_7/run_all.sh --report-only

# Run individual stages
python digihealth_risk/phase_7/train_trees_no_year.py
python digihealth_risk/phase_7/calibrate_trees_no_year.py
python digihealth_risk/phase_7/train_monotonic_no_year.py
python digihealth_risk/phase_7/compare_with_baseline.py
```

## Outputs

All written under `digihealth_risk/phase_7/outputs/`:

| File | Contents |
|---|---|
| `phase_7_no_year_trees_metrics.csv` | Test/train metrics, no-Year tree grid. |
| `phase_7_no_year_trees_test_predictions.csv` | Per-patient-year test probabilities. |
| `phase_7_no_year_calibration_metrics.csv` | Phase 4 calibrated metrics, no-Year. |
| `phase_7_no_year_calibration_final_recommendations.csv` | Best-of recommendations per horizon. |
| `phase_7_no_year_monotonic_metrics.csv` | Phase 5 monotonic metrics, no-Year. |
| `phase_7_compare_*.csv` | Side-by-side baseline vs no-Year deltas. |
| `phase_7_compare_summary.csv` | Aggregate deltas per comparison group. |
| `phase_7_year_ablation_report.md` | Final markdown report. |
| `no_year_monotonic_models/<family>/*.joblib` | Trained no-Year monotonic artifacts. |

## Reading the deltas

`delta_metric = no_year - baseline`.

- For PR-AUC and ROC-AUC: positive Δ = dropping Year helped.
- For Brier: positive Δ = dropping Year hurt (lower Brier is better).

If |Δ PR-AUC| < 0.005 across the grid, the Year features were doing very
little — the deployment-time fix is essentially free. If Δ is systematically
negative, the in-sample temporal-drift adjustment was load-bearing, and the
thesis needs to weigh that against the construct-validity argument in §6.

## Note on deployment

Phase 7 deliberately does **not** re-export phase 6 deployment artifacts.
This is a metrics-only ablation. If the deltas support dropping Year features
from production, modify `digihealth_risk/phase_6/api.py` and
`export_models.py` to omit `Year_centered` and `Year_centered_sq` from the
inference feature set, and re-run `export_models.py` against either the
existing phase 5 outputs (kept) or the phase 7 outputs (re-trained).
