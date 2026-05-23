# digihealth_risk: Modeling Pipeline (Phases 0-7)

End-to-end modeling pipeline for:
**"Longitudinal Diabetes Risk Prediction from 12-Year Thai Healthcare Data"**
(6,892 patients, 2005-2016, Thai NHES cohort)

This package is the production fork of the thesis `experiments_v2` tree. It
contains the curated v2 (feature-engineered) scripts for every phase, 0 through
7. The standalone serving slice lives in the sibling `deployment/` folder; see
the repository-root `README.md` for the overall layout.

---

## Prerequisites

```bash
# Python packages
pip install pandas numpy scikit-learn xgboost lightgbm catboost statsmodels \
            joblib fastapi uvicorn interpret

# Required data file (not committed — place in repo root)
datasets/df_final.pkl    # 5.6 MB, 6,892 patients, 121 columns
```

All commands must be run from the **repository root** (`longitudinal-diabetes-risk/`).

---

## Full run order

| Step | Phase | Script(s) | Key output | Thesis section |
|------|-------|-----------|------------|----------------|
| 1 | phase_0 | `build_modeling_tables.py` | 15 modeling tables | §3, §4.4.1 |
| 1a | phase_0 | `eda_depth.py` | Feature-engineering evidence | §3.4.2 |
| 2 | phase_1 | `gee_horizon_grid.py` | GEE predictions per N/M | §4.1.1 |
| 2 | phase_1 | `logistic_horizon_grid.py` | Logistic predictions per N/M | §4.1.1 |
| 2 | phase_1 | `glmm_exploratory.py` | GLMM (exploratory only) | §4.1.1 |
| 3 | phase_2 | `train_tree_models.py` | XGB/CB/LGBM predictions | §4.1.2 |
| 3 | phase_2 | `lmm_slope_features.py` | Slope-augmented table | §4.1.2 |
| 3 | phase_2 | `horizon_history_grid.py` | Full N×M grid | §5.2 |
| 4 | phase_3 | `landmark_cox.py` | Landmark Cox predictions | §4.1.3 |
| 4 | phase_3 | `two_stage_survival.py` (×3) | Two-stage M=1,3,5 predictions | §5.3 |
| 5 | phase_4 | `calibrate_trees.py` | Calibrated tree predictions | §4.2 |
| 5 | phase_4 | `threshold_optimization.py` | Threshold policy table | §4.4.4 |
| 5 | phase_4 | `cross_family_comparison.py` | **Final leaderboard** | §5.1 |
| 6 | phase_5 | `train_monotonic_*.py` (×5) | Monotonic model artifacts | §4.3 |
| 6 | phase_5 | `intervention_benchmark.py` | Intervention-safe leaderboard | §5.4 |
| 7 | phase_6 | `export_models.py` + `api.py` | REST API (15 models) | §7 |

Phase 0a (appendix EDA) can be run any time after Phase 0 Step 1.
Steps 2, 3, 4 can be run in any order — all depend only on Phase 0 outputs.
Step 5 depends on Steps 2, 3, and 4.
Step 6 depends on Step 5.

---

## Quick-start: reproduce the final leaderboard

The minimum set of commands to reproduce Table 5.1 (final leaderboard):

```bash
# 1. Build all modeling tables
for N in 1 2 3 4 5; do
  for M in 1 3 5; do
    python digihealth_risk/phase_0/build_modeling_tables.py \
      --horizon-years $N --history-years $M
  done
done

# 2. Statistical models (M=5 for leaderboard)
python digihealth_risk/phase_1/gee_horizon_grid.py --history-years 5
python digihealth_risk/phase_1/logistic_horizon_grid.py --history-years 5

# 3. Tree models (M=5 for leaderboard)
python digihealth_risk/phase_2/train_tree_models.py \
  --input-path digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_1_history_5.pkl
# Repeat for N=2..5 ...

# 4. Survival models
python digihealth_risk/phase_3/landmark_cox.py
python digihealth_risk/phase_3/two_stage_survival.py --history-window 3
python digihealth_risk/phase_3/two_stage_survival.py --history-window 5

# 5. Calibration and final comparison
python digihealth_risk/phase_4/calibrate_trees.py
python digihealth_risk/phase_4/threshold_optimization.py
python digihealth_risk/phase_4/cross_family_comparison.py
# → digihealth_risk/phase_4/outputs/phase_4_2_v2_cross_family_ranking.csv
```

---

## Output cascade (what each phase reads from prior phases)

```
datasets/df_final.pkl
  └── phase_0/outputs/
        ├── phase_0_modeling_table*.pkl   ← phase_1, phase_2, phase_4, phase_5
        └── patient_year_long.pkl         ← phase_3
              phase_1/outputs/
                └── *_test_predictions.csv ← phase_4
              phase_2/outputs/
                └── *_test_predictions.csv ← phase_4
              phase_3/outputs/
                └── *_test_predictions.csv ← phase_4
                    phase_4/outputs/
                      ├── phase_4_v2_test_predictions.csv ← phase_4 (threshold)
                      └── phase_4_2_v2_cross_family_ranking.csv ← phase_5
                            phase_5/outputs/
                              └── models_v2/ ← phase_6
```

---

## Canonical patient split

All phases share the same 60/20/20 patient-level split via `utils/patient_split.py`:
- Seed: `20260501`
- Split by `PatientId` (no temporal leakage)
- Cache: `digihealth_risk/phase_0/outputs/patient_split.csv` (auto-generated)

Import in any script:
```python
from digihealth_risk.utils.patient_split import apply_canonical_split
train_df, cal_df, test_df = apply_canonical_split(df, return_calibration=True)
```

---

## Primary metric
**PR-AUC** (precision-recall area under curve) — handles class imbalance and
prioritises detection of the at-risk minority class.

Secondary metrics: ROC-AUC, Brier score.

---

## Scope of this fork

This pipeline is forked from the thesis `experiments_v2` tree. The thesis
figure-generation scripts (`figures/`) are not included; every phase script
that produces a modeling result is. The original research repository retains
the full v1 development history.
