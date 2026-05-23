# Phase 6 — Deployment (FastAPI, dual-track)

## Thesis reference
Implements the dual-output deployment architecture argued in
`docs/thesis/06_DISCUSSION.md` §6.4 and `07_CONCLUSION_AND_FUTURE_WORK.md`
§7.1 — two complementary risk-scoring tracks built on a common
data-engineering foundation.

## Purpose
Exports 30 model artifacts and serves them via a stateless FastAPI:

| Track | Endpoint | Purpose | Source |
|-------|----------|---------|--------|
| Screening | `POST /predict` | Pure-prediction risk for passive screening | Thesis §5.1 leaderboard winners |
| Intervention | `POST /predict/interventions` | Intervention-safe what-if simulation | Thesis §5.4 monotonic winners |

The API accepts raw questionnaire + annual measurement inputs and returns a
risk score plus (for the intervention track) directionally safe what-if
scenarios. No patient ID, no patient lookup.

### Optional no-Year alternative

A parallel `/no_year/*` route tree exposes the phase_7 ablation variant
of the same 30 models, with `Year`, `Year_centered`, and `Year_centered_sq`
excluded from training and inference:

| Endpoint | Purpose | Source |
|----------|---------|--------|
| `POST /no_year/predict` | Screening, calendar-time invariant | Phase 7 ablation winners |
| `POST /no_year/predict/interventions` | Intervention, calendar-time invariant | Phase 7 ablation monotonic |

The two trees coexist — `/predict` represents the main thesis result and is
appropriate when the deployment cohort matches the 2005–2016 training window
or when retraining on fresh data is feasible. `/no_year/*` is the
construct-validity alternative for deployments without a retraining
pipeline: predictions become invariant to the calendar year a patient is
scored in, at a small documented PR-AUC cost (see
`digihealth_risk/phase_7/outputs/phase_7_year_ablation_report.md`).

## Per-horizon model family

| Horizon $N$ | Screening (`/predict`) | Intervention (`/predict/interventions`) |
|:-----------:|------------------------|-----------------------------------------|
| 1 | CatBoost | Monotonic EBM |
| 2 | Logistic | Monotonic CatBoost |
| 3 | XGBoost  | Monotonic XGBoost |
| 4 | Logistic | Monotonic CatBoost |
| 5 | Logistic\* | Monotonic CatBoost |

\* Thesis §5.1 screening winner at $N=5$ is GEE (PR-AUC 0.5282). statsmodels
GEE estimators do not serialize cleanly through joblib, so Logistic
(PR-AUC 0.5248 — a 0.0034 gap) is substituted for deployability. The
substitution is recorded explicitly in `outputs/model_registry.json`.

All 30 models are trained at history windows $M \in \{1, 3, 5\}$, matching
the surface area of the old deployment.

## Prerequisites
- `digihealth_risk/phase_0/outputs/` — all Phase 0 modeling tables for
  $N \in \{1..5\} \times M \in \{1,3,5\}$ must exist
- Python deps: `pandas numpy scikit-learn xgboost catboost interpret
  fastapi uvicorn joblib pydantic scipy`

## Step-by-step

### Step 1 — Export all 30 models
```bash
python digihealth_risk/phase_6/export_models.py
```
Writes:
- `outputs/models/{track}_{family}_n{N}_m{M}.joblib` — 30 artifacts
- `outputs/model_registry.json` — per-model metadata (track, family,
  horizon, history, threshold, test PR-AUC/ROC-AUC/Brier, intervention
  presets)
- `outputs/deployment_metrics.csv` — train/test metrics for all 30 fits

### Step 1b (optional) — Export the no-Year variant
```bash
python digihealth_risk/phase_6/export_models.py --no-year
```
Writes (parallel filenames so both variants coexist):
- `outputs/models_no_year/{track}_{family}_n{N}_m{M}.joblib`
- `outputs/model_registry_no_year.json`
- `outputs/deployment_metrics_no_year.csv`

Without this step, the `/no_year/*` endpoints will return 404. The main
`/predict` and `/predict/interventions` endpoints continue to work
regardless.

### Step 2 — Start the API server
```bash
uvicorn digihealth_risk.phase_6.api:app --reload --port 8000
```
Or directly:
```bash
python digihealth_risk/phase_6/api.py
```
Interactive docs: http://localhost:8000/docs

## Endpoints

### Main thesis tree (with-Year)

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Verify all 30 models are loaded; reports family-per-horizon for both tracks |
| GET  | `/models` | List every loaded artifact with track, family, horizon, history, threshold, feature count, presets |
| GET  | `/models/{key}` | Details for one artifact (e.g. `screening_catboost_n1_m5`, `intervention_ebm_n1_m5`) |
| POST | `/predict` | Screening track — returns risk score using the pure-prediction winner for the requested horizon |
| POST | `/predict/interventions` | Intervention track — returns baseline + per-preset scenarios using the monotonic winner |

### Construct-validity tree (no-Year, optional)

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/no_year/health` | Status + which Year features are excluded + rationale |
| GET  | `/no_year/models` | List of loaded no-Year artifacts |
| GET  | `/no_year/models/{key}` | Details for one no-Year artifact |
| POST | `/no_year/predict` | Screening track without Year features |
| POST | `/no_year/predict/interventions` | Intervention track without Year features |

The request body is identical across all `/predict*` endpoints; only the
routing-to-artifact and the calendar-time features differ. The intervention
endpoints also require a non-empty `presets` list.

The `/no_year/*` tree is only populated if you ran
`export_models.py --no-year`; otherwise it returns 404 with a hint pointing
back to the export command.

## Intervention presets

| Preset | Change |
|--------|--------|
| `reduce_sugary_to_zero` | `total_sugary_week = 0` |
| `reduce_sugary_50pct` | `total_sugary_week *= 0.5` |
| `increase_exercise_to_p75` | Exercise → training-set 75th percentile |
| `increase_activity_to_p75` | Physical activity → training-set 75th percentile |
| `increase_veg_fruit_to_p75` | Veg/fruit → training-set 75th percentile |
| `reduce_bmi_by_one` | `BMI -= 1` (clamped to training minimum) |
| `combined_lifestyle` | All favorable lifestyle changes together + BMI − 1 |

Each preset is stored inside every intervention-track artifact with
training-set-derived constants, so the API serves the same scenario
definitions the training cohort saw. The monotonic constraints guarantee
favorable changes never produce a higher risk score (thesis §5.4
"directionally correct rate = 1.0").

## Expected runtime
- Export (per variant): ~30–60 min wall clock (re-trains 30 models across the N×M grid).
- API startup: <10 sec for one variant, <20 sec if both `models/` and
  `models_no_year/` are populated.

## When to use each tree

| Situation | Recommended endpoint |
|---|---|
| Deployment cohort matches 2005–2016, or a periodic retraining pipeline on fresh data is in place | `/predict`, `/predict/interventions` (main thesis) |
| Deployment in 2024+ with no retraining pipeline, and same patient inputs must produce the same risk regardless of calendar year | `/no_year/predict`, `/no_year/predict/interventions` |
| Phase 7 reports a meaningful PR-AUC gap (e.g. N=1 horizon) and you can accept it for construct validity | `/no_year/*` for those horizons, `/predict*` for the others |

The phase_7 ablation report quantifies the trade-off per (horizon, history,
family). At N=3 and N=5 the two trees are within ~0.005 PR-AUC of each
other; at N=1 the no-Year variant loses ~0.02 PR-AUC.

## Differences vs legacy deployment

The original deployment used Monotonic
XGBoost uniformly for all 15 horizon/history combinations. That choice
predates the final thesis leaderboards and does not reflect the §5.1 /
§5.4 horizon-specific winners or the §6.4 dual-track argument. The
deployment in this directory replaces it.

## Notes

- The single substitution (Logistic for GEE at $N=5$) is the only place
  where the served model diverges from the thesis tables. It is recorded
  in `outputs/model_registry.json` under `tracks.screening.n5_substitution_note`.
- The intervention-track artifacts retain the same monotonic-constraint
  parameterization used in `digihealth_risk/phase_5/`, so the directional
  safety guarantees from the thesis §5.4 benchmark hold here too.
- Both tracks share the v2 feature engineering (FBS hinges,
  `Year_centered_sq`, `FBS_x_Age`, `MAX_FBS_x_Age`) from
  `digihealth_risk/phase_2/train_tree_models.py::engineer_features`.
- The `/no_year/*` tree uses the same module with
  `digihealth_risk/phase_7/year_ablation_utils.patch_drop_year_features()`
  applied — `Year`, `Year_centered`, and `Year_centered_sq` are removed at
  training time and not synthesized at inference. All other engineered
  features (FBS hinges, FBS × Age, MAX_FBS × Age) are identical between
  the two trees.
