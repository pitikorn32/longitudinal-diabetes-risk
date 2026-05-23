# deployment: DigiHealth Risk Score API

Standalone production slice of the longitudinal diabetes-risk model: a dual-track
FastAPI service plus the script that trains and exports the model artifacts it
serves. This folder has no imports into the `digihealth_risk/` phase tree; the
modeling helpers it needs are vendored in `modeling.py` and `patient_split.py`,
so it builds and runs on its own.

## Contents

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app: `/predict`, `/predict/interventions`, `/no_year/*`, `/logistic_only/*` |
| `schemas.py` | Pydantic request/response models (the wire contract) |
| `export_models.py` | Trains and exports the 30 model artifacts |
| `modeling.py` | Vendored feature engineering, preprocessing, monotone rules |
| `patient_split.py` | Vendored canonical 60/20/20 patient split |
| `Dockerfile` | Container image for the API |
| `requirements.txt` | Pinned serving dependencies |

## 1. Export the models

`export_models.py` trains 30 artifacts (2 tracks x 5 horizons x 3 history
windows) from the 15 phase-0 modeling tables.

```bash
pip install -r requirements.txt
python export_models.py
```

By default it reads the modeling tables from the sibling phase tree
(`../digihealth_risk/phase_0/outputs/`). Build them first if they are missing:

```bash
cd ..
for N in 1 2 3 4 5; do for M in 1 3 5; do
  python digihealth_risk/phase_0/build_modeling_tables.py \
    --horizon-years "$N" --history-years "$M"
done; done
```

Override the input location with `DIGIHEALTH_PHASE0_DIR`. Artifacts are written
to this folder: `models/`, `model_registry.json`, `deployment_metrics.csv`.

Optional construct-validity variant (powers the `/no_year/*` routes):

```bash
python export_models.py --no-year
```

Optional logistic-only variant (powers the `/logistic_only/*` routes):

```bash
python export_models.py --logistic-only              # with-Year
python export_models.py --logistic-only --no-year    # no-Year
```

Each `--logistic-only` invocation writes **30 artifacts**: 15 screening
artifacts using sklearn logistic regression at every horizon, plus 15
intervention artifacts using monotonic-constrained logistic regression at
every horizon. Outputs go to `models_logistic_only/` or
`models_logistic_only_no_year/`.

### Why logistic-only?

The default `/predict` route returns a per-horizon winning family (CatBoost at
N=1, XGBoost at N=3, Logistic at N=2/4/5), which means the response
`model_family` field varies by horizon. The `/logistic_only/*` route tree was
added for frontend consumers that want a uniform single-family output for
easier client-side post-processing (e.g. coefficient-based explanations, linear
score decomposition). Screening trades roughly **0.020 PR-AUC at N=1 and N=3**
against the mixed-family default and is within ~0.001 at N=2/4/5 (where the
default already uses Logistic).

Intervention is also exposed under `/logistic_only/*` via
**monotonic-constrained logistic regression**: coefficient sign bounds enforced
at fit time guarantee that any favorable preset (reduce sugar, increase
exercise, etc.) cannot raise the predicted risk. The closed-form sigmoid
prediction path is identical to the unconstrained logistic; only the fit
differs. See `digihealth_risk/phase_5/train_monotonic_logistic.py` for the
modeling validation of this technique.

## 2. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

## 3. Docker

```bash
# Export the artifacts first so they are baked into the image:
python export_models.py

docker build -t digihealth-risk-api .
docker run -p 8000:8000 digihealth-risk-api

# Or keep the image artifact-free and serve models from the host:
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" digihealth-risk-api
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Loaded-model status per track |
| GET | `/models` | List loaded artifacts |
| GET | `/models/{key}` | One artifact's metadata |
| POST | `/predict` | Passive-screening risk score (mixed family per horizon) |
| POST | `/predict/interventions` | Intervention-safe what-if simulation |
| GET / POST | `/no_year/*` | The same route tree with Year features excluded |
| POST | `/logistic_only/predict` | Screening using logistic at every horizon (uniform single-family output) |
| POST | `/logistic_only/predict/interventions` | Intervention using monotonic-constrained logistic at every horizon |
| POST | `/logistic_only/no_year/predict` | Logistic-only screening, Year features excluded |
| POST | `/logistic_only/no_year/predict/interventions` | Monotonic-logistic intervention, Year features excluded |
| GET | `/logistic_only/health`, `/logistic_only/models`, `/logistic_only/models/{key}` | Logistic-only registry surface |
| GET | `/logistic_only/no_year/health`, `/logistic_only/no_year/models`, `/logistic_only/no_year/models/{key}` | Logistic-only no-Year registry surface |

The `/no_year/*` routes return 404 until `export_models.py --no-year` has been
run; the `/logistic_only/*` and `/logistic_only/no_year/*` routes return 404
until `export_models.py --logistic-only` and `--logistic-only --no-year` have
been run respectively. `/predict` and `/predict/interventions` work regardless.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DIGIHEALTH_MODEL_DIR` | `./models` | With-Year mixed-family artifacts |
| `DIGIHEALTH_MODEL_DIR_NO_YEAR` | `./models_no_year` | No-Year mixed-family artifacts |
| `DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY` | `./models_logistic_only` | With-Year logistic-only artifacts |
| `DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY_NO_YEAR` | `./models_logistic_only_no_year` | No-Year logistic-only artifacts |
| `DIGIHEALTH_PHASE0_DIR` | `../digihealth_risk/phase_0/outputs` | Modeling tables for export |
| `DIGIHEALTH_DATA` | `../datasets/df_final.pkl` | Source cohort for the split |
| `DIGIHEALTH_SPLIT_CACHE` | `../digihealth_risk/phase_0/outputs/patient_split.csv` | Canonical split cache |
