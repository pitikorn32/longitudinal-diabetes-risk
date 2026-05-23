"""DigiHealth Diabetes Risk Score API: dual-track deployment.

Stateless, anonymous risk scoring with two complementary tracks:

    /predict                 Passive-screening track. Uses the best
                             pure-prediction model family per horizon.
    /predict/interventions   Active-simulation track. Uses the best
                             intervention-safe monotonic model family per
                             horizon, so favorable lifestyle changes can
                             never increase the returned risk score.

The two tracks share the same input schema (raw questionnaire plus annual
measurements) and the same server-side feature engineering. They differ only in
which artifact is selected for scoring.

A parallel /no_year/* route tree exposes the year-ablation variant: identical
model families and preprocessing, but with Year, Year_centered and
Year_centered_sq excluded from training and inference.

A /logistic_only/* route tree (with a nested /logistic_only/no_year/* tree)
serves a uniform logistic-regression screening model at every horizon, for
frontend consumers that want a single-family output for easier client-side
post-processing. Intervention scoring is not exposed under /logistic_only/*.

This is the standalone deployment slice. The model artifacts it serves are
produced by export_models.py in this folder.

Start the server (from this folder):
    uvicorn api:app --reload --port 8000

Or run directly:
    python api.py

Model directories default to ./models, ./models_no_year,
./models_logistic_only, and ./models_logistic_only_no_year; override with
the DIGIHEALTH_MODEL_DIR, DIGIHEALTH_MODEL_DIR_NO_YEAR,
DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY, and
DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY_NO_YEAR environment variables.

Endpoints:
    GET  /health
    GET  /models
    GET  /models/{key}
    POST /predict
    POST /predict/interventions
    GET  /no_year/health
    GET  /no_year/models
    GET  /no_year/models/{key}
    POST /no_year/predict
    POST /no_year/predict/interventions
    GET  /logistic_only/health
    GET  /logistic_only/models
    GET  /logistic_only/models/{key}
    POST /logistic_only/predict
    GET  /logistic_only/no_year/health
    GET  /logistic_only/no_year/models
    GET  /logistic_only/no_year/models/{key}
    POST /logistic_only/no_year/predict
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam
from scipy import special

from schemas import (
    ClinicalMeasurement,
    InterventionRequest,
    InterventionResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RiskResult,
    ScenarioResult,
)


HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("DIGIHEALTH_MODEL_DIR", str(HERE / "models")))
MODEL_DIR_NO_YEAR = Path(os.environ.get("DIGIHEALTH_MODEL_DIR_NO_YEAR", str(HERE / "models_no_year")))
MODEL_DIR_LOGISTIC_ONLY = Path(os.environ.get(
    "DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY", str(HERE / "models_logistic_only")
))
MODEL_DIR_LOGISTIC_ONLY_NO_YEAR = Path(os.environ.get(
    "DIGIHEALTH_MODEL_DIR_LOGISTIC_ONLY_NO_YEAR", str(HERE / "models_logistic_only_no_year")
))

CLINICAL_FEATURES = ["FBS", "BMI", "Pulse", "BL_pres1", "BL_pres2", "Waist"]
YEAR_REFERENCE = 2005  # min(Year) in training data

TRACK_SCREENING = "screening"
TRACK_INTERVENTION = "intervention"

VARIANT_WITH_YEAR = "with_year"
VARIANT_NO_YEAR = "no_year"
VARIANT_LOGISTIC_ONLY_WITH_YEAR = "logistic_only_with_year"
VARIANT_LOGISTIC_ONLY_NO_YEAR = "logistic_only_no_year"

# Per-horizon model family per track. Must match export_models.py.
SCREENING_FAMILY = {1: "catboost", 2: "logistic", 3: "xgboost", 4: "logistic", 5: "logistic"}
INTERVENTION_FAMILY = {1: "ebm", 2: "catboost", 3: "xgboost", 4: "catboost", 5: "catboost"}

# Logistic-only screening track: logistic at every horizon. Frontend-driven.
LOGISTIC_ONLY_SCREENING_FAMILY = {n: "logistic" for n in (1, 2, 3, 4, 5)}

# Loaded at startup. Keyed by model_key (e.g. "screening_catboost_n1_m5").
_models: dict[str, dict[str, Any]] = {}
_models_no_year: dict[str, dict[str, Any]] = {}
_models_logistic_only: dict[str, dict[str, Any]] = {}
_models_logistic_only_no_year: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _expected_keys() -> list[str]:
    keys: list[str] = []
    for track, family_map in ((TRACK_SCREENING, SCREENING_FAMILY), (TRACK_INTERVENTION, INTERVENTION_FAMILY)):
        for history in (1, 3, 5):
            for horizon in (1, 2, 3, 4, 5):
                family = family_map[horizon]
                keys.append(f"{track}_{family}_n{horizon}_m{history}")
    return keys


def _load_all_models() -> None:
    for key in _expected_keys():
        path = MODEL_DIR / f"{key}.joblib"
        if path.exists():
            _models[key] = joblib.load(path)
        else:
            print(f"WARNING: model not found — {path}")
    expected = len(_expected_keys())
    print(f"Loaded {len(_models)}/{expected} with-Year models.")


def _load_all_models_no_year() -> None:
    if not MODEL_DIR_NO_YEAR.exists():
        print(
            f"INFO: no-Year model directory missing — {MODEL_DIR_NO_YEAR}. "
            "Run `python export_models.py --no-year` "
            "to enable /no_year/* endpoints."
        )
        return
    for key in _expected_keys():
        path = MODEL_DIR_NO_YEAR / f"{key}.joblib"
        if path.exists():
            _models_no_year[key] = joblib.load(path)
        else:
            print(f"WARNING: no-Year model not found — {path}")
    expected = len(_expected_keys())
    print(f"Loaded {len(_models_no_year)}/{expected} no-Year models.")


def _expected_keys_logistic_only() -> list[str]:
    """Screening-only key set: logistic at every horizon, all history windows."""
    keys: list[str] = []
    for history in (1, 3, 5):
        for horizon in (1, 2, 3, 4, 5):
            keys.append(f"{TRACK_SCREENING}_logistic_n{horizon}_m{history}")
    return keys


def _load_all_models_logistic_only() -> None:
    if not MODEL_DIR_LOGISTIC_ONLY.exists():
        print(
            f"INFO: logistic-only model directory missing — {MODEL_DIR_LOGISTIC_ONLY}. "
            "Run `python export_models.py --logistic-only` "
            "to enable /logistic_only/* endpoints."
        )
        return
    for key in _expected_keys_logistic_only():
        path = MODEL_DIR_LOGISTIC_ONLY / f"{key}.joblib"
        if path.exists():
            _models_logistic_only[key] = joblib.load(path)
        else:
            print(f"WARNING: logistic-only model not found — {path}")
    expected = len(_expected_keys_logistic_only())
    print(f"Loaded {len(_models_logistic_only)}/{expected} logistic-only models.")


def _load_all_models_logistic_only_no_year() -> None:
    if not MODEL_DIR_LOGISTIC_ONLY_NO_YEAR.exists():
        print(
            f"INFO: logistic-only no-Year model directory missing — "
            f"{MODEL_DIR_LOGISTIC_ONLY_NO_YEAR}. "
            "Run `python export_models.py --logistic-only --no-year` "
            "to enable /logistic_only/no_year/* endpoints."
        )
        return
    for key in _expected_keys_logistic_only():
        path = MODEL_DIR_LOGISTIC_ONLY_NO_YEAR / f"{key}.joblib"
        if path.exists():
            _models_logistic_only_no_year[key] = joblib.load(path)
        else:
            print(f"WARNING: logistic-only no-Year model not found — {path}")
    expected = len(_expected_keys_logistic_only())
    print(f"Loaded {len(_models_logistic_only_no_year)}/{expected} logistic-only no-Year models.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_all_models()
    _load_all_models_no_year()
    _load_all_models_logistic_only()
    _load_all_models_logistic_only_no_year()
    yield


app = FastAPI(
    title="DigiHealth Risk Score API (dual-track)",
    description=(
        "Thesis-aligned diabetes risk scoring with two complementary tracks: "
        "passive screening (`/predict`) and intervention-safe what-if "
        "simulation (`/predict/interventions`). A `/logistic_only/predict` "
        "route (with a nested `/logistic_only/no_year/predict` variant) serves "
        "a uniform logistic-regression screening output for frontend consumers "
        "that need a single-family model. Send raw questionnaire + annual "
        "measurements — no patient ID needed."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Preprocessing — replicates Phase 0 logic without knowing patient identity
# ---------------------------------------------------------------------------

def _history_features(
    measurements: list[ClinicalMeasurement],
    history_years: int,
) -> dict[str, Any]:
    """Compute _hist_{M}y_ aggregates matching Phase 0 add_history_features()."""
    record: dict[str, Any] = {}

    for feature in CLINICAL_FEATURES:
        raw = [getattr(m, feature) for m in measurements]
        series = pd.Series(raw, dtype=float)
        observed = series.dropna()
        prefix = f"{feature}_hist_{history_years}y"

        record[f"{prefix}_observed_count"] = int(observed.size)
        record[f"{prefix}_missing_count"] = int(series.isna().sum())
        record[f"{prefix}_latest"] = float(observed.iloc[-1]) if not observed.empty else np.nan
        record[f"{prefix}_mean"] = float(observed.mean()) if not observed.empty else np.nan
        record[f"{prefix}_min"] = float(observed.min()) if not observed.empty else np.nan
        record[f"{prefix}_max"] = float(observed.max()) if not observed.empty else np.nan
        record[f"{prefix}_std"] = (
            float(observed.std(ddof=0)) if observed.size > 1
            else (0.0 if observed.size == 1 else np.nan)
        )
        record[f"{prefix}_range"] = float(observed.max() - observed.min()) if not observed.empty else np.nan

        if observed.size >= 2:
            x = observed.index.to_numpy(dtype=float)
            y = observed.to_numpy(dtype=float)
            record[f"{prefix}_slope"] = float(np.polyfit(x - x.min(), y, deg=1)[0])
        else:
            record[f"{prefix}_slope"] = np.nan

    return record


def build_modeling_row(req: PredictRequest) -> pd.DataFrame:
    """Convert anonymous patient data into a single modeling-table row.

    `MAX_FBS_up_to_year`, `years_since_last_fbs`, and `is_missing_last_year`
    are training-time features that Phase 0 derives from the full 2005-2016
    wide panel. If the request supplies the matching aggregate (`max_fbs_to_date`,
    `years_since_last_fbs`, `previous_year_fbs_missing`), it is used directly.
    Otherwise this function falls back to deriving each value from the
    submitted `measurements`, which matches training only when the window
    covers the patient's full prior FBS history.
    """
    current = req.measurements[-1]
    calendar_year = req.year or datetime.now(timezone.utc).year

    if req.max_fbs_to_date is not None:
        max_fbs = float(req.max_fbs_to_date)
    else:
        all_fbs = [m.FBS for m in req.measurements if m.FBS is not None]
        max_fbs = float(max(all_fbs)) if all_fbs else np.nan

    if req.years_since_last_fbs is not None:
        years_since_fbs: float = float(req.years_since_last_fbs)
    elif current.FBS is not None:
        years_since_fbs = 0.0
    else:
        gap = next(
            (i for i, m in enumerate(reversed(req.measurements[:-1]), start=1) if m.FBS is not None),
            None,
        )
        years_since_fbs = float(gap) if gap is not None else np.nan

    is_missing_last_year: bool | float
    if req.previous_year_fbs_missing is not None:
        is_missing_last_year = req.previous_year_fbs_missing
    elif len(req.measurements) >= 2:
        is_missing_last_year = req.measurements[-2].FBS is None
    else:
        is_missing_last_year = np.nan  # type: ignore[assignment]

    clinical_observed = sum(
        1 for f in CLINICAL_FEATURES if getattr(current, f) is not None
    )

    row: dict[str, Any] = {
        "Year": calendar_year,
        "Age": req.age,
        "gender": req.gender,
        "dm_first_degree_relative": req.dm_first_degree_relative,
        "cooking_method": req.cooking_method,
        "total_sugary_week": req.total_sugary_week,
        "total_veg_fruit_week": req.total_veg_fruit_week,
        "total_exercise_week": req.total_exercise_week,
        "total_phy_activity_week": req.total_phy_activity_week,
        "sleep_hours": req.sleep_hours,
        "sleep_quality": req.sleep_quality,
        "smoking_status": req.smoking_status,
        "alcohol_status": req.alcohol_status,
        "FBS": current.FBS,
        "BMI": current.BMI,
        "Pulse": current.Pulse,
        "BL_pres1": current.BL_pres1,
        "BL_pres2": current.BL_pres2,
        "Waist": current.Waist,
        "MAX_FBS_up_to_year": max_fbs,
        "has_fbs_this_year": int(current.FBS is not None),
        "years_since_last_fbs": years_since_fbs,
        "is_missing_last_year": is_missing_last_year,
        "clinical_observed_count": clinical_observed,
        "prediction_horizon_years": req.horizon_years,
        "history_window_years": req.history_years,
    }

    if req.history_years > 1:
        row.update(_history_features(req.measurements, req.history_years))

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Feature engineering (must match modeling.engineer_features)
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 2 v2 / Phase 5 derived features.

    Always overwrites FBS-derived columns so interventions that modify FBS
    get correct updated values.
    """
    df = df.copy()
    df["Year_centered"] = df["Year"] - YEAR_REFERENCE
    df["Year_centered_sq"] = df["Year_centered"] ** 2
    df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    df["FBS_x_Age"] = df["FBS"] * df["Age"]
    df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def _engineer_features_no_year(df: pd.DataFrame) -> pd.DataFrame:
    """Year-ablation variant: skip Year_centered and Year_centered_sq.

    The trained artifacts in models_no_year/ do not consume those columns
    (they were excluded from LEAKAGE_OR_METADATA_COLUMNS at export time),
    so omitting them here keeps inference symmetric with training.
    """
    df = df.copy()
    df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    df["FBS_x_Age"] = df["FBS"] * df["Age"]
    df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


# ---------------------------------------------------------------------------
# Intervention engine
# ---------------------------------------------------------------------------

def _apply_preset(
    row: pd.DataFrame,
    preset_def: dict[str, Any],
    ranges: dict[str, dict[str, float | None]],
    engineer: "Callable[[pd.DataFrame], pd.DataFrame]" = _engineer_features,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | None]]]:
    """Apply one named intervention preset to a fully-engineered feature row.

    `engineer` controls which feature engineering function is re-applied
    after the assignments — `_engineer_features` for the main thesis track
    or `_engineer_features_no_year` for the construct-validity variant.
    """
    adjusted = row.copy()
    changed: dict[str, dict[str, float | None]] = {}

    def clamp(feature: str, value: float) -> float:
        r = ranges.get(feature, {})
        lo, hi = r.get("min"), r.get("max")
        return float(np.clip(value, lo, hi)) if lo is not None and hi is not None else float(value)

    def orig(feature: str) -> float | None:
        v = adjusted[feature].iloc[0]
        return float(v) if pd.notna(v) else None

    for feature, value in preset_def.get("assignments", {}).items():
        if feature not in adjusted.columns or value is None:
            continue
        new_val = clamp(feature, value)
        changed[feature] = {"from": orig(feature), "to": new_val}
        adjusted.loc[:, feature] = new_val

    # max_assignments: ratchet upward only. Preserves the monotonic safety
    # guarantee (p' <= p0) for "increase to p75"-style presets when a patient
    # is already above the target.
    for feature, target in preset_def.get("max_assignments", {}).items():
        if feature not in adjusted.columns or target is None:
            continue
        current_val = orig(feature)
        new_val = target if current_val is None else max(current_val, target)
        new_val = clamp(feature, new_val)
        changed[feature] = {"from": current_val, "to": new_val}
        adjusted.loc[:, feature] = new_val

    for feature, delta in preset_def.get("delta_assignments", {}).items():
        if feature not in adjusted.columns:
            continue
        current_val = orig(feature) or 0.0
        new_val = current_val + delta
        floor = preset_def.get("floor_assignments", {}).get(feature)
        if floor is not None:
            new_val = max(new_val, floor)
        new_val = clamp(feature, new_val)
        changed[feature] = {"from": orig(feature), "to": new_val}
        adjusted.loc[:, feature] = new_val

    for feature, multiplier in preset_def.get("scale_assignments", {}).items():
        if feature not in adjusted.columns:
            continue
        current_val = orig(feature) or 0.0
        new_val = clamp(feature, current_val * multiplier)
        changed[feature] = {"from": orig(feature), "to": new_val}
        adjusted.loc[:, feature] = new_val

    adjusted = engineer(adjusted)
    return adjusted, changed


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _track_family(track: str, horizon: int) -> str:
    family_map = SCREENING_FAMILY if track == TRACK_SCREENING else INTERVENTION_FAMILY
    return family_map[horizon]


_VARIANT_STORES: dict[str, dict[str, dict[str, Any]]] = {
    VARIANT_WITH_YEAR: _models,
    VARIANT_NO_YEAR: _models_no_year,
    VARIANT_LOGISTIC_ONLY_WITH_YEAR: _models_logistic_only,
    VARIANT_LOGISTIC_ONLY_NO_YEAR: _models_logistic_only_no_year,
}

_VARIANT_EXPORT_CMD: dict[str, str] = {
    VARIANT_WITH_YEAR: "python export_models.py",
    VARIANT_NO_YEAR: "python export_models.py --no-year",
    VARIANT_LOGISTIC_ONLY_WITH_YEAR: "python export_models.py --logistic-only",
    VARIANT_LOGISTIC_ONLY_NO_YEAR: "python export_models.py --logistic-only --no-year",
}


def _get_artifact(track: str, horizon: int, history: int, variant: str = VARIANT_WITH_YEAR) -> dict[str, Any]:
    if variant in (VARIANT_LOGISTIC_ONLY_WITH_YEAR, VARIANT_LOGISTIC_ONLY_NO_YEAR):
        # Logistic-only variants serve only the screening track; family is
        # logistic regardless of horizon.
        family = "logistic"
    else:
        family = _track_family(track, horizon)
    key = f"{track}_{family}_n{horizon}_m{history}"
    store = _VARIANT_STORES[variant]
    artifact = store.get(key)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{key}' ({variant}) not loaded. "
                f"Run `{_VARIANT_EXPORT_CMD[variant]}` first."
            ),
        )
    return artifact


def _score(artifact: dict[str, Any], row: pd.DataFrame) -> float:
    missing = [f for f in artifact["feature_columns"] if f not in row.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Preprocessed row is missing features for '{artifact['model_key']}': {missing}. "
                "Ensure history_years matches the number of measurements provided."
            ),
        )
    x_raw = artifact["preprocessor"].transform(row[artifact["feature_columns"]].copy())

    if artifact["model_family"] == "logistic":
        x_scaled = (np.asarray(x_raw, dtype=float) - artifact["mean_"]) / artifact["scale_"]
        x = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=float), x_scaled])
        return float(special.expit(x @ artifact["coefficients"])[0])

    return float(artifact["model"].predict_proba(x_raw)[:, 1][0])


def _risk_score(prob: float) -> float:
    return round(float(np.clip(prob * 100.0, 0.0, 100.0)), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    expected = len(_expected_keys())
    return {
        "status": "ok",
        "models_loaded": len(_models),
        "expected": expected,
        "tracks": {
            "screening": {"family_per_horizon": SCREENING_FAMILY},
            "intervention": {"family_per_horizon": INTERVENTION_FAMILY},
        },
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            key=a["model_key"],
            track=a["track"],
            model_family=a["model_family"],
            horizon_years=a["horizon_years"],
            history_years=a["history_years"],
            threshold=round(a["threshold"], 6),
            feature_count=len(a["feature_columns"]),
            intervention_presets=list(a.get("intervention_presets", {}).keys()),
        )
        for a in sorted(
            _models.values(),
            key=lambda x: (x["track"], x["history_years"], x["horizon_years"]),
        )
    ]


@app.get("/models/{key}", response_model=ModelInfo)
def get_model(
    key: Annotated[str, PathParam(description="e.g. screening_catboost_n1_m5 or intervention_ebm_n1_m5")]
) -> ModelInfo:
    artifact = _models.get(key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found.")
    return ModelInfo(
        key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=artifact["horizon_years"],
        history_years=artifact["history_years"],
        threshold=round(artifact["threshold"], 6),
        feature_count=len(artifact["feature_columns"]),
        intervention_presets=list(artifact.get("intervention_presets", {}).keys()),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Passive-screening risk score (thesis pure-prediction winner per horizon)."""
    artifact = _get_artifact(TRACK_SCREENING, req.horizon_years, req.history_years)

    row = build_modeling_row(req)
    row = _engineer_features(row)

    prob = _score(artifact, row)
    score = _risk_score(prob)
    threshold = artifact["threshold"]

    return PredictResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        probability=round(prob, 6),
        risk_score=score,
        threshold=round(threshold, 6),
        at_risk_flag=bool(prob >= threshold),
    )


@app.post("/predict/interventions", response_model=InterventionResponse)
def predict_interventions(req: InterventionRequest) -> InterventionResponse:
    """Intervention-safe what-if simulation (monotonic models, thesis §5.4 winners)."""
    artifact = _get_artifact(TRACK_INTERVENTION, req.horizon_years, req.history_years)
    presets_store: dict[str, Any] = artifact.get("intervention_presets", {})

    unknown = [p for p in req.presets if p not in presets_store]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown presets: {unknown}. Available: {list(presets_store.keys())}",
        )

    row = build_modeling_row(req)
    row = _engineer_features(row)

    baseline_prob = _score(artifact, row)
    baseline_score = _risk_score(baseline_prob)
    threshold = artifact["threshold"]
    ranges = artifact.get("train_feature_ranges", {})

    scenarios: list[ScenarioResult] = []
    for preset_name in req.presets:
        preset_def = presets_store[preset_name]
        adjusted_row, changed = _apply_preset(row, preset_def, ranges)
        adj_prob = _score(artifact, adjusted_row)
        adj_score = _risk_score(adj_prob)
        scenarios.append(ScenarioResult(
            preset=preset_name,
            description=preset_def.get("description", ""),
            probability=round(adj_prob, 6),
            risk_score=adj_score,
            delta_risk_score=round(adj_score - baseline_score, 2),
            at_risk_flag=bool(adj_prob >= threshold),
            changed_features=changed,
        ))

    return InterventionResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        baseline=RiskResult(
            probability=round(baseline_prob, 6),
            risk_score=baseline_score,
            threshold=round(threshold, 6),
            at_risk_flag=bool(baseline_prob >= threshold),
        ),
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# Routes — /no_year/* (year-ablation variant: Year features excluded)
#
# Same input/output schema as the main routes; backed by artifacts in
# models_no_year/ and using _engineer_features_no_year. Run
# `export_models.py --no-year` to populate the underlying joblibs.
# ---------------------------------------------------------------------------

@app.get("/no_year/health")
def health_no_year() -> dict[str, Any]:
    expected = len(_expected_keys())
    return {
        "status": "ok" if _models_no_year else "models_not_loaded",
        "variant": VARIANT_NO_YEAR,
        "year_features_excluded": ["Year", "Year_centered", "Year_centered_sq"],
        "models_loaded": len(_models_no_year),
        "expected": expected,
        "tracks": {
            "screening": {"family_per_horizon": SCREENING_FAMILY},
            "intervention": {"family_per_horizon": INTERVENTION_FAMILY},
        },
        "rationale": (
            "Construct-validity alternative to /predict. Predictions are "
            "invariant to the calendar year a patient is scored in. Use this "
            "tree when deployment is years past the 2005-2016 training window "
            "and a retraining pipeline is not available."
        ),
    }


@app.get("/no_year/models", response_model=list[ModelInfo])
def list_models_no_year() -> list[ModelInfo]:
    return [
        ModelInfo(
            key=a["model_key"],
            track=a["track"],
            model_family=a["model_family"],
            horizon_years=a["horizon_years"],
            history_years=a["history_years"],
            threshold=round(a["threshold"], 6),
            feature_count=len(a["feature_columns"]),
            intervention_presets=list(a.get("intervention_presets", {}).keys()),
        )
        for a in sorted(
            _models_no_year.values(),
            key=lambda x: (x["track"], x["history_years"], x["horizon_years"]),
        )
    ]


@app.get("/no_year/models/{key}", response_model=ModelInfo)
def get_model_no_year(
    key: Annotated[str, PathParam(description="e.g. screening_catboost_n1_m5 or intervention_ebm_n1_m5")]
) -> ModelInfo:
    artifact = _models_no_year.get(key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' (no_year variant) not found.")
    return ModelInfo(
        key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=artifact["horizon_years"],
        history_years=artifact["history_years"],
        threshold=round(artifact["threshold"], 6),
        feature_count=len(artifact["feature_columns"]),
        intervention_presets=list(artifact.get("intervention_presets", {}).keys()),
    )


@app.post("/no_year/predict", response_model=PredictResponse)
def predict_no_year(req: PredictRequest) -> PredictResponse:
    """Construct-validity screening track — Year features excluded."""
    artifact = _get_artifact(
        TRACK_SCREENING, req.horizon_years, req.history_years, variant=VARIANT_NO_YEAR
    )

    row = build_modeling_row(req)
    row = _engineer_features_no_year(row)

    prob = _score(artifact, row)
    score = _risk_score(prob)
    threshold = artifact["threshold"]

    return PredictResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        probability=round(prob, 6),
        risk_score=score,
        threshold=round(threshold, 6),
        at_risk_flag=bool(prob >= threshold),
    )


@app.post("/no_year/predict/interventions", response_model=InterventionResponse)
def predict_interventions_no_year(req: InterventionRequest) -> InterventionResponse:
    """Construct-validity intervention track — Year features excluded."""
    artifact = _get_artifact(
        TRACK_INTERVENTION, req.horizon_years, req.history_years, variant=VARIANT_NO_YEAR
    )
    presets_store: dict[str, Any] = artifact.get("intervention_presets", {})

    unknown = [p for p in req.presets if p not in presets_store]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown presets: {unknown}. Available: {list(presets_store.keys())}",
        )

    row = build_modeling_row(req)
    row = _engineer_features_no_year(row)

    baseline_prob = _score(artifact, row)
    baseline_score = _risk_score(baseline_prob)
    threshold = artifact["threshold"]
    ranges = artifact.get("train_feature_ranges", {})

    scenarios: list[ScenarioResult] = []
    for preset_name in req.presets:
        preset_def = presets_store[preset_name]
        adjusted_row, changed = _apply_preset(row, preset_def, ranges, engineer=_engineer_features_no_year)
        adj_prob = _score(artifact, adjusted_row)
        adj_score = _risk_score(adj_prob)
        scenarios.append(ScenarioResult(
            preset=preset_name,
            description=preset_def.get("description", ""),
            probability=round(adj_prob, 6),
            risk_score=adj_score,
            delta_risk_score=round(adj_score - baseline_score, 2),
            at_risk_flag=bool(adj_prob >= threshold),
            changed_features=changed,
        ))

    return InterventionResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        baseline=RiskResult(
            probability=round(baseline_prob, 6),
            risk_score=baseline_score,
            threshold=round(threshold, 6),
            at_risk_flag=bool(baseline_prob >= threshold),
        ),
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# Routes — /logistic_only/* (uniform logistic screening track)
#
# Frontend-driven uniform-family screening: logistic regression at every
# horizon, so the response always has model_family == "logistic". Trades
# ~0.020 PR-AUC at N=1 and N=3 against the mixed-family default in exchange
# for an output that is easier to post-process client-side. No intervention
# endpoint here — intervention scoring stays on /predict/interventions.
# Run `python export_models.py --logistic-only` to populate the joblibs.
# ---------------------------------------------------------------------------

@app.get("/logistic_only/health")
def health_logistic_only() -> dict[str, Any]:
    expected = len(_expected_keys_logistic_only())
    return {
        "status": "ok" if _models_logistic_only else "models_not_loaded",
        "variant": VARIANT_LOGISTIC_ONLY_WITH_YEAR,
        "models_loaded": len(_models_logistic_only),
        "expected": expected,
        "tracks": {
            "screening": {"family_per_horizon": LOGISTIC_ONLY_SCREENING_FAMILY},
        },
        "rationale": (
            "Logistic-only screening alternative to /predict. Returns a uniform "
            "logistic model at every horizon for frontend post-processing. "
            "Intervention scoring is not exposed on this tree; use "
            "/predict/interventions for what-if simulation."
        ),
    }


@app.get("/logistic_only/models", response_model=list[ModelInfo])
def list_models_logistic_only() -> list[ModelInfo]:
    return [
        ModelInfo(
            key=a["model_key"],
            track=a["track"],
            model_family=a["model_family"],
            horizon_years=a["horizon_years"],
            history_years=a["history_years"],
            threshold=round(a["threshold"], 6),
            feature_count=len(a["feature_columns"]),
            intervention_presets=list(a.get("intervention_presets", {}).keys()),
        )
        for a in sorted(
            _models_logistic_only.values(),
            key=lambda x: (x["track"], x["history_years"], x["horizon_years"]),
        )
    ]


@app.get("/logistic_only/models/{key}", response_model=ModelInfo)
def get_model_logistic_only(
    key: Annotated[str, PathParam(description="e.g. screening_logistic_n3_m5")]
) -> ModelInfo:
    artifact = _models_logistic_only.get(key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' (logistic_only) not found.")
    return ModelInfo(
        key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=artifact["horizon_years"],
        history_years=artifact["history_years"],
        threshold=round(artifact["threshold"], 6),
        feature_count=len(artifact["feature_columns"]),
        intervention_presets=list(artifact.get("intervention_presets", {}).keys()),
    )


@app.post("/logistic_only/predict", response_model=PredictResponse)
def predict_logistic_only(req: PredictRequest) -> PredictResponse:
    """Logistic-only screening — uniform logistic family at every horizon."""
    artifact = _get_artifact(
        TRACK_SCREENING, req.horizon_years, req.history_years,
        variant=VARIANT_LOGISTIC_ONLY_WITH_YEAR,
    )

    row = build_modeling_row(req)
    row = _engineer_features(row)

    prob = _score(artifact, row)
    score = _risk_score(prob)
    threshold = artifact["threshold"]

    return PredictResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        probability=round(prob, 6),
        risk_score=score,
        threshold=round(threshold, 6),
        at_risk_flag=bool(prob >= threshold),
    )


@app.get("/logistic_only/no_year/health")
def health_logistic_only_no_year() -> dict[str, Any]:
    expected = len(_expected_keys_logistic_only())
    return {
        "status": "ok" if _models_logistic_only_no_year else "models_not_loaded",
        "variant": VARIANT_LOGISTIC_ONLY_NO_YEAR,
        "year_features_excluded": ["Year", "Year_centered", "Year_centered_sq"],
        "models_loaded": len(_models_logistic_only_no_year),
        "expected": expected,
        "tracks": {
            "screening": {"family_per_horizon": LOGISTIC_ONLY_SCREENING_FAMILY},
        },
        "rationale": (
            "Construct-validity variant of /logistic_only/predict. Combines the "
            "uniform-logistic screening output with the calendar-time-invariant "
            "year-ablation training. Use when both the frontend post-processing "
            "constraint and the no-Year construct-validity guarantee are needed."
        ),
    }


@app.get("/logistic_only/no_year/models", response_model=list[ModelInfo])
def list_models_logistic_only_no_year() -> list[ModelInfo]:
    return [
        ModelInfo(
            key=a["model_key"],
            track=a["track"],
            model_family=a["model_family"],
            horizon_years=a["horizon_years"],
            history_years=a["history_years"],
            threshold=round(a["threshold"], 6),
            feature_count=len(a["feature_columns"]),
            intervention_presets=list(a.get("intervention_presets", {}).keys()),
        )
        for a in sorted(
            _models_logistic_only_no_year.values(),
            key=lambda x: (x["track"], x["history_years"], x["horizon_years"]),
        )
    ]


@app.get("/logistic_only/no_year/models/{key}", response_model=ModelInfo)
def get_model_logistic_only_no_year(
    key: Annotated[str, PathParam(description="e.g. screening_logistic_n3_m5")]
) -> ModelInfo:
    artifact = _models_logistic_only_no_year.get(key)
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{key}' (logistic_only_no_year) not found."
        )
    return ModelInfo(
        key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=artifact["horizon_years"],
        history_years=artifact["history_years"],
        threshold=round(artifact["threshold"], 6),
        feature_count=len(artifact["feature_columns"]),
        intervention_presets=list(artifact.get("intervention_presets", {}).keys()),
    )


@app.post("/logistic_only/no_year/predict", response_model=PredictResponse)
def predict_logistic_only_no_year(req: PredictRequest) -> PredictResponse:
    """Logistic-only screening, Year features excluded (construct-validity variant)."""
    artifact = _get_artifact(
        TRACK_SCREENING, req.horizon_years, req.history_years,
        variant=VARIANT_LOGISTIC_ONLY_NO_YEAR,
    )

    row = build_modeling_row(req)
    row = _engineer_features_no_year(row)

    prob = _score(artifact, row)
    score = _risk_score(prob)
    threshold = artifact["threshold"]

    return PredictResponse(
        model_key=artifact["model_key"],
        track=artifact["track"],
        model_family=artifact["model_family"],
        horizon_years=req.horizon_years,
        history_years=req.history_years,
        probability=round(prob, 6),
        risk_score=score,
        threshold=round(threshold, 6),
        at_risk_flag=bool(prob >= threshold),
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
