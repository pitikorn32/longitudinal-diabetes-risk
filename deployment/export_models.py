"""Deployment model export — dual-track, thesis-aligned (standalone slice).

Trains and exports 30 model artifacts (15 per track x 5 horizons x 3 history
windows) for the deployment FastAPI in this folder.

    Screening track (/predict)            pure-prediction winners per horizon.
    Intervention track (/predict/...)     monotonic intervention-safe winners.

Per-horizon model family:

    Horizon N | Screening family   | Intervention family
    ----------|--------------------|--------------------
    1         | CatBoost           | Monotonic EBM
    2         | Logistic           | Monotonic CatBoost
    3         | XGBoost            | Monotonic XGBoost
    4         | Logistic           | Monotonic CatBoost
    5         | Logistic*          | Monotonic CatBoost

* The thesis screening winner at N=5 is GEE. statsmodels GEE estimators do not
  serialize cleanly via joblib, so Logistic (a 0.0034 PR-AUC gap) is substituted
  for deployability. The registry records the substitution explicitly.

History windows: M in {1, 3, 5}.

Run from this folder:
    python export_models.py

Reads the 15 phase-0 modeling tables. By default it looks in the sibling phase
tree (../digihealth_risk/phase_0/outputs/); override with DIGIHEALTH_PHASE0_DIR.

Outputs (this folder):
    models/{track}_{family}_n{N}_m{M}.joblib    30 model artifacts
    model_registry.json                          per-model metadata for the API
    deployment_metrics.csv                       train/test metrics for all 30

Optional no-Year variant:
    python export_models.py --no-year

Trains the same 30 configurations with Year, Year_centered and Year_centered_sq
excluded. Outputs to models_no_year/, model_registry_no_year.json,
deployment_metrics_no_year.csv. Powers the API's /no_year/* route tree.

Optional logistic-only screening variant:
    python export_models.py --logistic-only
    python export_models.py --logistic-only --no-year

Trains 15 screening-track artifacts using logistic regression at every
horizon (intervention track is skipped). Outputs to models_logistic_only/
or models_logistic_only_no_year/. Powers the API's /logistic_only/predict
and /logistic_only/no_year/predict routes. Frontend-driven: a uniform
single-family screening output for easier client-side post-processing.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import special
from sklearn.linear_model import LogisticRegression

import modeling
import patient_split
from modeling import (
    BASE_MONOTONE_RULES,
    HISTORY_MONOTONE_BASES,
    HISTORY_MONOTONE_SUFFIXES,
    MonotoneRule,
    RANDOM_SEED,
    classification_metrics,
    get_feature_columns,
    load_table,
    make_preprocessor,
)


HERE = Path(__file__).resolve().parent
SUBMODULE_ROOT = HERE.parent

PHASE0_OUT = Path(os.environ.get(
    "DIGIHEALTH_PHASE0_DIR",
    str(SUBMODULE_ROOT / "digihealth_risk" / "phase_0" / "outputs"),
))
OUT_DIR = HERE


def output_paths(no_year: bool, logistic_only: bool = False) -> tuple[Path, Path, Path]:
    """Return (model_dir, registry_path, metrics_path) for the chosen variant."""
    if logistic_only and no_year:
        return (
            OUT_DIR / "models_logistic_only_no_year",
            OUT_DIR / "model_registry_logistic_only_no_year.json",
            OUT_DIR / "deployment_metrics_logistic_only_no_year.csv",
        )
    if logistic_only:
        return (
            OUT_DIR / "models_logistic_only",
            OUT_DIR / "model_registry_logistic_only.json",
            OUT_DIR / "deployment_metrics_logistic_only.csv",
        )
    if no_year:
        return (
            OUT_DIR / "models_no_year",
            OUT_DIR / "model_registry_no_year.json",
            OUT_DIR / "deployment_metrics_no_year.csv",
        )
    return (
        OUT_DIR / "models",
        OUT_DIR / "model_registry.json",
        OUT_DIR / "deployment_metrics.csv",
    )


MODEL_DIR = OUT_DIR / "models"  # rebound at runtime by main() based on --no-year

HORIZONS = [1, 2, 3, 4, 5]
HISTORY_WINDOWS = [1, 3, 5]
RIDGE_ALPHA = 0.01  # matches digihealth_risk/phase_1/logistic.py

# Per-horizon model family per track. Each entry is the model_family slug
# used in the artifact filename and registry.
SCREENING_FAMILY = {
    1: "catboost",
    2: "logistic",
    3: "xgboost",
    4: "logistic",
    5: "logistic",  # substituted for GEE — see module docstring
}

INTERVENTION_FAMILY = {
    1: "ebm",
    2: "catboost",
    3: "xgboost",
    4: "catboost",
    5: "catboost",
}

TRACK_SCREENING = "screening"
TRACK_INTERVENTION = "intervention"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def phase0_path(horizon: int, history: int) -> Path:
    explicit = PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{history}.pkl"
    if explicit.exists():
        return explicit
    default = PHASE0_OUT / "phase_0_modeling_table.pkl"
    if horizon == 1 and history == 1 and default.exists():
        return default
    raise FileNotFoundError(
        f"No Phase 0 modeling table for horizon={horizon}, history={history}. "
        f"Expected: {explicit}. Set DIGIHEALTH_PHASE0_DIR or build the tables "
        f"with digihealth_risk/phase_0/build_modeling_tables.py first."
    )


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return patient_split.apply_canonical_split(df)


def model_key(track: str, family: str, horizon: int, history: int) -> str:
    return f"{track}_{family}_n{horizon}_m{history}"


def model_path(track: str, family: str, horizon: int, history: int) -> Path:
    return MODEL_DIR / f"{model_key(track, family, horizon, history)}.joblib"


# ---------------------------------------------------------------------------
# Monotonic constraints (parameterised for any history window)
# ---------------------------------------------------------------------------

def rule_for_feature(feature_name: str, history_years: int) -> MonotoneRule | None:
    if feature_name in BASE_MONOTONE_RULES:
        return BASE_MONOTONE_RULES[feature_name]
    for base, sign in HISTORY_MONOTONE_BASES.items():
        prefix = f"{base}_hist_{history_years}y"
        if feature_name.startswith(prefix) and feature_name.endswith(HISTORY_MONOTONE_SUFFIXES):
            return MonotoneRule(sign, f"History feature follows {base} monotonic direction.")
    return None


def monotone_constraints(feature_names: list[str], history_years: int) -> tuple[int, ...]:
    return tuple(
        0 if rule_for_feature(f, history_years) is None else rule_for_feature(f, history_years).sign
        for f in feature_names
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_catboost(constraints: tuple[int, ...] | None) -> Any:
    from catboost import CatBoostClassifier
    kwargs: dict[str, Any] = {
        "iterations": 400,
        "depth": 4,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": False,
    }
    if constraints is not None:
        kwargs["monotone_constraints"] = list(constraints)
    return CatBoostClassifier(**kwargs)


def build_xgboost(constraints: tuple[int, ...] | None) -> Any:
    from xgboost import XGBClassifier
    kwargs: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }
    if constraints is not None:
        kwargs["monotone_constraints"] = constraints
    return XGBClassifier(**kwargs)


def build_ebm(constraints: tuple[int, ...] | None) -> Any:
    from interpret.glassbox import ExplainableBoostingClassifier
    kwargs: dict[str, Any] = {
        "interactions": 0,
        "learning_rate": 0.03,
        "validation_size": 0.1,
        "outer_bags": 8,
        "inner_bags": 0,
        "max_rounds": 2000,
        "early_stopping_rounds": 50,
        "max_bins": 256,
        "max_leaves": 3,
        "min_samples_leaf": 4,
        "objective": "log_loss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if constraints is not None:
        kwargs["monotone_constraints"] = list(constraints)
    return ExplainableBoostingClassifier(**kwargs)


# ---------------------------------------------------------------------------
# Logistic fit / predict — kept simple (no monotonic constraints on screening
# track; the intervention track never uses logistic).
# ---------------------------------------------------------------------------

def fit_logistic_artifact(train_df: pd.DataFrame) -> dict[str, Any]:
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_columns = numeric_features + categorical_features
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_raw = preprocessor.fit_transform(train_df[feature_columns].copy()).astype(float)
    transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]

    mean_ = x_raw.mean(axis=0)
    scale_ = x_raw.std(axis=0)
    scale_[scale_ == 0.0] = 1.0
    x_scaled = (x_raw - mean_) / scale_

    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    model = LogisticRegression(
        C=max(1.0 / RIDGE_ALPHA, 1e-6),
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_scaled, y_train)
    coefficients = np.concatenate(
        [np.atleast_1d(model.intercept_).astype(float),
         np.ravel(model.coef_).astype(float)]
    )

    return {
        "preprocessor": preprocessor,
        "model": None,  # logistic is fully captured by coefficients/mean/scale
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_feature_names": transformed_names,
        "monotone_constraints": tuple([0] * len(transformed_names)),
        "mean_": mean_,
        "scale_": scale_,
        "coefficients": coefficients,
    }


def predict_logistic(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x_raw = artifact["preprocessor"].transform(df[artifact["feature_columns"]].copy()).astype(float)
    x_scaled = (x_raw - artifact["mean_"]) / artifact["scale_"]
    x = np.hstack([np.ones((x_scaled.shape[0], 1), dtype=float), x_scaled])
    return special.expit(x @ artifact["coefficients"])


# ---------------------------------------------------------------------------
# Generic tree/EBM fit / predict
# ---------------------------------------------------------------------------

def fit_estimator_artifact(
    train_df: pd.DataFrame,
    *,
    family: str,
    history_years: int,
    monotonic: bool,
) -> dict[str, Any]:
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_columns = numeric_features + categorical_features
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_train = preprocessor.fit_transform(train_df[feature_columns].copy())
    transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]

    constraints: tuple[int, ...] | None = None
    if monotonic:
        constraints = monotone_constraints(transformed_names, history_years)

    if family == "catboost":
        model = build_catboost(constraints)
    elif family == "xgboost":
        model = build_xgboost(constraints)
    elif family == "ebm":
        model = build_ebm(constraints)
    else:
        raise ValueError(f"Unsupported estimator family: {family}")

    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    model.fit(x_train, y_train)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_feature_names": transformed_names,
        "monotone_constraints": constraints if constraints is not None else tuple([0] * len(transformed_names)),
    }


def predict_estimator(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x = artifact["preprocessor"].transform(df[artifact["feature_columns"]].copy())
    return artifact["model"].predict_proba(x)[:, 1]


def predict(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    if artifact["model_family"] == "logistic":
        return predict_logistic(artifact, df)
    return predict_estimator(artifact, df)


# ---------------------------------------------------------------------------
# Intervention presets — computed from training-data statistics
# ---------------------------------------------------------------------------

def compute_intervention_presets(train_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    def p75(col: str) -> float | None:
        return float(train_df[col].quantile(0.75)) if col in train_df.columns else None

    def col_min(col: str) -> float | None:
        return float(train_df[col].min(skipna=True)) if col in train_df.columns else None

    return {
        "reduce_sugary_to_zero": {
            "assignments": {"total_sugary_week": 0.0},
            "expected_direction": "decrease_or_equal",
            "description": "Set sugary drink intake to zero.",
        },
        "reduce_sugary_50pct": {
            "scale_assignments": {"total_sugary_week": 0.5},
            "expected_direction": "decrease_or_equal",
            "description": "Reduce sugary drink intake by 50%.",
        },
        "increase_exercise_to_p75": {
            "max_assignments": {"total_exercise_week": p75("total_exercise_week")},
            "expected_direction": "decrease_or_equal",
            "description": "Raise exercise sessions to at least the population 75th percentile (never reduce).",
        },
        "increase_activity_to_p75": {
            "max_assignments": {"total_phy_activity_week": p75("total_phy_activity_week")},
            "expected_direction": "decrease_or_equal",
            "description": "Raise physical activity to at least the population 75th percentile (never reduce).",
        },
        "increase_veg_fruit_to_p75": {
            "max_assignments": {"total_veg_fruit_week": p75("total_veg_fruit_week")},
            "expected_direction": "decrease_or_equal",
            "description": "Raise vegetable/fruit servings to at least the population 75th percentile (never reduce).",
        },
        "reduce_bmi_by_one": {
            "delta_assignments": {"BMI": -1.0},
            "floor_assignments": {"BMI": col_min("BMI")},
            "expected_direction": "decrease_or_equal",
            "description": "Reduce BMI by 1 unit (clamped to training minimum).",
        },
        "combined_lifestyle": {
            "assignments": {"total_sugary_week": 0.0},
            "max_assignments": {
                "total_exercise_week": p75("total_exercise_week"),
                "total_phy_activity_week": p75("total_phy_activity_week"),
                "total_veg_fruit_week": p75("total_veg_fruit_week"),
            },
            "delta_assignments": {"BMI": -1.0},
            "floor_assignments": {"BMI": col_min("BMI")},
            "expected_direction": "decrease_or_equal",
            "description": "Combined: zero sugary drinks, exercise/activity/veg ratcheted up to at least p75, BMI minus 1.",
        },
    }


# ---------------------------------------------------------------------------
# Training + saving one model
# ---------------------------------------------------------------------------

def fit_and_save(track: str, horizon: int, history: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    family = SCREENING_FAMILY[horizon] if track == TRACK_SCREENING else INTERVENTION_FAMILY[horizon]
    key = model_key(track, family, horizon, history)

    df = load_table(phase0_path(horizon, history))
    df = modeling.engineer_features(df)
    train_df, test_df = split_by_patient(df)

    if family == "logistic":
        artifact = fit_logistic_artifact(train_df)
    else:
        artifact = fit_estimator_artifact(
            train_df,
            family=family,
            history_years=history,
            monotonic=(track == TRACK_INTERVENTION),
        )
    artifact["model_family"] = family
    artifact["track"] = track

    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    train_proba = predict(artifact, train_df)
    test_proba = predict(artifact, test_df)
    threshold = float(y_train.mean())

    train_metrics = classification_metrics(y_train, train_proba, threshold)
    test_metrics = classification_metrics(y_test, test_proba, threshold)

    feature_columns = artifact["feature_columns"]
    artifact.update({
        "model_key": key,
        "track": track,
        "model_family": family,
        "horizon_years": horizon,
        "history_years": history,
        "threshold": threshold,
        "train_positive_rate": threshold,
        "train_feature_ranges": {
            col: {
                "min": float(train_df[col].min(skipna=True)) if pd.api.types.is_numeric_dtype(train_df[col]) else None,
                "max": float(train_df[col].max(skipna=True)) if pd.api.types.is_numeric_dtype(train_df[col]) else None,
            }
            for col in feature_columns
        },
        "intervention_presets": compute_intervention_presets(train_df),
    })

    joblib.dump(artifact, model_path(track, family, horizon, history))

    metrics_rows = [
        {"model_key": key, "track": track, "model_family": family,
         "horizon_years": horizon, "history_years": history, "split": "train", **train_metrics},
        {"model_key": key, "track": track, "model_family": family,
         "horizon_years": horizon, "history_years": history, "split": "test", **test_metrics},
    ]

    registry_entry: dict[str, Any] = {
        "key": key,
        "track": track,
        "model_family": family,
        "horizon_years": horizon,
        "history_years": history,
        "threshold": round(threshold, 6),
        "test_pr_auc": round(float(test_metrics["pr_auc"]), 4),
        "test_roc_auc": round(float(test_metrics["roc_auc"]), 4),
        "test_brier": round(float(test_metrics["brier"]), 4),
        "feature_count": len(feature_columns),
        "model_path": str(model_path(track, family, horizon, history).relative_to(OUT_DIR)),
        "intervention_presets": list(artifact["intervention_presets"].keys()),
    }
    return pd.DataFrame(metrics_rows), registry_entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deployment artifacts.")
    parser.add_argument(
        "--no-year",
        action="store_true",
        help=(
            "Drop Year, Year_centered, and Year_centered_sq from training "
            "(phase_7 ablation). Outputs to models_no_year/ to power the "
            "API's /no_year/* route tree. Default trains with Year features."
        ),
    )
    parser.add_argument(
        "--logistic-only",
        action="store_true",
        help=(
            "Train only the screening track using logistic regression at every "
            "horizon (intervention track is skipped). Outputs 15 artifacts to "
            "models_logistic_only/ (or models_logistic_only_no_year/ when "
            "combined with --no-year). Powers the API's /logistic_only/* "
            "route tree. Frontend-driven uniform-family alternative."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MODEL_DIR, SCREENING_FAMILY  # type: ignore[misc]
    model_dir, registry_path, metrics_path = output_paths(args.no_year, args.logistic_only)
    MODEL_DIR = model_dir

    if args.no_year:
        modeling.patch_drop_year_features()
        print("[export_models] no-Year mode: Year, Year_centered, Year_centered_sq excluded.")

    tracks_to_train: tuple[str, ...] = (TRACK_SCREENING, TRACK_INTERVENTION)
    if args.logistic_only:
        SCREENING_FAMILY = {n: "logistic" for n in HORIZONS}
        tracks_to_train = (TRACK_SCREENING,)
        print(
            "[export_models] logistic-only mode: screening track uses logistic at every "
            "horizon; intervention track skipped."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    registry_entries: list[dict] = []

    for track in tracks_to_train:
        family_map = SCREENING_FAMILY if track == TRACK_SCREENING else INTERVENTION_FAMILY
        for history in HISTORY_WINDOWS:
            for horizon in HORIZONS:
                family = family_map[horizon]
                print(f"[{track:>12s}] N={horizon} M={history} family={family} …", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics_df, entry = fit_and_save(track, horizon, history)
                all_metrics.append(metrics_df)
                registry_entries.append(entry)
                test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]
                print(f"               PR-AUC={test_row['pr_auc']:.4f}  ROC-AUC={test_row['roc_auc']:.4f}")

    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)

    if args.logistic_only:
        variant_label = (
            "logistic_only_no_year" if args.no_year else "logistic_only_with_year"
        )
        tracks_block = {
            "screening": {
                "purpose": (
                    "Logistic-only alternative screening track for frontend "
                    "post-processing. Single-family, dependency-light uniform "
                    "model selection across all horizons."
                ),
                "family_per_horizon": SCREENING_FAMILY,
                "rationale": (
                    "Frontend-driven choice. Trades ~0.020 PR-AUC at N=1 and "
                    "N=3 against the mixed-family default in exchange for a "
                    "uniform logistic output that is easier to post-process "
                    "client-side."
                ),
            },
        }
    else:
        variant_label = "no_year" if args.no_year else "with_year"
        tracks_block = {
            "screening": {
                "purpose": "Pure-prediction risk score for passive screening (thesis §5.1, §6.4).",
                "family_per_horizon": SCREENING_FAMILY,
                "n5_substitution_note": (
                    "Thesis screening winner at N=5 is GEE (PR-AUC 0.5282). "
                    "Logistic (PR-AUC 0.5248) is substituted for deployability "
                    "since statsmodels GEE does not round-trip through joblib."
                ),
            },
            "intervention": {
                "purpose": "Monotonic intervention-safe risk score for what-if simulation (thesis §5.4, §6.4).",
                "family_per_horizon": INTERVENTION_FAMILY,
            },
        }

    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant_label,
        "year_features_excluded": ["Year", "Year_centered", "Year_centered_sq"] if args.no_year else [],
        "horizons": HORIZONS,
        "history_windows": HISTORY_WINDOWS,
        "tracks": tracks_block,
        "models": registry_entries,
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"\nExported {len(registry_entries)} models ({variant_label}) → {MODEL_DIR}")
    print(f"Registry → {registry_path}")


if __name__ == "__main__":
    main()
