#!/usr/bin/env bash
# ============================================================
# run_all.sh  (longitudinal-diabetes-risk repository root)
#
# Runs every thesis experiment phase in dependency order.
# Must be executed from the repository root:
#
#   bash run_all.sh [options]
#
# Options:
#   --from-phase N   Start from phase N (1-7).  Skips earlier phases.
#                    Phase numbering: 0=data, 0a=appendix EDA, 1=statistical,
#                    2=tree, 3=survival, 4=calibration+leaderboard,
#                    5=intervention-safe, 6=deployment, 7=year-features
#                    ablation + no-Year deployment.
#   --no-deploy      Skip Phase 6 deployment export (saves ~20 min).
#   --no-ablation    Skip Phase 7 year-features ablation + no-Year export
#                    (saves ~60-90 min).
#   --fail-fast      Stop immediately on any script error.
#   --force          Pass --force to scripts that support it (re-runs even if
#                    output already exists; e.g. GEE/logistic grid).
#   --dry-run        Print commands without executing them.
#
# Logs:
#   All output is tee'd to digihealth_risk/logs/run_YYYYMMDD_HHMMSS.log
# ============================================================

set -euo pipefail

# ── Parse arguments ─────────────────────────────────────────
FROM_PHASE=0
NO_DEPLOY=false
NO_ABLATION=false
FAIL_FAST=false
FORCE_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --from-phase=*)   FROM_PHASE="${arg#*=}" ;;
        --from-phase)     shift; FROM_PHASE="${1:-0}" ;;
        --no-deploy)      NO_DEPLOY=true ;;
        --no-ablation)    NO_ABLATION=true ;;
        --fail-fast)      FAIL_FAST=true ;;
        --force)          FORCE_FLAG="--force" ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)
            sed -n '3,32p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/digihealth_risk/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# ── Logging / execution helpers ──────────────────────────────
STEP=0
PASS=0
FAIL=0
SKIP=0
declare -a FAILED_STEPS=()

_log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

_sep() {
    echo "────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
}

_run() {
    STEP=$((STEP + 1))
    local label="$1"; shift
    local cmd=("$@")

    _sep
    _log "STEP $STEP  $label"
    _log "CMD  ${cmd[*]}"

    if $DRY_RUN; then
        _log "DRY-RUN — skipped"
        SKIP=$((SKIP + 1))
        return 0
    fi

    local t0=$SECONDS
    if "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        local elapsed=$((SECONDS - t0))
        _log "PASS $label  (${elapsed}s)"
        PASS=$((PASS + 1))
    else
        local elapsed=$((SECONDS - t0))
        _log "FAIL $label  (${elapsed}s)"
        FAIL=$((FAIL + 1))
        FAILED_STEPS+=("$STEP: $label")
        if $FAIL_FAST; then
            _log "Stopping (--fail-fast)"
            _summary
            exit 1
        fi
    fi
}

_phase_header() {
    local num="$1"; shift
    echo "" | tee -a "$LOG_FILE"
    _log "══════════════════════════════════════════════════════"
    _log "  PHASE $num — $*"
    _log "══════════════════════════════════════════════════════"
}

_summary() {
    echo "" | tee -a "$LOG_FILE"
    _sep
    _log "SUMMARY  pass=$PASS  fail=$FAIL  skip=$SKIP  total=$STEP"
    if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
        _log "Failed steps:"
        for s in "${FAILED_STEPS[@]}"; do _log "  ✗ $s"; done
    fi
    _log "Log: $LOG_FILE"
    _sep
}

_skip_phase() {
    _log "  → Skipping phase $1 (--from-phase=$FROM_PHASE)"
}

# ── Verify repo root ─────────────────────────────────────────
cd "$REPO_ROOT"

if [[ ! -f "datasets/df_final.pkl" ]]; then
    echo "ERROR: datasets/df_final.pkl not found. Place the 5.6 MB source file before running." >&2
    exit 1
fi

_log "DigiHealth 2025-IS — full experiment run"
_log "Repo root  : $REPO_ROOT"
_log "From phase : $FROM_PHASE"
_log "No deploy  : $NO_DEPLOY"
_log "No ablation: $NO_ABLATION"
_log "Fail fast  : $FAIL_FAST"
_log "Force      : ${FORCE_FLAG:-off}"
_log "Dry run    : $DRY_RUN"
_log "Log file   : $LOG_FILE"

# ── Build N×M path list helper ───────────────────────────────
# Generates --input-path flags for all 15 phase_0 modeling tables.
p0_input_args() {
    local args=()
    for N in 1 2 3 4 5; do
        for M in 1 3 5; do
            local path="digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_${N}_history_${M}.pkl"
            args+=(--input-path "$path")
        done
    done
    echo "${args[@]}"
}

# ═══════════════════════════════════════════════════════════
# PHASE 0 — Data Engineering (15 modeling tables)
# Thesis: §3, §4.4.1
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 0 ]]; then
    _phase_header 0 "Data Engineering — build 15 modeling tables (N×M grid)"

    for N in 1 2 3 4 5; do
        for M in 1 3 5; do
            _run "phase_0  N=${N} M=${M}" \
                python digihealth_risk/phase_0/build_modeling_tables.py \
                    --horizon-years "$N" \
                    --history-years "$M"
        done
    done
else
    _skip_phase 0
fi

# ═══════════════════════════════════════════════════════════
# PHASE 0a — Appendix EDA (depth analysis, Section 3.4.2)
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 0 ]]; then
    _phase_header "0a" "Appendix EDA — feature-engineering evidence"

    _run "phase_0  eda_depth" \
        python digihealth_risk/phase_0/eda_depth.py
fi

# ═══════════════════════════════════════════════════════════
# PHASE 1 — Statistical Models (GEE, Logistic, GLMM)
# Thesis: §4.1.1, §5.5.1
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 1 ]]; then
    _phase_header 1 "Statistical Models — GEE, Logistic (all M), GLMM"

    for M in 1 3 5; do
        _run "phase_1  GEE M=${M}" \
            python digihealth_risk/phase_1/gee_horizon_grid.py \
                --history-years "$M" \
                $FORCE_FLAG
    done

    for M in 1 3 5; do
        _run "phase_1  Logistic M=${M}" \
            python digihealth_risk/phase_1/logistic_horizon_grid.py \
                --history-years "$M" \
                $FORCE_FLAG
    done

    # GLMM: exploratory only (v1 features, no --force support)
    _run "phase_1  GLMM exploratory" \
        python digihealth_risk/phase_1/glmm_exploratory.py
else
    _skip_phase 1
fi

# ═══════════════════════════════════════════════════════════
# PHASE 2 — Tree Models
# Thesis: §4.1.2, §5.2, §5.5.2
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 2 ]]; then
    _phase_header 2 "Tree Models — train, LMM slopes, N×M horizon grid"

    # train_tree_models.py accepts multiple --input-path args; pass all 15 tables.
    # N=1,M=1 uses the no-suffix default filename; all others use horizon_N_history_M.
    INPUT_ARGS=()
    for N in 1 2 3 4 5; do
        for M in 1 3 5; do
            if [[ "$N" -eq 1 && "$M" -eq 1 ]]; then
                INPUT_ARGS+=(--input-path \
                    "digihealth_risk/phase_0/outputs/phase_0_modeling_table.pkl")
            else
                INPUT_ARGS+=(--input-path \
                    "digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_${N}_history_${M}.pkl")
            fi
        done
    done

    _run "phase_2  train_tree_models (all 15 N×M tables)" \
        python digihealth_risk/phase_2/train_tree_models.py \
            "${INPUT_ARGS[@]}"

    _run "phase_2  lmm_slope_features (all 15 N×M tables)" \
        python digihealth_risk/phase_2/lmm_slope_features.py \
            "${INPUT_ARGS[@]}"

    # horizon_history_grid re-builds phase0 tables if missing and runs the grid.
    _run "phase_2  horizon_history_grid (N×M grid, v2 features)" \
        python digihealth_risk/phase_2/horizon_history_grid.py
else
    _skip_phase 2
fi

# ═══════════════════════════════════════════════════════════
# PHASE 3 — Survival Models
# Thesis: §4.1.3, §5.3
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 3 ]]; then
    _phase_header 3 "Survival Models — Landmark Cox + Two-stage M=1,3,5"

    _run "phase_3  landmark_cox" \
        python digihealth_risk/phase_3/landmark_cox.py

    # Each M needs a distinct --output-prefix so files are not overwritten.
    # M=1 is expected to degenerate (ROC-AUC ≈ 0.5) — kept as negative result §5.3
    for M in 1 3 5; do
        _run "phase_3  two_stage_survival M=${M}" \
            python digihealth_risk/phase_3/two_stage_survival.py \
                --history-window "$M" \
                --output-prefix "phase_3_3_v2_m${M}"
    done
else
    _skip_phase 3
fi

# ═══════════════════════════════════════════════════════════
# PHASE 4 — Calibration, Threshold Policy, Final Leaderboard
# Thesis: §4.2, §4.4.4, §5.1
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 4 ]]; then
    _phase_header 4 "Calibration + Threshold Policy + Final Leaderboard"

    _run "phase_4  calibrate_trees" \
        python digihealth_risk/phase_4/calibrate_trees.py

    _run "phase_4  threshold_optimization" \
        python digihealth_risk/phase_4/threshold_optimization.py

    _run "phase_4  cross_family_comparison  →  FINAL LEADERBOARD" \
        python digihealth_risk/phase_4/cross_family_comparison.py
else
    _skip_phase 4
fi

# ═══════════════════════════════════════════════════════════
# PHASE 5 — Intervention-Safe Models (monotonic constraints)
# Thesis: §4.3, §5.4
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 5 ]]; then
    _phase_header 5 "Intervention-Safe Models — 5 monotonic families + benchmark"

    _run "phase_5  train_monotonic_xgboost" \
        python digihealth_risk/phase_5/train_monotonic_xgboost.py

    _run "phase_5  train_monotonic_lightgbm" \
        python digihealth_risk/phase_5/train_monotonic_lightgbm.py

    _run "phase_5  train_monotonic_catboost" \
        python digihealth_risk/phase_5/train_monotonic_catboost.py

    _run "phase_5  train_monotonic_ebm" \
        python digihealth_risk/phase_5/train_monotonic_ebm.py

    _run "phase_5  train_monotonic_logistic" \
        python digihealth_risk/phase_5/train_monotonic_logistic.py

    _run "phase_5  intervention_benchmark  →  INTERVENTION LEADERBOARD" \
        python digihealth_risk/phase_5/intervention_benchmark.py
else
    _skip_phase 5
fi

# ═══════════════════════════════════════════════════════════
# PHASE 6 — Deployment (FastAPI model export)
# Thesis: §7
# Skipped with --no-deploy.  ~15-30 min to retrain all 15 models.
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 6 ]] && ! $NO_DEPLOY; then
    _phase_header 6 "Deployment — export 15 monotonic XGBoost models"

    _run "phase_6  export_models (15 models)" \
        python digihealth_risk/phase_6/export_models.py

    _log ""
    _log "  To start the API after export:"
    _log "    uvicorn digihealth_risk.phase_6.api:app --reload --port 8000"
    _log "  Interactive docs: http://localhost:8000/docs"
elif $NO_DEPLOY; then
    _log ""
    _log "  Phase 6 deployment skipped (--no-deploy)."
    _log "  Run manually: python digihealth_risk/phase_6/export_models.py"
fi

# ═══════════════════════════════════════════════════════════
# PHASE 7 — Year-features Ablation + No-Year Deployment Export
# Thesis: §3.4.2, §6 (construct-validity discussion)
# Skipped with --no-ablation.  ~60-90 min combined.
# ═══════════════════════════════════════════════════════════
if [[ "$FROM_PHASE" -le 7 ]] && ! $NO_ABLATION; then
    _phase_header 7 "Year-features Ablation — retrain phases 2/4/5 without Year*"

    _run "phase_7  train_trees_no_year (uncalibrated grid)" \
        python digihealth_risk/phase_7/train_trees_no_year.py

    _run "phase_7  calibrate_trees_no_year (Platt / isotonic / raw)" \
        python digihealth_risk/phase_7/calibrate_trees_no_year.py

    _run "phase_7  train_monotonic_no_year (5 families × 5 horizons)" \
        python digihealth_risk/phase_7/train_monotonic_no_year.py

    _run "phase_7  compare_with_baseline  →  ABLATION REPORT" \
        python digihealth_risk/phase_7/compare_with_baseline.py

    # No-Year deployment artifacts for the API's /no_year/* route tree.
    # Tied to --no-deploy: if deployment is off, the no-Year deploy export is
    # also off; otherwise it's part of the ablation phase.
    if ! $NO_DEPLOY; then
        _run "phase_7  export_models --no-year (dual-track API artifacts)" \
            python digihealth_risk/phase_6/export_models.py --no-year

        _log ""
        _log "  After both phase 6 + phase 7 export, the API serves both trees:"
        _log "    /predict, /predict/interventions                (main thesis)"
        _log "    /no_year/predict, /no_year/predict/interventions (ablation alt.)"
    else
        _log ""
        _log "  No-Year deployment export skipped (--no-deploy)."
        _log "  Run manually: python digihealth_risk/phase_6/export_models.py --no-year"
    fi
elif $NO_ABLATION; then
    _log ""
    _log "  Phase 7 year-features ablation skipped (--no-ablation)."
    _log "  Run manually: bash digihealth_risk/phase_7/run_all.sh"
fi

# ═══════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════
_summary

# Exit with non-zero if any step failed
[[ $FAIL -eq 0 ]]
