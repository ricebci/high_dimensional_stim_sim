#!/usr/bin/env bash
# Grid search over simulation hyperparameters for the closed-loop no-encoder
# experiment.  Iterates over combinations of DC bias, excitatory PSP gain,
# inhibitory weight ratio, and background Poisson rate.
#
# Each combo runs a short experiment; results land in a per-combo output dir
# under outputs/grid_search/<tag>/.
#
# Usage:
#   bash run_grid_search.sh            # run all combos
#   bash run_grid_search.sh --dry-run  # print commands without running
#
# After the sweep, inspect STA response profiles in each output dir to find
# reasonable stimulation parameters.

set -euo pipefail
cd "$(dirname "$0")"

# ──────────────────────────────────────────────────────────────────────────────
# GRID DEFINITION — edit these arrays to control the search space
# ──────────────────────────────────────────────────────────────────────────────

# Additional DC current (pA) injected into every neuron.
# Positive = more excitable, negative = less excitable.
DC_AMP_EXTRA_VALUES=(0.0 50.0 100.0)

# Mean excitatory PSP (mV).  Default in network_params.py is 0.15.
PSP_EXC_MEAN_VALUES=(0.10 0.15 0.20)

# Relative inhibitory weight (negative).  Default is -4.
G_INH_VALUES=(-3 -4 -6)

# Poisson background input rate (spikes/s).  Default is 8.
BG_RATE_VALUES=(8)

# ──────────────────────────────────────────────────────────────────────────────
# FIXED EXPERIMENT SETTINGS — shared across all grid combos
# ──────────────────────────────────────────────────────────────────────────────

N_TRIALS=1                     # keep short for search
CLOSED_LOOP_INTERVAL_MS=1000
N_STIM_CHANNELS=128
STIM_AMPLITUDES="1 2 3 4 5 6"
SAME_SEED=42
N_SESSIONS=2
N_WORKERS=2
N_SCALING=""                   # empty = use default from network_params.py
K_SCALING=""                   # empty = use default from network_params.py
FAST_MODE=""                   # set to "--fast-mode" to enable
QUIET="--quiet"

# Probe volume bounds (um); empty = defaults (v: 0–1800, h: -380–380)
VOLUME_V_MIN=""
VOLUME_V_MAX=""
VOLUME_H_MIN=""
VOLUME_H_MAX=""

# ──────────────────────────────────────────────────────────────────────────────
# GRID EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — commands will be printed but not executed ==="
    echo
fi

TOTAL=0
for dc in "${DC_AMP_EXTRA_VALUES[@]}"; do
for psp in "${PSP_EXC_MEAN_VALUES[@]}"; do
for g in "${G_INH_VALUES[@]}"; do
for bg in "${BG_RATE_VALUES[@]}"; do
    TOTAL=$((TOTAL + 1))
done; done; done; done

echo "Grid search: ${TOTAL} parameter combinations"
echo "  DC_AMP_EXTRA : ${DC_AMP_EXTRA_VALUES[*]}"
echo "  PSP_EXC_MEAN : ${PSP_EXC_MEAN_VALUES[*]}"
echo "  G_INH        : ${G_INH_VALUES[*]}"
echo "  BG_RATE      : ${BG_RATE_VALUES[*]}"
echo

IDX=0
for dc in "${DC_AMP_EXTRA_VALUES[@]}"; do
for psp in "${PSP_EXC_MEAN_VALUES[@]}"; do
for g in "${G_INH_VALUES[@]}"; do
for bg in "${BG_RATE_VALUES[@]}"; do
    IDX=$((IDX + 1))

    # Build a human-readable tag for this combo
    TAG="dc${dc}_psp${psp}_g${g}_bg${bg}"
    OUTPUT_NAME="grid_search/${TAG}"

    echo "[$IDX/$TOTAL] ${TAG}"

    # Assemble the command
    CMD=(python closed_loop_no_encoder_experiment.py
        --closed-loop-interval-ms  "$CLOSED_LOOP_INTERVAL_MS"
        --same-seed                "$SAME_SEED"
        --stim-amplitudes-uA       $STIM_AMPLITUDES
        --n-trials                 "$N_TRIALS"
        --n-stim-channels          "$N_STIM_CHANNELS"
        --n-sessions               "$N_SESSIONS"
        --n-workers                "$N_WORKERS"
        --output-name              "$OUTPUT_NAME"
        --output-prefix            "$TAG"
        --dc-amp-extra             "$dc"
        --psp-exc-mean             "$psp"
        --g-inh                    "$g"
        --bg-rate                  "$bg"
    )

    # Append optional flags
    [[ -n "$N_SCALING" ]]    && CMD+=(--n-scaling    "$N_SCALING")
    [[ -n "$K_SCALING" ]]    && CMD+=(--k-scaling    "$K_SCALING")
    [[ -n "$VOLUME_V_MIN" ]] && CMD+=(--volume-v-min "$VOLUME_V_MIN")
    [[ -n "$VOLUME_V_MAX" ]] && CMD+=(--volume-v-max "$VOLUME_V_MAX")
    [[ -n "$VOLUME_H_MIN" ]] && CMD+=(--volume-h-min "$VOLUME_H_MIN")
    [[ -n "$VOLUME_H_MAX" ]] && CMD+=(--volume-h-max "$VOLUME_H_MAX")
    [[ -n "$FAST_MODE" ]]    && CMD+=($FAST_MODE)
    [[ -n "$QUIET" ]]        && CMD+=($QUIET)

    if $DRY_RUN; then
        echo "  ${CMD[*]}"
        echo
    else
        echo "  Starting at $(date '+%H:%M:%S') ..."
        if "${CMD[@]}"; then
            echo "  ✓ ${TAG} completed at $(date '+%H:%M:%S')"
        else
            echo "  ✗ ${TAG} FAILED (exit code $?)" >&2
        fi
        echo
    fi

done; done; done; done

echo "Grid search finished. ${IDX}/${TOTAL} combos processed."
echo "Results in: outputs/data_system_sim_*scale/grid_search/"
