#!/usr/bin/env bash
# Run closed-loop repertoire experiment targeting orientation 0°.
#
# Usage:
#   bash run_closed_loop_target_orientation.sh
#
# Outputs land in:
#   outputs/data_system_sim_0.05scale/closed_loop_orientation0_<run>/
# So far, assumes 3858 neurons + 32 stim electrodes
# A human-readable JSON config is written alongside the pickle at experiment end.

set -euo pipefail

cd "$(dirname "$0")"

# python closed_loop_repertoire_experiment.py \
#     --model-dir                outputs/models/20260312_012405/ \
#     --data-dir                 data/electrical/ \
#     --vis-dir                  outputs/data_system_sim_0.05scale/visual_orientations_8/ \
#     --total-duration-ms        200 \
#     --closed-loop-interval-ms  70 \
#     --n-repertoire-update-ms   50 \
#     --target-orientation-deg   90 \
#     --random-stim-prob         0.0 \
#     --ar-aggregation           average \
#     --cluster-centers-path     outputs/data_system_sim_0.05scale/visual_orientations_8/cluster_centers_analysis.json \
#     --output-name              closed_loop_orientation90_70msinterval_learning_90_small_aravg_DEBUG\
#     --output-prefix            closed_loop \
#     --device                   mps

# python closed_loop_no_encoder_experiment.py \
#     --closed-loop-interval-ms  1000 \
#     --same-seed                42 \
#     --stim-amplitudes-uA       1 2 3 4 5 6\
#     --n-trials                 4  \
#     --n-stim-channels          128 \
#     --n-sessions               8 \
#     --n-workers                8 \
#     --output-name              128_stimchannel_1sinterval_no_encoder_run1\
#     --output-prefix            run1\
#     --quiet


# python closed_loop_no_encoder_experiment.py \
#     --closed-loop-interval-ms  1000 \
#     --same-seed                42 \
#     --stim-amplitudes-uA       1 2 3 4 5 6\
#     --n-trials                 4  \
#     --n-stim-channels          32 \
#     --n-sessions               8 \
#     --n-workers                8 \
#     --output-name              32_stimchannel_1sinterval_no_encoder_run3\
#     --output-prefix            run3\
#     --quiet



# ── Hyperparameters (edit these) ──────────────────────────────────────────────
# Synaptic / network
PSP_EXC_MEAN=""          # mean excitatory PSP (mV); empty = default 0.15
G_INH=""                 # relative inhibitory weight; empty = default -4
BG_RATE=""               # Poisson background rate (spikes/s); empty = default 8
DC_AMP_EXTRA="0.0"       # extra DC current (pA) injected into all neurons

# Probe volume bounds (um); empty = defaults (v: 0–1800, h: -380–380)
VOLUME_V_MIN=""
VOLUME_V_MAX=""
VOLUME_H_MIN=""
VOLUME_H_MAX=""

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(python closed_loop_no_encoder_experiment.py
    --closed-loop-interval-ms  250
    --same-seed                42
    --stim-amplitudes-uA       6
    --n-trials                 2
    --n-stim-channels          32
    --n-sessions               1
    --n-workers                1
    --output-name              32_stimchannel_250msinterval_noscale_minimal_l4question
    --output-prefix            run1
    --k-scaling 1.00
    --n-scaling 1.00
    --dc-amp-extra "$DC_AMP_EXTRA"
    --quiet
)

[[ -n "$PSP_EXC_MEAN" ]]  && CMD+=(--psp-exc-mean  "$PSP_EXC_MEAN")
[[ -n "$G_INH" ]]          && CMD+=(--g-inh         "$G_INH")
[[ -n "$BG_RATE" ]]        && CMD+=(--bg-rate        "$BG_RATE")
[[ -n "$VOLUME_V_MIN" ]]   && CMD+=(--volume-v-min  "$VOLUME_V_MIN")
[[ -n "$VOLUME_V_MAX" ]]   && CMD+=(--volume-v-max  "$VOLUME_V_MAX")
[[ -n "$VOLUME_H_MIN" ]]   && CMD+=(--volume-h-min  "$VOLUME_H_MIN")
[[ -n "$VOLUME_H_MAX" ]]   && CMD+=(--volume-h-max  "$VOLUME_H_MAX")

"${CMD[@]}"
