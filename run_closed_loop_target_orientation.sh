#!/usr/bin/env bash
# Run closed-loop repertoire experiment targeting orientation 0°.
#
# Usage:
#   bash run_closed_loop_target_orientation.sh
#
# Outputs land in:
#   outputs/data_system_sim_0.05scale/closed_loop_orientation0_<run>/
#
# A human-readable JSON config is written alongside the pickle at experiment end.

set -euo pipefail

cd "$(dirname "$0")"

python closed_loop_repertoire_experiment.py \
    --model-dir                outputs/models/20260312_012405/ \
    --data-dir                 data/electrical/ \
    --vis-dir                  outputs/data_system_sim_0.05scale/visual_orientations_8/ \
    --total-duration-ms        100 \
    --closed-loop-interval-ms  10 \
    --n-repertoire-update-ms   50 \
    --target-orientation-deg   180 \
    --random-stim-prob         0.0 \
    --cluster-centers-path     outputs/data_system_sim_0.05scale/visual_orientations_8/cluster_centers_analysis.json \
    --output-name              closed_loop_orientation180_10msinterval_learning_180 \
    --output-prefix            closed_loop \
    --device                   mps
