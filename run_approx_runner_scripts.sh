#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Data directories and their stim channel counts
declare -A DATA_DIRS=(
    ["outputs/128_stimchannel_hv_space_1sinterval"]=128
    ["outputs/32_stimchannel_hv_space_1sinterval"]=32
    ["outputs/32_stimchannel_vonly_1sinterval"]=32
)

VISUAL_DIRS=(
    "outputs/visual_custom_stronger2/"
    "outputs/visual_orientations_8/"
)

# (window-bin-start, window-bin-end)
WINDOWS=(
    "0 7"
    "7 14"
    "4 11"
    "4 6"
    "6 8"
    "8 10"
)

for data_dir in "${!DATA_DIRS[@]}"; do
    n_ch="${DATA_DIRS[$data_dir]}"
    for visual_dir in "${VISUAL_DIRS[@]}"; do
        for window in "${WINDOWS[@]}"; do
            read -r wstart wend <<< "$window"
            echo "=== $data_dir | $visual_dir | wbin ${wstart}-${wend} ==="
            python run_approx_analysis.py "$data_dir" \
                --visual-dir "$visual_dir" \
                --n-stim-channels "$n_ch" \
                --k 0.0 \
                --window-bin-start "$wstart" \
                --window-bin-end "$wend"
            echo ""
        done
    done
done