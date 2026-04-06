#!/usr/bin/env bash
# Pull closed-loop no-encoder results to your local Mac.
#
# Run this ON YOUR MAC from the high_dimensional_stim_sim directory:
#   bash scp_results_to_local.sh
#
# Prerequisites:
#   - SSH access to the remote machine
#   - Same network or VPN connection

set -euo pipefail

REMOTE_USER="user"
REMOTE_HOST="192.168.50.161"
REMOTE_DIR="high_dimensional_stim_sim/outputs/data_system_sim_0.05scale"
LOCAL_DIR="./outputs/data_system_sim_0.05scale"

mkdir -p "$LOCAL_DIR"

echo "Pulling closed-loop no-encoder results from ${REMOTE_HOST}..."

scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/closed_loop_no_encoder_1sinterval*" \
    "$LOCAL_DIR/"

echo "Done. Results saved to ${LOCAL_DIR}/"
