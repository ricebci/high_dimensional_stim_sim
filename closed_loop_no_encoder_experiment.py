"""
Run closed-loop electrical stimulation using NoEncoderController.

Iterates through a fixed repertoire of single-electrode pulses (one per
electrode, pulse at t=0 ms) plus a no-stimulation slot in round-robin order.
No encoding model or target state is required.

Trials
------
Each "trial" is one complete pass through the N+1-element repertoire
(N electrodes + 1 no-stim slot).  Use --n-trials to set the number of
passes; total simulation time is computed automatically as:

    total_duration_ms = n_trials * n_repertoire * closed_loop_interval_ms

The config artifact stores n_trials and n_repertoire so downstream
analysis can recover trial_index = interval_index // n_repertoire and
within_trial_index = interval_index % n_repertoire without any extra
bookkeeping.

How to run
----------
    python closed_loop_no_encoder_experiment.py \\
        --n-trials 10 \\
        --closed-loop-interval-ms 500

Example (fast smoke test)
-------------------------
    python closed_loop_no_encoder_experiment.py \\
        --n-trials 3 \\
        --closed-loop-interval-ms 500 \\
        --fast-mode \\
        --output-name test_no_encoder_run
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import json
import pickle
from typing import Optional

import numpy as np

from closed_loop_repertoire_experiment import run_closed_loop_electrical_stim
from controller import NoEncoderController
from system import SystemNESTSim


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop electrical stimulation with NoEncoderController"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help=(
            "Number of complete passes through the repertoire. "
            "Total duration = n_trials × n_repertoire × closed_loop_interval_ms."
        ),
    )
    parser.add_argument("--closed-loop-interval-ms", type=float, default=500.0)
    parser.add_argument(
        "--bin-ms",
        type=float,
        default=5.0,
        help="Bin resolution (ms) used for spike binning in the closed-loop loop",
    )
    parser.add_argument(
        "--n-stim-channels",
        type=int,
        default=32,
        help="Number of stimulation electrodes (one repertoire entry per electrode)",
    )
    parser.add_argument(
        "--stim-amplitude-uA",
        type=float,
        default=2.0,
        help="Stimulation amplitude in µA for each single-electrode pulse",
    )
    parser.add_argument(
        "--visual-metadata-path",
        type=str,
        default="outputs/data_system_sim_0.05scale/visual_orientations_8/visual8_stim_metadata.pkl",
        help=(
            "Path to visual8_stim_metadata.pkl. When provided, per-neuron visual "
            "current generators are recreated in the NEST kernel."
        ),
    )
    parser.add_argument("--output-name", type=str, default="closed_loop_no_encoder")
    parser.add_argument("--output-prefix", type=str, default="no_encoder")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--fast-sim-resolution-ms", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.closed_loop_interval_ms <= 0:
        raise ValueError("--closed-loop-interval-ms must be > 0")

    # --- Build controller ---
    controller = NoEncoderController(
        n_stim_channels=args.n_stim_channels,
        stim_amplitude_uA=args.stim_amplitude_uA,
    )
    n_repertoire = len(controller._repertoire)   # n_stim_channels + 1
    total_duration_ms = args.n_trials * n_repertoire * args.closed_loop_interval_ms

    print(
        f"NoEncoderController: {n_repertoire} repertoire entries "
        f"({args.n_stim_channels} single-electrode + 1 no-stim)"
    )
    print(
        f"Trials: {args.n_trials}  |  "
        f"Intervals per trial: {n_repertoire}  |  "
        f"Total duration: {total_duration_ms:.0f} ms"
    )

    # --- Initialize NEST network (includes presim warmup) ---
    sim = SystemNESTSim(
        output_name=args.output_name,
        fast_mode=args.fast_mode,
        fast_sim_resolution_ms=args.fast_sim_resolution_ms,
    )

    # --- Recreate visual current generators if metadata path is provided ---
    if args.visual_metadata_path is not None and os.path.isfile(args.visual_metadata_path):
        print(f"Setting up visual generators from {args.visual_metadata_path} ...")
        sim.setup_visual_generators_from_metadata(args.visual_metadata_path)

    # --- Write partial config in case the run is interrupted ---
    partial_config = {
        "total_duration_ms": total_duration_ms,
        "closed_loop_interval_ms": args.closed_loop_interval_ms,
        "bin_ms": args.bin_ms,
        "n_stim_channels": args.n_stim_channels,
        "n_repertoire": n_repertoire,
        "n_trials": args.n_trials,
        "stim_amplitude_uA": args.stim_amplitude_uA,
        "visual_metadata_path": args.visual_metadata_path,
        "data_path": sim.sim_dict["data_path"],
    }
    partial_config_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_config.json"
    )
    with open(partial_config_path, "w") as f:
        json.dump(partial_config, f, indent=2)

    # --- Run the closed-loop experiment ---
    results = run_closed_loop_electrical_stim(
        sim=sim,
        total_duration_ms=total_duration_ms,
        closed_loop_interval_ms=args.closed_loop_interval_ms,
        controller=controller,
        initial_stim_pattern=None,
        output_prefix=args.output_prefix,
        realtime_progress=not args.no_progress,
        bin_ms=args.bin_ms,
        # Fine-tuning is a no-op for NoEncoderController (no model / no cache).
        n_repertoire_update_ms=total_duration_ms + 1.0,
        finetune_intervals=0,
    )

    # --- Save full config ---
    config = {**partial_config, "results": results}
    config_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_config.pkl"
    )
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    with open(partial_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Closed-loop no-encoder experiment complete")
    print("Data path:", sim.sim_dict["data_path"])
    print("Results:", results)


if __name__ == "__main__":
    main()
