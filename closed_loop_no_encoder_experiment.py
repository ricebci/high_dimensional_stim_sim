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

Sessions
--------
Use --n-sessions to parallelize across multiple stochastic runs (default 1).
When n_sessions > 1, each session runs independently with a different NEST
random seed, and results are saved to separate session-specific directories.
Each session can be run in parallel across available CPUs.

How to run
----------
    python closed_loop_no_encoder_experiment.py \\
        --n-trials 10 \\
        --closed-loop-interval-ms 500

Parallelized example (3 sessions on different CPUs)
-----------------------------------------------------
    python closed_loop_no_encoder_experiment.py \\
        --n-trials 10 \\
        --closed-loop-interval-ms 500 \\
        --n-sessions 3

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
from multiprocessing import Pool

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
        "--stim-amplitudes-uA",
        type=float,
        nargs="+",
        default=[2.0],
        help=(
            "One or more stimulation amplitudes in µA.  Each amplitude is "
            "paired with every electrode, expanding the repertoire to "
            "n_stim_channels × n_amplitudes + 1 entries per trial.  "
            "Example: --stim-amplitudes-uA 1 2 3 4 5 6"
        ),
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
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=1,
        help=(
            "Number of parallel sessions to run (default 1). "
            "Each session uses a different NEST random seed. "
            "When > 1, results are saved to session-specific directories."
        ),
    )
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--fast-sim-resolution-ms", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single session runner
# ---------------------------------------------------------------------------

def run_single_session(
    session_id: int,
    n_trials: int,
    closed_loop_interval_ms: float,
    bin_ms: float,
    n_stim_channels: int,
    stim_amplitudes_uA: list,
    visual_metadata_path: Optional[str],
    output_name: str,
    output_prefix: str,
    fast_mode: bool,
    fast_sim_resolution_ms: float,
    no_progress: bool,
    nest_seed: Optional[int] = None,
) -> dict:
    """
    Run a single session of the closed-loop no-encoder experiment.

    Parameters
    ----------
    session_id : int
        Session index (0-based), used for output naming
    nest_seed : int, optional
        Random seed for NEST. If None, NEST uses its default initialization.

    Returns
    -------
    dict
        Contains keys: 'session_id', 'data_path', 'config', 'results'
    """
    # Append session suffix to output names if running multiple sessions
    session_output_name = output_name if nest_seed is None else f"{output_name}_session_{session_id:03d}"
    session_output_prefix = output_prefix if nest_seed is None else f"{output_prefix}_session_{session_id:03d}"

    # Build controller
    controller = NoEncoderController(
        n_stim_channels=n_stim_channels,
        stim_amplitudes_uA=stim_amplitudes_uA,
    )
    n_repertoire = len(controller._repertoire)
    total_duration_ms = n_trials * n_repertoire * closed_loop_interval_ms

    if session_id == 0:  # Print info only once
        print(
            f"NoEncoderController: {n_repertoire} repertoire entries "
            f"({n_stim_channels} electrodes × {len(stim_amplitudes_uA)} amplitudes "
            f"{stim_amplitudes_uA} µA + 1 no-stim)"
        )
        print(
            f"Trials: {n_trials}  |  "
            f"Intervals per trial: {n_repertoire}  |  "
            f"Total duration: {total_duration_ms:.0f} ms"
        )

    # Initialize NEST network
    sim = SystemNESTSim(
        output_name=session_output_name,
        fast_mode=fast_mode,
        fast_sim_resolution_ms=fast_sim_resolution_ms,
    )

    # Set NEST random seed if provided
    if nest_seed is not None:
        import nest
        nest.SetKernelStatus({"rng_seeds": [nest_seed]})

    # Recreate visual current generators if metadata path is provided
    if visual_metadata_path is not None and os.path.isfile(visual_metadata_path):
        if session_id == 0:
            print(f"Setting up visual generators from {visual_metadata_path} ...")
        sim.setup_visual_generators_from_metadata(visual_metadata_path)

    # Write partial config in case the run is interrupted
    partial_config = {
        "session_id": session_id,
        "nest_seed": nest_seed,
        "total_duration_ms": total_duration_ms,
        "closed_loop_interval_ms": closed_loop_interval_ms,
        "bin_ms": bin_ms,
        "n_stim_channels": n_stim_channels,
        "n_repertoire": n_repertoire,
        "n_trials": n_trials,
        "stim_amplitudes_uA": stim_amplitudes_uA,
        "visual_metadata_path": visual_metadata_path,
        "data_path": sim.sim_dict["data_path"],
    }
    partial_config_path = os.path.join(
        sim.sim_dict["data_path"], f"{session_output_prefix}_config.json"
    )
    with open(partial_config_path, "w") as f:
        json.dump(partial_config, f, indent=2)

    # Run the closed-loop experiment
    results = run_closed_loop_electrical_stim(
        sim=sim,
        total_duration_ms=total_duration_ms,
        closed_loop_interval_ms=closed_loop_interval_ms,
        controller=controller,
        initial_stim_pattern=None,
        output_prefix=session_output_prefix,
        realtime_progress=not no_progress,
        bin_ms=bin_ms,
        n_repertoire_update_ms=total_duration_ms + 1.0,
        finetune_intervals=0,
    )

    # Save full config
    config = {**partial_config, "results": results}
    config_path = os.path.join(
        sim.sim_dict["data_path"], f"{session_output_prefix}_config.pkl"
    )
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    with open(partial_config_path, "w") as f:
        json.dump(config, f, indent=2)

    return {
        "session_id": session_id,
        "data_path": sim.sim_dict["data_path"],
        "config": config,
        "results": results,
    }


def run_session_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing.Pool.map.
    Unpacks arguments and calls run_single_session.
    """
    return run_single_session(*args_tuple)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.closed_loop_interval_ms <= 0:
        raise ValueError("--closed-loop-interval-ms must be > 0")
    if args.n_sessions <= 0:
        raise ValueError("--n-sessions must be > 0")

    # --- Single session mode ---
    if args.n_sessions == 1:
        result = run_single_session(
            session_id=0,
            n_trials=args.n_trials,
            closed_loop_interval_ms=args.closed_loop_interval_ms,
            bin_ms=args.bin_ms,
            n_stim_channels=args.n_stim_channels,
            stim_amplitudes_uA=args.stim_amplitudes_uA,
            visual_metadata_path=args.visual_metadata_path,
            output_name=args.output_name,
            output_prefix=args.output_prefix,
            fast_mode=args.fast_mode,
            fast_sim_resolution_ms=args.fast_sim_resolution_ms,
            no_progress=args.no_progress,
            nest_seed=None,
        )
        print("Closed-loop no-encoder experiment complete")
        print("Data path:", result["data_path"])
        print("Results:", result["results"])

    # --- Multiple sessions mode (parallelized) ---
    else:
        print(f"Running {args.n_sessions} parallel sessions...")

        # Prepare arguments for each session
        session_args = [
            (
                session_id,
                args.n_trials,
                args.closed_loop_interval_ms,
                args.bin_ms,
                args.n_stim_channels,
                args.stim_amplitudes_uA,
                args.visual_metadata_path,
                args.output_name,
                args.output_prefix,
                args.fast_mode,
                args.fast_sim_resolution_ms,
                args.no_progress,
                10000 + session_id,  # Unique nest seed for each session
            )
            for session_id in range(args.n_sessions)
        ]

        # Run sessions in parallel
        with Pool(processes=None) as pool:  # None = use all available CPUs
            results = pool.map(run_session_wrapper, session_args)

        # Print summary
        print(f"\nAll {args.n_sessions} sessions complete!")
        for result in results:
            print(f"  Session {result['session_id']}: {result['data_path']}")

        # Create a summary manifest
        manifest = {
            "n_sessions": args.n_sessions,
            "sessions": [
                {
                    "session_id": r["session_id"],
                    "data_path": r["data_path"],
                    "nest_seed": r["config"]["nest_seed"],
                }
                for r in results
            ],
        }
        print("\nSession manifest:")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
