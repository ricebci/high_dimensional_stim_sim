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
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse --quiet early (before any NEST import) to suppress startup banner.
if "--quiet" in sys.argv:
    os.environ["PYNEST_QUIET"] = "1"

import argparse
import json
import multiprocessing
import pickle
from typing import Optional
import threading
import time
from multiprocessing import Pool, Value

# Use 'spawn' so each child gets a fresh NEST kernel (fork inherits
# the parent's initialized C++ kernel state, causing hangs).
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass  # already set

import numpy as np


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
        default=None,
        help=(
            "Path to visual8_stim_metadata.pkl. When provided, per-neuron visual "
            "current generators are recreated in the NEST kernel. "
            "Default: auto-derived from --n-scaling."
        ),
    )
    parser.add_argument("--output-name", type=str, default="closed_loop_no_encoder")
    parser.add_argument(
        "--n-scaling",
        type=float,
        default=None,
        help="Override N_scaling (neuron count factor). Default: use network_params.py value.",
    )
    parser.add_argument(
        "--k-scaling",
        type=float,
        default=None,
        help="Override K_scaling (indegree factor). Default: use network_params.py value.",
    )
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help=(
            "Max number of parallel workers (default 4). "
            "Keeps NEST init from overwhelming the system. "
            "Sessions are processed in batches of this size."
        ),
    )
    parser.add_argument(
        "--same-seed",
        type=int,
        default=None,
        help=(
            "If set, all sessions use this exact NEST random seed "
            "(instead of a unique seed per session). Useful for "
            "reproducing identical network initial conditions."
        ),
    )
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--fast-sim-resolution-ms", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress NEST kernel output (set verbosity to M_ERROR).",
    )
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
    quiet: bool = False,
    shared_interval_counter=None,
    shared_init_counter=None,
    n_scaling: Optional[float] = None,
    k_scaling: Optional[float] = None,
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
    # Import heavy modules here (not at top level) so that with spawn
    # multiprocessing, each child gets a fresh NEST kernel.
    # Redirect stdout in worker to suppress all print output from NEST,
    # network_cortcol, system, etc.
    if quiet:
        os.environ["PYNEST_QUIET"] = "1"
        import sys as _sys
        _sys.stdout = open(os.devnull, "w")
    from closed_loop_repertoire_experiment import run_closed_loop_electrical_stim
    from controller import NoEncoderController
    from system import SystemNESTSim

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

    # Initialize NEST network with seed passed through to control both
    # NEST RNG and neuron placement consistently.
    sim = SystemNESTSim(
        output_name=session_output_name,
        fast_mode=fast_mode,
        fast_sim_resolution_ms=fast_sim_resolution_ms,
        n_stim_channels=n_stim_channels,
        n_scaling=n_scaling,
        k_scaling=k_scaling,
        rng_seed=nest_seed,
    )

    # Recreate visual current generators if metadata path is provided
    if visual_metadata_path is not None and os.path.isfile(visual_metadata_path):
        if session_id == 0:
            print(f"Setting up visual generators from {visual_metadata_path} ...")
        sim.setup_visual_generators_from_metadata(visual_metadata_path)

    # Write partial config in case the run is interrupted
    from system import volume_v_min, volume_v_max, volume_h_min, volume_h_max
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
        "completed": False,
        "electrode_volume_bounds_um": {
            "v_min": volume_v_min,
            "v_max": volume_v_max,
            "h_min": volume_h_min,
            "h_max": volume_h_max,
        },
    }
    partial_config_path = os.path.join(
        sim.sim_dict["data_path"], f"{session_output_prefix}_config.json"
    )
    with open(partial_config_path, "w") as f:
        json.dump(partial_config, f, indent=2)

    # Signal that NEST init is done, simulation loop is starting
    if shared_init_counter is not None:
        with shared_init_counter.get_lock():
            shared_init_counter.value += 1

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
        shared_interval_counter=shared_interval_counter,
    )

    # Save full config with completion marker
    config = {**partial_config, "completed": True, "results": results}
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


_shared_counter = None
_shared_init_counter = None
_shared_started_counter = None

def _pool_initializer(counter, init_counter, started_counter):
    """Store shared counters in each worker process."""
    global _shared_counter, _shared_init_counter, _shared_started_counter
    _shared_counter = counter
    _shared_init_counter = init_counter
    _shared_started_counter = started_counter

def run_session_wrapper(args_tuple):
    """
    Wrapper function for multiprocessing.Pool.map.
    Unpacks arguments and calls run_single_session.
    """
    # Signal that this task has been picked up by a worker
    if _shared_started_counter is not None:
        with _shared_started_counter.get_lock():
            _shared_started_counter.value += 1
    # Last three elements are quiet, n_scaling, k_scaling; rest are positional args
    *positional, quiet, n_scaling, k_scaling = args_tuple
    return run_single_session(
        *positional,
        quiet=quiet,
        shared_interval_counter=_shared_counter,
        shared_init_counter=_shared_init_counter,
        n_scaling=n_scaling,
        k_scaling=k_scaling,
    )


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

    # Resolve visual metadata path from scaling if not explicitly provided
    if args.visual_metadata_path is None:
        from corcol_params.network_params import net_dict as _nd
        _nscale = args.n_scaling if args.n_scaling is not None else _nd["N_scaling"]
        args.visual_metadata_path = (
            f"outputs/data_system_sim_{_nscale}scale/"
            f"visual_orientations_8/visual8_stim_metadata.pkl"
        )

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
            nest_seed=args.same_seed,
            quiet=args.quiet,
            n_scaling=args.n_scaling,
            k_scaling=args.k_scaling,
        )
        print("Closed-loop no-encoder experiment complete")
        print("Data path:", result["data_path"])
        print("Results:", result["results"])

    # --- Multiple sessions mode (parallelized) ---
    else:
        # Compute total intervals across all sessions for progress tracking
        n_stim_channels = args.n_stim_channels
        n_amplitudes = len(args.stim_amplitudes_uA)
        n_repertoire = n_stim_channels * n_amplitudes + 1
        intervals_per_session = args.n_trials * n_repertoire
        total_intervals = args.n_sessions * intervals_per_session

        n_workers = min(args.n_workers, args.n_sessions)
        print(f"Running {args.n_sessions} sessions with {n_workers} parallel workers...")
        print(
            f"  {intervals_per_session} intervals/session × "
            f"{args.n_sessions} sessions = {total_intervals} total intervals"
        )

        # Suppress per-interval progress in parallel mode (interleaved output is unreadable)
        parallel_no_progress = True

        # Shared counters for cross-process progress
        shared_counter = Value('i', 0)         # intervals completed
        shared_init_counter = Value('i', 0)    # sessions that finished NEST init
        shared_started_counter = Value('i', 0) # sessions that started (task picked up)

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
                parallel_no_progress,
                args.same_seed if args.same_seed is not None else 10000 + session_id,
                args.quiet,
                args.n_scaling,
                args.k_scaling,
            )
            for session_id in range(args.n_sessions)
        ]

        # tqdm progress bar updated from shared counter
        from tqdm import tqdm

        t0 = time.time()
        _monitor_stop = threading.Event()

        pbar = tqdm(
            total=total_intervals,
            desc="Intervals",
            unit="it",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        _last_pbar_n = 0

        def _progress_monitor():
            nonlocal _last_pbar_n
            while not _monitor_stop.wait(2):
                done = shared_counter.value
                if done > _last_pbar_n:
                    pbar.update(done - _last_pbar_n)
                    _last_pbar_n = done
                elif done == 0:
                    started = shared_started_counter.value
                    inited = shared_init_counter.value
                    pbar.set_postfix_str(
                        f"init {inited}/{args.n_sessions}  "
                        f"started {started}/{args.n_sessions}"
                    )

        monitor = threading.Thread(target=_progress_monitor, daemon=True)
        monitor.start()

        # Run sessions in parallel
        results = []
        with Pool(
            processes=n_workers,
            initializer=_pool_initializer,
            initargs=(shared_counter, shared_init_counter, shared_started_counter),
        ) as pool:
            for result in pool.imap_unordered(run_session_wrapper, session_args):
                results.append(result)
                pbar.set_postfix_str(
                    f"sessions {len(results)}/{args.n_sessions}"
                )

        # Final update
        done = shared_counter.value
        if done > _last_pbar_n:
            pbar.update(done - _last_pbar_n)
        pbar.close()
        _monitor_stop.set()
        monitor.join()

        results.sort(key=lambda r: r["session_id"])

        # Print summary
        total_time = time.time() - t0
        print(f"\nAll {args.n_sessions} sessions complete in {total_time:.0f}s!")
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
