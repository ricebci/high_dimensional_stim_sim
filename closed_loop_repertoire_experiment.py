"""
Run closed-loop electrical stimulation using a repertoire controller.

The RepertoireController uses a trained StimEncodingCNN to predict neural
responses to candidate stimulation patterns and selects the pattern whose
predicted response is closest to a target state (zero firing / silence).

How to run
----------
    python closed_loop_repertoire_experiment.py \\
        --model-dir ../../outputs/models/20260312_012405 \\
        --data-dir data/electrical \\
        --total-duration-ms 10000 \\
        --closed-loop-interval-ms 5 \\
        --device mps

Example (fast smoke test)
-------------------------
    python closed_loop_repertoire_experiment.py \\
        --model-dir ../../outputs/models/20260312_012405 \\
        --data-dir data/electrical \\
        --total-duration-ms 100 \\
        --closed-loop-interval-ms 5 \\
        --device mps \\
        --fast-mode \\
        --output-name test_repertoire_run
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import argparse
import glob as _glob
import json
import pickle
from typing import Callable, Dict, Iterable, Optional

import nest
import numpy as np
import torch

import helpers
from controller import RepertoireController
from models import NESTStimSpikeDataset, StimEncodingCNN
from system import SystemNESTSim

# Presimulation period used when generating the training data (ms).
# Spike and stim times from the .dat / JSON files are shifted by this amount
# so that t=0 corresponds to the start of the stimulus delivery period.
_SIM_INIT_PERIOD_MS: float = 1000.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop electrical stimulation with repertoire controller"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory containing meta.pt and model.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Path to the electrical data directory (data/electrical/) containing "
            "spike_recorder-*.dat files. The stim JSON is expected one level up."
        ),
    )
    parser.add_argument("--total-duration-ms", type=float, default=10_000.0)
    parser.add_argument("--closed-loop-interval-ms", type=float, default=5.0)
    parser.add_argument("--output-name", type=str, default="closed_loop_repertoire")
    parser.add_argument("--output-prefix", type=str, default="closed_loop")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string, e.g. 'cpu', 'mps', 'cuda'",
    )
    parser.add_argument(
        "--n-repertoire-update-ms",
        type=float,
        default=10_000.0,
        help="How often (ms of simulated time) to update repertoire from cached observations",
    )
    parser.add_argument(
        "--max-repertoire-len",
        type=int,
        default=100_000,
        help="Maximum number of entries in the stim repertoire",
    )
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--fast-sim-resolution-ms", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.total_duration_ms <= 0:
        raise ValueError("--total-duration-ms must be > 0")
    if args.closed_loop_interval_ms <= 0:
        raise ValueError("--closed-loop-interval-ms must be > 0")
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"--model-dir does not exist: {args.model_dir}")
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"--data-dir does not exist: {args.data_dir}")


# ---------------------------------------------------------------------------
# Repertoire pre-seeding
# ---------------------------------------------------------------------------

def _seed_repertoire(
    controller: RepertoireController,
    meta: dict,
    data_dir: str,
) -> None:
    """Pre-seed the repertoire from training stim data and spike recordings.

    Loads the original stim JSON and spike recorder files used to train the
    encoding model, rebuilds a NESTStimSpikeDataset in stim-locked mode, runs
    model inference on each sample, and adds the results to the repertoire.

    Parameters
    ----------
    controller : RepertoireController
        The controller whose repertoire should be seeded.
    meta : dict
        Checkpoint metadata (same dict passed to RepertoireController).
    data_dir : str
        Path to the electrical data directory (contains spike_recorder-*.dat).
        The stim JSON is expected at ``os.path.join(data_dir, "..", <json_name>)``.
    """
    import polars as pl

    bin_ms: float = float(meta["bin_ms"])
    n_stim_channels: int = meta["n_stim_channels"]
    n_input_bins: int = meta["n_input_bins"]
    n_output_bins: int = meta["n_output_bins"]
    history: int = meta.get("history", 0)
    kernel_sizes = meta.get("kernel_sizes", [3, 3])
    ctx = sum(k - 1 for k in kernel_sizes)

    # --- Load stim events from JSON (one level above data_dir) ---
    stim_json = os.path.join(
        data_dir, "..", "single_electrode_exponential_interarrival_duration.json"
    )
    stim_json = os.path.normpath(stim_json)
    if not os.path.isfile(stim_json):
        raise FileNotFoundError(
            f"Stim JSON not found at {stim_json}. "
            "Ensure --data-dir points to data/electrical/."
        )
    with open(stim_json, "r") as f:
        stims = json.load(f)
    stims_df = pl.DataFrame(stims).with_columns(
        (pl.col("times_ms") - _SIM_INIT_PERIOD_MS).alias("times_ms")
    )

    # --- Load spike recorder files ---
    dat_files = sorted(_glob.glob(os.path.join(data_dir, "spike_recorder-*.dat")))
    if not dat_files:
        raise FileNotFoundError(
            f"No spike_recorder-*.dat files found in {data_dir}"
        )
    spike_frames = []
    for path in dat_files:
        lf = pl.scan_csv(
            path,
            has_header=False,
            separator="\t",
            new_columns=["sender", "time"],
            skip_rows=3,
        ).drop_nulls().with_columns(
            (pl.col("time").cast(pl.Float32) - _SIM_INIT_PERIOD_MS).alias("time")
        )
        spike_frames.append(lf)
    spike_df = pl.collect_all(spike_frames)
    spike_df = pl.concat(spike_df)

    total_duration_ms = float(spike_df["time"].max())

    # --- Build stim-locked dataset (mirrors training setup) ---
    dataset = NESTStimSpikeDataset(
        stim_df=stims_df,
        spike_df=spike_df,
        total_duration_ms=total_duration_ms,
        bin_size_ms=bin_ms,
        n_input_bins=n_input_bins,
        n_output_bins=n_output_bins,
        n_stim_channels=n_stim_channels,
        presim_ms=0.0,
        history=history,
        kernel_sizes=kernel_sizes,
        mode="stim_locked",
        n_neurons=meta["n_neurons"],
    )

    print(
        f"_seed_repertoire: dataset has {len(dataset)} stim-locked samples "
        f"(bin_ms={bin_ms}, n_input_bins={n_input_bins})"
    )

    # --- Collect model inputs and reconstruct stim pattern dicts ---
    model_inputs = []
    stim_patterns = []
    for idx in range(len(dataset)):
        x, _ = dataset[idx]
        # x shape: (n_stim_channels [+ n_neurons if history>0], n_total_bins)
        # The first ctx columns are convolution context; stim starts at ctx.
        stim_bins = x[:n_stim_channels, ctx:].numpy()  # (n_stim_ch, n_input_bins)
        active = np.argwhere(stim_bins != 0)
        if active.size == 0:
            continue  # skip samples with no stimulation
        channels_list = active[:, 0].tolist()
        times_ms_list = (active[:, 1] * bin_ms).tolist()
        amplitudes_list = stim_bins[active[:, 0], active[:, 1]].tolist()
        model_inputs.append(x.numpy())
        stim_patterns.append(
            {
                "channels": channels_list,
                "times_ms": times_ms_list,
                "amplitudes_uA": amplitudes_list,
            }
        )

    controller.fit_stim_repertoire(model_inputs, stim_patterns)


# ---------------------------------------------------------------------------
# Closed-loop experiment loop
# ---------------------------------------------------------------------------

def run_closed_loop_electrical_stim(
    sim: SystemNESTSim,
    total_duration_ms: float,
    closed_loop_interval_ms: float,
    controller: Callable[..., Optional[Dict[str, Iterable[float]]]],
    initial_stim_pattern: Optional[Dict[str, Iterable[float]]] = None,
    output_prefix: str = "closed_loop",
    realtime_progress: bool = True,
) -> Dict[str, str]:
    # Validate duration parameters
    if total_duration_ms <= 0:
        raise ValueError("total_duration_ms must be > 0")
    if closed_loop_interval_ms <= 0:
        raise ValueError("closed_loop_interval_ms must be > 0")

    # Record simulation starting point (after presim) and network size
    loop_start_bio_ms = sim.get_current_biological_time_ms()
    n_neurons = sim.network.n_neurons

    # Snapshot current spike counts so we can detect new spikes each interval
    spike_trains = sim.network.get_spike_train_list()
    prev_spike_counts = np.array([len(train) for train in spike_trains], dtype=int)

    # Set up loop iteration state
    n_intervals = int(np.ceil(total_duration_ms / closed_loop_interval_ms))
    pending_pattern = initial_stim_pattern
    history = []

    for interval_index in range(n_intervals):
        # Compute relative and absolute timing for this interval
        interval_start_ms = interval_index * closed_loop_interval_ms
        interval_duration_ms = min(
            closed_loop_interval_ms,
            total_duration_ms - interval_start_ms,
        )
        interval_end_ms = interval_start_ms + interval_duration_ms
        abs_interval_start_ms = loop_start_bio_ms + interval_start_ms

        # Deliver pending stim pattern (if any) and advance simulation
        delivered = None
        if pending_pattern is not None:
            channels, times_ms, amplitudes_uA = sim.parse_stim_event_sequence(
                pending_pattern,
                sort_by_time=False,
            )

            sim.validate_event_arrays(
                channels,
                times_ms,
                amplitudes_uA,
                input_duration_ms=interval_duration_ms,
            )

            absolute_times_ms = abs_interval_start_ms + times_ms
            generators = sim.create_generators_from_channel_events(
                channels,
                absolute_times_ms,
                amplitudes_uA,
            )
            sim.network.simulate_current_input(
                generators,
                time_ms=float(interval_duration_ms),
            )
            delivered = {
                "channels": channels,
                "times_ms": times_ms,
                "amplitudes_uA": amplitudes_uA,
            }
        else:
            # No stim this interval; just advance the simulation
            nest.Simulate(float(interval_duration_ms))

        # Extract new spikes since last interval for each neuron
        spike_trains = sim.network.get_spike_train_list()
        new_spikes_by_neuron = []
        for neuron_index in range(n_neurons):
            train = np.asarray(spike_trains[neuron_index])
            new_spikes_abs = train[prev_spike_counts[neuron_index] :]
            prev_spike_counts[neuron_index] = len(train)
            new_spikes_rel = new_spikes_abs - loop_start_bio_ms
            new_spikes_by_neuron.append(new_spikes_rel)

        # Ask controller whether to stimulate in the next interval.
        # Pass delivered_pattern so the controller can track what was actually
        # delivered and keep its internal stim buffer accurate.
        pending_pattern = controller(
            interval_index=interval_index,
            interval_start_ms=interval_start_ms,
            interval_end_ms=interval_end_ms,
            interval_duration_ms=interval_duration_ms,
            new_spikes_by_neuron=new_spikes_by_neuron,
            history=history,
            delivered_pattern=delivered,
        )

        # Compute per-neuron spike count vector for L2 reporting
        spike_counts = np.array(
            [len(s) for s in new_spikes_by_neuron], dtype=np.float32
        )
        actual_l2 = float(np.linalg.norm(spike_counts))

        # Log this interval's outcome
        history.append(
            {
                "interval_index": interval_index,
                "interval_start_ms": interval_start_ms,
                "interval_end_ms": interval_end_ms,
                "n_new_spikes": int(spike_counts.sum()),
                "actual_l2": actual_l2,
                "delivered_pattern": delivered,
            }
        )

        if realtime_progress:
            # --- Delivered stim summary ---
            if delivered is not None:
                unique_chs = sorted(set(int(c) for c in delivered["channels"]))
                max_amp = float(np.max(delivered["amplitudes_uA"]))
                stim_str = f"stim=ch{unique_chs} amp={max_amp:.1f}µA"
            else:
                stim_str = "stim=none"

            # --- RepertoireController-specific state (duck-typed) ---
            ctrl_parts = []
            repertoire_entries = getattr(controller, "_entries", None)
            obs_cache = getattr(controller, "_obs_cache", None)
            if repertoire_entries is not None:
                ctrl_parts.append(f"repertoire={len(repertoire_entries)}")
            if obs_cache is not None:
                ctrl_parts.append(f"cache={len(obs_cache)}")
            ctrl_str = " ".join(ctrl_parts)

            print(
                f"[t={interval_end_ms:.1f}/{total_duration_ms:.1f}ms "
                f"iter={interval_index + 1}] "
                f"{stim_str} | "
                f"spikes={history[-1]['n_new_spikes']} L2={actual_l2:.2f}"
                + (f" | {ctrl_str}" if ctrl_str else ""),
                flush=True,
            )

    # Compute windowed spike rates over the full closed-loop duration
    spike_trains = sim.network.get_spike_train_list()
    spike_rates = helpers.compute_spike_rates(
        spike_trains,
        total_duration_ms,
        sim.window_ms,
        sim.overlap_ms,
        presim_time_ms=sim.sim_dict["t_presim"],
    )

    # Save spike rates and history to disk
    rates_path = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_spike_rates.pkl")
    history_path = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_history.pkl")

    with open(rates_path, "wb") as rates_file:
        pickle.dump(spike_rates, rates_file)
    with open(history_path, "wb") as history_file:
        pickle.dump(history, history_file)

    print(f"Saved closed-loop spike rates -> {rates_path}")
    print(f"Saved closed-loop history -> {history_path}")

    return {
        "spike_rates_path": rates_path,
        "history_path": history_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    # --- Load checkpoint metadata ---
    meta = torch.load(os.path.join(args.model_dir, "meta.pt"), weights_only=False)

    # Ensure bin_ms is present (may be absent in older checkpoints).
    # Fall back to the closed-loop interval, which must match the training bin.
    if "bin_ms" not in meta:
        meta["bin_ms"] = args.closed_loop_interval_ms

    # --- Build and load model ---
    model = StimEncodingCNN(
        n_stim_channels=meta["n_stim_channels"],
        n_neurons=meta["n_neurons"],
        n_input_bins=meta["n_input_bins"],
        n_output_bins=meta["n_output_bins"],
        conv_channels=meta.get("conv_channels", [256, 256]),
        kernel_sizes=meta.get("kernel_sizes", [3, 3]),
        fc_dims=meta.get("fc_dims", [256]),
        history=meta.get("history", 1),
        use_init_state=meta.get("use_init_state", True),
    ).to(args.device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "model.pt"),
            weights_only=True,
            map_location=args.device,
        )
    )
    model.eval()
    print(f"Loaded model from {args.model_dir}")

    # --- Target: drive toward silence (zero log-rate for all neurons) ---
    target = np.zeros(meta["n_neurons"], dtype=np.float32)

    # --- Build repertoire controller ---
    repertoire_controller = RepertoireController(
        model=model,
        meta=meta,
        target=target,
        max_len=args.max_repertoire_len,
        n_repertoire_update_ms=args.n_repertoire_update_ms,
        device=args.device,
    )

    # --- Pre-seed the repertoire from training data ---
    print("Pre-seeding repertoire from training data...")
    _seed_repertoire(repertoire_controller, meta, args.data_dir)

    # --- Initialize NEST network (includes presim warmup) ---
    sim = SystemNESTSim(
        output_name=args.output_name,
        fast_mode=args.fast_mode,
        fast_sim_resolution_ms=args.fast_sim_resolution_ms,
    )

    # --- Run the closed-loop experiment ---
    results = run_closed_loop_electrical_stim(
        sim=sim,
        total_duration_ms=args.total_duration_ms,
        closed_loop_interval_ms=args.closed_loop_interval_ms,
        controller=repertoire_controller,
        initial_stim_pattern=None,
        output_prefix=args.output_prefix,
        realtime_progress=not args.no_progress,
    )

    # --- Save full experiment configuration for reproducibility ---
    config = {
        "total_duration_ms": args.total_duration_ms,
        "closed_loop_interval_ms": args.closed_loop_interval_ms,
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "device": args.device,
        "n_repertoire_update_ms": args.n_repertoire_update_ms,
        "max_repertoire_len": args.max_repertoire_len,
        "results": results,
        "data_path": sim.sim_dict["data_path"],
    }

    config_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_config.pkl"
    )
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print("Closed-loop repertoire experiment complete")
    print("Data path:", sim.sim_dict["data_path"])
    print("Results:", results)
    print("Config:", config_path)


if __name__ == "__main__":
    main()
