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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import nest
import numpy as np
import torch
import torch.nn.functional as _F
from sklearn.decomposition import PCA as _PCA

# Fine-tune the encoding model every this many n_repertoire_update_ms periods.
FINETUNE_INTERVALS: int = 5

# Probability of drawing a random single-electrode pattern instead of using the model.
RANDOM_STIM_PROB: float = 0.0

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
    parser.add_argument("--closed-loop-interval-ms", type=float, default=250.0)
    parser.add_argument(
        "--cluster-centers-path",
        type=str,
        default=None,
        help=(
            "Path to cluster_centers_analysis.json produced by the orientation notebook. "
            "Required when --target-orientation-deg is set."
        ),
    )
    parser.add_argument(
        "--target-orientation-deg",
        type=int,
        default=None,
        help=(
            "Orientation angle (degrees) whose cluster center neural vector is used as the "
            "controller target. Requires --cluster-centers-path."
        ),
    )
    parser.add_argument(
        "--visual-metadata-path",
        type=str,
        default="outputs/data_system_sim_0.05scale/visual_orientations_8/visual8_stim_metadata.pkl",
        help=(
            "Path to visual8_stim_metadata.pkl (from run_visual_orientations_raster.py). "
            "When provided, per-neuron visual current generators are recreated in the NEST "
            "kernel using the same preferred orientations and target neurons as the original run."
        ),
    )
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
    parser.add_argument(
        "--random-stim-prob",
        type=float,
        default=RANDOM_STIM_PROB,
        help=(
            "Probability [0, 1) of drawing a random single-electrode pattern instead of "
            "using the greedy model. The chosen pattern is still cached for fine-tuning."
        ),
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help=(
            "Path to visual orientations output directory (contains visual8_spike_rates.pkl "
            "and visual8_run_config.pkl). Required for PC-space distance metric in controller."
        ),
    )
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--fast-sim-resolution-ms", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_pca_components(vis_dir: str, bin_ms: float, n_neurons: int):
    """Fit a 2-component PCA on visual-orientation epoch vectors (in Hz firing-rate space).

    Returns (pca_components, pca_mean) as float32 arrays shaped (2, n_neurons) and
    (n_neurons,), padded / truncated to match the model's neuron count.
    """
    with open(os.path.join(vis_dir, "visual8_spike_rates.pkl"), "rb") as f:
        vis_rates = pickle.load(f)
    with open(os.path.join(vis_dir, "visual8_run_config.pkl"), "rb") as f:
        vis_config = pickle.load(f)

    sample_info    = vis_config["sample_info"]
    vis_centers_ms = np.arange(vis_rates.shape[0]) * 100.0 + 250.0

    epoch_vectors = []
    for start_ms, end_ms, _, _ in sample_info:
        mask = (vis_centers_ms >= start_ms) & (vis_centers_ms < end_ms)
        if mask.sum() == 0:
            continue
        epoch_vectors.append(vis_rates[mask].mean(axis=0))
    epoch_vectors = np.array(epoch_vectors, dtype=np.float32)

    pca = _PCA(n_components=2)
    pca.fit(epoch_vectors)  # epoch_vectors are in Hz
    print(f"PCA explained variance (PC1–2): {pca.explained_variance_ratio_.round(3)}")

    components = pca.components_.astype(np.float32)  # (2, n_vis_neurons)
    mean       = pca.mean_.astype(np.float32)         # (n_vis_neurons,)
    n_vis      = components.shape[1]

    if n_vis < n_neurons:
        components = np.concatenate(
            [components, np.zeros((2, n_neurons - n_vis), dtype=np.float32)], axis=1
        )
        mean = np.concatenate([mean, np.zeros(n_neurons - n_vis, dtype=np.float32)])
    else:
        components = components[:, :n_neurons]
        mean       = mean[:n_neurons]

    return components, mean


def _finetune_model(
    controller,
    cache: List[Tuple[np.ndarray, np.ndarray]],
    n_epochs: int = 3,
    lr: float = 1e-4,
) -> None:
    """Fine-tune the encoding model on cached (model_input, spike_count_bins) pairs.

    cache entries: (x, y) where
        x : (C, T)                    model input built by _build_model_input
        y : (n_neurons, actual_bins)  raw spike counts for the response interval

    For history models the prediction-window history slots in x are filled with
    lag-1 of observed y (teacher forcing) before the forward pass.
    """
    from models import poisson_nll_loss as _poisson_nll

    if not cache:
        return

    ctx       = controller.total_conv_reduction
    n_stim_ch = controller.n_stim_channels
    n_out     = controller.model.n_output_bins
    use_hist  = controller.history > 0
    device    = controller.device

    xs_tf: List[np.ndarray] = []
    ys_tf: List[np.ndarray] = []

    for x_orig, y_bins in cache:
        # y_bins: (n_neurons, actual_bins) — truncate/pad to n_output_bins
        n_actual = y_bins.shape[1]
        if n_actual >= n_out:
            y = y_bins[:, :n_out]
        else:
            y = np.concatenate(
                [y_bins, np.zeros((y_bins.shape[0], n_out - n_actual), dtype=np.float32)],
                axis=1,
            )

        x_tf = x_orig.copy()
        if use_hist and n_out > 1:
            # Teacher forcing: fill prediction-window history slots with lag-1 of y.
            # Slot ctx+t should contain the observed count from time t-1.
            # Slot ctx (t=0) is already the last known count from spike_buf.
            x_tf[n_stim_ch:, ctx + 1 : ctx + n_out] = y[:, : n_out - 1]

        xs_tf.append(x_tf)
        ys_tf.append(y)

    xs = torch.from_numpy(np.stack(xs_tf)).to(device)   # (N, C, T)
    ys = torch.from_numpy(np.stack(ys_tf)).to(device)   # (N, n_neurons, n_out)

    controller.model.train()
    optimizer = torch.optim.Adam(controller.model.parameters(), lr=lr)
    loss_val = float("nan")
    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred     = controller.model(xs)      # (N, n_neurons, n_out)
        loss     = _poisson_nll(pred, ys)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    controller.model.eval()
    print(f"  [finetune] {len(cache)} samples, {n_epochs} epochs, loss={loss_val:.5f}")


def validate_args(args):
    if args.total_duration_ms <= 0:
        raise ValueError("--total-duration-ms must be > 0")
    if args.closed_loop_interval_ms <= 0:
        raise ValueError("--closed-loop-interval-ms must be > 0")
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"--model-dir does not exist: {args.model_dir}")
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"--data-dir does not exist: {args.data_dir}")
    if args.target_orientation_deg is not None and args.cluster_centers_path is None:
        raise ValueError("--cluster-centers-path is required when --target-orientation-deg is set")
    if args.cluster_centers_path is not None and not os.path.isfile(args.cluster_centers_path):
        raise ValueError(f"--cluster-centers-path does not exist: {args.cluster_centers_path}")
    if args.visual_metadata_path is not None and not os.path.isfile(args.visual_metadata_path):
        raise ValueError(f"--visual-metadata-path does not exist: {args.visual_metadata_path}")

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
    # Subtract the presim init period so stim times share the same time base as
    # spike times (t=0 = start of stimulus delivery). Mirrors explore.ipynb cell 0.
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
            # Subtract presim period — mirrors explore.ipynb cell 1 spike loading.
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

def _write_checkpoint(
    data_path: str,
    output_prefix: str,
    history: list,
    spike_counts_matrix: list,
    spike_bins_list: list,
    all_spike_rows: list,
    bin_ms: float,
    bins_per_interval: int,
    interval_index: int,
) -> None:
    """Flush all accumulated data to disk, overwriting previous checkpoint."""
    history_path = os.path.join(data_path, f"{output_prefix}_history.pkl")
    counts_path  = os.path.join(data_path, f"{output_prefix}_spike_counts.pkl")
    bins_path    = os.path.join(data_path, f"{output_prefix}_spike_bins.npz")
    dat_path     = os.path.join(data_path, "spike_recorder",
                                f"{output_prefix}_spike_recorder-0.dat")

    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    if spike_counts_matrix:
        with open(counts_path, "wb") as f:
            pickle.dump(np.stack(spike_counts_matrix, axis=0), f)

    if spike_bins_list:
        arr = np.stack(spike_bins_list, axis=0)
        np.savez_compressed(
            bins_path,
            spike_bins=arr.astype(np.int16),
            bin_ms=np.float32(bin_ms),
            bins_per_interval=np.int32(bins_per_interval),
            n_intervals_completed=np.int32(interval_index + 1),
        )

    sorted_rows = sorted(all_spike_rows, key=lambda x: x[1])
    with open(dat_path, "w") as f:
        f.write("# GID time\n# sender time_ms\n# ---\n")
        for _sender, _t in sorted_rows:
            f.write(f"{_sender}\t{_t:.4f}\n")


def run_closed_loop_electrical_stim(
    sim: SystemNESTSim,
    total_duration_ms: float,
    closed_loop_interval_ms: float,
    controller: Callable[..., Optional[Dict[str, Iterable[float]]]],
    initial_stim_pattern: Optional[Dict[str, Iterable[float]]] = None,
    output_prefix: str = "closed_loop",
    realtime_progress: bool = True,
    bin_ms: float = 25.0,
    n_repertoire_update_ms: float = 10_000.0,
    finetune_intervals: int = FINETUNE_INTERVALS,
    checkpoint_every_n_intervals: int = 10,
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
    bins_per_interval = int(np.ceil(closed_loop_interval_ms / bin_ms))
    pending_pattern = initial_stim_pattern
    history = []
    spike_counts_matrix = []  # (n_intervals, n_neurons) — total counts per interval
    spike_bins_list = []      # (n_intervals, bins_per_interval, n_neurons) — fine-grained

    # Fine-tuning state
    finetune_period_ms = n_repertoire_update_ms * finetune_intervals
    finetune_cache: List[Tuple[np.ndarray, np.ndarray]] = []
    last_finetune_ms: float = 0.0
    # Model input saved from the previous interval when stim was selected;
    # paired with the response observed in the following interval.
    _pending_finetune_x: Optional[np.ndarray] = None
    _pending_stim_delivered: bool = False

    # For writing a spike_recorder-compatible .dat file at the end.
    # Sender IDs follow the NEST convention used by load_spikes_robust:
    #   neuron_idx (1-based) = abs(sender - last_node_id) + 1
    # => sender = last_node_id - (neuron_index)  (neuron_index is 0-based)
    nodeids_path = os.path.join(sim.sim_dict["data_path"], "spike_recorder", "population_nodeids.dat")
    _nodeids = np.loadtxt(nodeids_path, dtype=int)
    if _nodeids.ndim == 1:
        _nodeids = _nodeids[None, :]
    _last_node_id = int(_nodeids[-1, -1])
    all_spike_rows: list = []   # (sender, time_ms_relative_to_loop_start)

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

            # Drop any events that fall outside this interval's window.
            # Patterns are built for the full model horizon (e.g. 250 ms) but
            # the closed-loop interval may be shorter (e.g. 10 ms).
            in_window = times_ms < interval_duration_ms
            channels, times_ms, amplitudes_uA = (
                channels[in_window], times_ms[in_window], amplitudes_uA[in_window]
            )

            if len(channels) > 0:
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
                # All events were outside the window — treat as no-stim
                nest.Simulate(float(interval_duration_ms))
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

        # Accumulate exact spike times for the raster dat file.
        for _ni, _spikes_rel in enumerate(new_spikes_by_neuron):
            if len(_spikes_rel) == 0:
                continue
            _sender = _last_node_id - _ni
            for _t in _spikes_rel:
                all_spike_rows.append((_sender, float(_t)))

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
        spike_counts_matrix.append(spike_counts)

        # Bin spikes into fine-grained model-resolution bins within this interval
        # spike times in new_spikes_by_neuron are relative to loop_start_bio_ms;
        # subtract interval_start_ms to get offset within this interval.
        interval_bins = np.zeros((bins_per_interval, n_neurons), dtype=np.float32)
        for neuron_index, spikes_rel in enumerate(new_spikes_by_neuron):
            if len(spikes_rel) == 0:
                continue
            t_within = np.asarray(spikes_rel) - interval_start_ms
            bin_indices = np.floor(t_within / bin_ms).astype(int)
            valid = (bin_indices >= 0) & (bin_indices < bins_per_interval)
            np.add.at(interval_bins[:, neuron_index], bin_indices[valid], 1)
        spike_bins_list.append(interval_bins)

        # ── Fine-tuning cache ──────────────────────────────────────────────
        # If last interval delivered stim, pair that model input with this
        # interval's observed response (raw spike counts per neuron per bin).
        if _pending_stim_delivered and _pending_finetune_x is not None:
            # interval_bins shape: (bins_per_interval, n_neurons) → transpose to (n_neurons, bins_per_interval)
            y_obs = interval_bins.T.astype(np.float32)
            finetune_cache.append((_pending_finetune_x, y_obs))
        _pending_stim_delivered = delivered is not None
        _pending_finetune_x = (
            getattr(controller, "_last_selected_input", None)
            if pending_pattern is not None else None
        )
        if _pending_finetune_x is not None:
            _pending_finetune_x = _pending_finetune_x.copy()

        # Trigger fine-tuning when enough time has elapsed
        if (finetune_cache
                and finetune_period_ms > 0
                and interval_end_ms - last_finetune_ms >= finetune_period_ms):
            print(f"[t={interval_end_ms:.0f}ms] Fine-tuning on {len(finetune_cache)} cached samples...")
            _finetune_model(controller, finetune_cache)
            last_finetune_ms = interval_end_ms

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

        # Periodic checkpoint — flush accumulated data to disk so a cancelled
        # run leaves usable partial output at the last checkpoint boundary.
        if checkpoint_every_n_intervals > 0 and (interval_index + 1) % checkpoint_every_n_intervals == 0:
            _write_checkpoint(
                data_path=sim.sim_dict["data_path"],
                output_prefix=output_prefix,
                history=history,
                spike_counts_matrix=spike_counts_matrix,
                spike_bins_list=spike_bins_list,
                all_spike_rows=all_spike_rows,
                bin_ms=bin_ms,
                bins_per_interval=bins_per_interval,
                interval_index=interval_index,
            )
            print(f"  [checkpoint] saved at interval {interval_index + 1}", flush=True)

        if realtime_progress:
            # --- Delivered stim summary ---
            if delivered is not None:
                unique_chs = sorted(set(int(c) for c in delivered["channels"]))
                unique_amps = sorted(set(float(a) for a in delivered["amplitudes_uA"]))
                amp_str = (f"{unique_amps[0]:.1f}" if len(unique_amps) == 1
                           else "[" + ", ".join(f"{a:.1f}" for a in unique_amps) + "]")
                was_random = getattr(controller, "_last_was_random", False)
                tag = " [random]" if was_random else ""
                stim_str = f"stim=ch{unique_chs} amp={amp_str}µA{tag}"
            else:
                stim_str = "stim=none"

            # --- RepertoireController-specific state (duck-typed) ---
            ctrl_parts = []
            repertoire_entries = getattr(controller, "_entries", None)
            if repertoire_entries is not None:
                ctrl_parts.append(f"repertoire={len(repertoire_entries)}")
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

    # Final flush — same paths as checkpoints, so this overwrites the last checkpoint
    # with the complete dataset (including spike_rates from the full NEST recording).
    _write_checkpoint(
        data_path=sim.sim_dict["data_path"],
        output_prefix=output_prefix,
        history=history,
        spike_counts_matrix=spike_counts_matrix,
        spike_bins_list=spike_bins_list,
        all_spike_rows=all_spike_rows,
        bin_ms=bin_ms,
        bins_per_interval=bins_per_interval,
        interval_index=n_intervals - 1,
    )

    rates_path   = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_spike_rates.pkl")
    history_path = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_history.pkl")
    counts_path  = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_spike_counts.pkl")
    bins_path    = os.path.join(sim.sim_dict["data_path"], f"{output_prefix}_spike_bins.npz")
    dat_path     = os.path.join(sim.sim_dict["data_path"], "spike_recorder",
                                f"{output_prefix}_spike_recorder-0.dat")

    with open(rates_path, "wb") as f:
        pickle.dump(spike_rates, f)

    spike_bins_arr = np.stack(spike_bins_list, axis=0)
    print(f"Saved closed-loop spike rates  -> {rates_path}")
    print(f"Saved closed-loop history      -> {history_path}")
    print(f"Saved closed-loop spike counts -> {counts_path}")
    print(f"Saved closed-loop spike bins   -> {bins_path}  "
          f"shape={spike_bins_arr.shape}  bin_ms={bin_ms}")
    print(f"Saved closed-loop spike dat    -> {dat_path}  "
          f"({len(all_spike_rows)} spikes)")

    return {
        "spike_rates_path": rates_path,
        "history_path": history_path,
        "spike_counts_path": counts_path,
        "spike_bins_path": bins_path,
        "spike_dat_path": dat_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    # --- Load checkpoint metadata ---
    meta_path = os.path.join(args.model_dir, "meta.json")
    if not os.path.isfile(meta_path):
        # fall back to legacy .pt format
        meta_path = os.path.join(args.model_dir, "meta.pt")
        meta = torch.load(meta_path, weights_only=False)
    else:
        with open(meta_path, "r") as _f:
            meta = json.load(_f)

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

    # --- Target: neural-space vector from an orientation cluster center, or silence ---
    if args.cluster_centers_path is not None and args.target_orientation_deg is not None:
        with open(args.cluster_centers_path, "r") as f:
            cluster_data = json.load(f)
        key = str(int(args.target_orientation_deg))
        if key not in cluster_data:
            raise ValueError(
                f"Orientation {args.target_orientation_deg}° not found in {args.cluster_centers_path}. "
                f"Available: {list(cluster_data.keys())}"
            )
        # full_neural_vector is in Hz (mean firing rate).
        hz_vector = np.array(cluster_data[key]["full_neural_vector"], dtype=np.float32)
        # Pad to model's neuron count if the cluster-center vector is shorter
        # (can happen when the highest-indexed neuron never spiked in the visual run).
        if len(hz_vector) < meta["n_neurons"]:
            hz_vector = np.pad(hz_vector, (0, meta["n_neurons"] - len(hz_vector)))
        target = hz_vector  # Hz, matching the PCA space
        print(
            f"Target: orientation {args.target_orientation_deg}° cluster center "
            f"(||neural||={np.linalg.norm(hz_vector):.2f} Hz, "
            f"||log-rate||={np.linalg.norm(target):.2f})"
        )
    else:
        # Default: drive toward silence (zero log-rate)
        target = np.zeros(meta["n_neurons"], dtype=np.float32)
        print("Target: silence (zero log-rate for all neurons)")


    # --- Load PCA components for PC-space distance metric ---
    if args.vis_dir is not None:
        pca_components, pca_mean = _load_pca_components(
            args.vis_dir, bin_ms=float(meta["bin_ms"]), n_neurons=meta["n_neurons"]
        )
    else:
        raise ValueError(
            "--vis-dir is required (used to fit the 2D PC-space distance metric). "
            "Point it at the visual_orientations_8 output directory."
        )

    # --- Build repertoire controller ---
    # n_relevant_output_bins: how many output bins correspond to the closed-loop
    # interval.  Greedy scoring uses the prediction at bin (n_relevant-1).
    _n_relevant = max(1, min(
        int(meta.get("n_output_bins", meta["n_input_bins"])),
        int(np.ceil(args.closed_loop_interval_ms / float(meta["bin_ms"]))),
    ))
    repertoire_controller = RepertoireController(
        model=model,
        meta=meta,
        target=target,
        pca_components=pca_components,
        pca_mean=pca_mean,
        max_len=args.max_repertoire_len,
        device=args.device,
        random_stim_prob=args.random_stim_prob,
        n_relevant_output_bins=_n_relevant,
    )
    print(f"Controller: n_relevant_output_bins={_n_relevant} "
          f"(interval={args.closed_loop_interval_ms}ms, bin={meta['bin_ms']}ms)")

    # --- Pre-seed the repertoire from training data ---
    print("Pre-seeding repertoire from training data...")
    _seed_repertoire(repertoire_controller, meta, args.data_dir)

    # --- Initialize NEST network (includes presim warmup) ---
    sim = SystemNESTSim(
        output_name=args.output_name,
        fast_mode=args.fast_mode,
        fast_sim_resolution_ms=args.fast_sim_resolution_ms,
    )

    # --- Recreate visual current generators from the orientation run ---
    # This wires per-neuron step_current_generators using the same preferred
    # orientations and target neuron indices as run_visual_orientations_raster.py,
    # so that stimulation during the closed loop goes through the same generators.
    if args.visual_metadata_path is not None:
        print(f"Setting up visual generators from {args.visual_metadata_path} ...")
        sim.setup_visual_generators_from_metadata(args.visual_metadata_path)

    # --- Write partial config now so it survives an OOM kill during the loop ---
    _partial_config = {
        "total_duration_ms": args.total_duration_ms,
        "closed_loop_interval_ms": args.closed_loop_interval_ms,
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "device": args.device,
        "target_orientation_deg": args.target_orientation_deg,
        "cluster_centers_path": args.cluster_centers_path,
        "visual_metadata_path": args.visual_metadata_path,
        "n_repertoire_update_ms": args.n_repertoire_update_ms,
        "max_repertoire_len": args.max_repertoire_len,
        "data_path": sim.sim_dict["data_path"],
    }
    _partial_config_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_config.json"
    )
    with open(_partial_config_path, "w") as _f:
        json.dump(_partial_config, _f, indent=2)

    # --- Run the closed-loop experiment ---
    results = run_closed_loop_electrical_stim(
        sim=sim,
        total_duration_ms=args.total_duration_ms,
        closed_loop_interval_ms=args.closed_loop_interval_ms,
        controller=repertoire_controller,
        initial_stim_pattern=None,
        output_prefix=args.output_prefix,
        realtime_progress=not args.no_progress,
        bin_ms=float(meta["bin_ms"]),
        n_repertoire_update_ms=args.n_repertoire_update_ms,
        finetune_intervals=FINETUNE_INTERVALS,
    )

    # --- Save full experiment configuration for reproducibility ---
    config = {
        "total_duration_ms": args.total_duration_ms,
        "closed_loop_interval_ms": args.closed_loop_interval_ms,
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "device": args.device,
        "target_orientation_deg": args.target_orientation_deg,
        "cluster_centers_path": args.cluster_centers_path,
        "visual_metadata_path": args.visual_metadata_path,
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

    config_json_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_config.json"
    )
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Closed-loop repertoire experiment complete")
    print("Data path:", sim.sim_dict["data_path"])
    print("Results:", results)
    print("Config (pkl):", config_path)
    print("Config (json):", config_json_path)


if __name__ == "__main__":

    main()
