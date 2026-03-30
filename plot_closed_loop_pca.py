"""
Visualize closed-loop experiment activity in PC-space.

Each closed-loop interval produces one point: the mean firing-rate vector
across all neurons during that interval, projected onto the top 2 PCs.
Points are colored by whether stimulation was delivered.

Usage
-----
    python plot_closed_loop_pca.py
    python plot_closed_loop_pca.py --data-path outputs/data_system_sim_0.05scale/closed_loop_experiment
    python plot_closed_loop_pca.py --history-path outputs/.../closed_loop_history.pkl
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from corcol_params.sim_params import sim_dict

PRESIM_TIME_MS = sim_dict["t_presim"]


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop PCA visualization")
    parser.add_argument(
        "--data-path",
        type=str,
        default="outputs/data_system_sim_0.05scale/closed_loop_experiment",
        help="Directory containing spike recorder .dat files",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Path to closed_loop_history.pkl (optional; auto-detected if present)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to closed_loop_config.pkl (optional; auto-detected if present)",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=5.0,
        help="Closed-loop interval duration in ms (used when no history file exists)",
    )
    parser.add_argument(
        "--total-duration-ms",
        type=float,
        default=None,
        help="Total experiment duration in ms (inferred from spikes if omitted)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of PCA components (2 or 3)",
    )
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Spike loading (mirrors the notebook helper)
# ---------------------------------------------------------------------------

def load_spikes(data_path, begin_ms, end_ms):
    """Load spike times and neuron indices from spike_recorder .dat files."""
    sd_dir = os.path.join(data_path, "spike_recorder")
    spike_files = sorted(glob.glob(os.path.join(sd_dir, "spike_recorder-*.dat")))
    if not spike_files:
        raise FileNotFoundError(f"No spike_recorder files in {sd_dir}")

    node_ids = np.loadtxt(os.path.join(sd_dir, "population_nodeids.dat"), dtype=int)
    if node_ids.ndim == 1:
        node_ids = node_ids[None, :]
    last_node_id = node_ids[-1, -1]

    all_times = []
    all_neurons = []

    for fpath in spike_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    sender = int(float(parts[0]))
                    time_ms = float(parts[1])
                except ValueError:
                    continue
                if begin_ms <= time_ms <= end_ms:
                    neuron_idx = abs(sender - last_node_id) + 1
                    all_times.append(time_ms)
                    all_neurons.append(neuron_idx)

    return np.asarray(all_times), np.asarray(all_neurons)


# ---------------------------------------------------------------------------
# Build per-interval mean activity vectors
# ---------------------------------------------------------------------------

def build_interval_vectors(times, neurons, intervals):
    """
    Compute a mean firing-rate vector (Hz) for each interval.

    Parameters
    ----------
    times : array of spike times (ms, stimulus-aligned)
    neurons : array of neuron indices (1-based)
    intervals : list of (start_ms, end_ms) tuples

    Returns
    -------
    vectors : (n_intervals, n_neurons) array of mean firing rates
    """
    n_neurons = int(np.max(neurons)) if len(neurons) > 0 else 0
    vectors = np.zeros((len(intervals), n_neurons))

    for i, (t_start, t_end) in enumerate(intervals):
        mask = (times >= t_start) & (times < t_end)
        spiking = neurons[mask].astype(int) - 1
        counts = np.bincount(spiking, minlength=n_neurons)
        duration_s = (t_end - t_start) / 1000.0
        vectors[i] = counts / duration_s if duration_s > 0 else counts

    return vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_path = args.data_path

    # Try to auto-detect history and config files
    history = None
    config = None
    history_path = args.history_path or os.path.join(data_path, "closed_loop_history.pkl")
    config_path = args.config_path or os.path.join(data_path, "closed_loop_config.pkl")

    if os.path.isfile(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        print(f"Loaded history: {history_path} ({len(history)} intervals)")

    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        print(f"Loaded config: {config_path}")

    # Determine interval timing
    if config is not None:
        interval_ms = config["closed_loop_interval_ms"]
        total_ms = config["total_duration_ms"]
    else:
        interval_ms = args.interval_ms
        total_ms = args.total_duration_ms

    # Load raw spikes covering the full experiment window
    begin_abs = PRESIM_TIME_MS
    # If total duration unknown, load all spikes after presim
    end_abs = (PRESIM_TIME_MS + total_ms) if total_ms else 1e9

    times, neurons = load_spikes(data_path, begin_abs, end_abs)
    if len(times) == 0:
        print("No spikes found. Exiting.")
        return

    # Shift to stimulus-aligned time (0 = start of closed-loop)
    times = times - PRESIM_TIME_MS

    # Infer total duration from spike data if not provided
    if total_ms is None:
        total_ms = float(np.max(times))
        print(f"Inferred total duration from spikes: {total_ms:.1f} ms")

    # Build interval boundaries
    n_intervals = int(np.ceil(total_ms / interval_ms))
    intervals = []
    for i in range(n_intervals):
        t_start = i * interval_ms
        t_end = min((i + 1) * interval_ms, total_ms)
        intervals.append((t_start, t_end))

    print(f"Epochs: {n_intervals} intervals of {interval_ms} ms")

    # Compute per-interval mean firing-rate vectors
    vectors = build_interval_vectors(times, neurons, intervals)

    # Determine which intervals had stimulation delivered
    stim_delivered = np.zeros(n_intervals, dtype=bool)
    if history is not None:
        for entry in history:
            idx = entry["interval_index"]
            if idx < n_intervals and entry.get("delivered_pattern") is not None:
                stim_delivered[idx] = True
        print(f"Stim delivered in {stim_delivered.sum()}/{n_intervals} intervals")

    # PCA projection
    n_components = min(args.n_components, vectors.shape[1], vectors.shape[0])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(vectors)

    print(
        f"Explained variance: "
        + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(pca.explained_variance_ratio_))
    )

    # Color by interval index for temporal ordering
    interval_indices = np.arange(n_intervals)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))

    # Non-stim intervals
    no_stim_mask = ~stim_delivered
    sc = ax.scatter(
        projected[no_stim_mask, 0],
        projected[no_stim_mask, 1],
        c=interval_indices[no_stim_mask],
        cmap="viridis",
        s=20,
        alpha=0.6,
        edgecolors="none",
        label="No stim",
    )

    # Stim intervals (highlighted)
    if stim_delivered.any():
        ax.scatter(
            projected[stim_delivered, 0],
            projected[stim_delivered, 1],
            c=interval_indices[stim_delivered],
            cmap="viridis",
            s=50,
            alpha=0.9,
            edgecolors="red",
            linewidths=1.2,
            label="Stim delivered",
            vmin=interval_indices[no_stim_mask].min() if no_stim_mask.any() else 0,
            vmax=interval_indices[no_stim_mask].max() if no_stim_mask.any() else n_intervals,
        )

    # Draw trajectory lines connecting consecutive epochs
    ax.plot(projected[:, 0], projected[:, 1], c="gray", alpha=0.2, linewidth=0.5, zorder=0)

    cbar = fig.colorbar(sc, ax=ax, label="Interval index (time →)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(
        f"Closed-loop activity in PC-space\n"
        f"({n_intervals} intervals × {interval_ms} ms, "
        f"{stim_delivered.sum()} stim epochs)"
    )
    ax.legend(loc="best")
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
