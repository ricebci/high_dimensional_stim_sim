#!/usr/bin/env python3
"""
Run approximation analysis for single-pulse stimulation data.

Given a stimulation data directory (DATA_DIR) and a visual response directory,
computes NNLS approximations, generates figures, and saves results.

Usage:
    python run_approx_analysis.py <data_dir> [--visual-dir <path>] [--k <float>]

Example:
    python run_approx_analysis.py outputs/32_stimchannel_hv_space_1sinterval
    python run_approx_analysis.py outputs/32_stimchannel_vonly_1sinterval --k 0.00043
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA


# ── Argument parsing ──────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Approximation analysis for stimulation data")
    parser.add_argument("data_dir", help="Path to stimulation data directory (e.g. outputs/32_stimchannel_hv_space_1sinterval)")
    parser.add_argument("--visual-dir", default="outputs/visual_orientations_8",
                        help="Path to visual orientations data (default: outputs/visual_orientations_8)")
    parser.add_argument("--k", type=float, default=0.0,
                        help="NNLS weighting parameter (default: 0.0, unweighted)")
    parser.add_argument("--n-stim-channels", type=int, default=32,
                        help="Number of stimulation channels (default: 32)")
    parser.add_argument("--prefix", default=None,
                        help="Session directory prefix (default: auto-detected from data_dir)")
    parser.add_argument("--window-bin-start", type=int, default=0,
                        help="STA window start bin index (default: 0)")
    parser.add_argument("--window-bin-end", type=int, default=14,
                        help="STA window end bin index (default: 14)")
    parser.add_argument("--visual-epoch-ms", type=float, default=70.0,
                        help="Visual epoch duration in ms for spike counting (default: 70.0)")
    return parser.parse_args()


# ── Visual response loading ──────────────────────────────────────────────
def load_spikes_robust(path, name_prefix, begin_ms, end_ms):
    spike_files = sorted(glob.glob(os.path.join(path, f"{name_prefix}-*.dat")))
    if len(spike_files) == 0:
        raise FileNotFoundError(f"No files found for prefix '{name_prefix}' in {path}")

    node_ids = np.loadtxt(os.path.join(path, "population_nodeids.dat"), dtype=int)
    if node_ids.ndim == 1:
        node_ids = node_ids[None, :]
    first_node_id = node_ids[0, 0]
    last_node_id = node_ids[-1, -1]
    n_neurons_total = last_node_id - first_node_id + 1
    print(f"{n_neurons_total} neuron IDs total")

    all_times = []
    all_neurons = []
    for file_path in spike_files:
        with open(file_path, "r", encoding="utf-8") as f:
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
                    all_times.append(time_ms)
                    all_neurons.append(sender)

    all_neurons = np.array(all_neurons)
    print(f"{np.unique(all_neurons).shape[0]} neurons spiked")
    non_firing_neurons = set(range(first_node_id, last_node_id + 1)) - set(all_neurons)
    return np.asarray(all_times), np.asarray(all_neurons), n_neurons_total, spike_files, non_firing_neurons


def load_visual_responses(visual_dir, visual_epoch_ms):
    """Load visual orientation responses and compute avg_vectors."""
    # Auto-detect run config: look for *_run_config.pkl or *_config.pkl
    _pkl_candidates = [f for f in os.listdir(visual_dir) if f.endswith("_run_config.pkl")]
    if not _pkl_candidates:
        _pkl_candidates = [f for f in os.listdir(visual_dir) if f.endswith("_config.pkl")]
    if not _pkl_candidates:
        raise FileNotFoundError(f"No *_run_config.pkl or *_config.pkl found in {visual_dir}")
    run_config_path = os.path.join(visual_dir, _pkl_candidates[0])
    print(f"Using run config: {run_config_path}")
    with open(run_config_path, "rb") as f:
        run_config = pickle.load(f)

    orientations_deg = run_config["orientations_deg"]
    n_repeats = run_config["n_repeats"]
    input_duration_ms = run_config["input_duration_ms"]
    sample_info = run_config["sample_info"]
    presim_ms = run_config["presim_ms"]
    data_path = run_config["data_path"]
    if not os.path.exists(data_path):
        data_path = visual_dir

    print(f"Visual dir: {visual_dir}")
    print(f"Orientations: {orientations_deg}")
    print(f"Repeats: {n_repeats}")

    # Load spikes
    raster_begin_ms = 0
    raster_end_ms = input_duration_ms
    begin_abs_ms = max(0.0, raster_begin_ms + presim_ms)
    end_abs_ms = raster_end_ms + presim_ms

    sd_dir = os.path.join(data_path, "spike_recorder")
    times_all, neurons_all, n_neurons_vis, _, _ = load_spikes_robust(
        sd_dir, "spike_recorder", begin_abs_ms, end_abs_ms
    )
    times_all = times_all - presim_ms

    # Compute avg_vectors using firstk mode
    n_neurons = n_neurons_vis
    unique_angles = sorted(np.unique([s[2] for s in sample_info]))
    trial_portion = visual_epoch_ms

    sample_vectors, sample_angles = [], []
    for start_ms, end_ms, angle_deg, repeat_idx in sample_info:
        ws = start_ms
        mask = (times_all >= ws) & (times_all < ws + trial_portion)
        counts = np.bincount(neurons_all[mask].astype(int) - 1, minlength=n_neurons)[:n_neurons]
        sample_vectors.append(counts)
        sample_angles.append(angle_deg)

    sample_vectors = np.asarray(sample_vectors)
    sample_angles = np.asarray(sample_angles)
    avg_vectors = np.array([sample_vectors[sample_angles == a].mean(0) for a in unique_angles])

    print(f"avg_vectors shape: {avg_vectors.shape}")
    print(f"unique angles: {[int(a) for a in unique_angles]}")

    # PCA projection matrix
    pca = PCA(n_components=2)
    pca.fit(avg_vectors)
    proj, _ = np.linalg.qr(np.array([avg_vectors.mean(0), pca.components_[0], pca.components_[1]]).T)

    # Orientation colormap
    ori_cmap = plt.get_cmap("hsv")
    ori_colors = {a: ori_cmap(i / len(orientations_deg)) for i, a in enumerate(orientations_deg)}

    return avg_vectors, orientations_deg, proj, pca, ori_colors, n_neurons_vis


# ── Electrical stim data loading ─────────────────────────────────────────
def detect_session_dirs(data_dir, prefix=None):
    """Auto-detect session subdirectories. If prefix is given, filter by it;
    otherwise find all subdirs containing '_session_'."""
    if prefix:
        return sorted([d for d in os.listdir(data_dir)
                       if d.startswith(prefix)
                       and os.path.isdir(os.path.join(data_dir, d))])
    return sorted([d for d in os.listdir(data_dir)
                   if "_session_" in d
                   and os.path.isdir(os.path.join(data_dir, d))])


def load_stim_sessions(data_dir, session_dirs, n_stim_channels):
    """Load and aggregate multi-session spike data."""
    if len(session_dirs) == 0:
        raise RuntimeError(f"No sessions found in {data_dir}")
    print(f"Loading {len(session_dirs)} sessions from {data_dir}")

    all_spike_bins_list = []
    all_histories = []
    bin_ms = None

    for session_dir in session_dirs:
        session_path = os.path.join(data_dir, session_dir)
        session_id = int(session_dir.split("_")[-1])
        print(f"  Session {session_id}: {session_dir}")

        # Load config
        _cfg_files = [f for f in os.listdir(session_path) if f.endswith("_config.json")]
        _pkl_files = [f for f in os.listdir(session_path) if f.endswith("_config.pkl")]
        if _cfg_files:
            with open(os.path.join(session_path, _cfg_files[0])) as f:
                session_config = json.load(f)
        elif _pkl_files:
            with open(os.path.join(session_path, _pkl_files[0]), "rb") as f:
                session_config = pickle.load(f)
        else:
            raise FileNotFoundError(f"No config file found in {session_path}")

        # Load history
        _hist_file = next(f for f in os.listdir(session_path) if f.endswith("_history.pkl"))
        with open(os.path.join(session_path, _hist_file), "rb") as f:
            session_history = pickle.load(f)

        # Load spike bins
        _bins_file = next(f for f in os.listdir(session_path) if f.endswith("_spike_bins.npz"))
        d = np.load(os.path.join(session_path, _bins_file))
        bin_ms = float(d["bin_ms"])
        session_spike_bins = d["spike_bins"].astype(np.float32)

        # Keep complete trials only
        n_intervals_session = session_spike_bins.shape[0]
        n_repertoire_session = int(session_config.get("n_repertoire", n_stim_channels + 1))
        n_complete_trials = n_intervals_session // n_repertoire_session
        n_complete_intervals = n_complete_trials * n_repertoire_session

        session_spike_bins = session_spike_bins[:n_complete_intervals]
        session_history = session_history[:n_complete_intervals]

        all_spike_bins_list.append(session_spike_bins)
        all_histories.extend(session_history)

    spike_bins_arr = np.concatenate(all_spike_bins_list, axis=0)
    # Neuron axis from get_spike_train_list uses abs(sender - last_node_id) + 1,
    # so index 0 = deepest. Flip to node-ID order: index 0 = shallowest.
    spike_bins_arr = spike_bins_arr[:, :, ::-1].copy()
    print(f"spike_bins_arr shape: {spike_bins_arr.shape} (neuron axis: node-ID order)")
    return spike_bins_arr, all_histories, bin_ms


# ── Electrode grid ───────────────────────────────────────────────────────
def make_electrode_grid(data_dir, session_dirs, n_stim_channels):
    """Reproduce electrode grid from config bounds."""
    _first_session_path = os.path.join(data_dir, session_dirs[0])
    _first_config_path = [f for f in os.listdir(_first_session_path) if f.endswith("config.json")][0]
    with open(os.path.join(_first_session_path, _first_config_path)) as f:
        bounds = json.load(f)["electrode_volume_bounds_um"]

    v_min, v_max = bounds["v_min"], bounds["v_max"]
    h_min, h_max = bounds["h_min"], bounds["h_max"]

    _n_ch = n_stim_channels
    _h_has_extent = (h_max != h_min)
    _v_has_extent = (v_max != v_min)
    if _h_has_extent and _v_has_extent:
        _n_h = int(np.sqrt(_n_ch))
        while _n_ch % _n_h != 0:
            _n_h -= 1
        _n_v = _n_ch // _n_h
    elif _v_has_extent:
        _n_h, _n_v = 1, _n_ch
    elif _h_has_extent:
        _n_h, _n_v = _n_ch, 1
    else:
        _n_h, _n_v = 1, _n_ch

    _h_vals = np.linspace(h_min, h_max, _n_h)
    _v_vals = np.linspace(v_min, v_max, _n_v)
    _hh, _vv = np.meshgrid(_h_vals, _v_vals)
    return _hh.ravel(), _vv.ravel()


def make_neuron_locations():
    """Reproduce neuron locations from network params."""
    from corcol_params.network_params import net_dict as _net_dict
    _pop_names = _net_dict["populations"]
    _pop_depths = _net_dict["pop_depth_ranges_um"]
    _horiz = _net_dict["pop_horiz_ranges_um"]
    _full_n = _net_dict["full_num_neurons"]
    _nscale = _net_dict["N_scaling"]
    _num_neurons_per_pop = np.round(_full_n * _nscale).astype(int)

    _rng = np.random.default_rng(seed=0)
    neuron_h_all, neuron_v_all = [], []
    for _i, _pname in enumerate(_pop_names):
        _nn = _num_neurons_per_pop[_i]
        _locs = _rng.random((_nn, 2))
        _locs[:, 0] = _locs[:, 0] * (_horiz[1] - _horiz[0]) + _horiz[0]
        _d = _pop_depths[_i]
        _locs[:, 1] = _locs[:, 1] * (_d[1] - _d[0]) + _d[0]
        neuron_h_all.append(_locs[:, 0])
        neuron_v_all.append(_locs[:, 1])

    return np.concatenate(neuron_h_all), np.concatenate(neuron_v_all)


# ── STA computation ─────────────────────────────────────────────────────
def compute_stas(spike_bins_arr, history, bin_ms, n_stim_channels,
                 electrode_h, electrode_v, neuron_h_all, neuron_v_all, viz_dir,
                 window_bin_start=0, window_bin_end=14):
    """Compute STAs and save heatmaps. Returns sta_spkcount dict and pattern_keys."""
    WINDOW_BIN_START, WINDOW_BIN_END = window_bin_start, window_bin_end
    n_intervals, bins_per_interval, n_neurons_total = spike_bins_arr.shape
    neuron_slice = slice(0, n_neurons_total)

    # Collect interval indices per pattern
    pattern_to_indices = defaultdict(list)
    for i, h in enumerate(history):
        pat = h.get("delivered_pattern")
        if pat is None or len(pat.get("channels", [])) == 0:
            continue
        ch = int(pat["channels"][0])
        amp = float(pat["amplitudes_uA"][0])
        pattern_to_indices[(ch, amp)].append(i)

    if len(pattern_to_indices) == 0:
        raise RuntimeError("No stimulated intervals found; cannot compute STA per pattern.")

    pattern_keys = sorted(pattern_to_indices.keys(), key=lambda k: (k[1], k[0]))

    # Layer boundaries
    from corcol_params.network_params import net_dict as _nd
    _pop_names_sta = _nd["populations"]
    _full_n_sta = _nd["full_num_neurons"]
    _nscale_sta = _nd["N_scaling"]
    _num_per_pop_sta = np.round(_full_n_sta * _nscale_sta).astype(int)
    _pop_boundaries = np.cumsum(_num_per_pop_sta)
    _pop_starts = np.concatenate([[0], _pop_boundaries[:-1]])
    _pop_mids = (_pop_starts + _pop_boundaries) / 2.0

    print(f"Computing STAs for {len(pattern_keys)} patterns...")

    STA_DPI = 100
    STA_VMIN, STA_VMAX = 0.0, 200.0
    STA_GAMMA = 1.0
    _sta_norm = PowerNorm(gamma=STA_GAMMA, vmin=STA_VMIN, vmax=STA_VMAX, clip=True)

    sta_spkcount = {}
    sta_dir = os.path.join(viz_dir, "sta_neuron_time")
    os.makedirs(sta_dir, exist_ok=True)

    for ch, amp in pattern_keys:
        idx = pattern_to_indices[(ch, amp)]
        sta_hz = spike_bins_arr[idx, WINDOW_BIN_START:WINDOW_BIN_END, neuron_slice].mean(axis=0) / float(bin_ms) * 1000.0
        sta_spkcount[(ch, amp)] = spike_bins_arr[idx, WINDOW_BIN_START:WINDOW_BIN_END, neuron_slice].mean(axis=0).sum(axis=0)

        fig, ax = plt.subplots(figsize=(9, 5))
        _window = sta_hz[WINDOW_BIN_START:WINDOW_BIN_END, :].T
        im = ax.imshow(_window, aspect="auto", origin="upper", cmap='Greys',
                       norm=_sta_norm, interpolation='bilinear')

        for _bi, _bnd in enumerate(_pop_boundaries[:-1]):
            ax.axhline(y=_bnd - 0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks([_pop_mids[_bi] for _bi in range(len(_pop_names_sta))])
        ax2.set_yticklabels(_pop_names_sta, fontsize=7, fontweight="bold")
        ax2.tick_params(length=0)

        _elec_h, _elec_v = electrode_h[ch], electrode_v[ch]
        _dists = np.sqrt((neuron_h_all - _elec_h)**2 + (neuron_v_all - _elec_v)**2)
        _nearest_idx = int(np.argmin(_dists))
        ax.axhline(y=_nearest_idx, color="red", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.text(_window.shape[1] - 0.05, _nearest_idx, f"nearest (#{_nearest_idx})",
                fontsize=6, color="red", ha="right", va="bottom")

        ax.set_title(f"STA Heatmap | ch{ch:02d}, {amp:g} uA | n={len(idx)} intervals")
        ax.set_xlabel("Time bin (5 ms)")
        ax.set_ylabel("Neuron index (shallow -> deep downwards)")
        ax.set_xticks(range(WINDOW_BIN_END - WINDOW_BIN_START))
        ax.set_xticklabels(range(WINDOW_BIN_START, WINDOW_BIN_END))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Firing rate (Hz)")

        out = os.path.join(sta_dir, f"sta_heatmap_ch{ch:02d}_amp{amp:g}uA.png")
        plt.tight_layout()
        plt.savefig(out, dpi=STA_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved STA heatmaps in: {sta_dir}")
    return sta_spkcount, pattern_keys


# ── NNLS solver ──────────────────────────────────────────────────────────
def solve_nonneg_least_squares(A, b, k=0.0):
    """Solve min_x || W(Ax - b) ||_2^2  subject to x >= 0"""
    n_patterns = A.shape[1]
    x = cp.Variable(n_patterns, nonneg=True)
    if k == 0:
        W = np.eye(len(b))
    else:
        W = np.diag(b + k)
    objective = cp.Minimize(cp.sum_squares(W @ (A @ x - b)))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    x_opt = x.value
    residual_norm = np.linalg.norm(A @ x_opt - b)
    return x_opt, residual_norm


def solve_mlp(A, b, k=0.0, hidden_sizes=(128, 64), lr=1e-3, n_iters=5000, seed=42):
    """
    Solve min_x || f(x) - b ||_2^2 subject to x >= 0, where f is a small MLP
    trained so that f(x) ≈ A @ x near the optimum but can capture nonlinearities.

    A: (n_features, n_patterns) — used to initialise the first layer
    b: (n_features,)
    Returns:
        x_opt: (n_patterns,) optimal non-negative coefficients
        residual_norm: scalar, || f(x_opt) - b ||_2
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_features, n_patterns = A.shape

    # Build MLP: n_patterns -> hidden -> ... -> n_features
    layers = []
    in_dim = n_patterns
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, n_features))
    mlp = nn.Sequential(*layers)

    # Initialise first layer from A so the MLP starts near the linear solution
    with torch.no_grad():
        proj_mat = torch.randn(hidden_sizes[0], n_features)
        mlp[0].weight.copy_(proj_mat @ torch.tensor(A, dtype=torch.float32))
        mlp[0].bias.zero_()

    b_t = torch.tensor(b, dtype=torch.float32)
    # Optimisable coefficients (unconstrained); softplus ensures x >= 0
    x_raw = nn.Parameter(torch.zeros(n_patterns))

    if k > 0:
        W = torch.tensor(np.diag(b + k), dtype=torch.float32)
    else:
        W = torch.eye(n_features)

    optimizer = torch.optim.Adam(list(mlp.parameters()) + [x_raw], lr=lr)

    for i in range(n_iters):
        optimizer.zero_grad()
        x_pos = nn.functional.softplus(x_raw)  # non-negative
        pred = mlp(x_pos)
        loss = ((W @ (pred - b_t)) ** 2).sum()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print(f"  iter {i+1}/{n_iters}  loss={loss.item():.4f}")

    with torch.no_grad():
        x_opt = nn.functional.softplus(x_raw).numpy()
        pred_final = mlp(torch.tensor(x_opt, dtype=torch.float32)).numpy()
        residual_norm = np.linalg.norm(pred_final - b)

    return x_opt, residual_norm


def run_nnls_approximation(A, avg_vectors, pattern_keys, k, proj):
    """Run NNLS for each orientation target. Returns approx_log, stim_log."""
    approx_log = []
    stim_log = []
    for itarget in range(avg_vectors.shape[0]):
        b = avg_vectors[itarget]
        print(f"  Target {itarget}: b sum={b.sum():.2f}")
        x_opt, residual_norm = solve_nonneg_least_squares(A, b, k=k)
        approx = (A @ x_opt)[None, :]
        print(f"    approx sum={approx.sum():.2f}, residual={residual_norm:.4f}")
        approx_log.append(approx)
        stim_log.append(x_opt)

    approx_log = np.squeeze(np.array(approx_log))
    approx_proj = approx_log.dot(proj)
    return approx_log, stim_log, approx_proj


# ── Visualization functions ──────────────────────────────────────────────
def plot_approx_vs_actual_sums(avg_vectors, approx_log, viz_dir):
    """Bar chart comparing actual vs approximation sums per orientation."""
    actual_sums = [avg_vectors[i].sum() for i in range(avg_vectors.shape[0])]
    approx_sums = [approx_log[i].sum() for i in range(approx_log.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 4))
    x_idx = np.arange(len(actual_sums))
    ax.bar(x_idx - 0.15, actual_sums, width=0.3, label="Actual (b)")
    ax.bar(x_idx + 0.15, approx_sums, width=0.3, label="Approx (Ax)")
    ax.set_xlabel("Orientation index")
    ax.set_ylabel("Sum of firing rates")
    ax.set_title("Approx sum vs. actual sum per orientation")
    ax.set_xticks(x_idx)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(viz_dir, "approx_vs_actual_sums.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_stim_coefficients(stim_log, pattern_keys, viz_dir):
    """Heatmaps of stimulation coefficients per orientation."""
    channels = sorted(set(k[0] for k in pattern_keys))
    amplitudes = sorted(set(k[1] for k in pattern_keys))
    n_ch, n_amp = len(channels), len(amplitudes)
    print(f"Electrodes: {n_ch}, Amplitudes: {n_amp}")

    # Line plot
    fig_line, ax_line = plt.subplots(figsize=(10, 4))
    ax_line.plot(np.array(stim_log).T)
    ax_line.set_xlabel("Pattern index")
    ax_line.set_ylabel("Coefficient")
    ax_line.set_title("Stimulation coefficients (all orientations)")
    plt.tight_layout()
    out_line = os.path.join(viz_dir, "stim_coefficients_line.png")
    plt.savefig(out_line, dpi=100, bbox_inches="tight")
    plt.close(fig_line)
    print(f"Saved: {out_line}")

    # Heatmaps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    vmax = max(np.max(s) for s in stim_log)
    for itarget, ax in enumerate(axes.flat):
        if itarget >= len(stim_log):
            ax.axis("off")
            continue
        grid = stim_log[itarget].reshape(n_amp, n_ch).T
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"Orientation {itarget}")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Electrode")
        ax.set_xticks(range(n_amp))
        ax.set_xticklabels([f"{a:.0f}" for a in amplitudes], rotation=45, fontsize=8)
        ax.set_yticks(range(0, n_ch, 4))
        ax.set_yticklabels([str(channels[i]) for i in range(0, n_ch, 4)], fontsize=8)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Coefficient")
    fig.suptitle("Stimulation coefficients per orientation (electrode x amplitude)", fontsize=14)
    out = os.path.join(viz_dir, "stimulation_coefficients_per_orientation.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_3d_projections(avg_vectors, approx_log, orientations_deg, ori_colors, proj, viz_dir):
    """3D scatter of actual vs approximation in PC space (saved as interactive HTML)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    avg_proj = avg_vectors.dot(proj)
    approx_proj = approx_log.dot(proj)

    def rgba_from_mpl(c, alpha=1.0):
        r, g, b, _ = c
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Full View (Actual + Approx)", "Approximation Only (Zoomed)"],
    )

    for i, angle in enumerate(orientations_deg):
        color = rgba_from_mpl(ori_colors[angle])
        fig.add_trace(go.Scatter3d(
            x=[avg_proj[i, 0]], y=[avg_proj[i, 1]], z=[avg_proj[i, 2]],
            mode="markers+text",
            marker=dict(size=8, color=color, line=dict(color="black", width=1)),
            text=[f"{angle}\u00b0"], textposition="top center",
            name=f"{angle}\u00b0", legendgroup=f"{angle}",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=[approx_proj[i, 0]], y=[approx_proj[i, 1]], z=[approx_proj[i, 2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="square", line=dict(color="black", width=1)),
            text=[f"{angle}\u00b0"], textposition="top center",
            name=f"Approx: {angle}\u00b0", legendgroup=f"approx_{angle}",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=[approx_proj[i, 0]], y=[approx_proj[i, 1]], z=[approx_proj[i, 2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="circle", line=dict(color="black", width=1)),
            text=[f"{angle}\u00b0"], textposition="top center",
            name=f"Approx: {angle}\u00b0", legendgroup=f"approx_{angle}",
            showlegend=False,
        ), row=1, col=2)

    for col in [1, 2]:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers",
            marker=dict(size=10, color="black", symbol="x"),
            name="Zero", showlegend=(col == 1),
        ), row=1, col=col)

    ap_min = approx_proj.min(axis=0)
    ap_max = approx_proj.max(axis=0)
    ap_pad = (ap_max - ap_min) * 0.2 + 1e-6

    scene_labels = dict(xaxis_title="Condition-Invariant", yaxis_title="PC1", zaxis_title="PC2")
    fig.update_layout(
        title="PCA of Average Visual Response Vectors",
        scene=dict(**scene_labels),
        scene2=dict(
            **scene_labels,
            xaxis=dict(range=[ap_min[0] - ap_pad[0], ap_max[0] + ap_pad[0]]),
            yaxis=dict(range=[ap_min[1] - ap_pad[1], ap_max[1] + ap_pad[1]]),
            zaxis=dict(range=[ap_min[2] - ap_pad[2], ap_max[2] + ap_pad[2]]),
        ),
        legend=dict(title="Orientation"),
        width=1500, height=650,
    )
    out = os.path.join(viz_dir, "3d_projections.html")
    fig.write_html(out)
    print(f"Saved: {out}")


def compute_principal_angles(avg_vectors, A):
    """Compute principal angles between orientation and stimulation subspaces."""
    n_components = min(8, avg_vectors.shape[0], A.shape[1])
    pca_orientations = PCA(n_components=n_components)
    pca_orientations.fit(avg_vectors)

    pca_stim = PCA(n_components=n_components)
    pca_stim.fit(A.T)

    angles = subspace_angles(pca_orientations.components_.T, pca_stim.components_.T)
    print(f"Principal angles (degrees): {np.degrees(angles)}")
    return angles


def compute_orientation_mses(avg_vectors, approx_log, orientations_deg):
    """Compute MSE between target and approximation per orientation."""
    mse_per_orientation = np.mean((avg_vectors - approx_log) ** 2, axis=1)
    for i, angle in enumerate(orientations_deg):
        print(f"  Orientation {angle}\u00b0: MSE = {mse_per_orientation[i]:.4f}")
    print(f"  Average MSE: {mse_per_orientation.mean():.4f}")
    return mse_per_orientation


def plot_mse_and_angles(mse_per_orientation, orientations_deg, angles, viz_dir):
    """Two-subplot figure: per-orientation MSEs on top, principal angles on bottom."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Top: MSE per orientation
    x_idx = np.arange(len(orientations_deg))
    ax1.bar(x_idx, mse_per_orientation, color="steelblue")
    ax1.axhline(mse_per_orientation.mean(), color="red", linestyle="--", linewidth=1,
                label=f"Mean MSE = {mse_per_orientation.mean():.4f}")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"{a}°" for a in orientations_deg], fontsize=8)
    ax1.set_xlabel("Orientation")
    ax1.set_ylabel("MSE")
    ax1.set_title("Approximation MSE per Orientation")
    ax1.legend(fontsize=8)

    # Bottom: principal angles
    angles_deg = np.degrees(angles)
    ax2.bar(np.arange(len(angles_deg)), angles_deg, color="darkorange")
    ax2.set_xticks(np.arange(len(angles_deg)))
    ax2.set_xticklabels([f"{i+1}" for i in range(len(angles_deg))], fontsize=8)
    ax2.set_xlabel("Component index")
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("Principal Angles Between Orientation and Stimulation Subspaces")

    plt.tight_layout()
    out = os.path.join(viz_dir, "mse_and_principal_angles.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_approx_config(approx_log, stim_log, pattern_keys, mse_per_orientation,
                       angles, orientations_deg, out_dir, visual_dir=None):
    """Save approximation results as JSON."""
    approx_config = {
        "approx_vectors": approx_log.tolist(),
        "stim_coefficients": np.array(stim_log).tolist(),
        "pattern_keys": [list(k) for k in pattern_keys],
        "orientation_mses": mse_per_orientation.tolist(),
        "mean_mse": mse_per_orientation.mean(),
        "principal_angles_rad": angles.tolist(),
        "principal_angles_deg": np.degrees(angles).tolist(),
        "orientations_deg": list(orientations_deg),
        **({"visual_dir": visual_dir} if visual_dir is not None else {}),
    }

    approx_config_path = os.path.join(out_dir, "approx_config.json")
    with open(approx_config_path, "w") as f:
        json.dump(approx_config, f)

    print(f"Saved approximation config to: {approx_config_path}")
    return approx_config_path


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    data_dir = args.data_dir
    visual_dir = args.visual_dir
    k = args.k
    n_stim_channels = args.n_stim_channels
    prefix = args.prefix
    window_bin_start = args.window_bin_start
    window_bin_end = args.window_bin_end
    visual_epoch_ms = args.visual_epoch_ms

    assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"
    assert os.path.exists(visual_dir), f"Visual directory not found: {visual_dir}"

    # Output directory encodes key parameters + visual dataset
    visual_label = os.path.basename(os.path.normpath(visual_dir))
    run_label = f"k{k}_wbin{window_bin_start}-{window_bin_end}_vis-{visual_label}"
    viz_dir = os.path.join(data_dir, "viz", run_label)
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Output directory: {viz_dir}")

    # 1. Load visual responses
    print("\n=== Loading visual responses ===")
    avg_vectors, orientations_deg, proj, pca, ori_colors, n_neurons_vis = load_visual_responses(
        visual_dir, visual_epoch_ms
    )

    # 2. Load stim sessions
    print("\n=== Loading stimulation sessions ===")
    session_dirs = detect_session_dirs(data_dir, prefix)
    assert len(session_dirs) > 0, f"No session directories found in {data_dir}"
    spike_bins_arr, history, bin_ms = load_stim_sessions(data_dir, session_dirs, n_stim_channels)

    # Validate window bins against epoch and bin resolution
    window_ms = bin_ms * window_bin_end
    assert window_ms <= visual_epoch_ms, (
        f"bin_ms * window_bin_end = {bin_ms} * {window_bin_end} = {window_ms} ms "
        f"exceeds visual_epoch_ms = {visual_epoch_ms} ms"
    )

    # 3. Electrode and neuron locations
    print("\n=== Setting up electrode and neuron locations ===")
    electrode_h, electrode_v = make_electrode_grid(data_dir, session_dirs, n_stim_channels)
    neuron_h_all, neuron_v_all = make_neuron_locations()

    # 4. Compute STAs
    print("\n=== Computing STAs ===")
    sta_spkcount, pattern_keys = compute_stas(
        spike_bins_arr, history, bin_ms, n_stim_channels,
        electrode_h, electrode_v, neuron_h_all, neuron_v_all, viz_dir,
        window_bin_start=window_bin_start, window_bin_end=window_bin_end,
    )

    # 5. Build dictionary matrix A
    A = np.column_stack([sta_spkcount[key] for key in pattern_keys])
    print(f"\nDictionary matrix A shape: {A.shape}")

    # 6. Run NNLS approximation
    print(f"\n=== Running NNLS approximation (k={k}) ===")
    approx_log, stim_log, approx_proj = run_nnls_approximation(A, avg_vectors, pattern_keys, k, proj)

    # 7. Visualizations
    print("\n=== Generating visualizations ===")
    plot_approx_vs_actual_sums(avg_vectors, approx_log, viz_dir)
    plot_stim_coefficients(stim_log, pattern_keys, viz_dir)
    plot_3d_projections(avg_vectors, approx_log, orientations_deg, ori_colors, proj, viz_dir)

    # 8. Principal angles
    print("\n=== Principal angles ===")
    angles = compute_principal_angles(avg_vectors, A)

    # 9. Orientation MSEs
    print("\n=== Orientation MSEs ===")
    mse_per_orientation = compute_orientation_mses(avg_vectors, approx_log, orientations_deg)

    # 10. MSE & principal angles plot
    plot_mse_and_angles(mse_per_orientation, orientations_deg, angles, viz_dir)

    # 11. Save config
    print("\n=== Saving results ===")
    save_approx_config(approx_log, stim_log, pattern_keys, mse_per_orientation,
                       angles, orientations_deg, viz_dir, visual_dir=visual_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
