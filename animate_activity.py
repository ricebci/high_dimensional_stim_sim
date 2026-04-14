#!/usr/bin/env python3
"""
Animate population activity showing neurons lighting up in response to stimuli.

Two modes:
  electrical  -- Shows electrode grid + neurons; electrodes flash at stim times,
                 nearby neurons light up when they fire.
  visual      -- Shows neurons only; color-codes visual stimulus epochs,
                 neurons light up when they fire.

Usage:
  python animate_activity.py electrical --session-dir outputs/128_stimchannel_hv_space_1sinterval/128_stimchannel_1sinterval_no_encoder_run1_session_001 --t-start 900 --t-end 2500 --fps 30 -o elec_anim.mp4

  python animate_activity.py visual --visual-dir outputs/visual_orientations_8 --t-start 50 --t-end 300 --fps 30 -o vis_anim.mp4
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Network geometry (deterministic, matches notebook) ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corcol_params.network_params import net_dict


def _build_neuron_positions(n_scaling=None, rng_seed=None):
    """Return (h, v, pop_labels) arrays for all neurons in the scaled network.

    Must match ``Network.__create_neuronal_populations`` in network_cortcol.py:
    same RNG seed (sim_params rng_seed) and depth-sorting within each population.

    Parameters
    ----------
    n_scaling : float, optional
        Override N_scaling. If None, uses the value from network_params.py.
    rng_seed : int, optional
        Override RNG seed. If None, uses sim_params default.
    """
    from corcol_params.sim_params import sim_dict
    pop_names = net_dict["populations"]
    pop_depths = net_dict["pop_depth_ranges_um"]
    horiz = net_dict["pop_horiz_ranges_um"]
    full_n = np.array(net_dict["full_num_neurons"])
    nscale = n_scaling if n_scaling is not None else net_dict["N_scaling"]
    num_neurons_per_pop = np.round(full_n * nscale).astype(int)

    seed = rng_seed if rng_seed is not None else sim_dict["rng_seed"]
    rng = np.random.default_rng(seed=seed)
    h_all, v_all, pop_all = [], [], []
    for i, pname in enumerate(pop_names):
        nn = num_neurons_per_pop[i]
        locs = rng.random((nn, 2))
        locs[:, 0] = locs[:, 0] * (horiz[1] - horiz[0]) + horiz[0]
        d = pop_depths[i]
        locs[:, 1] = locs[:, 1] * (d[1] - d[0]) + d[0]
        locs = locs[np.argsort(locs[:, 1])]
        h_all.append(locs[:, 0])
        v_all.append(locs[:, 1])
        pop_all.extend([pname] * nn)
    return np.concatenate(h_all), np.concatenate(v_all), pop_all


def _check_neuron_placement_consistency(h, v, metadata_path):
    """Warn if reconstructed neuron positions don't match saved metadata.

    Parameters
    ----------
    h, v : np.ndarray
        Reconstructed neuron horizontal and vertical positions.
    metadata_path : str
        Path to a ``*_stim_metadata.pkl`` that may contain ``neuron_locations``.
    """
    import warnings
    if not os.path.isfile(metadata_path):
        return
    try:
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
    except Exception:
        return
    saved_locations = meta.get("neuron_locations")
    if saved_locations is None:
        return
    reconstructed = np.column_stack([h, v])
    if reconstructed.shape != saved_locations.shape or not np.allclose(
        reconstructed, saved_locations
    ):
        warnings.warn(
            f"Reconstructed neuron positions do not match those saved in "
            f"{metadata_path}. This may be caused by a different RNG seed or "
            f"N_scaling. Visual analysis results may be incorrect.",
            UserWarning,
            stacklevel=2,
        )


def _build_electrode_grid(bounds, n_ch):
    """Reproduce electrode grid from config bounds (matches notebook logic)."""
    v_min, v_max = bounds["v_min"], bounds["v_max"]
    h_min, h_max = bounds["h_min"], bounds["h_max"]
    h_has = h_max != h_min
    v_has = v_max != v_min
    if h_has and v_has:
        n_h = int(np.sqrt(n_ch))
        while n_ch % n_h != 0:
            n_h -= 1
        n_v = n_ch // n_h
    elif v_has:
        n_h, n_v = 1, n_ch
    elif h_has:
        n_h, n_v = n_ch, 1
    else:
        n_h, n_v = 1, n_ch
    hh, vv = np.meshgrid(
        np.linspace(h_min, h_max, n_h), np.linspace(v_min, v_max, n_v)
    )
    return hh.ravel(), vv.ravel()


# ── Spike loading from .dat files ─────────────────────────────────────────

def _load_spikes_from_dat(spike_recorder_dir, t_start, t_end):
    """
    Load raw spike times from NEST .dat files.

    Returns
    -------
    spike_times : list[np.ndarray]
        Per-neuron (0-indexed) arrays of spike times in ms.
    n_neurons : int
    """
    # Read population node-id ranges
    nid_path = os.path.join(spike_recorder_dir, "population_nodeids.dat")
    node_ids = np.loadtxt(nid_path, dtype=int)  # (n_pops, 2)
    first_id = node_ids[0, 0]
    last_id = node_ids[-1, 1]
    n_neurons = last_id - first_id + 1

    # Gather all spike recorder dat files (skip population_nodeids.dat)
    dat_files = sorted(
        f
        for f in os.listdir(spike_recorder_dir)
        if f.endswith(".dat") and "population" not in f
    )

    dtype = np.dtype([("sender", "i4"), ("time_ms", "f8")])
    all_senders = []
    all_times = []
    for fn in dat_files:
        fpath = os.path.join(spike_recorder_dir, fn)
        try:
            data = np.loadtxt(fpath, skiprows=3, dtype=dtype)
        except ValueError:
            continue
        if data.size == 0:
            continue
        mask = (data["time_ms"] >= t_start) & (data["time_ms"] <= t_end)
        all_senders.append(data["sender"][mask])
        all_times.append(data["time_ms"][mask])

    if not all_senders:
        return [np.array([]) for _ in range(n_neurons)], n_neurons

    senders = np.concatenate(all_senders)
    times = np.concatenate(all_times)

    # Convert NEST node IDs to 0-based indices
    indices = senders - first_id

    spike_times = [np.array([]) for _ in range(n_neurons)]
    for idx in range(n_neurons):
        mask = indices == idx
        if mask.any():
            spike_times[idx] = np.sort(times[mask])

    return spike_times, n_neurons


def _load_spikes_from_bins(spike_bins, bin_ms, t_start, t_end):
    """
    Convert binned spike data into per-neuron spike time arrays.
    Uses bin centers as approximate spike times.
    """
    n_intervals, bins_per_interval, n_neurons = spike_bins.shape
    interval_ms = bins_per_interval * bin_ms

    spike_times = [[] for _ in range(n_neurons)]
    for interval_idx in range(n_intervals):
        interval_start = interval_idx * interval_ms
        interval_end = interval_start + interval_ms
        if interval_end < t_start or interval_start > t_end:
            continue
        for b in range(bins_per_interval):
            t = interval_start + (b + 0.5) * bin_ms
            if t < t_start or t > t_end:
                continue
            for n in range(n_neurons):
                if spike_bins[interval_idx, b, n] > 0:
                    spike_times[n].append(t)

    spike_times = [np.array(s) for s in spike_times]
    return spike_times, n_neurons


# ── Population colors ──────────────────────────────────────────────────────

POP_COLORS = {
    "L23E": "#1f77b4", "L23I": "#aec7e8",
    "L4E":  "#ff7f0e", "L4I":  "#ffbb78",
    "L5E":  "#2ca02c", "L5I":  "#98df8a",
    "L6E":  "#d62728", "L6I":  "#ff9896",
}
POP_COLORS_DIM = {k: c + "40" for k, c in POP_COLORS.items()}  # with alpha hex

ORIENTATION_COLORS = {
    0: "#e6194b", 45: "#3cb44b", 90: "#4363d8", 135: "#f58231",
    180: "#911eb4", 225: "#42d4f4", 270: "#f032e6", 315: "#bfef45",
}


# ── Animation: Electrical ─────────────────────────────────────────────────

def animate_electrical(session_dir, t_start, t_end, fps, output, bin_width_ms,
                       spike_window_ms, dpi, figsize, n_scaling=None):
    """Animate electrical stimulation with electrode flashes and neuron spikes."""
    # Load config
    config_files = [f for f in os.listdir(session_dir) if f.endswith("config.json")]
    with open(os.path.join(session_dir, config_files[0])) as f:
        config = json.load(f)

    n_stim_channels = config["n_stim_channels"]
    bounds = config["electrode_volume_bounds_um"]

    # Load history for stim events
    hist_files = [f for f in os.listdir(session_dir) if "history" in f and f.endswith(".pkl")]
    with open(os.path.join(session_dir, hist_files[0]), "rb") as f:
        history = pickle.load(f)

    # Build stim event lookup: time_ms -> channel index
    stim_events = {}  # interval_start_ms -> channel
    for entry in history:
        pat = entry["delivered_pattern"]
        if pat is not None:
            stim_events[entry["interval_start_ms"]] = pat["channels"][0]

    # Load spikes
    spike_rec_dir = os.path.join(session_dir, "spike_recorder")
    print(f"Loading spikes from {spike_rec_dir} for t=[{t_start}, {t_end}] ms ...")
    spike_times, n_neurons = _load_spikes_from_dat(spike_rec_dir, t_start, t_end)
    print(f"  {n_neurons} neurons, loaded spike times.")

    # Neuron & electrode geometry
    neur_h, neur_v, neur_pop = _build_neuron_positions(n_scaling=n_scaling)
    elec_h, elec_v = _build_electrode_grid(bounds, n_stim_channels)

    # Time bins for frames
    frame_times = np.arange(t_start, t_end, bin_width_ms)
    n_frames = len(frame_times)
    print(f"  {n_frames} frames at {bin_width_ms} ms/frame")

    # Pre-compute active neurons per frame (neurons that spiked within window)
    print("  Pre-computing active neurons per frame ...")
    active_per_frame = []
    for t in frame_times:
        window_start = t
        window_end = t + spike_window_ms
        active = set()
        for n_idx in range(n_neurons):
            st = spike_times[n_idx]
            if st.size > 0 and np.any((st >= window_start) & (st < window_end)):
                active.add(n_idx)
        active_per_frame.append(active)

    # Pre-compute active electrodes per frame
    active_elec_per_frame = []
    for t in frame_times:
        active_ch = None
        for stim_t, ch in stim_events.items():
            # Electrode lights up for a brief window after stim onset
            if stim_t <= t < stim_t + spike_window_ms * 2:
                active_ch = ch
                break
        active_elec_per_frame.append(active_ch)

    # ── Set up figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(min(neur_h) - 50, max(neur_h) + 50)
    ax.set_ylim(max(neur_v) + 50, min(neur_v) - 50)  # depth increases down
    ax.set_xlabel("Horizontal (μm)")
    ax.set_ylabel("Depth (μm)")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    title_text = ax.set_title("", color="white", fontsize=12)

    # Static dim neurons
    pop_names_unique = net_dict["populations"]
    dim_scatters = {}
    for pname in pop_names_unique:
        mask = np.array([p == pname for p in neur_pop])
        dim_scatters[pname] = ax.scatter(
            neur_h[mask], neur_v[mask],
            s=2, c="gray", alpha=0.15, zorder=1
        )

    # Active neurons (bright)
    bright_scatter = ax.scatter([], [], s=8, c="yellow", alpha=0.9, zorder=3)

    # Electrode markers
    elec_scatter = ax.scatter(
        elec_h, elec_v,
        s=30, c="white", marker="x", alpha=0.5, linewidths=0.8, zorder=2
    )
    # Active electrode highlight
    active_elec_scatter = ax.scatter([], [], s=200, c="red", marker="o",
                                      alpha=0.8, zorder=4, edgecolors="yellow",
                                      linewidths=2)

    # Legend
    legend_patches = [
        mpatches.Patch(color="yellow", label="Firing neuron"),
        plt.Line2D([0], [0], marker="x", color="white", markeredgecolor="white",
                    markersize=8, linestyle="None", label="Electrode"),
        plt.Line2D([0], [0], marker="o", color="red", markeredgecolor="yellow",
                    markersize=10, linestyle="None", label="Active electrode"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              facecolor="black", edgecolor="white", labelcolor="white")

    def update(frame_idx):
        t = frame_times[frame_idx]
        title_text.set_text(f"Electrical Stimulation  t = {t:.0f} ms")

        # Active neurons
        active = active_per_frame[frame_idx]
        if active:
            idxs = np.array(list(active))
            bright_scatter.set_offsets(np.column_stack([neur_h[idxs], neur_v[idxs]]))
            # Color by population
            colors = [POP_COLORS.get(neur_pop[i], "yellow") for i in idxs]
            bright_scatter.set_facecolors(colors)
        else:
            bright_scatter.set_offsets(np.empty((0, 2)))

        # Active electrode
        active_ch = active_elec_per_frame[frame_idx]
        if active_ch is not None and active_ch < len(elec_h):
            active_elec_scatter.set_offsets([[elec_h[active_ch], elec_v[active_ch]]])
        else:
            active_elec_scatter.set_offsets(np.empty((0, 2)))

        return bright_scatter, active_elec_scatter, title_text

    print(f"Rendering {n_frames} frames to {output} ...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    anim.save(output, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Done! Saved to {output}")


# ── Animation: Visual ─────────────────────────────────────────────────────

def animate_visual(visual_dir, t_start, t_end, fps, output, bin_width_ms,
                   spike_window_ms, dpi, figsize, n_scaling=None):
    """Animate visual stimulation with orientation-colored backgrounds and neuron spikes."""
    # Load run config
    cfg_files = [f for f in os.listdir(visual_dir) if "run_config" in f]
    with open(os.path.join(visual_dir, cfg_files[0]), "rb") as f:
        run_cfg = pickle.load(f)

    epoch_info = run_cfg["epoch_info"]  # list of (start_ms, end_ms, label)
    sample_info = run_cfg["sample_info"]  # list of (start_ms, end_ms, angle_deg, repeat)
    presim_ms = run_cfg["presim_ms"]

    # Adjust times: epoch_info times are post-presim
    # spike recorder times include presim offset
    # We'll work in the same coordinate system as the spike recorder

    # Load spikes
    spike_rec_dir = os.path.join(visual_dir, "spike_recorder")
    print(f"Loading spikes from {spike_rec_dir} for t=[{t_start}, {t_end}] ms ...")
    spike_times, n_neurons = _load_spikes_from_dat(spike_rec_dir, t_start, t_end)
    print(f"  {n_neurons} neurons, loaded spike times.")

    # Neuron geometry
    neur_h, neur_v, neur_pop = _build_neuron_positions(n_scaling=n_scaling)

    # Check neuron placement consistency against saved metadata if available
    meta_files = [f for f in os.listdir(visual_dir) if f.endswith("_stim_metadata.pkl")]
    for mf in meta_files:
        _check_neuron_placement_consistency(neur_h, neur_v, os.path.join(visual_dir, mf))

    # Time bins for frames
    frame_times = np.arange(t_start, t_end, bin_width_ms)
    n_frames = len(frame_times)
    print(f"  {n_frames} frames at {bin_width_ms} ms/frame")

    # Pre-compute active neurons per frame
    print("  Pre-computing active neurons per frame ...")
    active_per_frame = []
    for t in frame_times:
        window_start = t
        window_end = t + spike_window_ms
        active = set()
        for n_idx in range(n_neurons):
            st = spike_times[n_idx]
            if st.size > 0 and np.any((st >= window_start) & (st < window_end)):
                active.add(n_idx)
        active_per_frame.append(active)

    # Build epoch lookup: for each frame time, what orientation is being presented?
    # epoch_info times are relative to post-presim, spike times include presim
    def _get_orientation_at(t_ms):
        """Return orientation in degrees at time t_ms (spike recorder coords), or None."""
        t_post_presim = t_ms - presim_ms
        for start, end, label in epoch_info:
            if start <= t_post_presim < end:
                deg = int(label.replace("°", ""))
                return deg
        return None

    # ── Set up figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(min(neur_h) - 50, max(neur_h) + 50)
    ax.set_ylim(max(neur_v) + 50, min(neur_v) - 50)
    ax.set_xlabel("Horizontal (μm)")
    ax.set_ylabel("Depth (μm)")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    title_text = ax.set_title("", color="white", fontsize=12)

    # Static dim neurons
    for pname in net_dict["populations"]:
        mask = np.array([p == pname for p in neur_pop])
        ax.scatter(neur_h[mask], neur_v[mask], s=2, c="gray", alpha=0.15, zorder=1)

    # Active neurons
    bright_scatter = ax.scatter([], [], s=8, c="yellow", alpha=0.9, zorder=3)

    # Orientation indicator (colored bar at top)
    orient_bar = ax.axhspan(min(neur_v) - 50, min(neur_v) - 20, color="black",
                             alpha=0.6, zorder=5)

    # Legend for orientations
    legend_patches = [
        mpatches.Patch(color=c, label=f"{deg}°")
        for deg, c in sorted(ORIENTATION_COLORS.items())
    ]
    legend_patches.insert(0, mpatches.Patch(color="gray", label="Off (no stim)"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=6,
              facecolor="black", edgecolor="white", labelcolor="white",
              title="Orientation", title_fontsize=7)

    def update(frame_idx):
        t = frame_times[frame_idx]
        orient = _get_orientation_at(t)

        if orient is not None:
            title_text.set_text(f"Visual Stimulation  t = {t:.0f} ms  |  {orient}°")
            bg_color = ORIENTATION_COLORS.get(orient, "gray")
        else:
            title_text.set_text(f"Visual Stimulation  t = {t:.0f} ms  |  Off")
            bg_color = "#111111"

        # Tint the background subtly
        ax.set_facecolor(bg_color + "15")  # very faint tint

        # Active neurons
        active = active_per_frame[frame_idx]
        if active:
            idxs = np.array(list(active))
            bright_scatter.set_offsets(np.column_stack([neur_h[idxs], neur_v[idxs]]))
            colors = [POP_COLORS.get(neur_pop[i], "yellow") for i in idxs]
            bright_scatter.set_facecolors(colors)
        else:
            bright_scatter.set_offsets(np.empty((0, 2)))

        return bright_scatter, title_text

    print(f"Rendering {n_frames} frames to {output} ...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    anim.save(output, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Done! Saved to {output}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animate neural population activity in response to stimulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Electrical mode
    p_elec = sub.add_parser("electrical", help="Animate electrical stimulation response")
    p_elec.add_argument("--session-dir", required=True,
                        help="Path to session directory (contains config.json, history.pkl, spike_recorder/)")
    p_elec.add_argument("--t-start", type=float, default=0,
                        help="Start time in ms (default: 0)")
    p_elec.add_argument("--t-end", type=float, default=3000,
                        help="End time in ms (default: 3000)")

    # Visual mode
    p_vis = sub.add_parser("visual", help="Animate visual stimulation response")
    p_vis.add_argument("--visual-dir", required=True,
                       help="Path to visual output directory (contains run_config.pkl, spike_recorder/)")
    p_vis.add_argument("--t-start", type=float, default=1000,
                       help="Start time in ms (default: 1000, i.e. after presim)")
    p_vis.add_argument("--t-end", type=float, default=2200,
                       help="End time in ms (default: 2200)")

    # Common options
    for p in [p_elec, p_vis]:
        p.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
        p.add_argument("-o", "--output", default=None,
                       help="Output file (default: <mode>_activity.mp4)")
        p.add_argument("--bin-width", type=float, default=5,
                       help="Time per frame in ms (default: 5)")
        p.add_argument("--spike-window", type=float, default=5,
                       help="Window in ms to count a neuron as active (default: 5)")
        p.add_argument("--dpi", type=int, default=150, help="DPI (default: 150)")
        p.add_argument("--figsize", type=float, nargs=2, default=[7, 8],
                       help="Figure size in inches (default: 7 8)")
        p.add_argument("--n-scaling", type=float, default=None,
                       help="Override N_scaling (neuron count factor). Default: use network_params.py value.")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.mode}_activity.mp4"

    if args.mode == "electrical":
        animate_electrical(
            session_dir=args.session_dir,
            t_start=args.t_start,
            t_end=args.t_end,
            fps=args.fps,
            output=args.output,
            bin_width_ms=args.bin_width,
            spike_window_ms=args.spike_window,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
            n_scaling=args.n_scaling,
        )
    elif args.mode == "visual":
        animate_visual(
            visual_dir=args.visual_dir,
            t_start=args.t_start,
            t_end=args.t_end,
            fps=args.fps,
            output=args.output,
            bin_width_ms=args.bin_width,
            spike_window_ms=args.spike_window,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
            n_scaling=args.n_scaling,
        )


if __name__ == "__main__":
    main()
