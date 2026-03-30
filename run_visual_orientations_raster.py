"""
Generate visual-orientation simulation data for downstream notebook analysis.

How to run
----------
From the project root:

        python run_visual_orientations_raster.py

Example with custom arguments:

    python run_visual_orientations_raster.py \
        --orientations-deg 0 60 120 180 240 300 \
        --n-repeats 5 \
        --on-duration-ms 100 \
        --off-duration-ms 100 \
        --sparsity 0.05 \
        --output-name visual_custom \
        --output-prefix custom
tu
What this script does
---------------------
- Builds a repeated orientation input time-series.
- Runs `SystemNESTSim.visual_stim(...)` to generate spike output files.
- Saves run metadata to:
    `outputs/data_system_sim_0.05scale/visual_orientations_8/visual8_run_config.pkl`

After running this script
-------------------------
Open and run:

        run_visual_orientations_raster.ipynb

The notebook loads the saved metadata + spike files, then performs raster
plotting and PCA analysis.
"""

import os
import pickle
import argparse

import numpy as np

from corcol_params.sim_params import sim_dict
from system import SystemNESTSim


def build_orientation_time_series(
    orientations_deg,
    dt_ms,
    on_duration_ms,
    off_duration_ms,
    n_repeats,
):
    n_on_bins = int(round(on_duration_ms / dt_ms))
    n_off_bins = int(round(off_duration_ms / dt_ms))

    if not np.isclose(n_on_bins * dt_ms, on_duration_ms) or not np.isclose(
        n_off_bins * dt_ms, off_duration_ms
    ):
        raise ValueError(
            "on/off durations must be representable by the simulation resolution"
        )

    orientation_vectors = []
    epoch_info = []
    sample_info = []
    cursor_ms = 0.0

    for repeat_idx in range(n_repeats):
        for theta_deg in orientations_deg:
            theta = np.deg2rad(theta_deg)
            vec = [np.cos(theta), np.sin(theta)]

            orientation_vectors.extend([[0.0, 0.0]] * n_off_bins)
            cursor_ms += n_off_bins * dt_ms

            orientation_vectors.extend([vec] * n_on_bins)
            start_ms = cursor_ms
            end_ms = cursor_ms + n_on_bins * dt_ms
            epoch_info.append((start_ms, end_ms, f"{theta_deg}°"))
            sample_info.append((start_ms, end_ms, theta_deg, repeat_idx))
            cursor_ms = end_ms

    orientation_time_series = np.asarray(orientation_vectors, dtype=float)
    input_duration_ms = len(orientation_time_series) * dt_ms

    return (
        orientation_time_series,
        input_duration_ms,
        epoch_info,
        sample_info,
        n_on_bins,
        n_off_bins,
    )


def compute_epoch_activity(times_all, neurons_all, sample_info, n_neurons):
    sample_vectors = []
    sample_angles = []
    sample_repeats = []

    for start_ms, end_ms, angle_deg, repeat_idx in sample_info:
        in_window = (times_all >= start_ms) & (times_all < end_ms)
        spiking_neurons = neurons_all[in_window].astype(int) - 1

        spike_counts = np.bincount(spiking_neurons, minlength=n_neurons)
        duration_s = (end_ms - start_ms) / 1000.0
        mean_rates_hz = spike_counts / duration_s

        sample_vectors.append(mean_rates_hz)
        sample_angles.append(angle_deg)
        sample_repeats.append(repeat_idx)

    return (
        np.asarray(sample_vectors),
        np.asarray(sample_angles),
        np.asarray(sample_repeats),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual-orientation simulation data for notebook analysis."
    )
    parser.add_argument(
        "--orientations-deg",
        type=float,
        nargs="+",
        default=[0, 45, 90, 135, 180, 225, 270, 315],
        help="Orientation angles in degrees (default: 0 45 90 135 180 225 270 315)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of repeats of the full orientation sequence (default: 10)",
    )
    parser.add_argument(
        "--on-duration-ms",
        type=float,
        default=70,
        help="Stimulus-on duration per orientation in ms (default: 70)",
    )
    parser.add_argument(
        "--off-duration-ms",
        type=float,
        default=70,
        help="Blank duration before each orientation in ms (default: 70)",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=None,
        help="Time resolution for orientation series in ms (default: native sim resolution)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="visual_orientations_8",
        help="Output folder name under outputs/data_system_sim_* (default: visual_orientations_8)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="visual8",
        help="Prefix for output files (default: visual8)",
    )
    parser.add_argument(
        "--baseline-pA",
        type=float,
        default=0.0,
        help="Baseline current in pA for visual_stim (default: 0.0)",
    )
    parser.add_argument(
        "--gain-pA",
        type=float,
        default=80.0,
        help="Gain current in pA for visual_stim (default: 80.0)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Fraction of neurons receiving visual input (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast mode in SystemNESTSim",
    )
    parser.add_argument(
        "--fast-sim-resolution-ms",
        type=float,
        default=1.0,
        help="Simulation resolution in fast mode (default: 1.0)",
    )
    parser.add_argument(
        "--realtime-progress",
        action="store_true",
        help="Enable progress updates during simulation",
    )
    parser.add_argument(
        "--progress-step-ms",
        type=float,
        default=10.0,
        help="Progress chunk size in ms when realtime progress is enabled (default: 10.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    orientations_deg = args.orientations_deg
    n_repeats = args.n_repeats
    dt_ms = float(sim_dict["sim_resolution"]) if args.dt_ms is None else args.dt_ms
    on_duration_ms = args.on_duration_ms
    off_duration_ms = args.off_duration_ms

    (
        orientation_time_series,
        input_duration_ms,
        epoch_info,
        sample_info,
        n_on_bins,
        n_off_bins,
    ) = build_orientation_time_series(
        orientations_deg,
        dt_ms,
        on_duration_ms,
        off_duration_ms,
        n_repeats,
    )

    print("native dt_ms:", dt_ms)
    print("orientation_time_series shape:", orientation_time_series.shape)
    print("input_duration_ms:", input_duration_ms)
    print("n_on_bins:", n_on_bins, "n_off_bins:", n_off_bins)
    print("n_samples (angles x repeats):", len(sample_info))

    sim = SystemNESTSim(
        output_name=args.output_name,
        fast_mode=args.fast_mode,
        fast_sim_resolution_ms=args.fast_sim_resolution_ms,
    )

    result_paths = sim.visual_stim(
        orientation_time_series=orientation_time_series,
        baseline_pA=args.baseline_pA,
        gain_pA=args.gain_pA,
        sparsity=args.sparsity,
        seed=args.seed,
        output_prefix=args.output_prefix,
        realtime_progress=args.realtime_progress,
        progress_step_ms=args.progress_step_ms,
    )

    print("Simulation complete.")
    print("Data path:", sim.sim_dict["data_path"])
    print(result_paths)

    run_config = {
        "orientations_deg": orientations_deg,
        "n_repeats": n_repeats,
        "dt_ms": dt_ms,
        "on_duration_ms": on_duration_ms,
        "off_duration_ms": off_duration_ms,
        "input_duration_ms": input_duration_ms,
        "epoch_info": epoch_info,
        "sample_info": sample_info,
        "n_on_bins": n_on_bins,
        "n_off_bins": n_off_bins,
        "data_path": sim.sim_dict["data_path"],
        "presim_ms": sim.sim_dict["t_presim"],
        "result_paths": result_paths,
    }
    run_config_path = os.path.join(
        sim.sim_dict["data_path"], f"{args.output_prefix}_run_config.pkl"
    )
    with open(run_config_path, "wb") as config_file:
        pickle.dump(run_config, config_file)
    print(f"Saved run config -> {run_config_path}")
    print("Use run_visual_orientations_raster.ipynb for analysis and plotting.")


if __name__ == "__main__":
    main()
