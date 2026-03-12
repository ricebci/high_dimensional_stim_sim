"""
Run closed-loop electrical stimulation experiment.

How to run
----------
    python closed_loop_experiment.py

Example
-------
    python closed_loop_experiment.py \
        --total-duration-ms 1000 \
        --closed-loop-interval-ms 5 \
        --spike-threshold 40 \
        --stim-amplitude-uA 2.0 \
        --stim-channels 0 1 2 3
"""

import argparse
import os
import pickle
from typing import Callable, Dict, Iterable, Optional

import nest
import numpy as np

import helpers
from controller import SpikeThresholdController
from system import N_STIM_CHANNELS, SystemNESTSim


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop electrical stimulation")
    parser.add_argument("--total-duration-ms", type=float, default=1000.0)
    parser.add_argument("--closed-loop-interval-ms", type=float, default=5.0)
    parser.add_argument("--output-name", type=str, default="closed_loop_experiment")
    parser.add_argument("--output-prefix", type=str, default="closed_loop")

    parser.add_argument("--spike-threshold", type=int, default=50)
    parser.add_argument("--stim-amplitude-uA", type=float, default=2.0)
    parser.add_argument("--stim-time-ms", type=float, default=0.0)
    parser.add_argument("--refractory-intervals", type=int, default=0)
    parser.add_argument(
        "--stim-channels",
        type=int,
        nargs="+",
        default=[0],
        help="Channel indices to stimulate when controller triggers",
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
    if args.stim_time_ms < 0 or args.stim_time_ms > args.closed_loop_interval_ms:
        raise ValueError("--stim-time-ms must be in [0, closed_loop_interval_ms]")
    channels = np.asarray(args.stim_channels, dtype=int)
    if np.any(channels < 0) or np.any(channels >= N_STIM_CHANNELS):
        raise ValueError(f"--stim-channels must be in [0, {N_STIM_CHANNELS - 1}]")


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
    
    # This is where model and conotrller should affect the loop
    
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

        # Ask controller whether to stimulate in the next interval
        pending_pattern = controller(
            interval_index=interval_index,
            interval_start_ms=interval_start_ms,
            interval_end_ms=interval_end_ms,
            interval_duration_ms=interval_duration_ms,
            new_spikes_by_neuron=new_spikes_by_neuron,
            history=history,
        )

        # Log this interval's outcome
        history.append(
            {
                "interval_index": interval_index,
                "interval_start_ms": interval_start_ms,
                "interval_end_ms": interval_end_ms,
                "n_new_spikes": int(sum(len(s) for s in new_spikes_by_neuron)),
                "delivered_pattern": delivered,
            }
        )

        if realtime_progress:
            print(
                f"Closed-loop progress: {interval_end_ms:.1f}/{total_duration_ms:.1f} ms "
                f"(new spikes={history[-1]['n_new_spikes']})",
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


def main():
    args = parse_args()
    validate_args(args)

    # Initialize NEST network (includes presim warmup)
    sim = SystemNESTSim(
        output_name=args.output_name,
        fast_mode=args.fast_mode,
        fast_sim_resolution_ms=args.fast_sim_resolution_ms,
    )

    # Build feedback controller that decides when to stimulate
    controller = SpikeThresholdController(
        stim_channels=args.stim_channels,
        stim_amplitude_uA=args.stim_amplitude_uA,
        stim_time_ms=args.stim_time_ms,
        spike_threshold=args.spike_threshold,
        refractory_intervals=args.refractory_intervals,
    )

    # Run the closed-loop experiment
    results = run_closed_loop_electrical_stim(
        sim=sim,
        total_duration_ms=args.total_duration_ms,
        closed_loop_interval_ms=args.closed_loop_interval_ms,
        controller=controller.get_stim_pattern,
        initial_stim_pattern=None,
        output_prefix=args.output_prefix,
        realtime_progress=not args.no_progress,
    )

    # Save full experiment configuration for reproducibility
    config = {
        "total_duration_ms": args.total_duration_ms,
        "closed_loop_interval_ms": args.closed_loop_interval_ms,
        "controller": {
            "spike_threshold": args.spike_threshold,
            "stim_amplitude_uA": args.stim_amplitude_uA,
            "stim_time_ms": args.stim_time_ms,
            "refractory_intervals": args.refractory_intervals,
            "stim_channels": list(args.stim_channels),
        },
        "results": results,
        "data_path": sim.sim_dict["data_path"],
    }

    config_path = os.path.join(sim.sim_dict["data_path"], f"{args.output_prefix}_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print("Closed-loop experiment complete")
    print("Data path:", sim.sim_dict["data_path"])
    print("Results:", results)
    print("Config:", config_path)


if __name__ == "__main__":
    main()
