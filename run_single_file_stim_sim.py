"""
Run a single cortical-column simulation with stimulation events loaded from a JSON file.

How to run
----------
From the project root:

1) Run with a stimulation JSON file:
     python run_single_file_stim_sim.py --stim-file data/generated_stim_config.json

2) Run with a custom output folder name:
     python run_single_file_stim_sim.py \
             --stim-file data/my_stim_config.json \
             --output-name experiment_a

Inputs
------
The stimulation JSON file supports either:

- Array format:
    {
        "channels": [...],
        "times_ms": [...],
        "amplitudes_uA": [...]
    }

- Event list format:
    {
        "events": [
            {"channel": 0, "time_ms": 10.0, "amplitude_uA": 2.0},
            ...
        ]
    }
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes

# ====================PROBE=====================================================
# Layout: 1x32 spans 1800um
volume_v_min = 200  # um
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)

sigma_e_um = 2.76e-7
conductivity_constant = 10
N_STIM_CHANNELS = len(ch_coordinates)

PRESIM_TIME_MS = sim_dict["t_presim"]
SIM_TIME_MS = sim_dict["t_sim"]
WINDOW_MS = 500
OVERLAP_MS = 400


def amp_decay_func(amp_uA, dist_um):
    """Amplitude decay with distance from stimulation electrode."""
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )


stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}


def load_stimulation_events(
    stim_file: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load stimulation events from JSON.

    Supported JSON formats:

    1) Array format:
       {
         "channels": [0, 1, 2],
         "times_ms": [10.0, 11.5, 40.0],
         "amplitudes_uA": [2.0, 1.5, 2.5]
       }

    2) Event list format:
       {
         "events": [
           {"channel": 0, "time_ms": 10.0, "amplitude_uA": 2.0},
           {"channel": 1, "time_ms": 11.5, "amplitude_uA": 1.5}
         ]
       }
    """
    with open(stim_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "events" in payload:
        events = payload["events"]
        if len(events) == 0:
            raise ValueError("'events' list is empty.")
        channels = np.array([event["channel"] for event in events], dtype=int)
        times_ms = np.array([event["time_ms"] for event in events], dtype=float)
        amplitudes_uA = np.array(
            [event["amplitude_uA"] for event in events], dtype=float
        )
    else:
        required_keys = ("channels", "times_ms", "amplitudes_uA")
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(
                f"Missing required keys in stimulation file: {missing}. "
                "Expected either 'events' or all of 'channels', 'times_ms', 'amplitudes_uA'."
            )

        channels = np.asarray(payload["channels"], dtype=int)
        times_ms = np.asarray(payload["times_ms"], dtype=float)
        amplitudes_uA = np.asarray(payload["amplitudes_uA"], dtype=float)

    if not (len(channels) == len(times_ms) == len(amplitudes_uA)):
        raise ValueError("channels, times_ms and amplitudes_uA must have equal length.")

    if np.any(channels < 0) or np.any(channels >= N_STIM_CHANNELS):
        raise ValueError(
            f"Channel indices must be in range [0, {N_STIM_CHANNELS - 1}]."
        )

    if np.any(times_ms < 0):
        raise ValueError("Stimulation times must be non-negative.")

    if np.any(amplitudes_uA < 0):
        raise ValueError("Stimulation amplitudes must be non-negative.")

    order = np.argsort(times_ms)
    return channels[order], times_ms[order], amplitudes_uA[order]


def build_stimulations_from_events(
    electrodes: StimElectrodes,
    channels: np.ndarray,
    times_ms: np.ndarray,
    amplitudes_uA: np.ndarray,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Populate `electrodes.stimulations` from explicit event-level inputs.
    """
    stimulations = {ch: {"times": [], "amplitudes": []} for ch in range(electrodes.n_chs)}
    stim_onset_times_by_ch = {ch: [] for ch in range(electrodes.n_chs)}
    stim_amplitudes_by_ch = {ch: [] for ch in range(electrodes.n_chs)}

    for ch, t_ms, amp_uA in zip(channels, times_ms, amplitudes_uA):
        pulse_times, pulse_amps = electrodes.generate_biphasic_pulse(float(t_ms), float(amp_uA))
        stimulations[int(ch)]["times"].extend(pulse_times)
        stimulations[int(ch)]["amplitudes"].extend(pulse_amps)
        stim_onset_times_by_ch[int(ch)].append(float(t_ms))
        stim_amplitudes_by_ch[int(ch)].append(float(amp_uA))

    electrodes.stimulations = stimulations
    electrodes.stim_onset_times_by_ch = stim_onset_times_by_ch
    electrodes.stim_amplitudes_by_ch = stim_amplitudes_by_ch

    return stimulations


def run_single_simulation(stim_file: str, output_name: str = "custom_stim"):
    import nest
    from network_cortcol import Network

    nest.ResetKernel()

    channels, times_ms, amplitudes_uA = load_stimulation_events(stim_file)

    max_stim_time_ms = np.max(times_ms) if len(times_ms) > 0 else 0
    sim_time_ms = max_stim_time_ms + 2000

    Nscale = net_dict["N_scaling"]
    sim_time_s = int(sim_time_ms / 1000)
    base_path = os.path.join(
        os.getcwd(),
        "outputs",
        f"data_custom_stim_{Nscale}scale_{sim_time_s}s",
    )

    sim_dict["data_path"] = os.path.join(base_path, output_name)
    os.makedirs(sim_dict["data_path"], exist_ok=True)

    electrodes = StimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)

    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    print("Total # neurons simulated:", network.n_neurons)

    network.simulate_baseline(PRESIM_TIME_MS)

    build_stimulations_from_events(electrodes, channels, times_ms, amplitudes_uA)

    electrodes.compute_impulse_response_matrix(network.neuron_locations)
    electrodes.compute_stim_current_matrix()
    electrodes.calculate_induced_current_matrix()

    current_generators = electrodes.get_current_generators()
    network.simulate_current_input(current_generators, time_ms=sim_time_ms)

    stim_evoked_spike_trains = network.get_spike_train_list()
    stim_evoked_spike_rates = helpers.compute_spike_rates(
        stim_evoked_spike_trains,
        sim_time_ms,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )

    pkl_path_rates = os.path.join(sim_dict["data_path"], "stim_spike_rates.pkl")
    pkl_path_pulses = os.path.join(sim_dict["data_path"], "stim_pulses.pkl")

    with open(pkl_path_rates, "wb") as f:
        pickle.dump(stim_evoked_spike_rates, f)

    with open(pkl_path_pulses, "wb") as f:
        pickle.dump(electrodes.stim_onset_times_by_ch, f)

    print(f"Saved stim spike rates -> {pkl_path_rates}")
    print(f"Saved stim pulses -> {pkl_path_pulses}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one stimulation simulation using events from a JSON file."
    )
    parser.add_argument(
        "--stim-file",
        required=True,
        help="Path to JSON file containing stimulation channels/times/amplitudes.",
    )
    parser.add_argument(
        "--output-name",
        default="custom_stim",
        help="Output subfolder name under outputs/data_custom_stim_*.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_simulation(stim_file=args.stim_file, output_name=args.output_name)