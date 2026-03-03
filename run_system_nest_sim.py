"""
Class-based NEST simulation runner.

This module provides `SystemNESTSim`, which initializes and sets up the
network during class construction, and exposes:

- `electrical_stim(...)`: run stimulation from a current-amplitude sequence.
- `visual_stim(...)`: run orientation input given time-series of
  [cos(theta), sin(theta)].

Example
-------
    from run_system_nest_sim import SystemNESTSim

    sim = SystemNESTSim(output_name="system_experiment")

    electrical_sequence = {
        "channels": [0, 1, 2],
        "times_ms": [10, 20, 35],
        "amplitudes_uA": [2, 4, 3],
    }
    sim.electrical_stim(electrical_sequence, input_duration_ms=5000)

    # Orientation input over time: [cos(theta), sin(theta)]
    visual_series = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    sim.visual_stim(visual_series, input_duration_ms=3000, dt_ms=10)
"""

import copy
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import nest
import numpy as np

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
from network_cortcol import Network

# ====================PROBE=====================================================
# Layout: 1x32 spans 1800um
volume_v_min = 200  # um
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)

sigma_e_um = 2.76e-7
conductivity_constant = 10
N_STIM_CHANNELS = len(ch_coordinates)


def amp_decay_func(amp_uA, dist_um):
    """Amplitude decay with distance from stimulation electrode."""
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )


def build_stimulations_from_events(
    electrodes: StimElectrodes,
    channels: np.ndarray,
    times_ms: np.ndarray,
    amplitudes_uA: np.ndarray,
) -> Dict[int, Dict[str, List[float]]]:
    """Populate electrodes stimulation containers from event arrays."""
    stimulations = {
        channel_index: {"times": [], "amplitudes": []}
        for channel_index in range(electrodes.n_chs)
    }
    stim_onset_times_by_ch = {
        channel_index: [] for channel_index in range(electrodes.n_chs)
    }
    stim_amplitudes_by_ch = {
        channel_index: [] for channel_index in range(electrodes.n_chs)
    }

    for channel, time_ms, amplitude_uA in zip(channels, times_ms, amplitudes_uA):
        pulse_times, pulse_amps = electrodes.generate_biphasic_pulse(
            float(time_ms), float(amplitude_uA)
        )
        channel_index = int(channel)
        stimulations[channel_index]["times"].extend(pulse_times)
        stimulations[channel_index]["amplitudes"].extend(pulse_amps)
        stim_onset_times_by_ch[channel_index].append(float(time_ms))
        stim_amplitudes_by_ch[channel_index].append(float(amplitude_uA))

    electrodes.stimulations = stimulations
    electrodes.stim_onset_times_by_ch = stim_onset_times_by_ch
    electrodes.stim_amplitudes_by_ch = stim_amplitudes_by_ch
    return stimulations


class SystemNESTSim:
    """System wrapper for class-based NEST setup and stimulation runs."""

    def __init__(
        self,
        output_name: str = "system_sim",
        window_ms: int = 500,
        overlap_ms: int = 400,
        stim_pulse_params: Dict[str, float] = None,
    ):
        """
        Initialize NEST and set up the network.

        This constructor performs simulation setup immediately:
        - resets NEST kernel
        - creates/connects network
        - runs pre-simulation baseline (`t_presim`)
        """
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.stim_pulse_params = stim_pulse_params or {"pulse_width_ms": 0.2, "ipi_ms": 0.2}

        self.net_dict = copy.deepcopy(net_dict)
        self.sim_dict = copy.deepcopy(sim_dict)
        self.stim_dict = copy.deepcopy(stim_dict)

        nest.ResetKernel()

        nscale = self.net_dict["N_scaling"]
        base_path = os.path.join(
            os.getcwd(),
            "outputs",
            f"data_system_sim_{nscale}scale",
        )
        self.sim_dict["data_path"] = os.path.join(base_path, output_name)
        os.makedirs(self.sim_dict["data_path"], exist_ok=True)

        self.electrodes = StimElectrodes(
            ch_coordinates,
            self.stim_pulse_params,
            amp_decay_func,
        )

        self.network = Network(self.sim_dict, self.net_dict, self.stim_dict)
        self.network.create()
        self.network.connect()
        self.network.simulate_baseline(self.sim_dict["t_presim"])

        print("SystemNESTSim initialized.")
        print("Total # neurons simulated:", self.network.n_neurons)
        print("Data path:", self.sim_dict["data_path"])

    def _validate_event_arrays(
        self,
        channels: np.ndarray,
        times_ms: np.ndarray,
        amplitudes_uA: np.ndarray,
        input_duration_ms: float,
    ):
        if not (len(channels) == len(times_ms) == len(amplitudes_uA)):
            raise ValueError("channels, times_ms, amplitudes_uA must have equal length")
        if np.any(channels < 0) or np.any(channels >= N_STIM_CHANNELS):
            raise ValueError(f"Channel indices must be in [0, {N_STIM_CHANNELS - 1}]")
        if np.any(times_ms < 0):
            raise ValueError("times_ms must be non-negative")
        if np.any(amplitudes_uA < 0):
            raise ValueError("amplitudes_uA must be non-negative")
        if np.any(times_ms > input_duration_ms):
            raise ValueError("Found stimulation times beyond input_duration_ms")

    def _run_from_events(
        self,
        channels: np.ndarray,
        times_ms: np.ndarray,
        amplitudes_uA: np.ndarray,
        input_duration_ms: float,
        output_prefix: str,
    ) -> Dict[str, str]:
        order = np.argsort(times_ms)
        channels = channels[order]
        times_ms = times_ms[order]
        amplitudes_uA = amplitudes_uA[order]

        self._validate_event_arrays(channels, times_ms, amplitudes_uA, input_duration_ms)

        build_stimulations_from_events(self.electrodes, channels, times_ms, amplitudes_uA)

        self.electrodes.compute_impulse_response_matrix(self.network.neuron_locations)
        self.electrodes.compute_stim_current_matrix()
        self.electrodes.calculate_induced_current_matrix()

        current_generators = self.electrodes.get_current_generators()
        self.network.simulate_current_input(current_generators, time_ms=float(input_duration_ms))

        spike_trains = self.network.get_spike_train_list()
        spike_rates = helpers.compute_spike_rates(
            spike_trains,
            input_duration_ms,
            self.window_ms,
            self.overlap_ms,
            presim_time_ms=self.sim_dict["t_presim"],
        )

        rates_path = os.path.join(self.sim_dict["data_path"], f"{output_prefix}_spike_rates.pkl")
        pulses_path = os.path.join(self.sim_dict["data_path"], f"{output_prefix}_stim_pulses.pkl")

        with open(rates_path, "wb") as rates_file:
            pickle.dump(spike_rates, rates_file)
        with open(pulses_path, "wb") as pulses_file:
            pickle.dump(self.electrodes.stim_onset_times_by_ch, pulses_file)

        print(f"Saved spike rates -> {rates_path}")
        print(f"Saved stim pulses -> {pulses_path}")

        return {
            "spike_rates_path": rates_path,
            "stim_pulses_path": pulses_path,
        }

    def electrical_stim(
        self,
        current_amplitude_sequence: Dict[str, Iterable[float]],
        input_duration_ms: float,
        output_prefix: str = "electrical",
    ) -> Dict[str, str]:
        """
        Run electrical stimulation with explicit channel/time/amplitude events.

        Parameters
        ----------
        current_amplitude_sequence
            Dict containing `channels`, `times_ms`, `amplitudes_uA`.
        input_duration_ms
            Simulation duration for this stimulation input.
        output_prefix
            Prefix used in output pickle filenames.
        """
        required_keys = ("channels", "times_ms", "amplitudes_uA")
        missing_keys = [key for key in required_keys if key not in current_amplitude_sequence]
        if missing_keys:
            raise KeyError(f"Missing keys in current_amplitude_sequence: {missing_keys}")

        channels = np.asarray(current_amplitude_sequence["channels"], dtype=int)
        times_ms = np.asarray(current_amplitude_sequence["times_ms"], dtype=float)
        amplitudes_uA = np.asarray(current_amplitude_sequence["amplitudes_uA"], dtype=float)

        return self._run_from_events(
            channels,
            times_ms,
            amplitudes_uA,
            input_duration_ms,
            output_prefix,
        )

    def visual_stim(
        self,
        orientation_time_series: Iterable[Iterable[float]],
        input_duration_ms: float,
        dt_ms: float = 1.0,
        baseline_pA: float = 0.0,
        gain_pA: float = 80.0,
        sparsity: float = 0.1,
        seed: int = 0,
        output_prefix: str = "visual",
    ) -> Dict[str, str]:
        """
        Run visual stimulation from orientation vectors [cos(theta), sin(theta)].

        Parameters
        ----------
        orientation_time_series
            Time-series shaped (T, 2), each row [cos(theta), sin(theta)].
        input_duration_ms
            Total duration of this visual input.
        dt_ms
            Time step between rows in the time-series.
        baseline_pA
            Baseline current (pA) delivered to selected visual-input neurons.
        gain_pA
            Gain scaling for orientation projection onto neuron preferences.
        sparsity
            Fraction of neurons receiving visual input (0 < sparsity <= 1).
        seed
            Random seed for sparse-neuron and preference sampling.
        output_prefix
            Prefix used in output pickle filenames.
        """
        orientation_array = np.asarray(orientation_time_series, dtype=float)
        if orientation_array.ndim != 2 or orientation_array.shape[1] != 2:
            raise ValueError("orientation_time_series must have shape (T, 2)")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1]")

        max_steps = int(np.floor(input_duration_ms / dt_ms)) + 1
        num_steps = min(len(orientation_array), max_steps)
        orientation_array = orientation_array[:num_steps]

        event_times = np.arange(num_steps, dtype=float) * dt_ms
        if num_steps == 0:
            raise ValueError("orientation_time_series is empty after duration truncation")

        rng = np.random.default_rng(seed)
        num_neurons = self.network.n_neurons
        num_target_neurons = max(1, int(np.round(sparsity * num_neurons)))
        target_neuron_indices = rng.choice(
            num_neurons,
            size=num_target_neurons,
            replace=False,
        )

        preferred_thetas = rng.uniform(0.0, 2.0 * np.pi, size=num_target_neurons)
        preferred_vectors = np.stack(
            [np.cos(preferred_thetas), np.sin(preferred_thetas)],
            axis=1,
        )

        projections = orientation_array @ preferred_vectors.T
        target_currents = baseline_pA + gain_pA * projections

        current_generators = []
        empty_values = np.zeros(num_steps, dtype=float)

        target_lookup = {idx: i for i, idx in enumerate(target_neuron_indices)}
        for neuron_index in range(num_neurons):
            if neuron_index in target_lookup:
                amplitude_values = target_currents[:, target_lookup[neuron_index]]
            else:
                amplitude_values = empty_values

            generator = nest.Create(
                "step_current_generator",
                params={
                    "label": f"visual_input_neuron_{neuron_index}",
                    "amplitude_times": event_times,
                    "amplitude_values": amplitude_values,
                },
            )
            current_generators.append(generator)

        self.network.simulate_current_input(current_generators, time_ms=float(input_duration_ms))

        spike_trains = self.network.get_spike_train_list()
        spike_rates = helpers.compute_spike_rates(
            spike_trains,
            input_duration_ms,
            self.window_ms,
            self.overlap_ms,
            presim_time_ms=self.sim_dict["t_presim"],
        )

        rates_path = os.path.join(self.sim_dict["data_path"], f"{output_prefix}_spike_rates.pkl")
        meta_path = os.path.join(self.sim_dict["data_path"], f"{output_prefix}_stim_metadata.pkl")

        with open(rates_path, "wb") as rates_file:
            pickle.dump(spike_rates, rates_file)
        with open(meta_path, "wb") as meta_file:
            pickle.dump(
                {
                    "target_neuron_indices": target_neuron_indices,
                    "preferred_thetas": preferred_thetas,
                    "event_times_ms": event_times,
                },
                meta_file,
            )

        print(f"Saved spike rates -> {rates_path}")
        print(f"Saved visual stim metadata -> {meta_path}")
        print(
            f"Visual input targeted {num_target_neurons}/{num_neurons} neurons "
            f"(sparsity={sparsity:.3f})"
        )

        return {
            "spike_rates_path": rates_path,
            "stim_metadata_path": meta_path,
        }