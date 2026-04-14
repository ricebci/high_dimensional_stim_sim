"""
Class-based NEST simulation runner.

This module provides `SystemNESTSim`, which initializes and sets up the
network during class construction, and exposes:

- `electrical_stim(...)`: run stimulation from a current-amplitude sequence.
- `visual_stim(...)`: run orientation input given time-series of
  [cos(theta), sin(theta)].

Example
-------
    from system import SystemNESTSim

    sim = SystemNESTSim(output_name="system_experiment")

    electrical_sequence = {
        "channels": [0, 1, 2],
        "times_ms": [10, 20, 35],
        "amplitudes_uA": [2, 4, 3],
    }
    sim.electrical_stim(electrical_sequence, input_duration_ms=5000)

    # Orientation input over time: [cos(theta), sin(theta)]
    visual_series = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    sim.visual_stim(visual_series)
"""

import copy
import os
import pickle
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nest
import numpy as np

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
from network_cortcol import Network

# ====================PROBE=====================================================
volume_v_min = 0  # um
volume_v_max = 1800  # um
volume_h_min = -380  # um
volume_h_max = 380   # um


def make_electrode_grid(n_stim_channels: int, layout: str = "hex") -> np.ndarray:
    """Return (n_stim_channels, 2) array of [h, v] electrode coordinates.

    Parameters
    ----------
    n_stim_channels
        Number of stimulation channels to place.
    layout
        "hex" (default) for a staggered hex-like 2-D lattice, or "rect" for
        the previous rectangular grid behavior. If one axis has zero extent,
        channels are distributed along the non-collapsed axis regardless of
        layout.
    """
    if n_stim_channels <= 0:
        raise ValueError("n_stim_channels must be > 0")

    h_has_extent = (volume_h_max != volume_h_min)
    v_has_extent = (volume_v_max != volume_v_min)

    if h_has_extent and v_has_extent:
        if layout not in {"hex", "rect"}:
            raise ValueError("layout must be 'hex' or 'rect'")

        if layout == "rect":
            # 2-D rectangular grid: factor n_stim_channels into n_h x n_v.
            n_h = int(np.sqrt(n_stim_channels))
            while n_stim_channels % n_h != 0:
                n_h -= 1
            n_v = n_stim_channels // n_h

            h_vals = np.linspace(volume_h_min, volume_h_max, n_h)
            v_vals = np.linspace(volume_v_min, volume_v_max, n_v)
            hh, vv = np.meshgrid(h_vals, v_vals)
            return np.stack([hh.ravel(), vv.ravel()], axis=1)

        # 2-D staggered (hex-like) lattice with exact channel count.
        width = float(volume_h_max - volume_h_min)
        height = float(volume_v_max - volume_v_min)

        best = None
        for n_rows in range(1, n_stim_channels + 1):
            n_long = int(np.ceil((n_stim_channels + (n_rows // 2)) / n_rows))
            n_short = max(1, n_long - 1)

            n_long_rows = (n_rows + 1) // 2
            n_short_rows = n_rows // 2
            total = n_long_rows * n_long + n_short_rows * n_short
            if total < n_stim_channels:
                continue

            dx = width / max(n_long - 1, 1)
            dy = height / max(n_rows - 1, 1)
            ideal_dy = dx * (np.sqrt(3.0) / 2.0)

            dy_score = abs(dy - ideal_dy) / max(ideal_dy, 1e-12)
            overshoot = total - n_stim_channels
            candidate = (dy_score, overshoot, n_rows, n_long)

            if best is None or candidate < best:
                best = candidate

        _, _, n_rows, n_long = best
        n_short = max(1, n_long - 1)
        dx = width / max(n_long - 1, 1)

        row_counts = [n_long if (r % 2 == 0) else n_short for r in range(n_rows)]
        surplus = sum(row_counts) - n_stim_channels

        # Trim surplus points from lower rows while keeping >= 1 point per row.
        for r in range(n_rows - 1, -1, -1):
            if surplus == 0:
                break
            removable = row_counts[r] - 1
            if removable <= 0:
                continue
            drop = min(removable, surplus)
            row_counts[r] -= drop
            surplus -= drop

        y_vals = np.linspace(volume_v_min, volume_v_max, n_rows)
        coords = []
        for r, (y, count) in enumerate(zip(y_vals, row_counts)):
            if count == 1:
                x_vals = np.array([(volume_h_min + volume_h_max) / 2.0])
            elif r % 2 == 1 and n_long > 1:
                x_start = volume_h_min + 0.5 * dx
                x_stop = volume_h_max - 0.5 * dx
                x_vals = np.linspace(x_start, x_stop, count)
            else:
                x_vals = np.linspace(volume_h_min, volume_h_max, count)

            for x in x_vals:
                coords.append((x, y))

        return np.asarray(coords, dtype=float)
    elif v_has_extent:
        # Only vertical extent: single column
        n_h = 1
        n_v = n_stim_channels
    elif h_has_extent:
        # Only horizontal extent: single row
        n_h = n_stim_channels
        n_v = 1
    else:
        # Both axes collapsed: all electrodes at the same point
        n_h = 1
        n_v = n_stim_channels

    h_vals = np.linspace(volume_h_min, volume_h_max, n_h)
    v_vals = np.linspace(volume_v_min, volume_v_max, n_v)
    hh, vv = np.meshgrid(h_vals, v_vals)
    return np.stack([hh.ravel(), vv.ravel()], axis=1)

sigma_e_um = 2.76e-7
conductivity_constant = 10


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
        fast_mode: bool = False,
        fast_sim_resolution_ms: Optional[float] = None,
        n_stim_channels: int = 128,
        n_scaling: Optional[float] = None,
        k_scaling: Optional[float] = None,
        rng_seed: Optional[int] = None,
        psp_exc_mean: Optional[float] = None,
        g_inh: Optional[float] = None,
        bg_rate: Optional[float] = None,
        dc_amp_extra: float = 0.0,
        volume_bounds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize NEST and set up the network.

        This constructor performs simulation setup immediately:
        - resets NEST kernel
        - creates/connects network
        - runs pre-simulation baseline (`t_presim`)

        Parameters
        ----------
        fast_mode
            If True, use a coarser simulation resolution for faster iteration.
        fast_sim_resolution_ms
            Optional override for simulation resolution when fast_mode is True.
            Defaults to 1.0 ms if not provided.
        n_stim_channels
            Number of stimulation electrodes. Electrode coordinates are
            computed as an evenly-spaced grid within the probe volume.
        n_scaling
            Override for N_scaling (neuron count factor). If None, uses the
            value from network_params.py (default 0.05).
        k_scaling
            Override for K_scaling (indegree factor). If None, uses the
            value from network_params.py (default 0.05).
        rng_seed
            Override for the system RNG seed. This seed controls NEST's RNG
            and neuron placement. If None, uses sim_params.py default (42).
            This is the single source of truth for all downstream seeds.
        psp_exc_mean
            Override for mean excitatory PSP (mV). Controls overall synaptic
            gain. Default from network_params.py is 0.15 mV.
        g_inh
            Override for relative inhibitory synaptic weight (negative value).
            Default from network_params.py is -4.
        bg_rate
            Override for Poisson background input rate (spikes/s).
            Default from network_params.py is 8.
        dc_amp_extra
            Additional DC current (pA) injected into all neurons on top of
            any computed DC compensation. Use to shift the operating point.
        volume_bounds
            Override probe volume bounds (um) as a dict with keys
            ``v_min``, ``v_max``, ``h_min``, ``h_max``.  If None, uses
            module-level defaults.
        """
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.stim_pulse_params = stim_pulse_params or {"pulse_width_ms": 0.2, "ipi_ms": 0.2}

        self.net_dict = copy.deepcopy(net_dict)
        self.sim_dict = copy.deepcopy(sim_dict)
        self.stim_dict = copy.deepcopy(stim_dict)

        if rng_seed is not None:
            self.sim_dict["rng_seed"] = rng_seed

        if n_scaling is not None:
            self.net_dict["N_scaling"] = n_scaling
        if k_scaling is not None:
            self.net_dict["K_scaling"] = k_scaling

        # Hyperparameter overrides that require recomputing derived matrices
        _recompute_psp = False
        if psp_exc_mean is not None:
            self.net_dict["PSP_exc_mean"] = psp_exc_mean
            _recompute_psp = True
        if g_inh is not None:
            self.net_dict["g"] = g_inh
            _recompute_psp = True
        if _recompute_psp:
            from corcol_params.network_params import get_exc_inh_matrix
            num_pops = len(self.net_dict["populations"])
            psp_mat = get_exc_inh_matrix(
                self.net_dict["PSP_exc_mean"],
                self.net_dict["PSP_exc_mean"] * self.net_dict["g"],
                num_pops,
            )
            psp_mat[0, 2] = 2.0 * self.net_dict["PSP_exc_mean"]  # L4E→L23E doubled
            self.net_dict["PSP_matrix_mean"] = psp_mat

        if bg_rate is not None:
            self.net_dict["bg_rate"] = bg_rate

        self._dc_amp_extra = dc_amp_extra

        # Override module-level probe volume bounds if provided
        global volume_v_min, volume_v_max, volume_h_min, volume_h_max
        if volume_bounds is not None:
            volume_v_min = volume_bounds.get("v_min", volume_v_min)
            volume_v_max = volume_bounds.get("v_max", volume_v_max)
            volume_h_min = volume_bounds.get("h_min", volume_h_min)
            volume_h_max = volume_bounds.get("h_max", volume_h_max)

        if fast_mode:
            fast_resolution = (
                float(fast_sim_resolution_ms)
                if fast_sim_resolution_ms is not None
                else 1.0
            )
            if fast_resolution <= 0:
                raise ValueError("fast_sim_resolution_ms must be > 0")
            self.sim_dict["sim_resolution"] = fast_resolution

        nest.ResetKernel()
        if os.environ.get("PYNEST_QUIET"):
            nest.set_verbosity("M_ERROR")
            self.sim_dict["print_time"] = False

        nscale = self.net_dict["N_scaling"]
        base_path = os.path.join(
            os.getcwd(),
            "outputs",
            f"data_system_sim_{nscale}scale",
        )
        self.sim_dict["data_path"] = os.path.join(base_path, output_name)
        os.makedirs(self.sim_dict["data_path"], exist_ok=True)

        ch_coordinates = make_electrode_grid(n_stim_channels, layout="hex")
        self.n_stim_channels = n_stim_channels
        self.electrodes = StimElectrodes(
            ch_coordinates,
            self.stim_pulse_params,
            amp_decay_func,
        )

        self.network = Network(self.sim_dict, self.net_dict, self.stim_dict)
        self.network.create()

        # Apply extra DC bias to shift operating point of all neurons
        if self._dc_amp_extra != 0.0:
            for pop in self.network.pops:
                current_Ie = pop.get("I_e")
                if isinstance(current_Ie, (list, np.ndarray)):
                    pop.set(I_e=np.array(current_Ie) + self._dc_amp_extra)
                else:
                    pop.set(I_e=current_Ie + self._dc_amp_extra)

        self.network.connect()
        self.network.simulate_baseline(self.sim_dict["t_presim"])

        self._all_neurons = []
        for pop in self.network.pops:
            for neuron in pop:
                self._all_neurons.append(neuron)
        if len(self._all_neurons) != self.network.n_neurons:
            raise RuntimeError("Mismatch between flattened neuron count and network.n_neurons")

        self._impulse_response_cached = False

        print("SystemNESTSim initialized.")
        print("Total # neurons simulated:", self.network.n_neurons)
        print("Data path:", self.sim_dict["data_path"])
        print("Simulation resolution (ms):", self.sim_dict["sim_resolution"])

    def validate_event_arrays(
        self,
        channels: np.ndarray,
        times_ms: np.ndarray,
        amplitudes_uA: np.ndarray,
        input_duration_ms: float,
    ):
        if not (len(channels) == len(times_ms) == len(amplitudes_uA)):
            raise ValueError("channels, times_ms, amplitudes_uA must have equal length")
        if np.any(channels < 0) or np.any(channels >= self.n_stim_channels):
            raise ValueError(f"Channel indices must be in [0, {self.n_stim_channels - 1}]")
        if np.any(times_ms < 0):
            raise ValueError("times_ms must be non-negative")
        if np.any(amplitudes_uA < 0):
            raise ValueError("amplitudes_uA must be non-negative")
        if np.any(times_ms > input_duration_ms):
            raise ValueError("Found stimulation times beyond input_duration_ms")

    @staticmethod
    def get_current_biological_time_ms() -> float:
        """Return current NEST biological time in ms."""
        if hasattr(nest, "biological_time"):
            return float(nest.biological_time)

        try:
            return float(nest.GetKernelStatus("biological_time"))
        except Exception:
            return float(nest.GetKernelStatus("time"))

    def parse_stim_event_sequence(
        self,
        event_sequence: Dict[str, Iterable[float]],
        *,
        sort_by_time: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse an event-sequence dict into numpy arrays.

        Expected keys: `channels`, `times_ms`, `amplitudes_uA`.
        """
        required_keys = ("channels", "times_ms", "amplitudes_uA")
        missing_keys = [key for key in required_keys if key not in event_sequence]
        if missing_keys:
            raise KeyError(f"Missing keys in event_sequence: {missing_keys}")

        channels = np.asarray(event_sequence["channels"], dtype=int)
        times_ms = np.asarray(event_sequence["times_ms"], dtype=float)
        amplitudes_uA = np.asarray(event_sequence["amplitudes_uA"], dtype=float)

        if sort_by_time:
            order = np.argsort(times_ms)
            channels = channels[order]
            times_ms = times_ms[order]
            amplitudes_uA = amplitudes_uA[order]

        return channels, times_ms, amplitudes_uA

    def run_events(
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

        self.validate_event_arrays(channels, times_ms, amplitudes_uA, input_duration_ms)

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

    def create_generators_from_channel_events(
        self,
        channels: np.ndarray,
        absolute_times_ms: np.ndarray,
        amplitudes_uA: np.ndarray,
    ) -> List[Any]:
        """Create per-neuron step current generators from channel-level events."""
        if len(channels) == 0:
            return []

        build_stimulations_from_events(
            self.electrodes,
            channels,
            absolute_times_ms,
            amplitudes_uA,
        )

        if not self._impulse_response_cached:
            self.electrodes.compute_impulse_response_matrix(self.network.neuron_locations)
            self._impulse_response_cached = True

        self.electrodes.compute_stim_current_matrix()
        self.electrodes.calculate_induced_current_matrix()
        return self.electrodes.get_current_generators()

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
        channels, times_ms, amplitudes_uA = self.parse_stim_event_sequence(
            current_amplitude_sequence,
            sort_by_time=False,
        )

        return self.run_events(
            channels,
            times_ms,
            amplitudes_uA,
            input_duration_ms,
            output_prefix,
        )

    def visual_stim(
        self,
        orientation_time_series: Iterable[Iterable[float]],
        input_duration_ms: Optional[float] = None,
        dt_ms: Optional[float] = None,
        baseline_pA: float = 0.0,
        gain_pA: float = 80.0,
        sparsity: float = 0.1,
        seed: Optional[int] = None,
        output_prefix: str = "visual",
        realtime_progress: bool = False,
        progress_step_ms: float = 1000.0,
    ) -> Dict[str, str]:
        """
        Run visual stimulation from orientation vectors [cos(theta), sin(theta)].

        Parameters
        ----------
        orientation_time_series
            Time-series shaped (T, 2), each row [cos(theta), sin(theta)].
        input_duration_ms
            Total duration of this visual input. If None, inferred as
            `len(orientation_time_series) * dt_ms`.
        dt_ms
            Time step between rows in the time-series. If None, uses native
            simulation resolution (`sim_dict["sim_resolution"]`).
        baseline_pA
            Baseline current (pA) delivered to selected visual-input neurons.
        gain_pA
            Gain scaling for orientation projection onto neuron preferences.
        sparsity
            Fraction of neurons receiving visual input (0 < sparsity <= 1).
        seed
            Random seed for sparse-neuron and preference sampling.
            If None, derives from the system RNG seed (sim_dict["rng_seed"]).
        output_prefix
            Prefix used in output pickle filenames.
        realtime_progress
            If True, simulate in chunks and print progress updates.
        progress_step_ms
            Chunk size in ms for progress updates when realtime_progress is True.
        """
        orientation_array = np.asarray(orientation_time_series, dtype=float)
        if orientation_array.ndim != 2 or orientation_array.shape[1] != 2:
            raise ValueError("orientation_time_series must have shape (T, 2)")
        if dt_ms is None:
            dt_ms = float(self.sim_dict["sim_resolution"])
        if dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1]")
        if progress_step_ms <= 0:
            raise ValueError("progress_step_ms must be > 0")

        if input_duration_ms is None:
            input_duration_ms = float(len(orientation_array) * dt_ms)
        if input_duration_ms <= 0:
            raise ValueError("input_duration_ms must be > 0")

        # Number of input samples expected at native / requested dt.
        expected_steps = int(np.floor(float(input_duration_ms) / dt_ms))
        if expected_steps <= 0:
            raise ValueError("No simulation steps available for given input_duration_ms and dt_ms")

        max_steps = expected_steps
        num_steps = min(len(orientation_array), max_steps)
        orientation_array = orientation_array[:num_steps]

        if num_steps == 0:
            raise ValueError("orientation_time_series is empty after duration truncation")

        # step_current_generator times are absolute biological times in NEST.
        # Shift events to start after the current simulation time (post-presim).
        current_time_ms = self.get_current_biological_time_ms()
        time_offset_ms = max(float(nest.resolution), 1e-6)
        event_times = (
            current_time_ms + np.arange(num_steps, dtype=float) * dt_ms + time_offset_ms
        )

        # Keep only valid stimulation time points within this run's duration.
        valid_mask = event_times <= (current_time_ms + float(input_duration_ms))
        if not np.any(valid_mask):
            raise ValueError(
                "No valid event times remain within input_duration_ms after positive-time offset"
            )
        event_times = event_times[valid_mask]
        orientation_array = orientation_array[valid_mask]
        num_steps = len(event_times)

        if seed is None:
            seed = self.sim_dict["rng_seed"]
        rng = np.random.default_rng(seed)
        num_neurons = self.network.n_neurons
        num_target_neurons = max(1, int(np.round(sparsity * num_neurons)))
        target_neuron_indices = rng.choice(
            num_neurons,
            size=num_target_neurons,
            replace=False,
        )

        # for each neuron, assign a preferred theta and compute its projected response to the input orientation time-series
        preferred_thetas = rng.uniform(0.0, 2.0 * np.pi, size=num_target_neurons)
        preferred_vectors = np.stack(
            [np.cos(preferred_thetas), np.sin(preferred_thetas)],
            axis=1,
        )
        projections = orientation_array @ preferred_vectors.T # visual stims @ preferred directions
        target_currents = baseline_pA + gain_pA * projections

        all_neurons = []
        for pop in self.network.pops:
            for neuron in pop:
                all_neurons.append(neuron)

        if len(all_neurons) != num_neurons:
            raise RuntimeError("Mismatch between flattened neuron count and network.n_neurons")

        # Key line conecting assigned single-neuron preferred orientations to responses. 
        current_generators = []
        for i_target, neuron_index in enumerate(target_neuron_indices):
            amplitude_values = target_currents[:, i_target]
            generator = nest.Create(
                "step_current_generator",
                params={
                    "label": f"visual_input_neuron_{int(neuron_index)}",
                    "amplitude_times": event_times,
                    "amplitude_values": amplitude_values,
                },
            )
            current_generators.append(generator)
            nest.Connect(generator, all_neurons[int(neuron_index)])

        total_duration_ms = float(input_duration_ms)
        if realtime_progress:
            simulated_ms = 0.0
            print('Inside realtime_progress')
            while simulated_ms < total_duration_ms:
                chunk_ms = min(progress_step_ms, total_duration_ms - simulated_ms)
                print('Chunk duration', chunk_ms)
                nest.Simulate(chunk_ms)
                simulated_ms += chunk_ms
                print(
                    f"Visual sim progress: {simulated_ms:.0f}/{total_duration_ms:.0f} ms "
                    f"(biological time {self.get_current_biological_time_ms():.1f} ms)",
                    flush=True,
                )
        else:
            print('Total duration', total_duration_ms)
            nest.Simulate(total_duration_ms)

        print('Simulation over', flush=True)
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
                    "rng_seed": self.sim_dict["rng_seed"],
                    "visual_seed": seed,
                    "neuron_locations": self.network.neuron_locations.copy(),
                },
                meta_file,
            )

        print(f"Saved spike rates -> {rates_path}")
        print(f"Saved visual stim metadata -> {meta_path}")
        print(
            f"Visual input targeted {num_target_neurons}/{num_neurons} neurons "
            f"(sparsity={sparsity:.3f})"
        )
        print(f"Created {len(current_generators)} visual current generators")

        return {
            "spike_rates_path": rates_path,
            "stim_metadata_path": meta_path,
        }

    def setup_visual_generators_from_metadata(
        self,
        metadata_path: str,
    ) -> None:
        """Recreate per-neuron visual current generators from a saved stim metadata file.

        Loads ``target_neuron_indices`` and ``preferred_thetas`` from the pickle
        written by :meth:`visual_stim`, creates one ``step_current_generator``
        per target neuron with zero amplitude, and connects each to its neuron.

        Each neuron has its own preferred orientation (``preferred_thetas``).
        Amplitudes are set to zero here; call :meth:`update_visual_generators_for_orientation`
        each closed-loop interval to inject the current for a chosen orientation.

        Parameters
        ----------
        metadata_path
            Path to the ``*_stim_metadata.pkl`` file saved by :meth:`visual_stim`.
        """
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)

        target_neuron_indices = meta["target_neuron_indices"]
        preferred_thetas = meta["preferred_thetas"]

        # Check if the RNG seed used during the original visual stim run
        # matches the current system seed.  A mismatch means neuron placement
        # may differ, which would silently mis-route visual currents.
        saved_seed = meta.get("rng_seed")
        if saved_seed is not None and saved_seed != self.sim_dict["rng_seed"]:
            warnings.warn(
                f"System RNG seed ({self.sim_dict['rng_seed']}) differs from "
                f"the seed used in the original visual stimulation run "
                f"({saved_seed}). Neuron placement will differ, so visual "
                f"current generators may target the wrong neurons.",
                UserWarning,
                stacklevel=2,
            )

        # If the original run saved neuron locations, verify they match.
        saved_locations = meta.get("neuron_locations")
        if saved_locations is not None:
            if not np.allclose(saved_locations, self.network.neuron_locations):
                warnings.warn(
                    "Neuron locations in the current simulation do not match "
                    "those from the original visual stimulation run. This is "
                    "likely caused by a different RNG seed or N_scaling. "
                    "Visual current generators may target the wrong neurons.",
                    UserWarning,
                    stacklevel=2,
                )

        resolution = float(nest.GetKernelStatus("resolution"))
        current_time_ms = self.get_current_biological_time_ms()
        placeholder_time = np.array([current_time_ms + resolution])

        all_neurons = [neuron for pop in self.network.pops for neuron in pop]
        if len(all_neurons) != self.network.n_neurons:
            raise RuntimeError("Mismatch between flattened neuron count and network.n_neurons")

        self._visual_target_neuron_indices = target_neuron_indices
        self._visual_preferred_thetas = preferred_thetas
        self._visual_baseline_pA: float = 0.0
        self._visual_gain_pA: float = 80.0
        self._visual_generators = []

        for i_target, neuron_index in enumerate(target_neuron_indices):
            gen = nest.Create(
                "step_current_generator",
                params={
                    "label": f"visual_input_neuron_{int(neuron_index)}",
                    "amplitude_times": placeholder_time,
                    "amplitude_values": [0.0],
                },
            )
            self._visual_generators.append(gen)
            nest.Connect(gen, all_neurons[int(neuron_index)])

        print(
            f"setup_visual_generators_from_metadata: "
            f"{len(self._visual_generators)} generators connected "
            f"(zero amplitude; call update_visual_generators_for_orientation each interval)"
        )

    def update_visual_generators_for_orientation(
        self,
        theta_deg: float,
        interval_duration_ms: float,
        baseline_pA: Optional[float] = None,
        gain_pA: Optional[float] = None,
        dt_ms: Optional[float] = None,
    ) -> None:
        """Update visual generator amplitudes for a given orientation and simulate.

        Each neuron's amplitude is ``baseline + gain * dot(orient_vec, preferred_vec)``,
        where ``preferred_vec`` is that neuron's stored preferred orientation from the
        original visual_stim run.  The amplitudes are held constant across
        ``interval_duration_ms``.

        Parameters
        ----------
        theta_deg
            Input orientation angle in degrees.
        interval_duration_ms
            Simulation duration for this interval.
        baseline_pA, gain_pA
            Override stored baseline/gain; if None, uses values from
            :meth:`setup_visual_generators_from_metadata`.
        dt_ms
            Time step for amplitude events.  Defaults to simulation resolution.
        """
        if not hasattr(self, "_visual_generators"):
            raise RuntimeError("Call setup_visual_generators_from_metadata first.")

        if baseline_pA is None:
            baseline_pA = self._visual_baseline_pA
        if gain_pA is None:
            gain_pA = self._visual_gain_pA
        if dt_ms is None:
            dt_ms = float(self.sim_dict["sim_resolution"])

        resolution = float(nest.GetKernelStatus("resolution"))
        current_time_ms = self.get_current_biological_time_ms()
        time_offset_ms = max(resolution, 1e-6)

        num_steps = max(1, int(round(interval_duration_ms / dt_ms)))
        event_times = (
            current_time_ms
            + np.arange(num_steps, dtype=float) * dt_ms
            + time_offset_ms
        )
        valid_mask = event_times <= current_time_ms + interval_duration_ms
        event_times = event_times[valid_mask]
        if len(event_times) == 0:
            event_times = np.array([current_time_ms + time_offset_ms])
        event_times = np.round(event_times / resolution) * resolution

        theta = np.deg2rad(theta_deg)
        orient_vec = np.array([np.cos(theta), np.sin(theta)])
        preferred_vectors = np.stack(
            [np.cos(self._visual_preferred_thetas), np.sin(self._visual_preferred_thetas)],
            axis=1,
        )
        projections = orient_vec @ preferred_vectors.T   # (n_target,)
        amplitudes = baseline_pA + gain_pA * projections # (n_target,)

        for i_target, gen in enumerate(self._visual_generators):
            nest.SetStatus(
                gen,
                {
                    "amplitude_times": event_times,
                    "amplitude_values": np.full(len(event_times), float(amplitudes[i_target])),
                },
            )
