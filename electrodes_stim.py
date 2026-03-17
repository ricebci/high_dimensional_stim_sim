import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np

UA2PA = 1e6


class StimElectrodes:
    def __init__(
        self,
        ch_coordinates: np.ndarray,
        stim_pulse_params: dict,
        current_disperse_func: callable,
    ):
        """
        ch_coordinates : (n_chs, 2) ndarray of channel coordinates
        current_disperse_func : function(amp_uA, distance_um) -> amp_pA at distance; should support broadcasting
        """
        self.n_chs = len(ch_coordinates)
        self.ch_coordinates = ch_coordinates
        self.current_disperse_func = current_disperse_func
        self.stim_pulse_params = stim_pulse_params

        self.stimulations = {}
        self.stim_onset_times_by_ch = {}
        self.stim_amplitudes_by_ch = {}

    def generate_biphasic_pulse(self, t0, amplitude):
        """
        Generate biphasic pulse timing and amplitude values.
        """
        pulse_width_ms = self.stim_pulse_params.get("pulse_width_ms", 0.2)
        ipi_ms = self.stim_pulse_params.get("ipi_ms", 0.2)

        t1 = np.round(t0 + pulse_width_ms, 1)  # End of cathodic phase
        t2 = np.round(t1 + ipi_ms, 1)  # Start of anodic phase
        t3 = np.round(t2 + pulse_width_ms, 1)  # End of anodic phase

        # Amplitudes corresponding to those times: [-A, 0, A, 0]
        return [t0, t1, t2, t3], [-amplitude * UA2PA, 0, amplitude * UA2PA, 0]

    def generate_random_stimulation(
        self,
        channels,
        amplitude_range,
        duration_ms,
        stim_rate_hz=10,
        min_pulse_spacing=1,
        n_groups=1,
    ):
        """
        Generate random stimulation across channels with randomized amplitudes and Poisson-distributed stimulation times.
        """
        num_channels = len(channels)
        n_groups = min(n_groups, num_channels)

        # Divide channels into groups
        channels_per_group = np.array_split(channels, n_groups)

        lambda_poisson = stim_rate_hz / 1000  # Rate in events/ms

        for group in channels_per_group:
            # Generate Poisson stimulation times for the group
            stim_times = np.cumsum(
                np.random.exponential(
                    1 / lambda_poisson, int(duration_ms * lambda_poisson)
                )
            )
            stim_times = stim_times[stim_times < duration_ms]

            # Enforce minimum pulse spacing and round times afterward
            filtered_stim_times = []
            prev_time = -np.inf
            for t in stim_times:
                if t - prev_time >= min_pulse_spacing:
                    filtered_stim_times.append(t)
                    prev_time = t

            # Round to nearest simulation resolution and remove duplicates
            stim_times_rounded = np.unique(
                np.round(np.array(filtered_stim_times) / min_pulse_spacing)
                * min_pulse_spacing
            )

            if len(stim_times_rounded) == 0:
                continue

            # Randomized amplitudes
            amplitudes = np.random.choice(amplitude_range, size=len(stim_times_rounded))

            # For each channel in the group, store stimulation info
            for ch in group:
                biphasic_pulses = {"times": [], "amplitudes": []}

                # Generate biphasic pulses
                for stim_time, amplitude in zip(stim_times_rounded, amplitudes):
                    times, amps = self.generate_biphasic_pulse(stim_time, amplitude)

                    biphasic_pulses["times"].extend(times)
                    biphasic_pulses["amplitudes"].extend(amps)

                # Store each channel's stimulation details
                self.stimulations[ch] = biphasic_pulses
                self.stim_onset_times_by_ch[ch] = stim_times_rounded
                self.stim_amplitudes_by_ch[ch] = amplitudes

        return self.stimulations

    def generate_deterministic_stimulation(
        self,
        channels,
        times,
        amplitude_range,
        duration_ms,
        interpattern_time_ms,
        pulse_width_ms=0.2,
    ):
        """
        Generate deterministic stimulation with a user-defined pattern and repeat it.

        Parameters:
        - channels: List of channel indices for one pattern.
        - times: List of stimulation times in milliseconds for one pattern.
        - amplitude_range: List of possible stimulation amplitudes.
        - duration_ms: Total duration of the stimulation in milliseconds.
        - interpattern_time_ms: Time interval between repetitions of the pattern.
        - pulse_width_ms: Width of the biphasic pulse (default is 0.2 ms).
        """
        unique_channels = np.unique(channels)
        biphasic_pulses = {
            ch: {"times": [], "amplitudes": []} for ch in unique_channels
        }
        stim_onset_times_by_ch = {ch: [] for ch in unique_channels}
        stim_amplitudes_by_ch = {ch: [] for ch in unique_channels}

        current_time = 0
        while current_time < duration_ms:
            for ch, stim_time in zip(channels, times):
                pulse_start_time = current_time + stim_time
                if pulse_start_time >= duration_ms:
                    break

                # Randomly choose an amplitude for this pulse
                amplitude = np.random.choice(amplitude_range)

                # Generate biphasic pulse
                pulse_times, pulse_amps = self.generate_biphasic_pulse(
                    pulse_start_time, amplitude
                )

                # Append the biphasic pulse times and amplitudes
                biphasic_pulses[ch]["times"].extend(pulse_times)
                biphasic_pulses[ch]["amplitudes"].extend(pulse_amps)

                # Record onset times and amplitudes for each channel
                stim_onset_times_by_ch[ch].append(pulse_start_time)
                stim_amplitudes_by_ch[ch].append(amplitude)

            # Move to the next pattern interval based on the first pulse time
            current_time += interpattern_time_ms

        # Store the results
        self.stimulations = biphasic_pulses
        self.stim_onset_times_by_ch = stim_onset_times_by_ch
        self.stim_amplitudes_by_ch = stim_amplitudes_by_ch

        return self.stimulations

    def compute_impulse_response_matrix(self, neuron_locations):
        """
        Computes the impulse response matrix (H), which represents the induced current at each neuron
        in response to a unit stimulation current (1 µA) applied at each electrode channel.

        The matrix H is of size (N, M), where N is the number of neurons and M is the number of
        stimulation channels. Each element H[n, m] represents the induced current (in pA) at neuron n
        when a 1 µA stimulation current is applied at channel m.
        """
        # N x M matrix where N is num neurons and M is number of channels
        N = len(neuron_locations)
        M = self.n_chs

        self.H = np.zeros((N, M))

        ch_to_neuron_distances = np.sqrt(
            (self.ch_coordinates[:, 0, None] - neuron_locations[:, 0]) ** 2
            + (self.ch_coordinates[:, 1, None] - neuron_locations[:, 1]) ** 2
        )

        for ch_index in range(M):
            ch_i_to_neuron_distances = ch_to_neuron_distances[ch_index, :]
            induced_currents_at_neurons = self.current_disperse_func(
                1 * UA2PA, ch_i_to_neuron_distances
            ).T
            self.H[:, ch_index] = induced_currents_at_neurons

    def compute_stim_current_matrix(self):
        """
        Computes the stimulation current matrix (X), which represents the current applied to each channel
        across a series of unique time points.

        The matrix X is of size (M, T), where M is the number of stimulation channels and T is the number
        of unique time points when stimulation occurs. Each element X[m, t] corresponds to the current
        applied to channel m at time t.

        The amplitudes are scaled back from picoamperes (pA) to microamperes (µA) by dividing by UA2PA.
        """

        M = self.n_chs

        # Concatenate all the times from each channel
        all_times = np.concatenate(
            [self.stimulations[ch]["times"] for ch in np.arange(M)]
        )

        resolution = float(nest.GetKernelStatus("resolution"))
        rounded_all_times = np.round(all_times / resolution) * resolution
        self.unique_timestamps = np.unique(rounded_all_times)
        self.unique_timestamps = self.unique_timestamps[self.unique_timestamps > 0]

        T = len(self.unique_timestamps)

        # Initialize current matrix
        self.X = np.zeros((M, T), dtype="float16")

        for ch in range(M):
            # Round and find the indices in `unique_timestamps`
            ch_times = np.round(np.asarray(self.stimulations[ch]["times"]) / resolution) * resolution
            ch_amplitudes = np.array(self.stimulations[ch]["amplitudes"]) / UA2PA

            # Get the corresponding indices in the unique timestamps array
            indices_in_unique = np.searchsorted(self.unique_timestamps, ch_times)

            # Populate the matrix directly
            self.X[ch, indices_in_unique] = ch_amplitudes

    def calculate_induced_current_matrix(self):
        """
        Calculate the induced current matrix by multiplying the impulse response matrix (H) and the
        stimulation current matrix (X).

        This matrix represents the total induced current at each neuron over time. The resulting matrix is stored
        as 'induced_current_matrix' and is of size (N, T), where N is the number of neurons and T is the number
        of unique time points.
        """
        if not hasattr(self, "H") or not hasattr(self, "X"):
            raise AttributeError(
                "Impulse response matrix (H) or stimulation current matrix (X) not found."
            )

        self.induced_current_matrix = np.dot(self.H, self.X)

    def get_current_generators(self):
        """
        Generate NEST step current generators based on the induced current matrix, where each neuron is
        assigned its corresponding induced current over time.

        Returns
        -------
        current_generators : list of NEST step_current_generator objects
            A list of step current generators for each neuron, where each generator delivers the computed induced
            current at unique timestamps.
        """

        if (
            not hasattr(self, "H")
            or not hasattr(self, "X")
            or not hasattr(self, "induced_current_matrix")
        ):
            raise AttributeError(
                "Impulse response matrix (H) or stimulation current matrix (X) not found."
            )

        num_neurons = self.H.shape[0]

        if np.any(np.diff(self.unique_timestamps) <= 0.00001):
            raise ValueError("Non-increasing amplitude times detected!")
            
        if np.any(self.unique_timestamps <= 0):
            raise ValueError("Non-positive timestamps detected!")

        current_generators = [
            nest.Create(
                "step_current_generator",
                params=dict(
                    label=f"induced_current_at_neuron_{neuron_index}",
                    amplitude_times=self.unique_timestamps,
                    amplitude_values=self.induced_current_matrix[neuron_index, :],
                ),
            )
            for neuron_index in range(num_neurons)
        ]

        return current_generators

    def plot_stim_raster(self, time_range_ms=None, ax=None, title=None):
        fontsize = 12
        stim_channel_indices = list(self.stim_onset_times_by_ch.keys())
        num_stim_channels = len(stim_channel_indices)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))
            created_ax = True
        else:
            created_ax = False

        for i, ch_idx in enumerate(stim_channel_indices):
            stim_train = self.stim_onset_times_by_ch[ch_idx]
            # ax.plot(stim_train, [i + 1] * len(stim_train), "|", markersize=2, linewidth=40, color="k")
            ax.vlines(stim_train, i + 0.75, i + 1.25, colors="k", linewidth=1.5)
            
        if time_range_ms is not None:
            ax.set_xlim(time_range_ms[0], time_range_ms[1])

        ax.set_ylim(0, num_stim_channels + 1)
        step = max(1, num_stim_channels // 8)
        ax.set_yticks(range(1, num_stim_channels + 1, step))
        ax.set_yticklabels(
            [str(ch_idx) for ch_idx in stim_channel_indices[::step]], fontsize=fontsize
        )
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.set_xlabel("Time (ms)", fontsize=fontsize)
        ax.set_ylabel("Stim channel index", fontsize=fontsize)

        if title:
            ax.set_title(title)

        if created_ax:
            fig.tight_layout()  # Adjust layout to avoid clipping
