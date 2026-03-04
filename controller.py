from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class SpikeThresholdController:
    """
    Simple closed-loop controller based on interval spike count threshold.

    If the number of spikes observed in the just-finished interval is >=
    `spike_threshold`, this controller returns a stimulation pattern for the
    next interval; otherwise it returns None.
    """

    stim_channels: Iterable[int]
    stim_amplitude_uA: float = 2.0
    stim_time_ms: float = 0.0
    spike_threshold: int = 50
    refractory_intervals: int = 0

    def __post_init__(self):
        self.stim_channels = np.asarray(list(self.stim_channels), dtype=int)
        if len(self.stim_channels) == 0:
            raise ValueError("stim_channels must contain at least one channel")
        if self.stim_amplitude_uA < 0:
            raise ValueError("stim_amplitude_uA must be non-negative")
        if self.stim_time_ms < 0:
            raise ValueError("stim_time_ms must be non-negative")
        if self.spike_threshold < 0:
            raise ValueError("spike_threshold must be non-negative")
        if self.refractory_intervals < 0:
            raise ValueError("refractory_intervals must be non-negative")

        self._last_stim_interval = -np.inf

    def get_stim_pattern(
        self,
        *,
        interval_index: int,
        interval_start_ms: float,
        interval_end_ms: float,
        interval_duration_ms: float,
        new_spikes_by_neuron,
        history,
    ) -> Optional[Dict[str, Iterable[float]]]:
        total_new_spikes = int(sum(len(spikes) for spikes in new_spikes_by_neuron))

        if interval_index - self._last_stim_interval <= self.refractory_intervals:
            return None

        if total_new_spikes < self.spike_threshold:
            return None

        self._last_stim_interval = interval_index

        channels = self.stim_channels
        times_ms = np.full(len(channels), float(self.stim_time_ms), dtype=float)
        amplitudes_uA = np.full(
            len(channels),
            float(self.stim_amplitude_uA),
            dtype=float,
        )

        return {
            "channels": channels,
            "times_ms": times_ms,
            "amplitudes_uA": amplitudes_uA,
        }
