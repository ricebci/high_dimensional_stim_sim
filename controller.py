from dataclasses import dataclass
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


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


class RepertoireController:
    """
    Closed-loop controller using a StimEncodingCNN to predict neural responses
    and select stimulation patterns that drive the network toward a target state.

    Maintains a rolling buffer of binned (bin_ms-wide) spike counts and stim
    amplitudes.  Since closed_loop_interval_ms == bin_ms, each simulation
    interval corresponds to exactly one model bin.

    Every n_repertoire_update_ms of simulated time the repertoire is updated
    with new (state, stim) observations collected since the last update.

    Parameters
    ----------
    model : StimEncodingCNN
    meta : dict
        Checkpoint metadata from meta.pt.
    target : np.ndarray, shape (n_neurons,)
        Desired mean-over-time log-rate per neuron. The controller selects the
        stim pattern whose predicted response is closest (L2) to this target.
    max_len : int
        Maximum number of repertoire entries.
    n_repertoire_update_ms : float
        How often (ms of simulated time) to add cached observations to the
        repertoire.
    device : str
        Torch device string, e.g. 'cpu', 'mps', 'cuda'.
    """

    def __init__(
        self,
        model,
        meta: dict,
        target: np.ndarray,
        max_len: int = 100_000,
        n_repertoire_update_ms: float = 10_000.0,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.target = np.asarray(target, dtype=np.float32)
        self.max_len = max_len
        self.n_repertoire_update_ms = n_repertoire_update_ms

        self.n_stim_channels: int = meta["n_stim_channels"]
        self.n_neurons: int = meta["n_neurons"]
        self.n_input_bins: int = meta["n_input_bins"]
        self.n_output_bins: int = meta["n_output_bins"]
        self.bin_ms: float = float(meta["bin_ms"])
        self.history: int = meta.get("history", 0)
        self.kernel_sizes: List[int] = meta.get("kernel_sizes", [3, 3])
        self.total_conv_reduction: int = sum(k - 1 for k in self.kernel_sizes)
        self.n_total_bins: int = self.n_input_bins + self.total_conv_reduction

        # Rolling buffers — spike_buf[:, -1] is the most recently completed bin
        self.spike_buf = np.zeros((self.n_neurons, self.n_total_bins), dtype=np.float32)
        self.stim_buf = np.zeros((self.n_stim_channels, self.n_total_bins), dtype=np.float32)

        # Repertoire: list of {"stim_pattern": dict, "pred_response_mean": (n_neurons,)}
        self._entries: List[Dict] = []

        # Observation cache: (spike_buf_snap, stim_buf_snap, stim_pattern_dict)
        # captured each time the controller returns a non-None stim pattern
        self._obs_cache: List[Tuple[np.ndarray, np.ndarray, Dict]] = []
        self._last_update_ms: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_input(self, candidate_stim_bins: np.ndarray) -> torch.Tensor:
        """
        Construct the (1, C, T) model input from current buffer + candidate stim.

        candidate_stim_bins : (n_stim_channels, n_input_bins) float32
            The proposed stimulation pattern binned at bin_ms resolution.
        """
        ctx = self.total_conv_reduction
        n_total = self.n_total_bins  # == n_input_bins + ctx

        # Stim input: past ctx bins of stim history + candidate stim
        x_stim = np.zeros((self.n_stim_channels, n_total), dtype=np.float32)
        if ctx > 0:
            x_stim[:, :ctx] = self.stim_buf[:, -ctx:]
        x_stim[:, ctx:] = candidate_stim_bins

        if self.history > 0:
            # Spike history is lagged 1 bin relative to stim in the training data:
            # positions [0, ctx+1) are known past; positions [ctx+1, n_total) are
            # future (unknown) → leave as zero.
            h = np.zeros((self.n_neurons, n_total), dtype=np.float32)
            known = ctx + 1
            h[:, :known] = self.spike_buf[:, -known:]
            x = np.concatenate([x_stim, h], axis=0)
        else:
            x = x_stim

        return torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, C, T)

    @torch.no_grad()
    def _predict_mean(self, x_tensor: torch.Tensor) -> np.ndarray:
        """Run the model and return mean log-rate per neuron: (n_neurons,)."""
        y = self.model(x_tensor)          # (1, n_neurons, n_output_bins)
        return y[0].mean(dim=-1).cpu().numpy()

    def _stim_dict_to_bins(self, stim_dict: Optional[Dict]) -> np.ndarray:
        """Convert {channels, times_ms, amplitudes_uA} → (n_stim_channels, n_input_bins)."""
        bins = np.zeros((self.n_stim_channels, self.n_input_bins), dtype=np.float32)
        if stim_dict is None or len(stim_dict.get("channels", [])) == 0:
            return bins
        chs = np.asarray(stim_dict["channels"], dtype=int)
        ts = np.asarray(stim_dict["times_ms"], dtype=float)
        amps = np.asarray(stim_dict["amplitudes_uA"], dtype=float)
        b = np.floor(ts / self.bin_ms).astype(int)
        mask = (b >= 0) & (b < self.n_input_bins) & (chs >= 0) & (chs < self.n_stim_channels)
        np.add.at(bins, (chs[mask], b[mask]), amps[mask])
        return bins

    def _update_stim_buf_col(self, delivered: Optional[Dict]) -> np.ndarray:
        """Build a stim amplitude column (n_stim_channels,) for the just-completed bin."""
        col = np.zeros(self.n_stim_channels, dtype=np.float32)
        if delivered is not None and len(delivered.get("channels", [])) > 0:
            chs = np.asarray(delivered["channels"], dtype=int)
            amps = np.asarray(delivered["amplitudes_uA"], dtype=float)
            valid = (chs >= 0) & (chs < self.n_stim_channels)
            np.add.at(col, chs[valid], amps[valid])
        return col

    # ------------------------------------------------------------------
    # Repertoire population
    # ------------------------------------------------------------------

    def fit_stim_repertoire(
        self,
        model_inputs: List[np.ndarray],
        stim_patterns: List[Dict],
    ):
        """
        Seed the repertoire from pre-built model inputs and their stim pattern dicts.

        Parameters
        ----------
        model_inputs : list of (n_stim_channels + n_neurons, n_total_bins) float32 arrays
            Already-formatted model input tensors (e.g. from a NESTStimSpikeDataset).
        stim_patterns : list of {channels, times_ms, amplitudes_uA} dicts
            The stimulation pattern corresponding to each model input.
        """
        added = 0
        for x_np, pattern in zip(model_inputs, stim_patterns):
            if len(self._entries) >= self.max_len:
                logging.warning("Repertoire at capacity (%d); stopping fit.", self.max_len)
                break
            x_t = torch.from_numpy(
                np.asarray(x_np, dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            pred_mean = self._predict_mean(x_t)
            self._entries.append({"stim_pattern": pattern, "pred_response_mean": pred_mean})
            added += 1
        print(f"fit_stim_repertoire: added {added} entries (total {len(self._entries)}).")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_stim_pattern(self) -> Optional[Dict]:
        """
        Return the stim_pattern from the repertoire whose predicted mean response
        is closest (L2) to self.target.  Returns None if the repertoire is empty.
        """
        if not self._entries:
            return None
        preds = np.stack([e["pred_response_mean"] for e in self._entries])  # (N, n_neurons)
        dists = np.linalg.norm(preds - self.target[None, :], axis=1)
        best = int(np.argmin(dists))
        return self._entries[best]["stim_pattern"]

    def at_capacity(self) -> bool:
        return len(self._entries) >= self.max_len

    # ------------------------------------------------------------------
    # Periodic repertoire update from cached observations
    # ------------------------------------------------------------------

    def _flush_obs_cache(self):
        """Add cached (state, stim) observations to the repertoire via the encoder."""
        if not self._obs_cache:
            return
        added = 0
        for spike_snap, stim_snap, pattern in self._obs_cache:
            if self.at_capacity():
                break
            # Temporarily swap buffer to compute prediction from the snapshot state
            saved_spike, saved_stim = self.spike_buf, self.stim_buf
            self.spike_buf, self.stim_buf = spike_snap, stim_snap

            candidate_bins = self._stim_dict_to_bins(pattern)
            x_t = self._build_model_input(candidate_bins)
            pred_mean = self._predict_mean(x_t)

            self.spike_buf, self.stim_buf = saved_spike, saved_stim

            self._entries.append({"stim_pattern": pattern, "pred_response_mean": pred_mean})
            added += 1

        logging.info(
            "Repertoire update: added %d entries (total %d).", added, len(self._entries)
        )
        self._obs_cache.clear()

    # ------------------------------------------------------------------
    # Standard closed-loop controller interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        interval_index: int,
        interval_start_ms: float,
        interval_end_ms: float,
        interval_duration_ms: float,
        new_spikes_by_neuron,
        history,
        delivered_pattern: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Update the rolling buffer, periodically refresh the repertoire, then
        return the stim pattern to deliver in the next interval.

        `delivered_pattern` should be the stim that was applied in the interval
        that just ended (i.e. the pattern that caused `new_spikes_by_neuron`).
        Pass it from the experiment loop for accurate stim-buffer tracking.
        """
        # --- Update rolling spike buffer ---
        spike_counts = np.array(
            [len(s) for s in new_spikes_by_neuron], dtype=np.float32
        )
        self.spike_buf = np.roll(self.spike_buf, -1, axis=1)
        self.spike_buf[:, -1] = spike_counts

        # --- Update rolling stim buffer with what was just delivered ---
        stim_col = self._update_stim_buf_col(delivered_pattern)
        self.stim_buf = np.roll(self.stim_buf, -1, axis=1)
        self.stim_buf[:, -1] = stim_col

        # --- Periodic repertoire update ---
        if (
            interval_end_ms - self._last_update_ms >= self.n_repertoire_update_ms
            and not self.at_capacity()
        ):
            self._flush_obs_cache()
            self._last_update_ms = interval_end_ms

        # --- Query repertoire ---
        chosen = self.get_stim_pattern()

        # --- Cache this observation for the next update round ---
        if chosen is not None and not self.at_capacity():
            self._obs_cache.append((
                self.spike_buf.copy(),
                self.stim_buf.copy(),
                chosen,
            ))

        return chosen
