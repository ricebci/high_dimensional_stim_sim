from dataclasses import dataclass
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple
from scipy.special import logsumexp
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
    Closed-loop controller that greedily selects stimulation patterns from a
    fixed repertoire by running the encoding model on each candidate with the
    current neural state and choosing the one whose predicted next state is
    closest (L2) to the target.

    Parameters
    ----------
    model : StimEncodingCNN
    meta : dict
        Checkpoint metadata (meta.json / meta.pt).
    target : np.ndarray, shape (n_neurons,)
        Desired mean-over-time log-rate per neuron.
    max_len : int
        Maximum number of patterns stored in the repertoire.
    batch_size : int
        Number of candidate patterns to evaluate per forward pass.
    device : str
        Torch device string, e.g. 'cpu', 'mps', 'cuda'.
    """

    def __init__(
        self,
        model,
        meta: dict,
        target: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        max_len: int = 100_000,
        device: str = "cpu",
        random_stim_prob: float = 0.0,
        n_relevant_output_bins: Optional[int] = None,
        ar_aggregation: str = "most_recent",
        **kwargs,   # absorb deprecated kwargs (e.g. n_repertoire_update_ms)
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.target = np.asarray(target, dtype=np.float32)
        self.max_len = max_len

        # PCA projection into 2D orientation-tuning space for distance metric.
        # Components shape: (n_components, n_neurons); mean shape: (n_neurons,).
        # Both are in Hz firing-rate units; model predictions are exponentiated
        # to Hz before projection (see get_stim_pattern).
        self.pca_components = np.asarray(pca_components, dtype=np.float32)  # (K, N)
        self.pca_mean       = np.asarray(pca_mean,       dtype=np.float32)  # (N,)
        # Pre-project the target into PC space so we don't recompute it every call.
        self._target_pc = (self.target - self.pca_mean) @ self.pca_components.T  # (K,)

        self.n_stim_channels: int = meta["n_stim_channels"]
        self.n_neurons: int = meta["n_neurons"]
        self.n_input_bins: int = meta["n_input_bins"]
        self.bin_ms: float = float(meta["bin_ms"])
        self.history: int = meta.get("history", 0)
        self.kernel_sizes: List[int] = meta.get("kernel_sizes", [3, 3])
        self.total_conv_reduction: int = sum(k - 1 for k in self.kernel_sizes)
        self.n_total_bins: int = self.n_input_bins + self.total_conv_reduction

        # Rolling buffers for current neural context
        self.spike_buf = np.zeros((self.n_neurons, self.n_total_bins), dtype=np.float32)
        self.stim_buf  = np.zeros((self.n_stim_channels, self.n_total_bins), dtype=np.float32)

        self.random_stim_prob: float = float(random_stim_prob)
        # Only the output bin at index (n_relevant_output_bins - 1) is used when
        # scoring candidates.  Set to ceil(interval_ms / bin_ms); None = last bin.
        self.n_relevant_output_bins: Optional[int] = n_relevant_output_bins

        # Repertoire: list of {channels, times_ms, amplitudes_uA} dicts
        self._entries: List[Dict] = []

        # Amplitudes and max active bins observed in training — used for random exploration.
        self._seen_amplitudes: np.ndarray = np.array([], dtype=np.float32)
        self._max_active_bins: int = 1

        # Model input used for the most recently selected pattern (for fine-tuning).
        self._last_selected_input: Optional[np.ndarray] = None
        # Set to True when the last returned pattern was randomly explored.
        self._last_was_random: bool = False

        if ar_aggregation not in ("most_recent", "average"):
            raise ValueError(f"ar_aggregation must be 'most_recent' or 'average', got {ar_aggregation!r}")
        self.ar_aggregation: str = ar_aggregation


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stim_dict_to_bins(self, stim_dict: Optional[Dict]) -> np.ndarray:
        """Convert a stim pattern dict → (n_stim_channels, n_input_bins) array."""
        bins = np.zeros((self.n_stim_channels, self.n_input_bins), dtype=np.float32)
        if stim_dict is None or len(stim_dict.get("channels", [])) == 0:
            return bins
        chs  = np.asarray(stim_dict["channels"],      dtype=int)
        ts   = np.asarray(stim_dict["times_ms"],       dtype=float)
        amps = np.asarray(stim_dict["amplitudes_uA"], dtype=float)
        b    = np.floor(ts / self.bin_ms).astype(int)
        n_deliver = self.n_relevant_output_bins if self.n_relevant_output_bins is not None else self.n_input_bins
        mask = (b >= 0) & (b < n_deliver) & (chs >= 0) & (chs < self.n_stim_channels)
        np.add.at(bins, (chs[mask], b[mask]), amps[mask])
        return bins

    def _build_model_input(self, candidate_stim_bins: np.ndarray) -> np.ndarray:
        """
        Build a single (C, T) model input from the current rolling buffers plus
        a candidate stim pattern.

        candidate_stim_bins : (n_stim_channels, n_input_bins)
        """
        ctx     = self.total_conv_reduction
        n_total = self.n_total_bins

        x_stim = np.zeros((self.n_stim_channels, n_total), dtype=np.float32)
        if ctx > 0:
            x_stim[:, :ctx] = self.stim_buf[:, -ctx:]
        x_stim[:, ctx:] = candidate_stim_bins

        if self.history > 0:
            h = np.zeros((self.n_neurons, n_total), dtype=np.float32)
            known = ctx + 1
            h[:, :known] = self.spike_buf[:, -known:]
            return np.concatenate([x_stim, h], axis=0)   # (C+N, T)
        return x_stim   # (C, T)

    def _ar_forward(self, inputs: np.ndarray) -> np.ndarray:
        """Autoregressively run the model for n_relevant_output_bins steps.

        At each step k:
          - The predicted spike count at bin k is written into the neural history
            channel at position ctx+1+k for the next step.

        inputs    : (N, C, T) — modified in-place
        Returns   : (N, n_steps, n_neurons) — predictions at every AR step.
        """
        N         = len(inputs)
        n_steps   = self.n_relevant_output_bins or self.n_input_bins
        ctx       = self.total_conv_reduction
        n_stim_ch = self.n_stim_channels

        all_preds = np.empty((N, n_steps, self.n_neurons), dtype=np.float32)
        with torch.no_grad():
            start = 0
            end = N
            for k in range(n_steps):
                y   = self.model(
                    torch.from_numpy(inputs[start:end]).to(self.device)
                ).cpu().numpy()  # (B, n_neurons, n_out)
                all_preds[start:end, k] = y[:, :, k] # get the step-k prediction
                if k < n_steps - 1 and self.history > 0:
                    # fill lag-1 history for the next step
                    inputs[start:end, n_stim_ch:, ctx + 1 + k] = np.exp(
                        np.clip(y[:, :, k], -10.0, 10.0)
                    )
                    if N >= 2: 
                        import IPython; IPython.embed()  # DEBUG
                        print ("y:", y.shape, "inputs:", inputs.shape)
        return all_preds  # (N, n_steps, n_neurons)

    def _process_ar_preds(self, all_preds: np.ndarray, bin_ms: float) -> np.ndarray:
        """Aggregate AR predictions across steps into a single (N, n_neurons) vector.
        Args:
            all_preds: (N, n_steps, n_neurons) — log counts/bin
            bin_ms: duration of one bin in milliseconds
         Returns:
            (N, n_neurons) — firing rates in Hz
         """
        # encoder model output: log counts/bin
        # need to exponentiate to get counts/bin, multiply by 1000/bin_ms to get Hz, then do PCA projection and distance calculation in Hz space.

        if self.ar_aggregation == "most_recent": 
            return np.exp(all_preds[:, -1, :])  / bin_ms * 1000.0  # use the most recent step's prediction
        else:  # "average"
            # 1. Log-sum-exp across the time axis (axis 1)
            # 2. Subtract log(N) to get the log of the mean counts
            log_mean_counts = logsumexp(all_preds, axis=1) - np.log(all_preds.shape[1]) # for stability
            return np.exp(log_mean_counts) / bin_ms * 1000.0  # convert to Hz

    def _update_stim_buf_col(self, delivered: Optional[Dict]) -> np.ndarray:
        """Summarise the just-delivered stim into a (n_stim_channels,) column."""
        col = np.zeros(self.n_stim_channels, dtype=np.float32)
        if delivered is not None and len(delivered.get("channels", [])) > 0:
            chs  = np.asarray(delivered["channels"],      dtype=int)
            amps = np.asarray(delivered["amplitudes_uA"], dtype=float)
            valid = (chs >= 0) & (chs < self.n_stim_channels)
            np.add.at(col, chs[valid], amps[valid])
        return col

    # ------------------------------------------------------------------
    # Repertoire population
    # ------------------------------------------------------------------

    def fit_stim_repertoire(
        self,
        model_inputs,           # unused — kept for API compatibility
        stim_patterns: List[Dict],
    ):
        """Store stimulation patterns from training data as the repertoire.

        If n_relevant_output_bins < n_input_bins (i.e. the closed-loop interval
        is shorter than the training horizon), each training pattern is split
        into sub-patterns of n_relevant_output_bins length.  Times within each
        chunk are re-zeroed to the start of that chunk.  Duplicate patterns
        (same channels/bins/amps tuple) are removed.

        model_inputs is accepted but ignored; predictions are computed live
        from the current neural state at call time.
        """
        n_deliver = self.n_relevant_output_bins if self.n_relevant_output_bins is not None else self.n_input_bins
        window_ms = n_deliver * self.bin_ms

        # Collect unique amplitudes and max event count for random exploration.
        all_amps: set = set()
        max_events: int = 0
        for p in stim_patterns:
            all_amps.update(float(a) for a in p.get("amplitudes_uA", []))
            max_events = max(max_events, len(p.get("channels", [])))
        self._seen_amplitudes = np.array(sorted(all_amps), dtype=np.float32)
        self._max_active_bins = max(1, max_events)

        if n_deliver >= self.n_input_bins:
            # No splitting needed — use patterns as-is.
            self._entries = list(stim_patterns[: self.max_len])
        else:
            n_chunks = self.n_input_bins // n_deliver  # integer number of non-overlapping windows
            seen_keys: set = set()
            expanded: List[Dict] = []
            for p in stim_patterns:
                chs  = np.asarray(p.get("channels",      []), dtype=int)
                ts   = np.asarray(p.get("times_ms",      []), dtype=float)
                amps = np.asarray(p.get("amplitudes_uA", []), dtype=float)
                for chunk in range(n_chunks):
                    t_start = chunk * window_ms
                    t_end   = t_start + window_ms
                    mask = (ts >= t_start) & (ts < t_end)
                    if not np.any(mask):
                        continue
                    new_chs  = chs[mask]
                    new_ts   = ts[mask] - t_start   # re-zero to chunk start
                    new_amps = amps[mask]
                    # Deduplicate via a hashable key: sorted (ch, bin, amp) tuples
                    bins_i = np.floor(new_ts / self.bin_ms).astype(int)
                    key = tuple(sorted(zip(new_chs.tolist(), bins_i.tolist(),
                                          (new_amps * 100).round().astype(int).tolist())))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    expanded.append({
                        "channels":      new_chs.tolist(),
                        "times_ms":      new_ts.tolist(),
                        "amplitudes_uA": new_amps.tolist(),
                    })
                    if len(expanded) >= self.max_len:
                        break
                if len(expanded) >= self.max_len:
                    break
            self._entries = expanded

        print(f"fit_stim_repertoire: stored {len(self._entries)} unique patterns "
              f"(n_deliver_bins={n_deliver}, seen amps={self._seen_amplitudes.tolist()}, "
              f"max_active_bins={self._max_active_bins}).")

    def _random_stim_pattern(self) -> Optional[Dict]:
        """Sample a random single-channel pattern for exploration.

        Channel is drawn uniformly. Amplitude is drawn uniformly from the set
        seen in training. Number of active bins is drawn uniformly from
        [1, _max_active_bins]. Bin indices are drawn without replacement.
        """
        if len(self._seen_amplitudes) == 0:
            return None
        channel       = int(np.random.randint(0, self.n_stim_channels))
        n_active      = int(np.random.randint(1, self._max_active_bins + 1))
        # Restrict to bins that fall within the deliverable closed-loop window.
        n_deliverable = self.n_relevant_output_bins or self.n_input_bins
        n_active      = min(n_active, n_deliverable)
        bins          = np.sort(np.random.choice(n_deliverable, size=n_active, replace=False))
        amps      = np.random.choice(self._seen_amplitudes, size=n_active)
        times_ms  = bins.astype(float) * self.bin_ms
        return {
            "channels":      [channel] * n_active,
            "times_ms":      times_ms.tolist(),
            "amplitudes_uA": amps.tolist(),
        }

    # ------------------------------------------------------------------
    # Greedy selection
    # ------------------------------------------------------------------

    def get_stim_pattern(self) -> Optional[Dict]:
        """
        For every pattern in the repertoire, build a model input using the
        current rolling state, run the encoder, and return the pattern whose
        predicted mean log-rate is closest (L2) to self.target — but only if
        that pattern improves on the no-stimulation baseline prediction.
        Returns None if the repertoire is empty or if no candidate reduces the
        distance to the target compared to delivering no stimulation.
        """
        if not self._entries:
            return None

        # Random exploration: bypass the model with probability random_stim_prob.
        # Still build and cache the model input so fine-tuning can pair it with
        # the observed response on the next interval.
        if self.random_stim_prob > 0.0 and np.random.random() < self.random_stim_prob:
            pattern = self._random_stim_pattern()
            if pattern is not None:
                stim_bins = self._stim_dict_to_bins(pattern)
                inp = self._build_model_input(stim_bins)[None].copy()  # (1, C, T)
                self._process_ar_preds(self._ar_forward(inp))  # fills history slots in-place
                self._last_selected_input = inp[0]
                self._last_was_random = True
                if len(self._entries) < self.max_len:
                    self._entries.append(pattern)
                return pattern

        # Compute no-stim baseline distance in 2D PC space via AR forward
        null_input = self._build_model_input(
            np.zeros((self.n_stim_channels, self.n_input_bins), dtype=np.float32)
        )[None].copy()                                         # (1, C, T)
        null_pred    = self._process_ar_preds(self._ar_forward(null_input))[0]  # (n_neurons,) log counts/bin
        null_pc      = (null_pred_hz - self.pca_mean) @ self.pca_components.T
        no_stim_dist = float(np.linalg.norm(null_pc - self._target_pc))

        # Build all candidate inputs and run AR forward (modifies inputs in-place)
        inputs = np.stack(
            [self._build_model_input(self._stim_dict_to_bins(p)) for p in self._entries]
        )  # (N, C, T)
        final_preds    = self._process_ar_preds(np.exp(self._ar_forward(inputs)))  # (N, n_neurons) log counts/bin

        # Score in 2D PC space
        preds_pc = (final_preds_hz - self.pca_mean[None, :]) @ self.pca_components.T  # (N, K)
        dists    = np.linalg.norm(preds_pc - self._target_pc[None, :], axis=1)     # (N,)
        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        # Only stimulate if the best candidate actually improves on doing nothing
        if best_dist >= no_stim_dist:
            self._last_selected_input = None
            self._last_was_random = False
            return None

        # inputs[best_idx] has its history slots filled up to step n_steps-2 —
        # the AR-refined input appropriate for fine-tuning.
        self._last_selected_input = inputs[best_idx]
        self._last_was_random = False
        return self._entries[best_idx]

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
        1. Advance the rolling spike/stim buffers with observations from the
           interval that just finished.
        2. Run the greedy repertoire search with the updated state.
        3. Return the chosen stim pattern (or None if repertoire is empty).
        """
        # Update rolling spike buffer
        spike_counts = np.array(
            [len(s) for s in new_spikes_by_neuron], dtype=np.float32
        )
        self.spike_buf = np.roll(self.spike_buf, -1, axis=1)
        self.spike_buf[:, -1] = spike_counts

        # Update rolling stim buffer with what was just delivered
        stim_col = self._update_stim_buf_col(delivered_pattern)
        self.stim_buf = np.roll(self.stim_buf, -1, axis=1)
        self.stim_buf[:, -1] = stim_col

        return self.get_stim_pattern()


class NoEncoderController:
    """A dummy controller that iteratively stimulates each pattern in the repertoire without using the encoding model at all.
    Useful for ablation and debugging.

    Repertoire: one single-pulse pattern per electrode (pulse at time 0 ms,
    i.e. the first bin) for each of ``n_stim_channels`` electrodes, plus one
    no-stimulation slot (None), for a total of ``n_stim_channels + 1`` entries.
    Patterns are delivered in round-robin order.
    """

    def __init__(
        self,
        n_stim_channels: int = 32,
        stim_amplitude_uA: float = 2.0,
    ):
        self.n_stim_channels = n_stim_channels
        self.stim_amplitude_uA = stim_amplitude_uA

        # Build repertoire: one single-pulse per electrode + None for no-stim
        self._repertoire: List[Optional[Dict]] = [
            {
                "channels":      [ch],
                "times_ms":      [0.0],
                "amplitudes_uA": [float(stim_amplitude_uA)],
            }
            for ch in range(n_stim_channels)
        ] + [None]  # no-stimulation slot

        self._index: int = 0

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
        pattern = self._repertoire[self._index % len(self._repertoire)]
        self._index += 1
        return pattern
    