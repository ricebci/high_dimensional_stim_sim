"""
Encoding models: electrical stimulation → neural spike rate prediction.

Adapted from closed_loop_stim/patterns_5k/models.py for the NEST cortical
column simulation.  Main differences:
- Stimulus inputs are continuous amplitudes (µA per channel per time bin),
  not categorical indices, so no embedding layer is needed.
- Raw data comes from NEST spike recorders (spike times in ms) and stim
  event dicts ({channels, times_ms, amplitudes_uA}), so a dedicated
  dataset class bins them into (channels × time) tensors.

Input:  (batch, n_stim_channels, n_input_bins)   — binned stim amplitudes
Output: (batch, n_neurons, n_output_bins)         — predicted log firing rates

The model is trained with Poisson negative log-likelihood so that
exp(output) gives predicted spike counts per bin.
"""

import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# Dataset: bin NEST stim events + spike trains into aligned tensors
# =====================================================================

class NESTStimSpikeDataset(Dataset):
    """Dataset over binned stimulation and spike responses.

    Supports two data-input modes:

    - **Legacy (numpy)**: pass ``stim_events`` dict + ``spike_trains``
      list.  Binning uses vectorised numpy (``np.add.at`` /
      ``np.bincount``).
    - **Polars (fast)**: pass ``stim_df`` and ``spike_df`` polars
      DataFrames.  Binning is done with polars group-by operations
      (Rust-parallel), which is significantly faster for large NEST
      datasets.

    And two sampling modes:

    - **Sliding window** (default): every valid position across the
      recording.
    - **Trial-based**: pass ``trials`` (list of dicts with
      ``"start_bin"``).  Use :meth:`with_trials` to create lightweight
      views that share the same binned arrays.

    Parameters
    ----------
    stim_events : dict, optional
        ``{"channels": array, "times_ms": array, "amplitudes_uA": array}``
    spike_trains : list[np.ndarray], optional
        Per-neuron spike-time arrays (ms).
    stim_df : polars.DataFrame, optional
        Columns: ``channels`` (int), ``times_ms`` (float),
        ``amplitudes_uA`` (float).
    spike_df : polars.DataFrame, optional
        Columns: ``sender`` (neuron ID), ``time`` (ms, absolute NEST
        time including presim).
    total_duration_ms : float
        Recording duration in ms *excluding* presim.
    bin_size_ms : float
    n_input_bins, n_output_bins : int
    n_stim_channels : int
    output_offset : int
    presim_ms : float
    history : int
        When >0, ``n_input_bins`` bins of preceding spike activity are
        concatenated as extra input channels (zero-padded at the start
        of the recording).
    trials : list[dict], optional
        Each dict must have a ``"start_bin"`` key.
    n_neurons : int, optional
        Override automatic neuron count (useful when some neurons are
        silent and absent from the data).
    """

    def __init__(
        self,
        stim_events: Optional[Dict[str, np.ndarray]] = None,
        spike_trains: Optional[list] = None,
        total_duration_ms: Optional[float] = None,
        bin_size_ms: float = 1.0,
        n_input_bins: int = 60,
        n_output_bins: int = 10,
        n_stim_channels: int = 32,
        output_offset: int = 0,
        presim_ms: float = 0.0,
        history: int = 0,
        stim_df=None,
        spike_df=None,
        trials: Optional[list] = None,
        n_neurons: Optional[int] = None,
        kernel_sizes: Optional[List[int]] = None,
        mode: str = "sliding",
    ):
        super().__init__()
        self.bin_size_ms = bin_size_ms
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.n_stim_channels = n_stim_channels
        self.output_offset = output_offset
        self.history = history
        self.mode = mode

        # Context bins for valid convolution
        if kernel_sizes is not None:
            self.n_initial_state_bins = sum(k - 1 for k in kernel_sizes)
        else:
            self.n_initial_state_bins = 0
        self.kernel_sizes = kernel_sizes

        n_total_bins = int(np.floor(total_duration_ms / bin_size_ms))
        self.n_total_bins = n_total_bins

        # ---- Bin stimulation ------------------------------------------------
        if stim_df is not None:
            self._bin_stim_from_polars(stim_df, n_total_bins, bin_size_ms,
                                       n_stim_channels)
        elif stim_events is not None:
            self._bin_stim_from_arrays(stim_events, n_total_bins, bin_size_ms,
                                       n_stim_channels)
        else:
            raise ValueError("Provide either stim_df or stim_events")

        # ---- Bin spikes -----------------------------------------------------
        if spike_df is not None:
            self._bin_spikes_from_polars(spike_df, n_total_bins, bin_size_ms,
                                         presim_ms, n_neurons)
        elif spike_trains is not None:
            self._bin_spikes_from_arrays(spike_trains, n_total_bins,
                                         bin_size_ms, presim_ms)
            if n_neurons is not None:
                self.n_neurons = n_neurons
        else:
            raise ValueError("Provide either spike_df or spike_trains")

        # ---- Store stim_df for stim_locked sample enumeration ---------------
        self._stim_df = stim_df
        self._stim_events = stim_events

        # ---- Enumerate sample positions -------------------------------------
        if trials is not None:
            self.sample_starts = [t["start_bin"] for t in trials]
        elif mode == "stim_locked":
            self.sample_starts = self._build_stim_locked_starts()
        else:
            # sliding window: every valid position
            context = self.n_initial_state_bins
            history_pad = max(history, 0)
            min_start = max(context, history_pad)
            max_start = n_total_bins - max(
                n_input_bins, output_offset + n_output_bins
            )
            self.sample_starts = list(range(min_start, max_start + 1))

    def _build_stim_locked_starts(self):
        """Build sample_starts so t=0 of each sample is a stim event.

        Each stim event becomes one sample whose output window starts at
        the stim time bin.  Stim events that are too close to the edges
        (not enough context or output room) are skipped.
        """
        if self._stim_df is not None:
            import polars as pl
            times = self._stim_df["times_ms"].to_numpy().astype(float)
        elif self._stim_events is not None:
            times = np.asarray(self._stim_events["times_ms"], dtype=float)
        else:
            return []

        starts = []
        context = self.n_initial_state_bins
        history_pad = max(self.history, 0)
        min_start = max(context, history_pad)
        max_end = self.n_total_bins
        for t_ms in times:
            t_bin = int(np.floor(t_ms / self.bin_size_ms))
            if t_bin < min_start:
                continue
            out_end = t_bin + self.output_offset + self.n_output_bins
            if out_end > max_end:
                continue
            starts.append(t_bin)
        return sorted(set(starts))

    # -----------------------------------------------------------------
    # Polars binning (fast path)
    # -----------------------------------------------------------------
    def _bin_stim_from_polars(self, stim_df, n_total_bins, bin_size_ms,
                              n_stim_channels):
        import polars as pl
        binned = (
            stim_df
            .select(["channels", "times_ms", "amplitudes_uA"])
            .with_columns(
                (pl.col("times_ms").cast(pl.Float64) / bin_size_ms)
                .floor().cast(pl.Int64).alias("bin")
            )
            .filter((pl.col("bin") >= 0) & (pl.col("bin") < n_total_bins))
            .group_by(["channels", "bin"])
            .agg(pl.col("amplitudes_uA").sum().alias("amp"))
        )
        self.stim_binned = np.zeros(
            (n_stim_channels, n_total_bins), dtype=np.float32
        )
        if len(binned) > 0:
            ch = binned["channels"].to_numpy().astype(np.intp)
            bidx = binned["bin"].to_numpy().astype(np.intp)
            amps = binned["amp"].to_numpy().astype(np.float32)
            self.stim_binned[ch, bidx] = amps

    def _bin_spikes_from_polars(self, spike_df, n_total_bins, bin_size_ms,
                                presim_ms, n_neurons_override):
        import polars as pl
        unique_senders = spike_df.select("sender").unique().sort("sender")
        inferred_n = len(unique_senders)
        self.n_neurons = (n_neurons_override
                          if n_neurons_override is not None
                          else inferred_n)
        sender_lut = unique_senders.with_row_index("neuron_idx")
        binned = (
            spike_df
            .select(["sender", "time"])
            .join(sender_lut, on="sender")
            .with_columns(
                ((pl.col("time").cast(pl.Float64) - presim_ms) / bin_size_ms)
                .floor().cast(pl.Int64).alias("bin")
            )
            .filter((pl.col("bin") >= 0) & (pl.col("bin") < n_total_bins))
            .group_by(["neuron_idx", "bin"])
            .len()
        )
        self.spikes_binned = np.zeros(
            (self.n_neurons, n_total_bins), dtype=np.float32
        )
        if len(binned) > 0:
            nidx = binned["neuron_idx"].to_numpy().astype(np.intp)
            bidx = binned["bin"].to_numpy().astype(np.intp)
            counts = binned["len"].to_numpy().astype(np.float32)
            self.spikes_binned[nidx, bidx] = counts

    # -----------------------------------------------------------------
    # Legacy numpy binning
    # -----------------------------------------------------------------
    def _bin_stim_from_arrays(self, stim_events, n_total_bins, bin_size_ms,
                              n_stim_channels):
        self.stim_binned = np.zeros(
            (n_stim_channels, n_total_bins), dtype=np.float32
        )
        channels = np.asarray(stim_events["channels"], dtype=int)
        times = np.asarray(stim_events["times_ms"], dtype=float)
        amps = np.asarray(stim_events["amplitudes_uA"], dtype=float)
        bin_indices = np.floor(times / bin_size_ms).astype(int)
        valid = (bin_indices >= 0) & (bin_indices < n_total_bins)
        np.add.at(
            self.stim_binned,
            (channels[valid], bin_indices[valid]),
            amps[valid],
        )

    def _bin_spikes_from_arrays(self, spike_trains, n_total_bins, bin_size_ms,
                                presim_ms):
        self.n_neurons = len(spike_trains)
        self.spikes_binned = np.zeros(
            (self.n_neurons, n_total_bins), dtype=np.float32
        )
        for neuron_idx, train in enumerate(spike_trains):
            train = np.asarray(train, dtype=float) - presim_ms
            spike_bins = np.floor(train / bin_size_ms).astype(int)
            spike_bins = spike_bins[
                (spike_bins >= 0) & (spike_bins < n_total_bins)
            ]
            counts = np.bincount(spike_bins, minlength=n_total_bins)
            self.spikes_binned[neuron_idx] = counts[:n_total_bins]

    # -----------------------------------------------------------------
    # Trial-based views
    # -----------------------------------------------------------------
    def with_trials(self, trials):
        """Return a lightweight view sampling from specific trial positions.

        The underlying binned arrays are shared (no data copy).

        Parameters
        ----------
        trials : list[dict]
            Each dict must have a ``"start_bin"`` key.
        """
        view = object.__new__(NESTStimSpikeDataset)
        view.stim_binned = self.stim_binned
        view.spikes_binned = self.spikes_binned
        view.bin_size_ms = self.bin_size_ms
        view.n_input_bins = self.n_input_bins
        view.n_output_bins = self.n_output_bins
        view.n_stim_channels = self.n_stim_channels
        view.output_offset = self.output_offset
        view.history = self.history
        view.n_neurons = self.n_neurons
        view.n_total_bins = self.n_total_bins
        view.n_initial_state_bins = self.n_initial_state_bins
        view.kernel_sizes = self.kernel_sizes
        view.mode = self.mode
        view._stim_df = self._stim_df
        view._stim_events = self._stim_events
        view.sample_starts = [t["start_bin"] for t in trials]
        return view

    def __len__(self):
        return len(self.sample_starts)

    def __getitem__(self, idx):
        t = self.sample_starts[idx]
        ctx = self.n_initial_state_bins  # bins of real context for valid conv
        history_lag = max(self.history, 0)

        # Width of the stim/history time axis sent to the model.
        # With valid conv the model needs ctx extra bins on the left that
        # the convolutions will "burn through" to produce n_input_bins outputs.
        x_width = ctx + self.n_input_bins

        # --- Stim input: prepend ctx real bins of stim context ---------------
        stim_start = t - ctx
        if stim_start >= 0:
            x_stim = self.stim_binned[:, stim_start : stim_start + x_width].copy()
        else:
            # Near start of recording — zero-pad what we can't look back for
            available = t
            x_stim = np.zeros((self.n_stim_channels, x_width), dtype=np.float32)
            x_stim[:, ctx - available :] = self.stim_binned[:, 0 : available + self.n_input_bins]

        # --- Spike output window (unchanged) ---------------------------------
        out_start = t + self.output_offset
        y = self.spikes_binned[
            :, out_start : out_start + self.n_output_bins
        ].copy()

        # --- Teacher-forced spike history (lagged by 1 bin) ------------------
        if self.history > 0:
            # History spans the same x_width as stim, but shifted 1 bin earlier
            h_start = stim_start - 1
            h_end = h_start + x_width
            if h_start >= 0:
                h = self.spikes_binned[:, h_start : h_end].copy()
            else:
                h = np.zeros((self.n_neurons, x_width), dtype=np.float32)
                valid_from = -h_start
                h[:, valid_from:] = self.spikes_binned[:, 0 : h_end]
            x = np.concatenate([x_stim, h], axis=0).astype(np.float32)
        else:
            x = x_stim.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


# =====================================================================
# Convenience loader from saved experiment artifacts
# =====================================================================

def load_dataset_from_experiment(
    data_path: str,
    history_path: Optional[str] = None,
    spike_rates_path: Optional[str] = None,
    bin_size_ms: float = 1.0,
    n_input_bins: int = 60,
    n_output_bins: int = 10,
    n_stim_channels: int = 32,
    output_offset: int = 0,
    presim_ms: float = 1000.0,
    history: int = 0,
) -> NESTStimSpikeDataset:
    """Build a dataset from closed-loop experiment output files.

    Reconstructs stimulation events from the history pickle and loads
    raw spike times from ``spike_recorder-*.dat`` files.
    """
    # --- Load history to reconstruct stim events ---
    if history_path is None:
        candidates = glob.glob(os.path.join(data_path, "*_history.pkl"))
        if candidates:
            history_path = candidates[0]
    if history_path is None:
        raise FileNotFoundError("No history pickle found; cannot reconstruct stim events")

    with open(history_path, "rb") as f:
        cl_history = pickle.load(f)

    all_channels = []
    all_times = []
    all_amps = []
    total_duration_ms = 0.0
    for entry in cl_history:
        total_duration_ms = max(total_duration_ms, entry["interval_end_ms"])
        dp = entry.get("delivered_pattern")
        if dp is not None:
            offset = entry["interval_start_ms"]
            chs = np.asarray(dp["channels"], dtype=int)
            ts = np.asarray(dp["times_ms"], dtype=float) + offset
            amps = np.asarray(dp["amplitudes_uA"], dtype=float)
            all_channels.append(chs)
            all_times.append(ts)
            all_amps.append(amps)

    if all_channels:
        stim_events = {
            "channels": np.concatenate(all_channels),
            "times_ms": np.concatenate(all_times),
            "amplitudes_uA": np.concatenate(all_amps),
        }
    else:
        stim_events = {
            "channels": np.array([], dtype=int),
            "times_ms": np.array([], dtype=float),
            "amplitudes_uA": np.array([], dtype=float),
        }

    # --- Load raw spike trains from recorder .dat files ---
    spike_trains = _load_spike_trains_from_dat(data_path, presim_ms)

    return NESTStimSpikeDataset(
        stim_events=stim_events,
        spike_trains=spike_trains,
        total_duration_ms=total_duration_ms,
        bin_size_ms=bin_size_ms,
        n_input_bins=n_input_bins,
        n_output_bins=n_output_bins,
        n_stim_channels=n_stim_channels,
        output_offset=output_offset,
        presim_ms=presim_ms,
        history=history,
    )


def _load_spike_trains_from_dat(
    data_path: str, presim_ms: float
) -> list:
    """Load spike recorder .dat files into a list of per-neuron spike arrays."""
    spike_files = sorted(glob.glob(os.path.join(data_path, "spike_recorder-*.dat")))
    if not spike_files:
        raise FileNotFoundError(f"No spike_recorder files in {data_path}")

    node_ids = np.loadtxt(
        os.path.join(data_path, "population_nodeids.dat"), dtype=int
    )
    if node_ids.ndim == 1:
        node_ids = node_ids[None, :]
    last_node_id = node_ids[-1, -1]
    first_node_id = node_ids[0, 0]
    n_neurons = last_node_id - first_node_id + 1

    trains: Dict[int, list] = {i: [] for i in range(n_neurons)}

    for fpath in spike_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    sender = int(float(parts[0]))
                    time_ms = float(parts[1])
                except ValueError:
                    continue
                idx = sender - first_node_id
                if 0 <= idx < n_neurons:
                    trains[idx].append(time_ms)

    return [np.asarray(trains[i]) for i in range(n_neurons)]


# =====================================================================
# Model: Causal CNN  (continuous stim input → log spike rates)
# =====================================================================

class StimEncodingCNN(nn.Module):
    """Causal 1-D CNN mapping binned stim amplitudes to log spike rates.

    Architecture mirrors ``SimpleCausalSpikeCNN`` from the reference
    codebase but replaces the categorical embedding with a linear
    projection, since stimulation amplitudes are continuous.

    Supports two convolution modes controlled by ``use_init_state``:

    - **use_init_state=True** (valid conv): no padding in the conv
      layers.  The dataset must prepend ``total_conv_reduction`` real
      context bins so the output length equals ``n_input_bins``.
    - **use_init_state=False** (causal padding, default): each conv
      layer left-pads with zeros so output length == input length.

    Parameters
    ----------
    n_stim_channels : int
        Number of electrode channels (32).
    n_neurons : int
        Number of neurons to predict.
    n_input_bins, n_output_bins : int
        Temporal lengths of input / output windows.
    proj_dim : int
        Dimension to project each channel's amplitude into (0 = skip,
        feed raw amplitudes).
    conv_channels : list[int]
        Number of filters per conv layer.
    kernel_sizes : list[int]
        Kernel width per conv layer.
    fc_dims : list[int]
        Hidden-layer widths for the per-timestep FC head.
    dropout : float
        Dropout rate.
    use_batch_norm : bool
        Whether to add BatchNorm after each conv layer.
    history : int
        If > 0 the model expects ``n_stim_channels + n_neurons`` input
        channels (spike history concatenated after stim channels).
    use_init_state : bool
        If True, use valid convolution (no padding).  The dataset must
        prepend ``total_conv_reduction`` bins of real context.
    """

    def __init__(
        self,
        n_stim_channels: int = 32,
        n_neurons: int = 3906,
        n_input_bins: int = 60,
        n_output_bins: int = 10,
        proj_dim: int = 0,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        fc_dims: List[int] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        history: int = 0,
        use_init_state: bool = False,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 5]
        if fc_dims is None:
            fc_dims = [256]

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have equal length")

        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.history = history
        self.use_init_state = use_init_state

        # Optional per-channel linear projection
        raw_in = n_stim_channels + (n_neurons if history > 0 else 0)
        if proj_dim > 0:
            self.proj = nn.Linear(raw_in, proj_dim)
            in_channels = proj_dim
        else:
            self.proj = None
            in_channels = raw_in

        # Conv stack
        #   use_init_state=True  → valid conv (no padding; dataset prepends context)
        #   use_init_state=False → causal conv (left-only zero padding)
        self.kernel_sizes_list = list(kernel_sizes)
        layers: list = []
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            if use_init_state:
                layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=ks))
            else:
                layers.append(nn.ConstantPad1d((ks - 1, 0), 0))
                layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=ks))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # Per-timestep FC head
        fc_layers: list = []
        curr = conv_channels[-1]
        for fd in fc_dims:
            fc_layers.extend([nn.Linear(curr, fd), nn.ReLU(), nn.Dropout(dropout)])
            curr = fd
        fc_layers.append(nn.Linear(curr, n_neurons))
        self.fc = nn.Sequential(*fc_layers)

    @property
    def total_conv_reduction(self) -> int:
        """Time bins consumed by valid convolution (sum of kernel_size-1).

        When ``use_init_state=True``, the dataset must prepend this many
        bins of real context so that the output length equals
        ``n_input_bins``.
        """
        return sum(k - 1 for k in self.kernel_sizes_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)
            When use_init_state=False, T = n_input_bins.
            When use_init_state=True,  T = n_input_bins + total_conv_reduction.

        Returns
        -------
        (B, n_neurons, n_input_bins)  — log firing rates
        """
        if self.proj is not None:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        features = self.conv_stack(x)              # (B, conv_ch, T')
        features = features.transpose(1, 2)        # (B, T', conv_ch)
        y = self.fc(features)                      # (B, T', n_neurons)
        return y.transpose(1, 2)                   # (B, n_neurons, T')



# =====================================================================
# Factory
# =====================================================================

def get_model(model_type: str, **kwargs) -> nn.Module:
    """Create a model by name.

    Parameters
    ----------
    model_type : {'cnn', 'history_cnn'}
    """
    models = {
        "cnn": StimEncodingCNN,
        "history_cnn": HistoryStimEncodingCNN,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )
    return models[model_type](**kwargs)


# =====================================================================
# Loss
# =====================================================================

def poisson_nll_loss(
    pred_log_rates: torch.Tensor, target_counts: torch.Tensor
) -> torch.Tensor:
    """Poisson negative log-likelihood (mean-reduced).

    loss = mean( exp(log_rate) - target * log_rate )
    """
    return torch.mean(torch.exp(pred_log_rates) - target_counts * pred_log_rates)


# =====================================================================
# Correlation metric
# =====================================================================

def compute_correlation(
    pred_log_rates: torch.Tensor, target_counts: torch.Tensor
) -> float:
    """Mean per-neuron Pearson correlation between predicted rates and targets."""
    pred = torch.exp(pred_log_rates).detach().cpu().numpy()
    target = target_counts.detach().cpu().numpy()

    # (B, n_neurons, T) → per-neuron correlation over (B × T)
    n_neurons = pred.shape[1]
    pred_flat = pred.transpose(1, 0, 2).reshape(n_neurons, -1)
    target_flat = target.transpose(1, 0, 2).reshape(n_neurons, -1)

    corrs = []
    for i in range(n_neurons):
        p, t = pred_flat[i], target_flat[i]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        corrs.append(np.corrcoef(p, t)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


# =====================================================================
# Training utilities
# =====================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_history: bool = False,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()

        if use_history and isinstance(model, HistoryStimEncodingCNN):
            stim_input = bx[:, : model.n_stim_channels, :]
            pred = model(stim_input, spike_context=by, mode="teacher_forcing")
        else:
            pred = model(bx)

        # Align output length to target length
        T_out = min(pred.shape[2], by.shape[2])
        loss = poisson_nll_loss(pred[:, :, :T_out], by[:, :, :T_out])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_history: bool = False,
) -> dict:
    """Evaluate model. Returns dict with loss and correlation."""
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    n_batches = 0

    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)

        if use_history and isinstance(model, HistoryStimEncodingCNN):
            stim_input = bx[:, : model.n_stim_channels, :]
            pred = model(stim_input, spike_context=by, mode="teacher_forcing")
        else:
            pred = model(bx)

        T_out = min(pred.shape[2], by.shape[2])
        pred = pred[:, :, :T_out]
        by = by[:, :, :T_out]

        total_loss += poisson_nll_loss(pred, by).item()
        all_pred.append(pred.cpu())
        all_true.append(by.cpu())
        n_batches += 1

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "correlation": compute_correlation(all_pred, all_true),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = None,
    use_history: bool = False,
    verbose: bool = True,
) -> dict:
    """Full training loop with optional validation.

    Returns
    -------
    dict with 'train_losses', 'val_losses', 'val_correlations', 'best_state_dict'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_corrs = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        t_loss = train_epoch(model, train_loader, optimizer, device, use_history)
        train_losses.append(t_loss)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, use_history)
            val_losses.append(metrics["loss"])
            val_corrs.append(metrics["correlation"])
            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{n_epochs}  "
                    f"train_loss={t_loss:.4f}  "
                    f"val_loss={metrics['loss']:.4f}  "
                    f"val_corr={metrics['correlation']:.4f}"
                )
        else:
            if verbose:
                print(f"Epoch {epoch+1:3d}/{n_epochs}  train_loss={t_loss:.4f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_correlations": val_corrs,
        "best_state_dict": best_state,
    }
