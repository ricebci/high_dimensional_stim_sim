import numpy as np
import os
import nest
from system import SystemNESTSim
import helpers
import pickle

# ── Spontaneous activity baseline: run simulation with NO stimulation ──────

# Match the same parameters as the closed-loop experiments
data_prefix = "spontaneous_baseline_noduration"
spontaneous_duration_ms = 100000.0  # 10 seconds of spontaneous activity
bin_ms = 5.0  # bin resolution (must match the data we'll compare against)
closed_loop_interval_ms = 500.0  # match the interval from the original experiment
bins_per_interval = int(np.ceil(closed_loop_interval_ms / bin_ms))

print(f"Running {spontaneous_duration_ms:.0f} ms of spontaneous activity (no stimulation)...")
print(f"bin_ms={bin_ms}, closed_loop_interval_ms={closed_loop_interval_ms}, bins_per_interval={bins_per_interval}")

# Initialize NEST network with same parameters as the closed-loop experiments
sim = SystemNESTSim(
    output_name=data_prefix,
    fast_mode=False,
    fast_sim_resolution_ms=None,
)

print(f"NEST network initialized. Data path: {sim.sim_dict['data_path']}")

# Run simulation with NO stimulation — just spontaneous activity
n_intervals = int(np.ceil(spontaneous_duration_ms / closed_loop_interval_ms))
n_neurons = sim.network.n_neurons
spike_counts_matrix = []
spike_bins_list = []
history = []

print(f"Running {n_intervals} intervals of {closed_loop_interval_ms:.0f} ms each (spontaneous only)...")

# Get initial spike counts for detecting new spikes
spike_trains = sim.network.get_spike_train_list()
prev_spike_counts = np.array([len(train) for train in spike_trains], dtype=int)
loop_start_bio_ms = sim.get_current_biological_time_ms()

for interval_index in range(n_intervals):
    interval_start_ms = interval_index * closed_loop_interval_ms
    interval_duration_ms = min(
        closed_loop_interval_ms,
        spontaneous_duration_ms - interval_start_ms,
    )
    interval_end_ms = interval_start_ms + interval_duration_ms
    
    # Simulate with NO stimulation
    nest.Simulate(float(interval_duration_ms))
    
    # Extract new spikes since last interval for each neuron
    spike_trains = sim.network.get_spike_train_list()
    new_spikes_by_neuron = []
    for neuron_index in range(n_neurons):
        train = np.asarray(spike_trains[neuron_index])
        new_spikes_abs = train[prev_spike_counts[neuron_index]:]
        prev_spike_counts[neuron_index] = len(train)
        new_spikes_rel = new_spikes_abs - loop_start_bio_ms
        new_spikes_by_neuron.append(new_spikes_rel)
    
    # Store spike counts per neuron for this interval
    spike_counts = np.array(
        [len(s) for s in new_spikes_by_neuron], dtype=np.float32
    )
    spike_counts_matrix.append(spike_counts)
    
    # Bin spikes into fine-grained bins within this interval
    interval_bins = np.zeros((bins_per_interval, n_neurons), dtype=np.float32)
    for neuron_index, spikes_rel in enumerate(new_spikes_by_neuron):
        if len(spikes_rel) == 0:
            continue
        t_within = np.asarray(spikes_rel) - interval_start_ms
        bin_indices = np.floor(t_within / bin_ms).astype(int)
        valid = (bin_indices >= 0) & (bin_indices < bins_per_interval)
        np.add.at(interval_bins[:, neuron_index], bin_indices[valid], 1)
    spike_bins_list.append(interval_bins)
    
    # Log this interval
    history.append({
        "interval_index": interval_index,
        "interval_start_ms": interval_start_ms,
        "interval_end_ms": interval_end_ms,
        "n_new_spikes": int(spike_counts.sum()),
        "delivered_pattern": None,  # No stimulation
    })
    
    if (interval_index + 1) % 5 == 0 or interval_index == 0:
        print(f"  Interval {interval_index + 1}/{n_intervals}: "
              f"t={interval_end_ms:.0f} ms, spikes={history[-1]['n_new_spikes']}")

# Save spike counts and spike bins to disk (same format as closed-loop experiments)
data_path = sim.sim_dict["data_path"]
counts_path = os.path.join(data_path, f"{data_prefix}_spike_counts.pkl")
bins_path = os.path.join(data_path, f"{data_prefix}_spike_bins.npz")
history_path = os.path.join(data_path, f"{data_prefix}_history.pkl")

with open(counts_path, "wb") as f:
    pickle.dump(np.stack(spike_counts_matrix, axis=0), f)

np.savez_compressed(
    bins_path,
    spike_bins=np.stack(spike_bins_list, axis=0).astype(np.int16),
    bin_ms=bin_ms,
)

with open(history_path, "wb") as f:
    pickle.dump(history, f)

