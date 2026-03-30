import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import helpers
from corcol_params.sim_params import sim_dict

# %%
n_groups = 2
currents = [0.5, 1, 1.5, 2, 2.5]
pop_names = ["L2/3E", "L2/3I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
color_list = np.tile(["#595289", "#af143c"], 4)
example_ch = 16

for current in currents:
    data_path = f"/home/robin/stim_sim/outputs/data_8Hz_k10_scale005_[{current}]uA/data_randstim_{n_groups}groups/"
    """
    Outputs grouped by layers
    Sender is neuron index
    """
    begin = sim_dict["t_presim"]
    end = sim_dict["t_sim"] + sim_dict["t_presim"]

    sd_names, node_ids, data = helpers.load_spike_times(
        os.path.join(data_path, "spike_recorder"), "spike_recorder", begin, end
    )

    # Need to get stim times for specific channel
    pkl_path = os.path.join(sim_dict["data_path"], f"{n_groups}groups_stim_pulses.pkl")
    with open(pkl_path, "rb") as f:
        stim_pulse_data = pickle.load(f)

    # Look at responses to specific channel or channels that stimulate with this channel
    stim_pulses = stim_pulse_data[example_ch]

    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    win_ms = 50
    # Loop over each population and plot the histogram
    for pop_index in range(len(sd_names)):
        pop_data = data[pop_index]
        spikes = pop_data["time_ms"]
        spikes_in_windows = []

        # Extract spikes in the time window around each pulse
        for pulse in stim_pulses:
            psth_start = pulse - win_ms
            psth_end = pulse + win_ms
            spike_indices = np.where((spikes > psth_start) & (spikes < psth_end))[0]
            spikes_in_window = spikes[spike_indices] - psth_start
            spikes_in_windows.append(spikes_in_window)

        # Handle cases where there are no spikes in the window
        if len(spikes_in_windows) > 0:
            flattened_spikes = np.concatenate(spikes_in_windows)
        else:
            flattened_spikes = np.array([])

        # Plot the histogram for the current population
        ax = axes[pop_index]
        ax.hist(
            flattened_spikes - win_ms,
            bins=50,
            density=True,
            color=color_list[pop_index],
        )
        ax.set_xlabel("Time (ms) relative to pulse", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_title(f"{pop_names[pop_index]}", fontsize=8)

    fig.delaxes(axes[8])
    fig.suptitle(f"PSTH with {n_groups} groups at {current} uA")
    plt.tight_layout()

    output_filename = os.path.join(f"outputs/analysis/psth_{current}uA_{n_groups}.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
