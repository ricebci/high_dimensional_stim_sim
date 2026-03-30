import os
import pickle
import shutil
import re

import matplotlib.pyplot as plt
import nest
import numpy as np
from sklearn.decomposition import PCA

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
from network_cortcol import Network

volume_v_min = 200  # um
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
# stim configurations
sigma_e_um = 2.76e-7
conductivity_constant = 10
STIM_CHANNELS = np.arange(32)
STIM_AMPLITUDES = [1.5]  # uA
STIM_POISSON_RATE_HZ = 8
RANDOM_STIM = True  # else, deterministic

PRESIM_TIME_MS = sim_dict["t_presim"]
WINDOW_MS = 500
OVERLAP_MS = 400
RASTER_PLOT_TIME_MS = 500
N_GROUPS_LIST = [1, 2, 4, 8, 16, 32]  # number of stim electrode groups

def amp_decay_func(amp_uA, dist_um):
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )
#%% Fig 1b

output_name = 'data_10Hz_0.05scale_2uA_60s'
base_path = os.path.join(
    os.getcwd(), "data", output_name
)
save_path = 'figures'


SIM_TIME_MS = int(re.search(r'(\d+)(?=s$)', output_name).group(1)) * 1000
SCALING = float(re.search(r'([\d\.]+)(?=scale)', output_name).group(1))
RASTER_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + RASTER_PLOT_TIME_MS]
FIRING_RATE_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + SIM_TIME_MS]
sim_dict['t_sim'] = SIM_TIME_MS
net_dict['N_scaling'] = SCALING
net_dict['K_scaling'] = SCALING

# Baseline
nest.ResetKernel()

sim_dict["data_path"] = os.path.join(base_path, "data_baseline")
network = Network(sim_dict, net_dict, stim_dict)

baseline_spike_trains = network.get_spike_train_list()
baseline_spike_rates = helpers.compute_spike_rates(
    baseline_spike_trains,
    SIM_TIME_MS,
    WINDOW_MS,
    OVERLAP_MS,
    presim_time_ms=PRESIM_TIME_MS,
)

fig, ax = plt.subplots(1, 1, figsize=(3, 2.25))
helpers.plot_raster(
    os.path.join(sim_dict["data_path"], "spike_recorder"),
    "spike_recorder",
    RASTER_INTERVAL[0],
    RASTER_INTERVAL[1],
    SCALING,
    title='Baseline activity',
    ax=ax
)
begin, end = 1000, 1500
xticks = np.linspace(begin, end, num=6)
ax.set_xticks(xticks)
# Relabel them from 0 to 500
ax.set_xticklabels([f"{int(tick - begin)}" for tick in xticks])
ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "baseline_raster.svg"))

#%% Plot stim condition raster plots (Fig 1b continued) 
stim_spike_rates_list = []
stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}
NN_GROUPS_LIST = [1, 2, 4, 8, 16, 32]  # number of stim electrode groups

for n_groups in N_GROUPS_LIST:
    sim_dict["data_path"] = os.path.join(base_path, f"data_randstim_{n_groups}groups/")
    pkl_path_stim_pulses = os.path.join(
        sim_dict["data_path"], f"{n_groups}groups_stim_pulses.pkl"
    )
    with open(pkl_path_stim_pulses, "rb") as f:
        stim_onset_times_by_ch = pickle.load(f)
    nest.ResetKernel()
    network = Network(sim_dict, net_dict, stim_dict)
    electrodes = StimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)
    electrodes.stim_onset_times_by_ch = stim_onset_times_by_ch
    
    stim_evoked_spike_trains = network.get_spike_train_list()
    stim_evoked_spike_rates = helpers.compute_spike_rates(
        stim_evoked_spike_trains,
        SIM_TIME_MS,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(3, 4))
    plot_title = f"{n_groups} electrode group{'s' if n_groups > 1 else ''}"
    helpers.plot_raster(
        os.path.join(sim_dict["data_path"], "spike_recorder"),
        "spike_recorder",
        RASTER_INTERVAL[0],
        RASTER_INTERVAL[1],
        net_dict["N_scaling"],
        title=plot_title,
        ax=axes[0],
    )
    
    
    begin, end = 1000, 1500
    xticks = np.linspace(begin, end, num=6)
    axes[0].set_xticks(xticks)
    # Relabel them from 0 to 500
    axes[0].set_xticklabels([f"{int(tick - begin)}" for tick in xticks])
    
    electrodes.plot_stim_raster(ax=axes[1], time_range_ms=RASTER_INTERVAL)
    if n_groups != 1:
        axes[1].set_ylabel('')
    xticks = np.linspace(begin, end, num=6)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([f"{int(tick - begin)}" for tick in xticks])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"stim_{n_groups}_groups_raster.svg"))    
    stim_spike_rates_list.append(stim_evoked_spike_rates)

#%% Plot PCA overlap (Fig 1c)

variance_threshold = 0.85
pca = PCA(n_components=3)
# Fit PCA on baseline spike rates and transform the data
pca.fit(baseline_spike_rates)  
baseline_pca = pca.transform(baseline_spike_rates) 
baseline_num_components = helpers.get_dimensionality(
    baseline_spike_rates, variance_threshold
)

stim_projected_list = []
stim_num_components_list = []

for i, stim_spike_rates in enumerate(stim_spike_rates_list):
    # Project stimulus spike rates onto the PCA components derived from baseline
    stim_projected = pca.transform(stim_spike_rates)
    stim_projected_list.append(stim_projected)

    num_components = helpers.get_dimensionality(stim_spike_rates, variance_threshold)
    stim_num_components_list.append(num_components)

views = [(15, -135)]

helpers.plot_projections(
    baseline_pca,
    stim_projected_list,
    sim_dict["data_path"],
    "pca_projection",
    views=views,
    xlim=[-0.04, 0.03],
    ylim=[0.05, -0.03],
    zlim=[-0.025, 0.03],
)

overlap_list, _, _ = helpers.compute_all_overlaps(baseline_pca, stim_projected_list)

stim_amps_str = STIM_AMPLITUDES[0] if len(STIM_AMPLITUDES) > 0 else STIM_AMPLITUDES
plt.suptitle("")

plt.savefig(os.path.join(save_path, "pca_projection_ellipsoids.svg"))
plt.close()
# %% Fig 1d
stim_channels = [1, 2, 4, 8, 16, 32]  # Number of stimulation channels

plt.figure(figsize=(4, 3.5))
plt.plot(np.arange(6), overlap_list, marker="o")
plt.xticks(np.arange(6), labels=stim_channels, fontsize=10)
plt.xlabel("Number of Electrode Groups", fontsize=10)
plt.ylabel("Volume Overlap\n(Jaccard Index)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "overlap.svg"))
#%% Fig 1e
plt.figure(figsize=(4, 3.5))
plt.plot(
    np.arange(6), stim_num_components_list, marker="o", label="Evoked Dimensionality"
)
plt.axhline(
    baseline_num_components, color="r", linestyle="--", label="Baseline Dimensionality"
)  

plt.yticks(fontsize=10)
plt.xticks(np.arange(6), labels=stim_channels, fontsize=10)
plt.xlabel("Number of Electrode Groups", fontsize=10)
plt.ylabel("Dimensionality", fontsize=10)
plt.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "dimensionality.svg"))
