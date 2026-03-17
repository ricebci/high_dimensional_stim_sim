# -*- coding: utf-8 -*-
#
# helpers.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""PyNEST Microcircuit: Helper Functions
-------------------------------------------

Helper functions for network construction, simulation and evaluation of the
microcircuit.

"""

import os
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

if "DISPLAY" not in os.environ:
    import matplotlib

    matplotlib.use("Agg")


def num_synapses_from_conn_probs(conn_probs, popsize1, popsize2):
    """Computes the total number of synapses between two populations from
    connection probabilities.

    Here it is irrelevant which population is source and which target.

    Parameters
    ----------
    conn_probs
        Matrix of connection probabilities.
    popsize1
        Size of first population.
    popsize2
        Size of second population.

    Returns
    -------
    num_synapses
        Matrix of synapse numbers.
    """
    prod = np.outer(popsize1, popsize2)
    num_synapses = np.log(1.0 - conn_probs) / np.log((prod - 1.0) / prod)
    return num_synapses


def postsynaptic_potential_to_current(C_m, tau_m, tau_syn):
    r"""Computes a factor to convert postsynaptic potentials to currents.

    The time course of the postsynaptic potential ``v`` is computed as
    :math: `v(t)=(i*h)(t)`
    with the exponential postsynaptic current
    :math:`i(t)=J\mathrm{e}^{-t/\tau_\mathrm{syn}}\Theta (t)`,
    the voltage impulse response
    :math:`h(t)=\frac{1}{\tau_\mathrm{m}}\mathrm{e}^{-t/\tau_\mathrm{m}}\Theta (t)`,
    and
    :math:`\Theta(t)=1` if :math:`t\geq 0` and zero otherwise.

    The ``PSP`` is considered as the maximum of ``v``, i.e., it is
    computed by setting the derivative of ``v(t)`` to zero.
    The expression for the time point at which ``v`` reaches its maximum
    can be found in Eq. 5 of [1]_.

    The amplitude of the postsynaptic current ``J`` corresponds to the
    synaptic weight ``PSC``.

    References
    ----------
    .. [1] Hanuschkin A, Kunkel S, Helias M, Morrison A and Diesmann M (2010)
           A general and efficient method for incorporating precise spike times
           in globally time-driven simulations.
           Front. Neuroinform. 4:113.
           DOI: `10.3389/fninf.2010.00113 <https://doi.org/10.3389/fninf.2010.00113>`__.

    Parameters
    ----------
    C_m
        Membrane capacitance (in pF).
    tau_m
        Membrane time constant (in ms).
    tau_syn
        Synaptic time constant (in ms).

    Returns
    -------
    PSC_over_PSP
        Conversion factor to be multiplied to a `PSP` (in mV) to obtain a `PSC`
        (in pA).

    """
    sub = 1.0 / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub

    PSC_over_PSP = 1.0 / (pre * (frac**tau_m - frac**tau_syn))
    return PSC_over_PSP


def dc_input_compensating_poisson(bg_rate, K_ext, tau_syn, PSC_ext):
    """Computes DC input if no Poisson input is provided to the microcircuit.

    Parameters
    ----------
    bg_rate
        Rate of external Poisson generators (in spikes/s).
    K_ext
        External indegrees.
    tau_syn
        Synaptic time constant (in ms).
    PSC_ext
        Weight of external connections (in pA).

    Returns
    -------
    DC
        DC input (in pA) which compensates lacking Poisson input.
    """
    DC = bg_rate * K_ext * PSC_ext * tau_syn * 0.001
    return DC


def adjust_weights_and_input_to_synapse_scaling(
    full_num_neurons,
    full_num_synapses,
    K_scaling,
    mean_PSC_matrix,
    PSC_ext,
    tau_syn,
    full_mean_rates,
    DC_amp,
    poisson_input,
    bg_rate,
    K_ext,
):
    """Adjusts weights and external input to scaling of indegrees.

    The recurrent and external weights are adjusted to the scaling
    of the indegrees. Extra DC input is added to compensate for the
    scaling in order to preserve the mean and variance of the input.

    Parameters
    ----------
    full_num_neurons
        Total numbers of neurons.
    full_num_synapses
        Total numbers of synapses.
    K_scaling
        Scaling factor for indegrees.
    mean_PSC_matrix
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn
        Synaptic time constant (in ms).
    full_mean_rates
        Firing rates of the full network (in spikes/s).
    DC_amp
        DC input current (in pA).
    poisson_input
        True if Poisson input is used.
    bg_rate
        Firing rate of Poisson generators (in spikes/s).
    K_ext
        External indegrees.

    Returns
    -------
    PSC_matrix_new
        Adjusted weight matrix (in pA).
    PSC_ext_new
        Adjusted external weight (in pA).
    DC_amp_new
        Adjusted DC input (in pA).

    """
    PSC_matrix_new = mean_PSC_matrix / np.sqrt(K_scaling)
    PSC_ext_new = PSC_ext / np.sqrt(K_scaling)

    # recurrent input of full network
    indegree_matrix = full_num_synapses / full_num_neurons[:, np.newaxis]
    input_rec = np.sum(mean_PSC_matrix * indegree_matrix * full_mean_rates, axis=1)

    DC_amp_new = DC_amp + 0.001 * tau_syn * (1.0 - np.sqrt(K_scaling)) * input_rec

    if poisson_input:
        input_ext = PSC_ext * K_ext * bg_rate
        DC_amp_new += 0.001 * tau_syn * (1.0 - np.sqrt(K_scaling)) * input_ext
    return PSC_matrix_new, PSC_ext_new, DC_amp_new


def plot_raster(path, name, begin, end, N_scaling, title=None, ax=None):
    """Creates a spike raster plot of the network activity.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start plotting spikes (included).
    end
        Time point (in ms) to stop plotting spikes (included).
    N_scaling
        Scaling factor for number of neurons.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes._axes.Axes, optional
        Axis object for plotting.

    Returns
    -------
    None

    """
    fontsize = 12
    ylabels = ["L2/3", "L4", "L5", "L6"]
    color_list = np.tile(["#595289", "#af143c"], 4)

    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    last_node_id = node_ids[-1, -1]
    mod_node_ids = np.abs(node_ids - last_node_id) + 1

    label_pos = [
        (mod_node_ids[i, 0] + mod_node_ids[i + 1, 1]) / 2.0 for i in np.arange(0, 8, 2)
    ]

    stp = 1
    if N_scaling > 0.1:
        stp = int(10.0 * N_scaling)
        print("  Only spikes of neurons in steps of {} are shown.".format(stp))

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for i, n in enumerate(sd_names):
        times = data[i]["time_ms"]
        neurons = np.abs(data[i]["sender"] - last_node_id) + 1
        ax.plot(times[::stp], neurons[::stp], "|", color=color_list[i], markersize=1)
    # ax.set_xlabel("time [ms]", fontsize=fs)
    ax.set_xlim([begin, end]) 
    
    ax.set_yticks(label_pos)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_yticklabels(ylabels, fontsize=fontsize)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    # output_path = os.path.join(path, f"raster_plot_{name}.png")
    # plt.savefig(output_path)
    # plt.close()
    # save the firing stamps for each unit
    all_neuron_stamps = {}
    for i, n in enumerate(sd_names):
        times = data[i]["time_ms"]
        neurons = np.abs(data[i]["sender"] - last_node_id) + 1
        for neuron_id, spike_indices in groupby(
            sorted(range(len(neurons)), key=neurons.__getitem__),
            key=neurons.__getitem__,
        ):
            # one spike stamp file per neuron
            stamp = times[list(spike_indices)]
            neuron_name = "neuron%d" % (neuron_id)
            # neuron_name = "neuron%d_%s"%(neuron_id, n)
            assert neuron_name not in all_neuron_stamps
            all_neuron_stamps[neuron_name] = stamp
            # if neuron_name not in all_neuron_stamps:
            #     all_neuron_stamps[neuron_name] = [stamp]
            # else:
            #     all_neuron_stamps[neuron_name].append(stamp)
    # for k, v in all_neuron_stamps.items():
    #     all_neuron_stamps[k] = np.concatenate(v)
    print("Total neurons:", len(all_neuron_stamps))
    # np.savez(os.path.join(path, "spike_stamp_msec_%s.npz"%(name)), **all_neuron_stamps)


def psth_from_stamps(
    ts, evs, binsize_ms, end_time, return_hz=False, avg_across_pop=False
):
    if len(ts) > 0:
        t = max(np.max(ts), end_time)
    else:
        t = end_time
    nbins = int(np.ceil(t / binsize_ms))
    bin_edges = np.arange(0, nbins + 1) * binsize_ms
    if len(ts) == 0:
        return np.zeros(len(bin_edges) - 1), bin_edges
    ret, _ = np.histogram(ts, bin_edges)
    if avg_across_pop:
        ret = ret / np.unique(evs).shape[0]
    if return_hz:
        return ret / binsize_ms * 1000, bin_edges
    else:
        return ret, bin_edges


def plot_psth(path, name, begin, end, title=None):
    fs = 12  # fontsize
    pop_names = ["L2/3E", "L2/3I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    n_pops = len(pop_names)
    # color_list = np.tile(['#595289', '#af143c'], 4)

    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    # last_node_id = node_ids[-1, -1]
    spk_stamps_by_pop = [[] for _ in range(node_ids.shape[0])]
    spk_labels_by_pop = [[] for _ in range(node_ids.shape[0])]
    n_neurons_by_pop = [y - x + 1 for (x, y) in node_ids]
    for i_, n in enumerate(sd_names):
        times = data[i_]["time_ms"]
        senders = data[i_]["sender"]
        for i_pop_, (lbl_beg, lbl_end) in enumerate(node_ids):
            tmp_pop_mask = (senders >= lbl_beg) & (senders <= lbl_end)
            spk_stamps_by_pop[i_pop_].extend(times[tmp_pop_mask])
            spk_labels_by_pop[i_pop_].extend(senders[tmp_pop_mask])

    psth_by_pop = []
    for i_pop_ in range(len(spk_stamps_by_pop)):
        ids_sorted_ = np.argsort(spk_stamps_by_pop[i_pop_])
        tmp_stamps = np.array(spk_stamps_by_pop[i_pop_])[ids_sorted_]
        tmp_labels = np.array(spk_labels_by_pop[i_pop_])[ids_sorted_]
        psth_tmp, bin_edges = psth_from_stamps(
            tmp_stamps, tmp_labels, 5, end, return_hz=True, avg_across_pop=False
        )
        psth_tmp = psth_tmp / n_neurons_by_pop[i_pop_]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        psth_by_pop.append((bin_centers, psth_tmp))

    plt.figure(figsize=(8, 6))
    # max_psth_val = max([max(k) for _, k in psth_by_pop])
    for i_pop_, (bin_times, psth_rates) in enumerate(psth_by_pop):
        plt.subplot(n_pops, 1, i_pop_ + 1)
        plt.plot(bin_times, psth_rates, color="k", label=pop_names[i_pop_])
        # plt.ylim([0, max_psth_val])

        plt.xlim([begin, end])
        plt.legend()
        if i_pop_ < n_pops - 1:
            plt.xticks([])
        plt.yticks([np.min(psth_rates), np.max(psth_rates)], fontsize=fs)
    plt.xlabel("Time [ms]", fontsize=fs)
    plt.ylabel("Avg. firing rate [Hz]", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    if title:
        plt.suptitle(title)
    plt.savefig(os.path.join(path, "psth_%s.png" % (name)), dpi=300)
    plt.close()


# def plot_psth_thal(path, name, begin, end, N_scaling):
#     fs = 18  # fontsize
#     sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
#     last_node_id = node_ids[-1, -1]

#     plt.figure(figsize=(8, 6))
#     for i, n in enumerate(sd_names):
#         times = data[i]['time_ms']
#         neurons = np.abs(data[i]['sender'] - last_node_id) + 1
#     plt.plot(times[::stp], neurons[::stp], '.')
#     plt.xlabel('time [ms]', fontsize=fs)
#     plt.ylabel("Firing rate")
#     plt.xticks(fontsize=fs)
#     plt.savefig(os.path.join(path, 'psth_thal.png'), dpi=300)
#     plt.close()


def plot_thal(rate_times, rate_values, data_path):
    plt.figure(figsize=(6, 1))
    plt.plot(rate_times, rate_values, color="k")
    plt.xlim([0, 1400])
    plt.xticks([])
    plt.gca().set_yticks([0, 30, 60])
    plt.gca().set_yticklabels([None, None, None])
    plt.savefig(os.path.join(data_path, "thal_fr_designed.png"))
    plt.close()


def firing_rates(path, name, begin, end):
    """Computes mean and standard deviation of firing rates per population.

    The firing rate of each neuron in each population is computed and stored
    in a .dat file in the directory of the spike recorders. The mean firing
    rate and its standard deviation are printed out for each population.

    Parameters
    -----------
    path
        Path where the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start calculating the firing rates (included).
    end
        Time point (in ms) to stop calculating the firing rates (included).

    Returns
    -------
    None

    """
    sd_names, node_ids, data = __load_spike_times(path, name, begin, end)
    all_mean_rates = []
    all_std_rates = []
    for i, n in enumerate(sd_names):
        senders = data[i]["sender"]
        # 1 more bin than node ids per population
        bins = np.arange(node_ids[i, 0], node_ids[i, 1] + 2)
        spike_count_per_neuron, _ = np.histogram(senders, bins=bins)
        rate_per_neuron = spike_count_per_neuron * 1000.0 / (end - begin)
        np.savetxt(os.path.join(path, ("rate" + str(i) + ".dat")), rate_per_neuron)
        # zeros are included
        all_mean_rates.append(np.mean(rate_per_neuron))
        all_std_rates.append(np.std(rate_per_neuron))
    print("Mean rates: {} spikes/s".format(np.around(all_mean_rates, decimals=3)))
    print(
        "Standard deviation of rates: {} spikes/s".format(
            np.around(all_std_rates, decimals=3)
        )
    )


def boxplot(path, populations, title=None):
    """Creates a boxplot of the firing rates of all populations.

    Parameters
    ----------
    path : str
        Path where the firing rates are stored.
    populations : list
        Names of neuronal populations.
    title : str, optional
        Title of the plot.
    """
    fs = 18
    pop_names = [string.replace("23", "2/3") for string in populations]
    label_pos = np.arange(1, len(populations) + 1)
    color_list = ["#af143c", "#595289"]

    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="lightgray")

    rates_per_neuron_rev = []
    for i in range(len(populations)):
        rates_per_neuron_rev.append(
            np.loadtxt(os.path.join(path, f"rate{i}.dat"))
        )

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(
        rates_per_neuron_rev,
        vert=False,  # Horizontal boxplot
        patch_artist=True,  # Enables colored boxes
        medianprops=medianprops,
        meanprops=meanprops,
        meanline=True,
        showmeans=True,
    )

    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    # Apply colors to boxes
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(color_list[i % len(color_list)])  # Alternate colors

    plt.xlabel("Firing Rate [spikes/s]", fontsize=fs)
    plt.yticks(label_pos, pop_names, fontsize=fs)
    plt.xticks(fontsize=fs)

    if title:
        plt.suptitle(title)

    plt.savefig(os.path.join(path, "box_plot.png"), dpi=300)
    plt.close()


def __gather_metadata(path, name):
    """Reads names and ids of spike recorders and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike recorder per MPI-process/thread is extracted.

    Parameters
    ------------
    path
        Path where the spike recorder files are stored.
    name
        Name of the spike recorder, typically ``spike_recorder``.

    Returns
    -------
    sd_files
        Names of all files written by spike recorders.
    sd_names
        Names of all spike recorders.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(name):
            sd_files.append(fn)
            # spike recorder name and its ID
            fnsplit = "-".join(fn.split("-")[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)

    # load node IDs
    node_idfile = open(os.path.join(path, "population_nodeids.dat"), "r")
    node_ids = []
    for node_id in node_idfile:
        node_ids.append(node_id.split())
    node_ids = np.array(node_ids, dtype="i4")
    return sd_files, sd_names, node_ids


def __load_spike_times(path, name, begin, end):
    """Loads spike times of each spike recorder.

    Parameters
    ----------
    path
        Path where the files with the spike times are stored.
    name
        Name of the spike recorder.
    begin
        Time point (in ms) to start loading spike times (included).
    end
        Time point (in ms) to stop loading spike times (included).

    Returns
    -------
    data
        Dictionary containing spike times in the interval from ``begin``
        to ``end``.

    """
    sd_files, sd_names, node_ids = __gather_metadata(path, name)
    data = {}
    dtype = {"names": ("sender", "time_ms"), "formats": ("i4", "f8")}  # as in header
    for i, name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                # skip header while loading
                ld = np.loadtxt(os.path.join(path, f), skiprows=3, dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order="time_ms")
        # begin and end are included if they exist
        low = np.searchsorted(data_i_raw["time_ms"], v=begin, side="left")
        high = np.searchsorted(data_i_raw["time_ms"], v=end, side="right")
        data[i] = data_i_raw[low:high]
    return sd_names, node_ids, data


load_spike_times = __load_spike_times


def compute_smoothed_firing_rate(spike_train, sim_time_ms, sigma=50, time_resolution=1):
    """
    Compute smoothed firing rate for spike train using Gaussian kernel.

    Parameters
    ----------
    spike_train : np.array
        Array of spike times in milliseconds for a single neuron.
    sim_time_ms : int
        Total simulation time in milliseconds.
    sigma : int, optional
        Standard deviation of the Gaussian kernel for smoothing the firing rate. The default is 50.
    time_resolution : int, optional
        Resolution of time bins in millisecond. The default is 1.

    Returns
    -------
    smoothed_rate : np.array
        Array of the smoothed firing rate over time.
    time_bins : np.array
        Array of time bin centers used for calculating the firing rate.

    """

    time_bins = np.arange(0, sim_time_ms, time_resolution)
    spike_counts, _ = np.histogram(spike_train, bins=time_bins)

    # Gaussian smoothing kernel
    kernel_width = int(3 * sigma / time_resolution)
    kernel = np.exp(
        -np.linspace(-kernel_width, kernel_width, 2 * kernel_width + 1) ** 2
        / (2 * sigma**2)
    )
    kernel /= np.sum(kernel)

    # Smooth the spike counts
    smoothed_rate = np.convolve(spike_counts, kernel, mode="same")
    return smoothed_rate, time_bins[:-1]


def compute_spike_rates(
    spike_trains, sim_time_ms, window_ms, overlap_ms, presim_time_ms=0, sigma=20
):
    """
    Compute smoothed spike rates for each neuron over overlapping time windows.

    Parameters
    ----------
    spike_trains : list of np.array
        List where each element is an array of spike times in milliseconds for a single neuron.
    sim_time_ms : int
        Total simulation time in milliseconds.
    window_ms : int
        Duration of each window in ms to calculate spike rates.
    overlap_ms : int
        Overlap duration between consecutive windows in milliseconds.
    presim_time_ms: int
        Pre-simulation time to subtract from spike trains.
    sigma : float, optional
        Standard deviation of the Gaussian kernel for smoothing the firing rate. The default is 20.

    Returns
    -------
    spike_rates : np.ndarray
        2D array with shape (num_windows, num_neurons) representing smoothed spike rates.
        Rows represent different time windows, and columns represent different neurons.

    """
    step_ms = window_ms - overlap_ms
    if sim_time_ms < window_ms:
        window_ms = sim_time_ms
        overlap_ms = 0
        step_ms = window_ms
    num_windows = int(np.floor((sim_time_ms - window_ms) / step_ms)) + 1
    num_neurons = len(spike_trains)
    windows = np.linspace(0, sim_time_ms - window_ms, num_windows)
    spike_rates = np.zeros((num_windows, num_neurons))

    # Adjust for pre-simulation time
    if presim_time_ms > 0:
        spike_trains = [
            np.array(spike_train) - presim_time_ms for spike_train in spike_trains
        ]

    for i, spike_train in enumerate(spike_trains):
        smoothed_rate, _ = compute_smoothed_firing_rate(
            spike_train, sim_time_ms, sigma=sigma
        )
        for j, start_time in enumerate(windows):
            end_time = start_time + window_ms
            spike_rates[j, i] = np.mean(smoothed_rate[int(start_time) : int(end_time)])

    return spike_rates


def get_dimensionality(spike_rates, variance_threshold, plot_scree=False):
    """
    Compute the number of principal components required to explain 95% of the variance in spike rates.

    Parameters
    ----------
    spike_rates : np.ndarray
        2D array where rows represent different time windows and columns represent neurons.
    plot_scree : bool, optional
        If True, plot a scree plot showing cumulative explained variance. Default is True.

    Returns
    -------
    num_components : int
        Number of PCA components explaining 95% of the variance.

    """
    pca = PCA()
    pca.fit(spike_rates)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(explained_variance >= variance_threshold) + 1
    print(
        f"Number of components explaining {variance_threshold*100}% variance: {num_components}"
    )

    if plot_scree:
        plt.figure()
        plt.plot(explained_variance, label="Cumulative Explained Variance")
        plt.axvline(
            num_components,
            color="k",
            linestyle="--",
            label=f"{num_components} components",
        )
        # Annotate the number of components explaining 95% variance
        plt.text(
            num_components + 1,
            variance_threshold,
            f"{num_components} components",
            verticalalignment="center",
            color="k",
        )

        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.show()

    return num_components


def plot_trajectories(
    baseline_pca, stim_projected=None, sigma=1, stim_color=None, ax=None
):
    """
    Plots the baseline PCA and stimulus projections on the same 3D axis with Gaussian smoothing.

    Parameters:
        baseline_pca (ndarray): Transformed baseline data projected onto top 3 PCA components.
        sigma (float): Standard deviation for Gaussian kernel to smooth the trajectories.
        stim_projected (ndarray): Stimulus data projected onto the PCA components.
        stim_color (str): Color for the stimulus trajectory.
        ax (matplotlib axis): Existing 3D axis to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Apply Gaussian filter to baseline PCA components
    baseline_smoothed = np.empty_like(baseline_pca)
    for i in range(3):  # Loop over each PCA component (x, y, z)
        baseline_smoothed[:, i] = gaussian_filter1d(baseline_pca[:, i], sigma=sigma)

    # Plot the smoothed baseline trajectory
    ax.plot(
        baseline_smoothed[:, 0],
        baseline_smoothed[:, 1],
        baseline_smoothed[:, 2],
        color="k",
        linewidth=0.5,
        alpha=0.7,
        label="Baseline Smoothed",
    )

    if stim_projected is not None:
        # Apply Gaussian filter to stimulus components
        stim_smoothed = np.empty_like(stim_projected)
        for i in range(3):
            stim_smoothed[:, i] = gaussian_filter1d(stim_projected[:, i], sigma=sigma)

        # Plot the smoothed stimulus trajectory
        ax.plot(
            stim_smoothed[:, 0],
            stim_smoothed[:, 1],
            stim_smoothed[:, 2],
            color=stim_color,
            linewidth=0.5,
            alpha=0.7,
            label="Stimulus Smoothed",
        )

    ax.set_title("3D Projection with Smoothed Trajectories")
    
    ax.set_xlabel("PC1", fontsize=8, labelpad=2)
    ax.set_ylabel("PC2", fontsize=8, labelpad=0)
    ax.set_zlabel("PC3", fontsize=8, labelpad=-4)
    
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))  
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))  
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=2))  
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    ax.tick_params(axis="x", labelsize=8, pad=0)
    ax.tick_params(axis="y", labelsize=8, pad=-2)
    ax.tick_params(axis="z", labelsize=8, pad=0)

    return ax


def plot_gaussian_ellipsoid(
    mean, cov, ax=None, n_std=2, color="k", alpha=0.3, wireframe=True
):
    """
    Plots an ellipsoid representing the Gaussian defined by mean and covariance matrix.

    Parameters:
    - mean: Center of the ellipsoid (mean of Gaussian).
    - cov: Covariance matrix of the Gaussian.
    - ax: Existing 3D axis to plot on. Creates new one if None.
    - n_std: Number of standard deviations to scale the ellipsoid radii. (2 stds includes 95% of points)
    - color: Color of the ellipsoid.
    - alpha: Transparency of the ellipsoid.
    - wireframe: Plot wireframe
    """
    # Create a 3D grid of points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale and rotate the points using the covariance matrix
    sphere_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Decompose the covariance matrix to obtain radii and rotation
    radii, rotation = np.linalg.eigh(cov)
    # Scale the standard deviation by n_std (e.g., 2 for 95% CI)
    radii = n_std * np.sqrt(radii)
    ellipsoid_points = sphere_points @ np.diag(radii) @ rotation.T

    # Translate the points to the mean
    ellipsoid_points += mean

    # Reshape the points for plotting
    x_ellipsoid = ellipsoid_points[:, 0].reshape(x.shape)
    y_ellipsoid = ellipsoid_points[:, 1].reshape(y.shape)
    z_ellipsoid = ellipsoid_points[:, 2].reshape(z.shape)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot the ellipsoid surface
    if wireframe:
        ax.plot_wireframe(
            x_ellipsoid,
            y_ellipsoid,
            z_ellipsoid,
            color=color,
            alpha=alpha,
            rcount=20,
            ccount=20,
            linewidth=1,
        )
    else:
        ax.plot_surface(
            x_ellipsoid,
            y_ellipsoid,
            z_ellipsoid,
            color=color,
            alpha=alpha,
            rstride=4,
            cstride=4,
            linewidth=0,
        )

    return ax


def fit_gaussian_model(data):
    """Fit a Gaussian Mixture Model and return mean and covariance."""
    gmm = GaussianMixture(n_components=1, covariance_type="full")
    gmm.fit(data)
    return gmm.means_[0], gmm.covariances_[0]


def compute_all_overlaps(baseline, stim_projected_list):
    """Calculate overlaps between baseline and stimulus projections."""
    overlap_list = []
    baseline_mean, baseline_cov = fit_gaussian_model(baseline)

    for stim_projected in stim_projected_list:
        stim_mean, stim_cov = fit_gaussian_model(stim_projected)
        overlap, _, _ = compute_jaccard_overlap(
            baseline_mean, baseline_cov, stim_mean, stim_cov
        )
        overlap_list.append(overlap)

    return overlap_list, baseline_mean, baseline_cov


def plot_projection(
    ax,
    baseline,
    stim_projected,
    baseline_mean,
    baseline_cov,
    stim_mean,
    stim_cov,
    xlim=None,
    ylim=None,
    zlim=None,
    view=None,
):
    """Plot trajectories and ellipsoids on a 3D axis."""
    plot_trajectories(baseline, stim_projected, stim_color="C0", ax=ax)
    plot_gaussian_ellipsoid(
        stim_mean, stim_cov, ax=ax, n_std=2, color="k", alpha=0.2, wireframe=True
    )
    plot_gaussian_ellipsoid(
        baseline_mean,
        baseline_cov,
        ax=ax,
        n_std=2,
        color="k",
        alpha=0.2,
        wireframe=True,
    )

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)
    if view:
        ax.view_init(elev=view[0], azim=view[1])


def plot_projections(
    baseline,
    stim_projected_list,
    output_dir,
    file_name,
    views,
    N_GROUPS_LIST=[1, 2, 4, 8, 16, 32],
    xlim=None,
    ylim=None,
    zlim=None,
):
    """Plot projections for different views and save the figures without returning any data."""
    # Calculate overlaps and fit baseline GMM model
    overlap_list, baseline_mean, baseline_cov = compute_all_overlaps(
        baseline, stim_projected_list
    )

    for view_index, view in enumerate(views):
        fig = plt.figure(figsize=(7, 5))
        gs = GridSpec(2, 3, figure=fig, hspace=0, wspace=0.32)

        for i, stim_projected in enumerate(stim_projected_list):
            ax = fig.add_subplot(gs[i], projection="3d")
            stim_mean, stim_cov = fit_gaussian_model(stim_projected)

            # Plot trajectories and ellipsoids
            plot_projection(
                ax,
                baseline,
                stim_projected,
                baseline_mean,
                baseline_cov,
                stim_mean,
                stim_cov,
                xlim,
                ylim,
                zlim,
                view,
            )

            ax.set_title(f"{N_GROUPS_LIST[i]} electrode groups", fontsize=12)

        
        # Save the figure for the current view
        file_path = os.path.join(output_dir, f"{file_name}_view_{view_index + 1}.png")
        plt.savefig(file_path)
        plt.show()


def compute_jaccard_overlap(mean_1, cov_1, mean_2, cov_2, n_samples=10000):
    """
    Compute an estimate of the volume overlap between two Gaussian distributions using the Jaccard Index.

    Parameters
    ----------
    mean_1 : ndarray
        Mean of the first Gaussian distribution.
    cov_1 : ndarray
        Covariance matrix of the first Gaussian distribution.
    mean_2 : ndarray
        Mean of the second Gaussian distribution.
    cov_2 : ndarray
        Covariance matrix of the second Gaussian distribution.
    n_samples : int, optional
        Number of samples to use for Monte Carlo estimation. Default is 10000.

    Returns
    -------
    jaccard_index : float
        Estimated Jaccard Index between the two Gaussians.
    samples_1 : ndarray
        Samples drawn from the first Gaussian distribution.
    samples_2 : ndarray
        Samples drawn from the second Gaussian distribution.
    """
    # Generate random samples from both Gaussian distributions
    samples_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples)
    samples_2 = np.random.multivariate_normal(mean_2, cov_2, n_samples)

    # Evaluate the densities of the samples under both Gaussians
    density_1_samples_1 = multivariate_normal.pdf(samples_1, mean=mean_1, cov=cov_1)
    density_2_samples_1 = multivariate_normal.pdf(samples_1, mean=mean_2, cov=cov_2)

    density_1_samples_2 = multivariate_normal.pdf(samples_2, mean=mean_1, cov=cov_1)
    density_2_samples_2 = multivariate_normal.pdf(samples_2, mean=mean_2, cov=cov_2)

    # Estimate the intersection volumes for both sets of samples
    intersection_1 = np.sum(np.minimum(density_1_samples_1, density_2_samples_1))
    intersection_2 = np.sum(np.minimum(density_1_samples_2, density_2_samples_2))
    intersection_volume = (intersection_1 + intersection_2) / 2

    # Estimate the union volumes for both sets of samples
    union_1 = np.sum(np.maximum(density_1_samples_1, density_2_samples_1))
    union_2 = np.sum(np.maximum(density_1_samples_2, density_2_samples_2))
    union_volume = (union_1 + union_2) / 2

    # Calculate the Jaccard Index as Intersection / Union
    jaccard_index = intersection_volume / union_volume

    return jaccard_index, samples_1, samples_2


    