import os
import pickle

import nest
import numpy as np

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
from network_cortcol import Network

# ====================PROBE=====================================================
# Layout: 1x32 spans 1800um
volume_v_min = 200  # um
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)

# stim configurations
sigma_e_um = 2.76e-7
conductivity_constant = 10
STIM_CHANNELS = np.arange(32)
STIM_AMPLITUDES = [2]  # uA
GAP_MS = 1000  # ms gap between consecutive pairs
MAX_STIM_DURATION_MS = 60000  # ms total stimulation duration

# Delay values to sweep
SEQ_DELAY_MS_LIST = [5] #[0, 5, 10, 20, 30, 40, 50, 100, 200, 300]  # ms

PRESIM_TIME_MS = sim_dict["t_presim"]
SIM_TIME_MS = sim_dict["t_sim"]
WINDOW_MS = 500
OVERLAP_MS = 400


def amp_decay_func(amp_uA, dist_um):
    """Amplitude decay with distance from stimulation electrode."""
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )


stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}

nest.ResetKernel()

Nscale = net_dict["N_scaling"]
sim_time_s = int(sim_dict["t_sim"] / 1000)
stim_amp = STIM_AMPLITUDES[0]

# Shared output root (not tied to a specific delay)
base_path = os.path.join(
    os.getcwd(), "outputs",
    f"data_multi_delay_{Nscale}scale_{stim_amp}uA_{sim_time_s}s",
)

# =============================================================================
# Baseline simulation (shared across all delay conditions)
# =============================================================================
sim_dict["data_path"] = os.path.join(base_path, "data_baseline")
pkl_path_baseline = os.path.join(sim_dict["data_path"], "baseline_spike_rates.pkl")

if not os.path.exists(pkl_path_baseline):
    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    print("Total # neurons simulated:", network.n_neurons)

    network.simulate_baseline(PRESIM_TIME_MS)  # startup transient
    network.simulate_baseline(SIM_TIME_MS)

    baseline_spike_trains = network.get_spike_train_list()
    baseline_spike_rates = helpers.compute_spike_rates(
        baseline_spike_trains,
        SIM_TIME_MS,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )

    with open(pkl_path_baseline, "wb") as f:
        pickle.dump(baseline_spike_rates, f)
    print("Baseline saved.")

with open(pkl_path_baseline, "rb") as f:
    baseline_spike_rates = pickle.load(f)

# =============================================================================
# Stimulation sweep over SEQ_DELAY_MS values
# =============================================================================
for seq_delay_ms in SEQ_DELAY_MS_LIST:
    print(f"\n\n{'='*60}")
    print(f"  SEQ_DELAY_MS = {seq_delay_ms} ms")
    print(f"{'='*60}\n")

    stim_data_path = os.path.join(base_path, f"data_detstim_paired_{seq_delay_ms}ms")
    sim_dict["data_path"] = stim_data_path

    pkl_path = os.path.join(stim_data_path, "stim_spike_rates.pkl")
    pkl_path_stim_pulses = os.path.join(stim_data_path, "stim_pulses.pkl")

    if not os.path.exists(pkl_path):
        nest.ResetKernel()

        electrodes = StimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)

        network = Network(sim_dict, net_dict, stim_dict)
        network.create()
        network.connect()

        network.simulate_baseline(PRESIM_TIME_MS)  # startup transient

        # Build paired stimulation pattern:
        # Pulse 1: all channels at offset 0 ms
        # Pulse 2: all channels at offset seq_delay_ms ms
        # Pair repeats every (seq_delay_ms + GAP_MS) ms
        pair_channels = np.tile(STIM_CHANNELS, 2)
        pair_times = np.concatenate([
            np.zeros(len(STIM_CHANNELS)),
            np.full(len(STIM_CHANNELS), float(seq_delay_ms)),
        ])
        interpattern_time_ms = float(seq_delay_ms + GAP_MS)

        electrodes.generate_deterministic_stimulation(
            pair_channels,
            pair_times,
            STIM_AMPLITUDES,
            MAX_STIM_DURATION_MS,
            interpattern_time_ms=interpattern_time_ms,
        )

        electrodes.compute_impulse_response_matrix(network.neuron_locations)
        electrodes.compute_stim_current_matrix()
        electrodes.calculate_induced_current_matrix()

        current_generators = electrodes.get_current_generators()

        network.simulate_current_input(current_generators, time_ms=SIM_TIME_MS)

        stim_evoked_spike_trains = network.get_spike_train_list()
        stim_evoked_spike_rates = helpers.compute_spike_rates(
            stim_evoked_spike_trains,
            SIM_TIME_MS,
            WINDOW_MS,
            OVERLAP_MS,
            presim_time_ms=PRESIM_TIME_MS,
        )

        with open(pkl_path, "wb") as f:
            pickle.dump(stim_evoked_spike_rates, f)

        with open(pkl_path_stim_pulses, "wb") as f:
            pickle.dump(electrodes.stim_onset_times_by_ch, f)

        print(f"  Saved stim spike rates → {pkl_path}")

    else:
        print(f"  Already exists, skipping: {pkl_path}")

print("\n\nAll delay conditions complete.")
