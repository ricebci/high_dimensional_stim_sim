"""
Generate a randomized stimulation configuration JSON file.

Stimulation events are generated with:
- Single mode: one random channel per stimulation time
- All mode: all channels stimulated at each stimulation time
- Exponentially-distributed inter-stimulus intervals
- Random amplitude per event sampled from a set of integer current levels
- Time stamps discretized to integer milliseconds

How to run
----------
From the project root:

1) Use defaults (single-electrode mode):
    python generate_stim_config.py

2) Use all-electrode mode (all channels at each stimulation time):
    python generate_stim_config.py --mode all

3) Use custom duration/rate/range and output path:
    python generate_stim_config.py \
         --total-duration-ms 6000 \
         --rate-ms 300 \
         --current-levels-uA 2 3 4 5 6 \
         --mode single \
         --output data/my_stim_config_1ch.json

4) Reproducible generation with seed:
    python generate_stim_config.py --seed 42
"""

import argparse
import json
import numpy as np


def generate_stim_config(
    total_duration_ms: float = 20000,
    rate_ms: float = 500,
    current_levels_uA=None,
    mode: str = "single",
    n_channels: int = 32,
    seed: int = None,
    output_file: str = None,
):
    """
    Generate a stimulation configuration with exponentially-distributed
    inter-stimulus intervals.

    Parameters
    ----------
    total_duration_ms : float
        Total experiment duration in milliseconds (default 20000 = 20 seconds)
    rate_ms : float
        Mean interval between successive stimulations in milliseconds (default 500)
    current_levels_uA : list[int], optional
        Set of integer stimulation amplitudes (uA) to sample from per event
    mode : str
        Stimulation mode: "single" (one random channel per event) or
        "all" (all channels at each event time)
    n_channels : int
        Number of stimulation channels (0 to n_channels-1)
    seed : int, optional
        Random seed for reproducibility
    output_file : str, optional
        Path to save the JSON file. If None, returns dict instead.

    Returns
    -------
    dict or None
        If output_file is None, returns the config dict. Otherwise saves to file.
    """
    if seed is not None:
        np.random.seed(seed)

    if current_levels_uA is None:
        current_levels_uA = [2, 3, 4, 5, 6]

    if len(current_levels_uA) == 0:
        raise ValueError("current_levels_uA must contain at least one value")

    current_levels_uA = [int(level) for level in current_levels_uA]

    if mode not in {"single", "all"}:
        raise ValueError("mode must be either 'single' or 'all'")

    # Generate inter-stimulus intervals from exponential distribution
    inter_stimulus_intervals = np.random.exponential(rate_ms, 
                                                     size=10 * int(total_duration_ms / rate_ms))

    # Compute cumulative stimulus times
    stim_times = np.cumsum(inter_stimulus_intervals)
    stim_times = stim_times[stim_times < total_duration_ms]

    # Discretize to integer millisecond resolution
    stim_times = np.rint(stim_times).astype(int)

    n_stimuli = len(stim_times)

    if mode == "single":
        # One random channel per stimulation time
        channels = np.random.randint(0, n_channels, size=n_stimuli)
        times_out = stim_times
    else:
        # All channels at each stimulation time
        channels = np.tile(np.arange(n_channels), n_stimuli)
        times_out = np.repeat(stim_times, n_channels)

    # Random amplitude for each channel-time event
    amplitudes = np.random.choice(current_levels_uA, size=len(times_out))

    config = {
        "channels": channels.tolist(),
        "times_ms": times_out.tolist(),
        "amplitudes_uA": amplitudes.tolist(),
    }

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Generated {n_stimuli} stimulation events")
        print(f"Total duration: {total_duration_ms} ms")
        print(f"Mean inter-stimulus interval: {rate_ms} ms")
        print(f"Current levels: {sorted(set(current_levels_uA))} uA")
        print(f"Mode: {mode}")
        print(f"Total channel-time entries: {len(times_out)}")
        print(f"Saved to: {output_file}")
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a randomized stimulation configuration JSON file."
    )
    parser.add_argument(
        "--total-duration-ms",
        type=float,
        default=20000,
        help="Total experiment duration in milliseconds (default 20000)",
    )
    parser.add_argument(
        "--rate-ms",
        type=float,
        default=500,
        help="Mean interval between stimulations in ms (default 500)",
    )
    parser.add_argument(
        "--current-levels-uA",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="Integer current levels in uA to sample from (default: 2 3 4 5 6)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "all"],
        default="single",
        help="Stimulation mode: single random channel per event or all channels per event time",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=32,
        help="Number of stimulation channels (default 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated_stim_config.json",
        help="Output JSON file path (default data/generated_stim_config.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_stim_config(
        total_duration_ms=args.total_duration_ms,
        rate_ms=args.rate_ms,
        current_levels_uA=args.current_levels_uA,
        mode=args.mode,
        n_channels=args.n_channels,
        seed=args.seed,
        output_file=args.output,
    )
