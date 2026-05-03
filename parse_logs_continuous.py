import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Define the three curriculum phases and their corresponding RLlib log files.
LOG_FILES_CONFIG = {
    "Phase 1 (vs Random)": [
        "logs/soccer_shaped_ma-4095503.out",
        "logs/soccer_shaped_ma-4096743.out"
    ],
    "Phase 2 (vs Baseline)": [
        "logs/soccer_shaped_vs_baseline-4107863.out"
    ],
    "Phase 3 (Self-Play)": [
        "logs/soccer_shaped_vs_self-4123721.out",
        "logs/soccer_shaped_vs_self-4153101.out"
    ],
}


# --- Parsing logic ---
def parse_rllib_log(filepath):
    """Parse a single Ray RLlib .out log file."""
    if not os.path.exists(filepath):
        print(f"Warning: file not found: {filepath}")
        return []

    data_points = []
    reward_pattern = re.compile(r"episode_reward_mean:\s*([\-0-9\.]+)")
    timestep_pattern = re.compile(r"timesteps_total:\s*([0-9]+)")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    iterations = content.split("Result for")

    for iter_block in iterations:
        reward_match = reward_pattern.search(iter_block)
        timestep_match = timestep_pattern.search(iter_block)

        if reward_match and timestep_match:
            try:
                reward_val = float(reward_match.group(1))
                timestep_val = int(timestep_match.group(1))
                data_points.append((timestep_val, reward_val))
            except ValueError:
                continue

    # Keep the original training order within the log file.
    return data_points


def estimate_typical_step(timesteps):
    """Estimate the typical logging interval from a timestep sequence."""
    if len(timesteps) < 2:
        return 0

    diffs = [timesteps[i] - timesteps[i - 1] for i in range(1, len(timesteps))]
    positive_diffs = [d for d in diffs if d > 0]

    if not positive_diffs:
        return 0

    return int(np.median(positive_diffs))


def merge_phase_logs_as_continuous(filepaths):
    """
    Merge multiple log files within one phase into a continuous local phase axis.

    This function does not assume that different jobs share the same global
    timestep counter. Each log file is shifted so that it starts immediately
    after the previous log file in the same phase.
    """
    phase_points = []
    next_start = 0
    previous_step = 0

    for path in filepaths:
        points = parse_rllib_log(path)
        print(f"  -> Reading file: {path}")
        print(f"     Found {len(points)} data points.")

        if not points:
            continue

        # Sort points inside each file to avoid accidental ordering issues.
        points = sorted(points, key=lambda x: x[0])
        raw_timesteps = [t for t, _ in points]
        raw_rewards = [r for _, r in points]

        local_start = raw_timesteps[0]
        typical_step = estimate_typical_step(raw_timesteps)

        # Start this log after the previous one. For the first log, start at 0.
        start_for_this_log = next_start
        shifted_points = [
            (start_for_this_log + (t - local_start), r)
            for t, r in zip(raw_timesteps, raw_rewards)
        ]

        phase_points.extend(shifted_points)

        previous_step = typical_step if typical_step > 0 else previous_step
        next_start = shifted_points[-1][0] + previous_step

    if not phase_points:
        return [], []

    timesteps, rewards = zip(*phase_points)
    return list(timesteps), list(rewards)


# --- Plotting logic ---
def plot_curves(all_data):
    """Plot the parsed training curves with Matplotlib."""
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 3.5))

    for label, data in all_data.items():
        timesteps, rewards = data
        if not timesteps:
            continue

        # Smooth the reward curve with a moving average.
        window_size = 5
        if len(rewards) >= window_size:
            rewards_smooth = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            timesteps_smooth = timesteps[window_size - 1:]
        else:
            rewards_smooth = rewards
            timesteps_smooth = timesteps

        ax.plot(timesteps_smooth, rewards_smooth, label=label, linewidth=2.5)

    ax.set_title("Training Performance Across Curriculum Phases", fontsize=16)
    ax.set_xlabel("Continuous Timesteps Across Phases", fontsize=12)
    ax.set_ylabel("Mean Episode Reward (Smoothed)", fontsize=12)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True)

    output_filename = "training_curve_continuous.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved successfully as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    final_data = {}
    global_offset = 0
    previous_global_step = 0

    for label, filepaths in LOG_FILES_CONFIG.items():
        print(f"--- Parsing phase: {label} ---")

        phase_timesteps, phase_rewards = merge_phase_logs_as_continuous(filepaths)

        if not phase_timesteps:
            final_data[label] = ([], [])
            print(f"  => Phase '{label}' has no valid data points.\n")
            continue

        phase_step = estimate_typical_step(phase_timesteps)
        if phase_step <= 0:
            phase_step = previous_global_step

        # Shift the entire phase so that Phase 1, Phase 2, and Phase 3 are
        # displayed sequentially instead of overlapping on the x-axis.
        shifted_timesteps = [t + global_offset for t in phase_timesteps]
        final_data[label] = (shifted_timesteps, phase_rewards)

        print(f"  => Phase '{label}' data points: {len(shifted_timesteps)}")
        print(f"     Continuous timestep range: {shifted_timesteps[0]:,} -> {shifted_timesteps[-1]:,}\n")

        previous_global_step = phase_step if phase_step > 0 else previous_global_step
        global_offset = shifted_timesteps[-1] + previous_global_step

    plot_curves(final_data)
