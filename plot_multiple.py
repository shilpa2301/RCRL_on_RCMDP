import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

def plot_mean_std_multiple_dirs(data_dirs, labels, num_runs_list, save=False, filename="aggregated_plot.png", smooth_window=10):
    """
    Plot mean and standard deviation of rewards and max costs from multiple directories.
    
    Parameters:
        data_dirs (list of str): List of data directories to read from.
        labels (list of str): List of labels for each data directory.
        num_runs_list (list of int): List of the number of runs for each directory.
        save (bool): Whether to save the plot.
        filename (str): Filename to save the plot as.
        smooth_window (int): Window size for smoothing.
    """
    assert len(data_dirs) == len(labels) == len(num_runs_list), "data_dirs, labels, and num_runs_list must have the same length"

    plt.figure(figsize=(12, 8))

    # Plot Rewards
    plt.subplot(2, 1, 1)
    for data_dir, label, num_runs in zip(data_dirs, labels, num_runs_list):
        all_rewards = []
        min_length = float('inf')  # Track the shortest run length

        for run in range(1, num_runs + 1):
            run_dir = f"{data_dir}/run{run}"
            rewards = np.load(f"{run_dir}/episode_rewards.npy")
            all_rewards.append(rewards)
            min_length = min(min_length, len(rewards))  # Update the shortest length

        # Truncate all arrays to the shortest length
        all_rewards = np.array([r[:min_length] for r in all_rewards])

        # Compute mean and std deviation
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        # Apply smoothing
        smoothed_mean_rewards = smooth(mean_rewards, smooth_window)
        smoothed_std_rewards = smooth(std_rewards, smooth_window)

        # Adjust x-axis for the smoothed plots
        smoothed_x = range(len(smoothed_mean_rewards))

        # Plot raw and smoothed rewards
        # plt.plot(mean_rewards, label=f"{label} (Raw)", alpha=0.3)
        plt.plot(smoothed_x, smoothed_mean_rewards, label=f"{label}", linewidth=2)
        plt.fill_between(smoothed_x, smoothed_mean_rewards - smoothed_std_rewards, smoothed_mean_rewards + smoothed_std_rewards, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mean and Smoothed Total Rewards")
    plt.legend()

    # Plot Max Costs
    plt.subplot(2, 1, 2)
    for data_dir, label, num_runs in zip(data_dirs, labels, num_runs_list):
        all_max_costs = []
        min_length = float('inf')  # Track the shortest run length

        for run in range(1, num_runs + 1):
            run_dir = f"{data_dir}/run{run}"
            max_costs = np.load(f"{run_dir}/episode_max_costs.npy")
            all_max_costs.append(max_costs)
            min_length = min(min_length, len(max_costs))  # Update the shortest length

        # Truncate all arrays to the shortest length
        all_max_costs = np.array([c[:min_length] for c in all_max_costs])

        # Compute mean and std deviation
        mean_max_costs = np.mean(all_max_costs, axis=0)
        std_max_costs = np.std(all_max_costs, axis=0)

        # Apply smoothing
        smoothed_mean_max_costs = smooth(mean_max_costs, smooth_window)
        smoothed_std_max_costs = smooth(std_max_costs, smooth_window)

        # Adjust x-axis for the smoothed plots
        smoothed_x = range(len(smoothed_mean_max_costs))

        # Plot raw and smoothed max costs
        # plt.plot(mean_max_costs, label=f"{label} (Raw)", alpha=0.3)
        plt.plot(smoothed_x, smoothed_mean_max_costs, label=f"{label}", linewidth=2)
        plt.fill_between(smoothed_x, smoothed_mean_max_costs - smoothed_std_max_costs, smoothed_mean_max_costs + smoothed_std_max_costs, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.title("Mean and Smoothed Max Costs")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(filename)
    # plt.show()
    plt.close()

# Example usage
data_dirs = [
    "./plot_data/CartPoleCostEnv",
    "./plot_data/CartPolePerturbedEnv"
]
labels = ["CartPoleCostEnv", "CartPolePerturbedEnv"]
num_runs_list = [2, 2]

plot_mean_std_multiple_dirs(data_dirs, labels, num_runs_list, save=True, filename="train_plots/combined_plot.png", smooth_window=80)
