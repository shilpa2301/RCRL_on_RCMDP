import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

def plot_mean_std_multiple_dirs(data_dirs, labels, num_runs_list, save=False, base_filename="aggregated_plot", smooth_window=10):
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

    plt.rcParams.update({'font.size': 85, 'lines.linewidth': 7})
    fig_size = 28

    # Plot Rewards
    plt.figure(figsize=(fig_size, fig_size))  # Square figure
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

        # Plot smoothed rewards
        plt.plot(smoothed_x, smoothed_mean_rewards, label=f"{label}", linewidth=2)
        plt.fill_between(smoothed_x, smoothed_mean_rewards - smoothed_std_rewards, smoothed_mean_rewards + smoothed_std_rewards, alpha=0.2)

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # Optional: Add a baseline at y=0
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(labels))  # Legend above the plot
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches='tight')
    plt.close()

    # Plot Max Costs
    plt.figure(figsize=(fig_size, fig_size))  # Square figure
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

        # Plot smoothed max costs
        plt.plot(smoothed_x, smoothed_mean_max_costs, label=f"{label}", linewidth=2)
        plt.fill_between(smoothed_x, smoothed_mean_max_costs - smoothed_std_max_costs, smoothed_mean_max_costs + smoothed_std_max_costs, alpha=0.2)

    # Always plot the threshold line at y=2.0
    plt.axhline(y=2.0, color='black', linestyle='--', label="Safety Relaxation")

    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(labels))  # Legend above the plot
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches='tight')
    plt.close()
    

# Example usage
# data_dirs = [
#     "./plot_data/CartPoleCostEnv",
#     "./plot_data/CartPolePerturbedEnv",
#     "./plot_data/CartPoleCostEnv_PD"
# ]
# labels = ["CartPoleCostEnv", "CartPolePerturbedEnv", "Vanilla_Primal_Dual"]
# num_runs_list = [2, 2, 3]

# data_dirs = [
#     "./plot_data/CartPoleCostEnv",
#     # "./plot_data/CartPolePerturbedEnv",
#     "./plot_data/CartPoleCostEnv_PD_RCRL"
# ]
# labels = ["Surrogate Obj", "PD (RCRL)"]
# num_runs_list = [3, 2]

# data_dirs = [
#     # "./plot_data/CartPoleCostEnv",
#     "./plot_data/CartPolePerturbedEnv",
#     # "./plot_data/CartPoleCostEnv_PD_RCRL"
# ]
# labels = ["Ours"]
# num_runs_list = [3]

data_dirs = [
    # "./plot_data/CartPoleCostEnv",
    "./plot_data/CartPolePerturbedEnv_Cost3",
    # "./plot_data/CartPoleCostEnv_PD_RCRL"
]
labels = ["Ours-C2"]
num_runs_list = [3]

plot_mean_std_multiple_dirs(data_dirs, labels, num_runs_list, save=True, base_filename="train_plots/New_Ours_C2", smooth_window=80)
