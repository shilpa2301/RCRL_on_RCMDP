import numpy as np
import matplotlib.pyplot as plt

# def plot_mean_std(data_dir, num_runs, save=False, filename="aggregated_plot.png"):
#     all_rewards = []
#     all_max_costs = []

#     for run in range(1, num_runs + 1):
#         run_dir = f"{data_dir}/run{run}"
#         rewards = np.load(f"{run_dir}/episode_rewards.npy")
#         max_costs = np.load(f"{run_dir}/max_costs.npy")
#         all_rewards.append(rewards)
#         all_max_costs.append(max_costs)

#     # Convert to numpy arrays
#     all_rewards = np.array(all_rewards)
#     all_max_costs = np.array(all_max_costs)

#     # Compute mean and std deviation
#     mean_rewards = np.mean(all_rewards, axis=0)
#     std_rewards = np.std(all_rewards, axis=0)
#     mean_max_costs = np.mean(all_max_costs, axis=0)
#     std_max_costs = np.std(all_max_costs, axis=0)

#     # Plot results
#     plt.figure(figsize=(12, 6))

#     # Plot rewards
#     plt.subplot(2, 1, 1)
#     plt.plot(mean_rewards, label="Mean Reward", color="blue")
#     plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color="blue", alpha=0.2, label="Std Dev")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.title("Mean and Std Dev of Total Rewards")
#     plt.legend()

#     # Plot max costs
#     plt.subplot(2, 1, 2)
#     plt.plot(mean_max_costs, label="Mean Max Cost", color="red")
#     plt.fill_between(range(len(mean_max_costs)), mean_max_costs - std_max_costs, mean_max_costs + std_max_costs, color="red", alpha=0.2, label="Std Dev")
#     plt.xlabel("Episode")
#     plt.ylabel("Max Cost")
#     plt.title("Mean and Std Dev of Max Costs")
#     plt.legend()

#     plt.tight_layout()
#     if save:
#         plt.savefig(filename)
#     plt.show()

def plot_mean_std(data_dir, num_runs, save=False, filename="aggregated_plot.png"):
    all_rewards = []
    all_max_costs = []

    min_length = float('inf')  # Track the shortest run length

    for run in range(1, num_runs + 1):
        run_dir = f"{data_dir}/run{run}"
        rewards = np.load(f"{run_dir}/episode_rewards.npy")
        max_costs = np.load(f"{run_dir}/max_costs.npy")
        all_rewards.append(rewards)
        all_max_costs.append(max_costs)
        min_length = min(min_length, len(rewards), len(max_costs))  # Update the shortest length

    # Truncate all arrays to the shortest length
    all_rewards = np.array([r[:min_length] for r in all_rewards])
    all_max_costs = np.array([c[:min_length] for c in all_max_costs])

    # Compute mean and std deviation
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_max_costs = np.mean(all_max_costs, axis=0)
    std_max_costs = np.std(all_max_costs, axis=0)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color="blue", alpha=0.2, label="Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Mean and Std Dev of Total Rewards")
    plt.legend()

    # Plot max costs
    plt.subplot(2, 1, 2)
    plt.plot(mean_max_costs, label="Mean Max Cost", color="red")
    plt.fill_between(range(len(mean_max_costs)), mean_max_costs - std_max_costs, mean_max_costs + std_max_costs, color="red", alpha=0.2, label="Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.title("Mean and Std Dev of Max Costs")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(filename)
    # plt.show()
    plt.close()

# Example usage
plot_mean_std(data_dir="./plot_data", num_runs=3, save=True, filename="aggregated_ipm_rcrl_plot.png")
