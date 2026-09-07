import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


def plot_mean_std_multiple_runs(
    plot_specs,
    save=False,
    base_filename="aggregated_plot",
    smooth_window=10
):
    """
    Plot mean and standard deviation of rewards and max costs from multiple methods.

    Parameters:
        plot_specs (list of dict): Each dict should contain:
            {
                "data_dir": str,
                "label": str,
                "runs": list of int
            }
        save (bool): Whether to save the plot.
        base_filename (str): Base filename to save the plots.
        smooth_window (int): Window size for smoothing.
    """

    baseline_cost_threshold = 0.1

    plt.rcParams.update({
        'font.size': 100,
        'lines.linewidth': 15,
        'font.weight': 'bold'
    })

    fig_size = 28
    label_font = 130
    max_episodes = 16000

    # =====================
    # Plot Rewards
    # =====================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for spec in plot_specs:
        data_dir = spec["data_dir"]
        label = spec["label"]
        runs = spec["runs"]

        all_rewards = []

        for run in runs:
            run_dir = f"{data_dir}/run{run}"
            rewards = np.load(f"{run_dir}/episode_rewards.npy")
            all_rewards.append(rewards[:max_episodes])

        min_length = min(len(r) for r in all_rewards)
        all_rewards = np.array([r[:min_length] for r in all_rewards])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        smoothed_mean_rewards = smooth(mean_rewards, smooth_window)
        smoothed_std_rewards = smooth(std_rewards, smooth_window)

        smoothed_x = range(len(smoothed_mean_rewards))

        plt.plot(smoothed_x, smoothed_mean_rewards, label=label)
        plt.fill_between(
            smoothed_x,
            smoothed_mean_rewards - smoothed_std_rewards,
            smoothed_mean_rewards + smoothed_std_rewards,
            alpha=0.2
        )

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Reward", fontweight='bold', fontsize=label_font)
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(-0.25, 1.25),
        ncol=2,
        frameon=False
    )

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches='tight')

    plt.close()

    # =====================
    # Plot Max Costs
    # =====================
    plt.figure(figsize=(fig_size + 8, fig_size))

    plt.axhspan(baseline_cost_threshold, 12.5, color='red', alpha=0.1)
    plt.axhspan(-1, baseline_cost_threshold, color='blue', alpha=0.1)

    for spec in plot_specs:
        data_dir = spec["data_dir"]
        label = spec["label"]
        runs = spec["runs"]

        all_max_costs = []

        for run in runs:
            run_dir = f"{data_dir}/run{run}"
            max_costs = np.load(f"{run_dir}/episode_max_costs.npy")
            all_max_costs.append(max_costs[:max_episodes])

        min_length = min(len(c) for c in all_max_costs)
        all_max_costs = np.array([c[:min_length] for c in all_max_costs])

        mean_max_costs = np.mean(all_max_costs, axis=0)
        std_max_costs = np.std(all_max_costs, axis=0)

        smoothed_mean_max_costs = smooth(mean_max_costs, smooth_window)
        smoothed_std_max_costs = smooth(std_max_costs, smooth_window)

        smoothed_x = range(len(smoothed_mean_max_costs))

        plt.plot(smoothed_x, smoothed_mean_max_costs, label=label)
        plt.fill_between(
            smoothed_x,
            smoothed_mean_max_costs - smoothed_std_max_costs,
            smoothed_mean_max_costs + smoothed_std_max_costs,
            alpha=0.2
        )

    plt.axhline(
        y=baseline_cost_threshold,
        color='black',
        linestyle='--',
        label="Baseline"
    )

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Max Cost", fontweight='bold', fontsize=label_font)
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(-0.25, 1.25),
        ncol=3,
        frameon=False
    )

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches='tight')

    plt.close()
plot_specs = [
    {
        "data_dir": "./plot_data/HalfCheetahWithPos",
        "label": "Surrogate Obj",
        "runs": [1, 2, 3]
    },
    {
        "data_dir": "./plot_data/HalfCheetahWithPosPerturbed",
        "label": "Ours",
        "runs": [1, 2]
    },
    {
        "data_dir": "./plot_data/HalfCheetahWithPos",
        "label": "RCRL",
        "runs": [101, 102]
    },
    {
        "data_dir": "./plot_data/HalfCheetahWithPos",
        "label": "PD",
        "runs": [103, 104]
    }
]

plot_mean_std_multiple_runs(
    plot_specs,
    save=True,
    base_filename="train_plots/HC_training_all",
    smooth_window=80
)
