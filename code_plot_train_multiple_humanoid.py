import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )
    return smoothed_data


# ============================================================
# Fixed color scheme used across all experiment plots
# ============================================================
METHOD_COLORS = {
    "Ours": "#1f77b4",           # blue
    "Surrogate Obj": "#ff7f0e",  # orange
    "PD": "#2ca02c",             # green
    "FAC": "#d62728",            # red
    "RCRL": "#9467bd",           # purple
    "RESPO": "#8c564b",          # brown
    "Baseline": "black"
}


def get_method_color(label):
    """
    Return fixed color for known methods.
    If label is unknown, return gray.
    """
    return METHOD_COLORS.get(label, "#7f7f7f")


def save_separate_legends(
    plot_specs,
    base_filename,
    line_width=15,
    font_size=90
):
    """
    Save separate horizontal and vertical legends.
    Baseline is not included.
    """

    labels = [spec["label"] for spec in plot_specs]

    # Remove duplicate labels while preserving order.
    labels = list(dict.fromkeys(labels))

    handles = [
        Line2D(
            [0],
            [0],
            color=get_method_color(label),
            linewidth=line_width,
            label=label
        )
        for label in labels
    ]

    # ============================================================
    # Horizontal legend
    # ============================================================
    fig_horizontal = plt.figure(figsize=(36, 4))
    ax_horizontal = fig_horizontal.add_subplot(111)
    ax_horizontal.axis("off")

    ax_horizontal.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        fontsize=font_size
    )

    fig_horizontal.savefig(
        f"{base_filename}_legend_horizontal.png",
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig_horizontal)

    # ============================================================
    # Vertical legend
    # ============================================================
    fig_vertical = plt.figure(figsize=(12, 16))
    ax_vertical = fig_vertical.add_subplot(111)
    ax_vertical.axis("off")

    ax_vertical.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=font_size
    )

    fig_vertical.savefig(
        f"{base_filename}_legend_vertical.png",
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig_vertical)


def plot_mean_std_multiple_runs(
    plot_specs,
    save=False,
    base_filename="aggregated_plot",
    smooth_window=10,
    save_legends=True,
    max_episodes=5500
):
    """
    Plot mean and standard deviation of rewards and max costs from multiple methods.

    Parameters:
        plot_specs: list of dicts:
            {
                "data_dir": str,
                "label": str,
                "runs": list of int
            }

        save: bool
        base_filename: str
        smooth_window: int
        save_legends: bool
        max_episodes: int
    """

    # Same as the newer plotting code.
    # Change this if your Humanoid threshold should be 1.0 instead.
    baseline_cost_threshold = 1.0

    plt.rcParams.update({
        "font.size": 100,
        "lines.linewidth": 15,
        "font.weight": "bold"
    })

    fig_size = 28
    label_font = 130

    if save:
        save_dir = os.path.dirname(base_filename)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    if save and save_legends:
        save_separate_legends(
            plot_specs=plot_specs,
            base_filename=base_filename,
            line_width=15,
            font_size=90
        )

    # ============================================================
    # Plot Rewards without legend
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for spec in plot_specs:
        data_dir = spec["data_dir"]
        label = spec["label"]
        runs = spec["runs"]
        color = get_method_color(label)

        all_rewards = []

        for run in runs:
            run_dir = f"{data_dir}/run{run}"
            rewards_path = f"{run_dir}/episode_rewards.npy"

            rewards = np.load(rewards_path)
            all_rewards.append(rewards[:max_episodes])

        min_length = min(len(r) for r in all_rewards)
        all_rewards = np.array([r[:min_length] for r in all_rewards])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        smoothed_mean_rewards = smooth(mean_rewards, smooth_window)
        smoothed_std_rewards = smooth(std_rewards, smooth_window)

        smoothed_x = range(len(smoothed_mean_rewards))

        plt.plot(
            smoothed_x,
            smoothed_mean_rewards,
            label=label,
            color=color
        )

        plt.fill_between(
            smoothed_x,
            smoothed_mean_rewards - smoothed_std_rewards,
            smoothed_mean_rewards + smoothed_std_rewards,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Reward", fontweight="bold", fontsize=label_font)

    # No legend inside the plot.

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches="tight")

    plt.close()

    # ============================================================
    # Plot Max Costs without legend
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    plt.axhspan(
        baseline_cost_threshold,
        12.5,
        color="red",
        alpha=0.1
    )

    plt.axhspan(
        -1,
        baseline_cost_threshold,
        color="blue",
        alpha=0.1
    )

    for spec in plot_specs:
        data_dir = spec["data_dir"]
        label = spec["label"]
        runs = spec["runs"]
        color = get_method_color(label)

        all_max_costs = []

        for run in runs:
            run_dir = f"{data_dir}/run{run}"
            max_costs_path = f"{run_dir}/episode_max_costs.npy"

            max_costs = np.load(max_costs_path)
            all_max_costs.append(max_costs[:max_episodes])

        min_length = min(len(c) for c in all_max_costs)
        all_max_costs = np.array([c[:min_length] for c in all_max_costs])

        mean_max_costs = np.mean(all_max_costs, axis=0)
        std_max_costs = np.std(all_max_costs, axis=0)

        smoothed_mean_max_costs = smooth(mean_max_costs, smooth_window)
        smoothed_std_max_costs = smooth(std_max_costs, smooth_window)

        smoothed_x = range(len(smoothed_mean_max_costs))

        plt.plot(
            smoothed_x,
            smoothed_mean_max_costs,
            label=label,
            color=color
        )

        plt.fill_between(
            smoothed_x,
            smoothed_mean_max_costs - smoothed_std_max_costs,
            smoothed_mean_max_costs + smoothed_std_max_costs,
            color=color,
            alpha=0.2
        )

    # Baseline threshold line, but no legend label.
    plt.axhline(
        y=baseline_cost_threshold,
        color=METHOD_COLORS["Baseline"],
        linestyle="--"
    )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    # No legend inside the plot.

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches="tight")

    plt.close()


# ============================================================
# Humanoid plot specs
# Fill the missing paths and run numbers if needed.
# Known paths are taken from your original Humanoid code.
# ============================================================
humanoid_plot_specs = [
    {
        "data_dir": "./plot_data/HumanoidWithCost",
        "label": "Surrogate Obj",
        "runs": [1, 2]
    },
    {
        "data_dir": "./plot_data/HumanoidWithCostPerturbed",
        "label": "Ours",
        "runs": [1, 2]
    },

    # ========================================================
    # Fill these paths/runs if you have Humanoid versions
    # ========================================================
    {
        "data_dir": "./plot_data/HumanoidWithCost_PD_RESPO",
        "label": "RESPO",
        "runs": [1,2]
    },
    {
        "data_dir": "./plot_data/HumanoidWithCost",
        "label": "PD",
        "runs": [103, 104]
    },
    {
        "data_dir": "./plot_data/HumanoidWithCost",
        "label": "FAC",
        "runs": [501, 502]
    },
    {
        "data_dir": "./plot_data/HumanoidWithCost",
        "label": "RCRL",
        "runs": [101, 102]
    },
]


plot_mean_std_multiple_runs(
    plot_specs=humanoid_plot_specs,
    save=True,
    base_filename="train_plots/humanoid_training_all",
    smooth_window=80,
    save_legends=False,
    max_episodes=5500
)
