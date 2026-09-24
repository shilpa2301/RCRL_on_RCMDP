import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth(data, window_size):
    """
    Smooth the data using a simple moving average.
    If the data is shorter than the window size, return the original data.
    """
    data = np.asarray(data)

    if window_size <= 1:
        return data

    if len(data) < window_size:
        return data

    smoothed_data = np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )
    return smoothed_data


# ============================================================
# Fixed color scheme for feasible vs infeasible comparison
# ============================================================
METHOD_COLORS = {
    "Infeasible": "#1f77b4",  # blue
    "Feasible": "#d62728",    # red
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
    fig_horizontal = plt.figure(figsize=(24, 4))
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
    fig_vertical = plt.figure(figsize=(10, 10))
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


def load_metric_for_runs(
    data_dir,
    runs,
    filename,
    max_episodes=None
):
    """
    Load one metric for multiple runs.

    Example:
        filename = "episode_rewards.npy"
        filename = "episode_max_costs.npy"
    """

    all_data = []

    for run in runs:
        run_dir = f"{data_dir}/run{run}"
        path = f"{run_dir}/{filename}"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        data = np.load(path)

        if max_episodes is not None:
            data = data[:max_episodes]

        all_data.append(data)

    min_length = min(len(x) for x in all_data)
    all_data = np.array([x[:min_length] for x in all_data])

    return all_data


def plot_mean_std_multiple_runs(
    plot_specs,
    save=False,
    base_filename="swimmer_sample_complexity_feasible_vs_infeasible",
    smooth_window=80,
    save_legends=True,
    max_episodes=8000,
    baseline_cost_threshold=0.1
):
    """
    Plot mean and standard deviation of rewards and max costs
    for feasible vs infeasible Swimmer runs.

    plot_specs format:
        [
            {
                "data_dir": "./plot_data/SwimmerWithPosPerturbed",
                "label": "Infeasible",
                "runs": [11, 12, 13],
            },
            {
                "data_dir": "./plot_data/SwimmerWithPosPerturbed",
                "label": "Feasible",
                "runs": [701, 702],
            },
        ]
    """

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

        all_rewards = load_metric_for_runs(
            data_dir=data_dir,
            runs=runs,
            filename="episode_rewards.npy",
            max_episodes=max_episodes
        )

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

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    plt.tick_params(width=10, length=25)

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

        all_max_costs = load_metric_for_runs(
            data_dir=data_dir,
            runs=runs,
            filename="episode_max_costs.npy",
            max_episodes=max_episodes
        )

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

    plt.axhline(
        y=baseline_cost_threshold,
        color=METHOD_COLORS["Baseline"],
        linestyle="--"
    )
    plt.ylim(-0.05, 3.0)
    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    plt.tick_params(width=10, length=25)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches="tight")

    plt.close()


# ============================================================
# Swimmer feasible vs infeasible plot specs
# ============================================================

BASE_DATA_DIR = "./plot_data/SwimmerWithPosPerturbed"

swimmer_feasible_infeasible_specs = [
    {
        "data_dir": BASE_DATA_DIR,
        "label": "Infeasible",
        "runs": [1, 2]
    },
    {
        "data_dir": BASE_DATA_DIR,
        "label": "Feasible",
        "runs": [ 702]
    },
]


plot_mean_std_multiple_runs(
    plot_specs=swimmer_feasible_infeasible_specs,
    save=True,
    base_filename="train_plots/swimmer_sample_complexity_feasible_vs_infeasible",
    smooth_window=80,
    save_legends=True,
    max_episodes=8000,
    baseline_cost_threshold=0.1
)
