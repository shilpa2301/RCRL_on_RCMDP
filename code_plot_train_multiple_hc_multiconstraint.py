import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Smoothing
# ============================================================
def smooth(data, window_size):
    """Smooth 1D data using a simple moving average."""
    data = np.asarray(data, dtype=np.float32).squeeze()

    if data.ndim != 1:
        raise ValueError(f"smooth() expected 1D data, got shape {data.shape}")

    if window_size <= 1:
        return data

    if len(data) < window_size:
        return data

    smoothed_data = np.convolve(
        data,
        np.ones(window_size, dtype=np.float32) / window_size,
        mode="valid",
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
    "Baseline": "black",
}


def get_method_color(label):
    return METHOD_COLORS.get(label, "#7f7f7f")


# ============================================================
# Cost names for HalfCheetahMultiConstraintCost
# ============================================================
COST_NAMES = [
    "Pitch (C1)",
    "Joint Speed (C2)",
]


# ============================================================
# Loading helpers
# ============================================================
def load_1d_array(path, max_episodes=None, name="array"):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32).squeeze()

    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D {name}, got shape {arr.shape} from file {path}"
        )

    if max_episodes is not None:
        arr = arr[:max_episodes]

    return arr


def load_cost_array(path, cost_dim=2, max_episodes=None, name="costs"):
    """
    Loads episode_costs.npy or episode_max_costs.npy.

    Expected shape:
        [episodes, cost_dim]

    If shape is [episodes], it converts to [episodes, 1].
    """
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D {name}, got shape {arr.shape} from file {path}"
        )

    if arr.shape[1] != cost_dim:
        raise ValueError(
            f"Expected cost_dim={cost_dim}, got shape {arr.shape} from file {path}"
        )

    if max_episodes is not None:
        arr = arr[:max_episodes]

    return arr


def check_file(path, required=True):
    if not os.path.exists(path):
        msg = f"Missing file: {path}"
        if required:
            raise FileNotFoundError(msg)
        else:
            print(f"[WARNING] {msg}")
            return False
    return True


# ============================================================
# Legend saving
# ============================================================
def save_separate_legends(
    plot_specs,
    base_filename,
    line_width=15,
    font_size=90,
):
    labels = [spec["label"] for spec in plot_specs]
    labels = list(dict.fromkeys(labels))

    handles = [
        Line2D(
            [0],
            [0],
            color=get_method_color(label),
            linewidth=line_width,
            label=label,
        )
        for label in labels
    ]

    # Horizontal legend
    fig_horizontal = plt.figure(figsize=(36, 4))
    ax_horizontal = fig_horizontal.add_subplot(111)
    ax_horizontal.axis("off")

    ax_horizontal.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        fontsize=font_size,
    )

    fig_horizontal.savefig(
        f"{base_filename}_legend_horizontal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig_horizontal)

    # Vertical legend
    fig_vertical = plt.figure(figsize=(12, 16))
    ax_vertical = fig_vertical.add_subplot(111)
    ax_vertical.axis("off")

    ax_vertical.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=font_size,
    )

    fig_vertical.savefig(
        f"{base_filename}_legend_vertical.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig_vertical)


# ============================================================
# Main plotting function
# ============================================================
def plot_halfcheetah_multiconstraint_training(
    plot_specs,
    save=False,
    base_filename="halfcheetah_multiconstraint_training",
    smooth_window=10,
    save_legends=True,
    max_episodes=5500,
    cost_dim=2,
    persistent_eps=0.1,
):
    """
    Plot mean/std across multiple runs for HalfCheetah multi-constraint training.

    This version produces only 3 saved plots:

        1. {base_filename}_rewards.png
        2. {base_filename}_worst_max_costs.png
        3. {base_filename}_individual_constraints.png

    Required per run:
        episode_rewards.npy
        episode_max_costs.npy

    Notes:
        episode_max_costs.npy should have shape [episodes, cost_dim].
        Worst-case max cost is max over cost dimensions per episode.
    """

    plt.rcParams.update(
        {
            "font.size": 100,
            "lines.linewidth": 15,
            "font.weight": "bold",
        }
    )

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
            font_size=90,
        )

    # ============================================================
    # 1. Reward plot
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

            check_file(rewards_path)

            rewards = load_1d_array(
                rewards_path,
                max_episodes=max_episodes,
                name="episode_rewards",
            )

            all_rewards.append(rewards)

        min_length = min(len(r) for r in all_rewards)
        all_rewards = np.asarray([r[:min_length] for r in all_rewards])

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        smoothed_mean_rewards = smooth(mean_rewards, smooth_window)
        smoothed_std_rewards = smooth(std_rewards, smooth_window)
        smoothed_x = range(len(smoothed_mean_rewards))

        plt.plot(
            smoothed_x,
            smoothed_mean_rewards,
            label=label,
            color=color,
        )

        plt.fill_between(
            smoothed_x,
            smoothed_mean_rewards - smoothed_std_rewards,
            smoothed_mean_rewards + smoothed_std_rewards,
            color=color,
            alpha=0.2,
        )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Reward", fontweight="bold", fontsize=label_font)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches="tight")

    plt.close()

    # ============================================================
    # 2. Worst-case max cost plot
    #    worst per episode = max over constraint dimensions
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    plt.axhspan(
        persistent_eps,
        12.5,
        color="red",
        alpha=0.1,
    )

    plt.axhspan(
        -1,
        persistent_eps,
        color="blue",
        alpha=0.1,
    )

    for spec in plot_specs:
        data_dir = spec["data_dir"]
        label = spec["label"]
        runs = spec["runs"]
        color = get_method_color(label)

        all_worst_max_costs = []

        for run in runs:
            run_dir = f"{data_dir}/run{run}"
            max_costs_path = f"{run_dir}/episode_max_costs.npy"

            check_file(max_costs_path)

            max_costs = load_cost_array(
                max_costs_path,
                cost_dim=cost_dim,
                max_episodes=max_episodes,
                name="episode_max_costs",
            )

            # Worst constraint per episode
            worst_max_cost = np.max(max_costs, axis=1)

            all_worst_max_costs.append(worst_max_cost)

        min_length = min(len(c) for c in all_worst_max_costs)
        all_worst_max_costs = np.asarray(
            [c[:min_length] for c in all_worst_max_costs]
        )

        mean_worst_max_costs = np.mean(all_worst_max_costs, axis=0)
        std_worst_max_costs = np.std(all_worst_max_costs, axis=0)

        smoothed_mean = smooth(mean_worst_max_costs, smooth_window)
        smoothed_std = smooth(std_worst_max_costs, smooth_window)
        smoothed_x = range(len(smoothed_mean))

        plt.plot(
            smoothed_x,
            smoothed_mean,
            label=label,
            color=color,
        )

        plt.fill_between(
            smoothed_x,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=color,
            alpha=0.2,
        )

    plt.axhline(
        y=persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
    )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    if save:
        plt.savefig(f"{base_filename}_worst_max_costs.png", bbox_inches="tight")

    plt.close()

    # ============================================================
    # 3. Individual constraints in one image
    #    One subplot per constraint dimension
    # ============================================================
    # fig, axes = plt.subplots(
    #     cost_dim,
    #     1,
    #     figsize=(fig_size + 8, fig_size * cost_dim),
    #     sharex=True,
    # )

    # ============================================================
    # 3. Individual constraints in one horizontal image
    # ============================================================
    fig, axes = plt.subplots(
        1,
        cost_dim,
        figsize=(fig_size * cost_dim, fig_size),
        sharey=True,
    )


    if cost_dim == 1:
        axes = [axes]

    for ci in range(cost_dim):
        ax = axes[ci]

        ax.axhspan(
            persistent_eps,
            12.5,
            color="red",
            alpha=0.1,
        )

        ax.axhspan(
            -1,
            persistent_eps,
            color="blue",
            alpha=0.1,
        )

        for spec in plot_specs:
            data_dir = spec["data_dir"]
            label = spec["label"]
            runs = spec["runs"]
            color = get_method_color(label)

            all_cost_dim_values = []

            for run in runs:
                run_dir = f"{data_dir}/run{run}"
                max_costs_path = f"{run_dir}/episode_max_costs.npy"

                check_file(max_costs_path)

                max_costs = load_cost_array(
                    max_costs_path,
                    cost_dim=cost_dim,
                    max_episodes=max_episodes,
                    name="episode_max_costs",
                )

                all_cost_dim_values.append(max_costs[:, ci])

            min_length = min(len(c) for c in all_cost_dim_values)
            all_cost_dim_values = np.asarray(
                [c[:min_length] for c in all_cost_dim_values]
            )

            mean_ci = np.mean(all_cost_dim_values, axis=0)
            std_ci = np.std(all_cost_dim_values, axis=0)

            smoothed_mean = smooth(mean_ci, smooth_window)
            smoothed_std = smooth(std_ci, smooth_window)
            smoothed_x = range(len(smoothed_mean))

            ax.plot(
                smoothed_x,
                smoothed_mean,
                label=label,
                color=color,
            )

            ax.fill_between(
                smoothed_x,
                smoothed_mean - smoothed_std,
                smoothed_mean + smoothed_std,
                color=color,
                alpha=0.2,
            )

        ax.axhline(
            y=persistent_eps,
            color=METHOD_COLORS["Baseline"],
            linestyle="--",
        )

        cost_name = COST_NAMES[ci] if ci < len(COST_NAMES) else f"C{ci+1}"

        ax.set_ylabel(
            f"Max {cost_name}",
            fontweight="bold",
            fontsize=label_font-20,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(15)

        ax.tick_params(width=8, length=20)

    # axes[-1].set_xlabel(
    #     "Episode",
    #     fontweight="bold",
    #     fontsize=label_font,
    # )

    # plt.tight_layout()

    fig.text(
        0.5,
        0.03,
        "Episode",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=label_font - 20,
    )

    plt.subplots_adjust(
        bottom=0.18,
        wspace=0.25,
    )



    if save:
        plt.savefig(
            f"{base_filename}_individual_constraints_horizontal.png",
            bbox_inches="tight",
        )

    plt.close()



# ============================================================
# HalfCheetah Multi-Constraint Plot Specs
# ============================================================
# IMPORTANT:
# Your training wrapper saves into:
#
# ./plot_data/{args.env}/{MULTI_CONSTRAINT_COST_VERSION}/run{run}/
#
# Example:
# ./plot_data/HalfCheetahMultiConstraintCost/multi_constraint_pitch_joint_speed_v1/run1/
#
# Replace the version string below with your actual MULTI_CONSTRAINT_COST_VERSION.
# Check the folder name in plot_data if unsure.
# ============================================================

HALFCHEETAH_ENV_NAME = "HalfCheetahMultiConstraintCost"
HALFCHEETAH_ENV_NAME_PERTURBED = "HalfCheetahMultiConstraintCostPerturbed"

# Change this if your actual folder name is different.
# It must match MULTI_CONSTRAINT_COST_VERSION from half_cheetah_multi_constraint_cost.py.
HALFCHEETAH_COST_VERSION = "multi_constraint_pitch_joint_v1"
# HALFCHEETAH_COST_VERSION = "multi_constraint_pitch_joint_speed_v1"

HALFCHEETAH_DATA_DIR = f"./plot_data/{HALFCHEETAH_ENV_NAME}/{HALFCHEETAH_COST_VERSION}"
HALFCHEETAH_DATA_DIR_PERTURBED = f"./plot_data/{HALFCHEETAH_ENV_NAME_PERTURBED}/{HALFCHEETAH_COST_VERSION}"


halfcheetah_plot_specs = [
    {
        "data_dir": HALFCHEETAH_DATA_DIR,
        "label": "Surrogate Obj",
        "runs": [1, 2],
    },
    {
        "data_dir": HALFCHEETAH_DATA_DIR_PERTURBED,
        "label": "Ours",
        "runs": [1, 2],
    },
    {
        "data_dir": HALFCHEETAH_DATA_DIR,
        "label": "FAC",
        "runs": [501, 502],
    },
    {
        "data_dir": HALFCHEETAH_DATA_DIR,
        "label": "RCRL",
        "runs": [101, 102],
    },
]


plot_halfcheetah_multiconstraint_training(
    plot_specs=halfcheetah_plot_specs,
    save=True,
    base_filename="train_plots/halfcheetah_multiconstraint_training_all",
    smooth_window=80,
    save_legends=True,
    max_episodes=25000,
    cost_dim=2,
    persistent_eps=0.1,
)
