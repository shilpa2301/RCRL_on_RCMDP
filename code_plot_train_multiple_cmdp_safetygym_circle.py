import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def smooth(data, window_size):
    """
    Smooth data using moving average.
    If data is shorter than window_size, return data unchanged.
    """
    data = np.asarray(data, dtype=np.float32).reshape(-1)

    if window_size <= 1:
        return data

    if len(data) < window_size:
        return data

    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )


# ============================================================
# Fixed color scheme
# ============================================================
METHOD_COLORS = {
    "Ours": "#1f77b4",       # blue
    "CMDP": "#ff7f0e",       # orange
    "RCMDP": "#2ca02c",      # green
    "Baseline": "black",
}


def get_method_color(label):
    return METHOD_COLORS.get(label, "#7f7f7f")


def load_first_existing(paths):
    """
    Load the first existing npy file from a list of possible paths.
    """
    for path in paths:
        if os.path.exists(path):
            return np.load(path, allow_pickle=True), path

    raise FileNotFoundError(
        "None of these files exist:\n" + "\n".join(paths)
    )


def save_separate_legends(
    plot_specs,
    base_filename,
    line_width=15,
    font_size=90
):
    labels = [spec["label"] for spec in plot_specs]
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
        fontsize=font_size
    )

    fig_horizontal.savefig(
        f"{base_filename}_legend_horizontal.png",
        bbox_inches="tight",
        pad_inches=0.1
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
        fontsize=font_size
    )

    fig_vertical.savefig(
        f"{base_filename}_legend_vertical.png",
        bbox_inches="tight",
        pad_inches=0.1
    )
    plt.close(fig_vertical)


# ============================================================
# Reward loading
# ============================================================
def get_reward_metric_paths(spec, run):
    """
    Training reward path for all methods.
    """
    plot_data_dir = spec["plot_data_dir"]

    return [
        f"{plot_data_dir}/run{run}/episode_rewards.npy",
    ]


def collect_reward_runs(plot_specs, max_episodes=None):
    """
    Collect training episode rewards.

    Returns:
        dict[label] = np.array with shape [num_runs, min_length]
    """
    collected = {}

    for spec in plot_specs:
        label = spec["label"]
        runs = spec["runs"]

        all_data = []

        for run in runs:
            candidate_paths = get_reward_metric_paths(spec, run)

            data, used_path = load_first_existing(candidate_paths)

            data = np.asarray(data, dtype=np.float32).reshape(-1)

            if max_episodes is not None:
                data = data[:max_episodes]

            all_data.append(data)

            print(
                f"[Loaded Reward] {label} run{run}: {used_path}, "
                f"length={len(data)}"
            )

        min_length = min(len(x) for x in all_data)
        all_data = np.array([x[:min_length] for x in all_data])

        collected[label] = all_data

    return collected


# ============================================================
# Mixed max-cost loading
# ============================================================
def get_cost_metric_paths(spec, run):
    """
    Choose the correct cost file depending on the method.

    Important:
        Ours uses evaluate_max_costs.npy.
        CMDP/RCMDP use evaluate_costs.npy because their total cost corresponds
        to the max-cost constraint quantity in this experiment.
    """
    label = spec["label"]
    plot_data_dir = spec["plot_data_dir"]
    eval_data_dir = spec["eval_data_dir"]

    if label == "Ours":
        return [
            f"{eval_data_dir}/run{run}/evaluate_max_costs.npy",
            f"{plot_data_dir}/run{run}/evaluate_max_costs.npy",
        ]

    elif label in ["CMDP", "RCMDP"]:
        return [
            f"{eval_data_dir}/run{run}/evaluate_costs.npy",
            f"{plot_data_dir}/run{run}/evaluate_costs.npy",
        ]

    else:
        raise ValueError(
            f"Unknown label '{label}'. Please specify whether it should use "
            "evaluate_max_costs.npy or evaluate_costs.npy."
        )


def collect_mixed_cost_runs(plot_specs, max_evals=None):
    """
    Collect one unified cost metric:

        Ours  -> evaluate_max_costs.npy
        CMDP  -> evaluate_costs.npy
        RCMDP -> evaluate_costs.npy

    Returns:
        dict[label] = np.array with shape [num_runs, min_length]
    """
    collected = {}

    for spec in plot_specs:
        label = spec["label"]
        runs = spec["runs"]

        all_data = []

        for run in runs:
            candidate_paths = get_cost_metric_paths(spec, run)

            data, used_path = load_first_existing(candidate_paths)

            data = np.asarray(data, dtype=np.float32).reshape(-1)

            if max_evals is not None:
                data = data[:max_evals]

            all_data.append(data)

            print(
                f"[Loaded Cost] {label} run{run}: {used_path}, "
                f"length={len(data)}"
            )

        min_length = min(len(x) for x in all_data)
        all_data = np.array([x[:min_length] for x in all_data])

        collected[label] = all_data

    return collected


# ============================================================
# Reward plot
# ============================================================
def plot_sg_circle_rewards(
    plot_specs,
    base_filename="train_plots/SG_circle_training_all",
    smooth_window=80,
    max_episodes=5500,
    ylabel="Reward",
):
    """
    Generate SG Circle reward plot.

    Uses:
        plot_data/.../runX/episode_rewards.npy

    Output:
        {base_filename}_rewards.png
    """

    data_by_label = collect_reward_runs(
        plot_specs=plot_specs,
        max_episodes=max_episodes,
    )

    plt.figure(figsize=(36, 28))

    for spec in plot_specs:
        label = spec["label"]
        color = get_method_color(label)

        all_data = data_by_label[label]

        mean_data = np.mean(all_data, axis=0)
        std_data = np.std(all_data, axis=0)

        smoothed_mean = smooth(mean_data, smooth_window)
        smoothed_std = smooth(std_data, smooth_window)

        x = np.arange(len(smoothed_mean))

        plt.plot(
            x,
            smoothed_mean,
            label=label,
            color=color
        )

        plt.fill_between(
            x,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=color,
            alpha=0.2
        )

    plt.xlabel(
        "Episode",
        fontweight="bold",
        fontsize=130
    )

    plt.ylabel(
        ylabel,
        fontweight="bold",
        fontsize=130
    )

    plt.xticks(fontsize=90, fontweight="bold")
    plt.yticks(fontsize=90, fontweight="bold")

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    save_path = f"{base_filename}_rewards.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# Mixed max-cost plot
# ============================================================
def plot_sg_circle_mixed_max_cost(
    plot_specs,
    base_filename="train_plots/SG_circle_training_all",
    smooth_window=5,
    max_evals=None,
    persistent_eps=0.5,
    ylabel="Max Cost",
    y_upper_shade=None,
):
    """
    Generate one SG Circle cost plot.

    File mapping:
        Ours:
            evaluate_max_costs.npy

        CMDP:
            evaluate_costs.npy

        RCMDP:
            evaluate_costs.npy

    Output:
        {base_filename}_max_cost.png
    """

    data_by_label = collect_mixed_cost_runs(
        plot_specs=plot_specs,
        max_evals=max_evals,
    )

    plt.figure(figsize=(36, 28))

    # ============================================================
    # Safe / unsafe shaded regions
    # ============================================================
    if y_upper_shade is None:
        all_values = np.concatenate([
            arr.reshape(-1) for arr in data_by_label.values()
        ])
        y_upper_shade = max(float(np.nanmax(all_values)), persistent_eps + 1.0)

    plt.axhspan(
        persistent_eps,
        y_upper_shade,
        color="red",
        alpha=0.1
    )

    plt.axhspan(
        -1,
        persistent_eps,
        color="blue",
        alpha=0.1
    )

    # ============================================================
    # Plot each method
    # ============================================================
    for spec in plot_specs:
        label = spec["label"]
        color = get_method_color(label)

        all_data = data_by_label[label]

        mean_data = np.mean(all_data, axis=0)
        std_data = np.std(all_data, axis=0)

        smoothed_mean = smooth(mean_data, smooth_window)
        smoothed_std = smooth(std_data, smooth_window)

        x = np.arange(len(smoothed_mean))

        plt.plot(
            x,
            smoothed_mean,
            label=label,
            color=color
        )

        plt.fill_between(
            x,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=color,
            alpha=0.2
        )

    # ============================================================
    # Safety threshold
    # ============================================================
    plt.axhline(
        y=persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
        linewidth=12,
    )

    plt.xlabel(
        "Evaluation Checkpoint",
        fontweight="bold",
        fontsize=130
    )

    plt.ylabel(
        ylabel,
        fontweight="bold",
        fontsize=130
    )

    plt.xticks(fontsize=90, fontweight="bold")
    plt.yticks(fontsize=90, fontweight="bold")

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    save_path = f"{base_filename}_max_cost.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# Combined wrapper
# ============================================================
def plot_sg_circle_rewards_and_cost(
    plot_specs,
    base_filename="train_plots/SG_circle_training_all",
    reward_smooth_window=80,
    cost_smooth_window=5,
    max_episodes=5500,
    max_evals=None,
    persistent_eps=0.5,
    save_legends=True,
):
    """
    Generate:
        1. Reward plot
        2. One mixed max-cost plot
    """

    plt.rcParams.update({
        "font.size": 100,
        "lines.linewidth": 15,
        "font.weight": "bold"
    })

    save_dir = os.path.dirname(base_filename)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    if save_legends:
        save_separate_legends(
            plot_specs=plot_specs,
            base_filename=base_filename,
            line_width=15,
            font_size=90
        )

    plot_sg_circle_rewards(
        plot_specs=plot_specs,
        base_filename=base_filename,
        smooth_window=reward_smooth_window,
        max_episodes=max_episodes,
        ylabel="Reward",
    )

    plot_sg_circle_mixed_max_cost(
        plot_specs=plot_specs,
        base_filename=base_filename,
        smooth_window=cost_smooth_window,
        max_evals=max_evals,
        persistent_eps=persistent_eps,
        ylabel="Max Cost",
    )


# ============================================================
# SG Circle plot specs
# ============================================================
sg_circle_plot_specs = [
    {
        "label": "Ours",
        "plot_data_dir": "./plot_data/SafetyCarCircle2-v0",
        "eval_data_dir": "./data_train/SafetyCarCircle2-v0",
        "runs": [100, 101],
    },
    {
        "label": "CMDP",
        "plot_data_dir": "./plot_data/SafetyCarCircle2-v0",
        "eval_data_dir": "./data_train/SafetyCarCircle2-v0",
        "runs": [1, 2],
    },
    {
        "label": "RCMDP",
        "plot_data_dir": "./plot_data/SafetyCarCircle2-v0",
        "eval_data_dir": "./data_train/SafetyCarCircle2-v0",
        "runs": [10, 11],
    },
]


plot_sg_circle_rewards_and_cost(
    plot_specs=sg_circle_plot_specs,
    base_filename="train_plots/CMDP_SG_circle_training_all",
    reward_smooth_window=80,
    cost_smooth_window=5,
    save_legends=True,
    max_episodes=5500,
    max_evals=None,
    persistent_eps=0.5,
)
