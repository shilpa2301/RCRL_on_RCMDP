import os
import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from code_ipm_rcmdp_rcrl_max_swimmer import (
    Actor_Beta,
    Actor_Gaussian,
    Actor_Discrete,
    Critic,
    CostCritic,
    Robust_RCAC_NPG,
    Normalization,
    RunningMeanStd,
    RewardScaling
)

# Adjust this import if your file name is different.
from envs.swimmer import SwimmerWithCostPerturbedTest


# ============================================================
# Fixed color scheme shared with training plots
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
    Unknown labels get gray.
    """
    return METHOD_COLORS.get(label, "#7f7f7f")


def smooth(data, window_size):
    """Smooth data using a simple moving average."""
    if window_size <= 1 or len(data) < window_size:
        return np.asarray(data)

    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid"
    )


def load_agent(args, save_path):
    """
    Load trained agent from saved files.
    """
    agent = Robust_RCAC_NPG(args)
    state_norm = None
    reward_scaling = None

    actor_path = f"{save_path}_actor"
    rcritic_path = f"{save_path}_Rcritic"
    ccritic_path = f"{save_path}_Ccritic"

    agent.actor.load(actor_path)
    agent.Rcritic.load(rcritic_path)
    agent.Ccritic.load(ccritic_path)

    if args.use_state_norm:
        print("Loading state norm")
        with open(f"{save_path}_state_norm", "rb") as file1:
            state_norm = pickle.load(file1)
        print(state_norm.running_ms.mean, state_norm.running_ms.std)

    if args.use_reward_scaling:
        print("Loading reward scaling")
        with open(f"{save_path}_reward_scaling", "rb") as file2:
            reward_scaling = pickle.load(file2)

    print("Agent and normalization objects loaded successfully!")
    return agent, state_norm, reward_scaling


def safe_reset_env(env, seed=None):
    """
    Safely handle different reset formats:
        obs
        obs, info
        nested obs
    """
    if seed is not None:
        reset_out = env.reset(seed=seed)
    else:
        reset_out = env.reset()
    # print(reset_out)

    if isinstance(reset_out, tuple):
        state = reset_out[0]
    else:
        state = reset_out

    state = np.asarray(state)

    while state.ndim > 1:
        state = state[0]

    return state


def test_agent_single_model(
    args,
    save_path,
    env,
    num_episodes=100
):
    """
    Evaluate one saved Swimmer model.
    """

    rewards = []
    costs = []
    max_costs = []

    agent, state_norm, reward_scaling = load_agent(args, save_path)

    for episode in range(num_episodes):
        state = safe_reset_env(env)

        if args.use_state_norm:
            state = state_norm(state, update=False)

        total_reward = 0.0
        total_cost = 0.0
        max_cost = float("-inf")

        done = False

        while not done:
            action = agent.evaluate(state)

            if agent.policy_dist == "Beta":
                action = 2 * (action - 0.5) * agent.max_action

            step_out = env.step(action)

            next_state, reward, cost, truncated, terminated, info = step_out

            done = truncated or terminated

            next_state = np.asarray(next_state)
            while next_state.ndim > 1:
                next_state = next_state[0]

            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            total_reward += reward
            total_cost += cost
            max_cost = max(max_cost, cost)

            state = next_state

        rewards.append(total_reward)
        costs.append(total_cost)
        max_costs.append(max_cost)

        print(
            f"Episode {episode + 1}: "
            f"Total Reward = {total_reward}, "
            f"Max Cost = {max_cost}, "
            f"Total Cost = {total_cost}"
        )

    return rewards, costs, max_costs


def test_single_models(
    args,
    model_specs,
    perturbation_stds,
    num_episodes=100
):
    """
    Test one model per method across perturbation stds.

    Returns:
        results[label][std_index] = {
            "rewards": rewards,
            "costs": costs,
            "max_costs": max_costs
        }
    """

    results = {}

    for spec in model_specs:
        label = spec["label"]
        model_path = spec["model_path"]

        results[label] = []

        for std in perturbation_stds:
            print(f"\nTesting method: {label}, perturbation std = {std}")
            print(f"  Loading model: {model_path}")

            # For swimmer, std corresponds to viscosity perturbation.
            env = SwimmerWithCostPerturbedTest(
                sigma_viscosity=std,
                max_steps=1000,
                seed=args.seed
            )

            env.reset(seed=args.seed)
            env.action_space.seed(args.seed)

            args.max_action = float(env.action_space.high[0])
            args.state_dim = env.observation_space.shape[0]
            args.action_dim = env.action_space.shape[0]
            args.sigma_viscosity = std

            rewards, costs, max_costs = test_agent_single_model(
                args=args,
                save_path=model_path,
                env=env,
                num_episodes=num_episodes
            )

            results[label].append({
                "rewards": rewards,
                "costs": costs,
                "max_costs": max_costs
            })

    return results


def save_separate_legends(
    labels,
    base_filename,
    line_width=15,
    font_size=90
):
    """
    Save separate horizontal and vertical legends.
    Baseline is not included.
    """

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

    save_dir = os.path.dirname(base_filename)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

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


def plot_evaluation_single(
    args,
    results,
    labels,
    perturbation_stds,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10,
    save_legends=True
):
    """
    Plot evaluation results for one Swimmer model per method.

    Saves:
        {base_filename}_rewards.png
        {base_filename}_max_costs.png
        {base_filename}_legend_horizontal.png
        {base_filename}_legend_vertical.png

    Main plots contain no legends.
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
            labels=labels,
            base_filename=base_filename,
            line_width=15,
            font_size=90
        )

    def style_axes():
        plt.grid(False)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(15)

    def get_metric(label, std_index, metric_name):
        metric = np.array(results[label][std_index][metric_name])
        smoothed_metric = smooth(metric, smooth_window)
        x = range(len(smoothed_metric))
        return x, smoothed_metric

    # ============================================================
    # Plot Rewards without legend
    # ============================================================
    plt.figure(figsize=(fig_size + 16, fig_size))

    for label in labels:
        color = get_method_color(label)

        for std_index, std in enumerate(perturbation_stds):
            x, rewards = get_metric(label, std_index, "rewards")

            plt.plot(
                x,
                rewards,
                color=color,
                label=label
            )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        plt.savefig(
            f"{base_filename}_rewards.png",
            bbox_inches="tight"
        )

    plt.close()

    # ============================================================
    # Plot Max Costs without legend
    # ============================================================
    plt.figure(figsize=(fig_size + 14, fig_size))

    for label in labels:
        color = get_method_color(label)

        for std_index, std in enumerate(perturbation_stds):
            x, max_costs = get_metric(label, std_index, "max_costs")

            plt.plot(
                x,
                max_costs,
                color=color,
                label=label
            )

    y_min, y_max = plt.gca().get_ylim()

    plt.axhline(
        y=args.persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
        linewidth=15
    )

    plt.axhspan(
        args.persistent_eps,
        y_max,
        color="red",
        alpha=0.1
    )

    plt.axhspan(
        y_min,
        args.persistent_eps,
        color="blue",
        alpha=0.1
    )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        plt.savefig(
            f"{base_filename}_max_costs.png",
            bbox_inches="tight"
        )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")

    parser.add_argument("--env", type=str, default="SwimmerWithPos")
    parser.add_argument("--uncer_set", type=str, default="IPM")
    parser.add_argument("--next_steps", type=int, default=2)
    parser.add_argument("--random_steps", type=int, default=int(25e3))
    parser.add_argument("--max_train_steps", type=int, default=int(16e3))
    parser.add_argument("--evaluate_freq", type=float, default=1e2)
    parser.add_argument("--save_freq", type=int, default=20)

    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--hidden_width", type=int, default=64)

    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)

    parser.add_argument("--persistent_eps", type=float, default=1.0)

    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=False)

    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=float, default=True)
    parser.add_argument("--use_tanh", type=float, default=True)

    parser.add_argument("--adaptive_alpha", type=float, default=False)
    parser.add_argument("--weight_reg", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=4)

    parser.add_argument("--GAMMA", type=str, default="0")
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=30000.0)
    parser.add_argument("--run", type=int, default=1)

    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=150)

    # For swimmer perturbation.
    parser.add_argument("--sigma_viscosity", type=float, default=0.7)

    parser.add_argument("--lr_cost", type=float, default=1e-3)

    args = parser.parse_args()

    # ============================================================
    # Single Swimmer evaluation only.
    #
    # Based on your Swimmer training plot:
    #
    #   SwimmerWithPos:
    #       Surrogate Obj runs [1, 2]
    #
    #   SwimmerWithPosPerturbed:
    #       Ours runs [1, 2]
    #
    # If you also trained these methods, update paths/runs as needed:
    #   PD    -> run103
    #   FAC   -> run501
    #   RCRL  -> run101
    #   RESPO -> run1 from SwimmerWithPos_PD_RESPO
    #
    # If any model path does not exist, comment that block out.
    # ============================================================
    model_specs = [
        {
            "label": "Ours",
            "model_path": "./models/SwimmerWithPosPerturbed/run1/Best_RCAC"
        },
        {
            "label": "Surrogate Obj",
            "model_path": "./models/SwimmerWithPos/run2/Best_RCAC"
        },

        # Uncomment/adjust these if you have the trained checkpoints.

        {
            "label": "PD",
            "model_path": "./models/SwimmerWithPos/run103/Best_RCAC"
        },
        {
            "label": "FAC",
            "model_path": "./models/SwimmerWithPos/run501/Best_RCAC"
        },
        {
            "label": "RCRL",
            "model_path": "./models/SwimmerWithPos/run102/Best_RCAC"
        },
        {
            "label": "RESPO",
            "model_path": "./models/SwimmerWithPos_PD_RESPO/run1/Best_RCAC"
        },
    ]

    labels = [spec["label"] for spec in model_specs]

    # Only one evaluation perturbation.
    # For Swimmer, this is viscosity perturbation.
    perturbation_stds = [1.0]

    results = test_single_models(
        args=args,
        model_specs=model_specs,
        perturbation_stds=perturbation_stds,
        num_episodes=100
    )

    plot_evaluation_single(
        args=args,
        results=results,
        labels=labels,
        perturbation_stds=perturbation_stds,
        save=True,
        base_filename="plot_inference/swimmer_inference_all",
        smooth_window=20,
        save_legends=False
    )
