import os

if os.name == "nt":
    os.add_dll_directory(r"C:\Users\rinki\.mujoco\mujoco210\bin")
    os.add_dll_directory(r"C:\Users\rinki\miniconda3\envs\rpcrl_env\Library\bin")

os.environ["MUJOCO_PY_MUJOCO_PATH"] = r"C:\Users\rinki\.mujoco\mujoco210"

import torch
import numpy as np
from code_ipm_rcmdp_rcrl_max_humanoid import (
    Actor_Beta,
    Actor_Gaussian,
    Actor_Discrete,
    Critic,
    CostCritic,
    Robust_RCAC_NPG,
    Normalization,
    RunningMeanStd,
    RewardScaling,
)
import argparse
import pickle
import matplotlib.pyplot as plt
import os
import glob

from envs.humanoid import (
    HumanoidWithCostPerturbed,
    HumanoidWithCostPerturbedTest,
    HumanoidCMDPPerturbedTest,
)


def load_agent(args, save_path):
    """
    Load the trained agent from saved files.

    Args:
        args: Argument parser with required parameters.
        save_path: Base path where the models were saved.

    Returns:
        agent: Loaded Robust_RCAC_NPG agent with weights.
        state_norm: State normalization object.
        reward_scaling: Reward scaling object.
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


def test_agent_multiple_models(args, save_paths, env, num_episodes=100, is_cmdp=False):
    """
    Evaluate one model or an ensemble of models.

    For non-CMDP:
        max_cost = max returned cost over trajectory

    For CMDP:
        max_cost = sum(info["incremental_max_cost"] over trajectory)
    """

    rewards = []
    costs = []
    max_costs = []

    agents = []

    if isinstance(save_paths, str):
        save_paths = [save_paths]

    for save_path in save_paths:
        agent, state_norm, reward_scaling = load_agent(args, save_path)
        agents.append((agent, state_norm, reward_scaling))

    for episode in range(num_episodes):
        reset_out = env.reset()

        if isinstance(reset_out, tuple):
            state = reset_out[0]
        else:
            state = reset_out

        if args.use_state_norm:
            state = state_norm(state, update=False)

        total_reward = 0.0
        total_cost = 0.0

        if is_cmdp:
            max_cost = 0.0
        else:
            max_cost = float("-inf")

        done = False

        while not done:
            actions = []

            for agent, _, _ in agents:
                action = agent.evaluate(state)

                if agent.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * agent.max_action

                actions.append(action)

            mean_action = np.mean(actions, axis=0)

            next_state, reward, cost, truncated, terminated, info = env.step(mean_action)

            done = truncated or terminated

            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            total_reward += reward
            total_cost += cost

            if is_cmdp:
                max_cost += info.get("incremental_max_cost", 0.0)
            else:
                max_cost = max(max_cost, cost)

            state = next_state

        rewards.append(total_reward)
        costs.append(total_cost)
        max_costs.append(max_cost)

        if is_cmdp:
            print(
                f"Episode {episode + 1}: "
                f"Total Reward = {total_reward}, "
                f"CMDP Max Cost=sum(incremental_max_cost) = {max_cost}"
            )
        else:
            print(
                f"Episode {episode + 1}: "
                f"Total Reward = {total_reward}, "
                f"Max Cost = {max_cost}"
            )

    return rewards, costs, max_costs


def smooth(data, window_size):
    """
    Smooth the data using simple moving average.
    """

    data = np.asarray(data)

    if len(data) < window_size:
        return data

    smoothed_data = np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid",
    )

    return smoothed_data


def test_multiple_dirs(args, model_specs, perturbation_stds, num_episodes=100):
    """
    Test multiple Humanoid models across gravity perturbation stds.

    For each std:
        sample one shared gravity perturbation
        use it for all models

    model_specs format:
        [
            {
                "label": "Surrogate Obj(NP)",
                "model_path": "./models/HumanoidWithCost/run2/Best_RCAC",
                "env_type": "cost"
            },
            {
                "label": "Ours(P+R)",
                "model_path": "./models/HumanoidWithCostPerturbed/run1/Best_RCAC",
                "env_type": "cost"
            },
            {
                "label": "SO-CMDP",
                "model_path": "./models/HumanoidCMDP/run1/Best_RCAC",
                "env_type": "cmdp"
            }
        ]
    """

    results = {}

    for std in perturbation_stds:
        rng = np.random.default_rng(args.seed)

        shared_gravity_perturbation = float(rng.normal(0.0, std))

        print(
            f"\nShared gravity perturbation for std={std}: "
            f"{shared_gravity_perturbation}"
        )

        for spec in model_specs:
            label = spec["label"]
            save_path = spec["model_path"]
            env_type = spec.get("env_type", "cost")

            is_cmdp = env_type == "cmdp"

            if label not in results:
                results[label] = []

            print(f"\nTesting method: {label}, perturbation std = {std}")
            print(f"  Loading model: {save_path}")

            if is_cmdp:
                env = HumanoidCMDPPerturbedTest(
                    sigma_gravity=std,
                    shared_gravity_perturbation=shared_gravity_perturbation,
                    seed=args.seed,
                )
            else:
                env = HumanoidWithCostPerturbedTest(
                    sigma_gravity=std,
                    shared_gravity_perturbation=shared_gravity_perturbation,
                    seed=args.seed,
                )

            env.reset(seed=args.seed)
            env.action_space.seed(args.seed)

            args.max_action = float(env.action_space.high[0])
            args.state_dim = env.observation_space.shape[0]
            args.action_dim = env.action_space.shape[0]
            args.gravity_std = std

            rewards, costs, max_costs = test_agent_multiple_models(
                args,
                save_path,
                env,
                num_episodes=num_episodes,
                is_cmdp=is_cmdp,
            )

            results[label].append({
                "rewards": rewards,
                "costs": costs,
                "max_costs": max_costs,
            })

    return results


def save_legend(legend_elements, labels, filename, horizontal=True):
    """
    Save a separate legend image.
    """

    fig = plt.figure(figsize=(20, 5) if horizontal else (5, 20))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.legend(
        handles=legend_elements,
        labels=labels,
        loc="center",
        ncol=len(legend_elements) if horizontal else 1,
        frameon=False,
    )

    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_evaluation(
    args,
    results,
    labels,
    perturbation_stds,
    color_map=None,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10,
):
    """
    Plot evaluation results.

    Plots:
        1. Cumulative Reward
        2. Max Cost

    Total cost plot removed.
    """

    plt.rcParams.update({
        "font.size": 100,
        "lines.linewidth": 15,
        "font.weight": "bold",
    })

    fig_size = 28
    label_font = 130

    if color_map is None:
        color_map = {}

    def style_axes():
        plt.grid(False)

        plt.gca().spines["top"].set_linewidth(15)
        plt.gca().spines["right"].set_linewidth(15)
        plt.gca().spines["left"].set_linewidth(15)
        plt.gca().spines["bottom"].set_linewidth(15)

    def get_metric(label, std_index, metric_name):
        metric = np.asarray(results[label][std_index][metric_name])

        if len(metric) < smooth_window:
            smoothed_metric = metric
        else:
            smoothed_metric = smooth(metric, smooth_window)

        x = range(len(smoothed_metric))
        return x, smoothed_metric

    legend_elements = []
    legend_labels = []

    # ============================================================
    # Plot cumulative rewards
    # ============================================================
    plt.figure(figsize=(fig_size + 16, fig_size))

    for label in labels:
        for i, std in enumerate(perturbation_stds):
            x, rewards = get_metric(label, i, "rewards")

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} (std={std})"

            line, = plt.plot(
                x,
                rewards,
                label=plot_label,
                color=color_map.get(label, None),
            )

            legend_elements.append(line)
            legend_labels.append(plot_label)

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches="tight")

    plt.close()

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_rewards_legend_horizontal.png",
        horizontal=True,
    )

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_rewards_legend_vertical.png",
        horizontal=False,
    )

    # ============================================================
    # Plot max costs
    # ============================================================
    plt.figure(figsize=(fig_size + 14, fig_size))

    for label in labels:
        for i, std in enumerate(perturbation_stds):
            x, max_costs = get_metric(label, i, "max_costs")

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} (std={std})"

            plt.plot(
                x,
                max_costs,
                label=plot_label,
                color=color_map.get(label, None),
            )

    y_min, y_max = plt.gca().get_ylim()

    plt.axhline(
        y=args.persistent_eps,
        color="black",
        linestyle="--",
        linewidth=15,
        label="Baseline",
    )

    plt.axhspan(args.persistent_eps, y_max, color="red", alpha=0.1)
    plt.axhspan(y_min, args.persistent_eps, color="blue", alpha=0.1)

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")

    parser.add_argument(
        "--env",
        type=str,
        default="SwimmerWithPos",
        help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv",
    )
    parser.add_argument("--uncer_set", type=str, default="IPM", help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument(
        "--random_steps",
        type=int,
        default=int(25e3),
        help="Uniformly sample action within random steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(16e3),
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=1e2,
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument(
        "--policy_dist",
        type=str,
        default="Gaussian",
        help="Beta or Gaussian or Discrete",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--hidden_width",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the neural network",
    )
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")

    parser.add_argument(
        "--persistent_eps",
        type=float,
        default=1.0,
        help="Persistent Safety Perturbation",
    )

    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
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
    parser.add_argument("--lambda_", type=int, default=1.0)
    parser.add_argument("--beta", type=float, default=30000.0)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=150)
    parser.add_argument("--sigma_gravity", type=float, default=0.7)
    parser.add_argument("--lr_cost", type=float, default=1e-3)

    args = parser.parse_args()

    # ============================================================
    # Model specifications
    # ============================================================
    model_specs = [
        {
            "label": "Surrogate Obj(NP)",
            "model_path": "./models/HumanoidWithCost/run2/Best_RCAC",
            "env_type": "cost",
        },
        {
            "label": "Ours(P+R)",
            "model_path": "./models/HumanoidWithCostPerturbed/run1/Best_RCAC",
            "env_type": "cost",
        },

        # If you later want CMDP models, uncomment/adapt these:
        {
            "label": "SO-CMDP",
            "model_path": "./models/HumanoidCMDP/run1/Best_RCAC",
            "env_type": "cmdp",
        },
        {
            "label": "RPCRL-CMDP",
            "model_path": "./models/HumanoidCMDPPerturbed/run1/Best_RCAC",
            "env_type": "cmdp",
        },
    ]

    labels = [spec["label"] for spec in model_specs]

    # Fixed colors for each label
    color_map = {
        "Surrogate Obj(NP)": "tab:blue",
        "Ours(P+R)": "tab:orange",
        "SO-CMDP": "tab:purple",
        "RPCRL-CMDP": "tab:brown",
    }

    perturbation_stds = [2.0]

    results = test_multiple_dirs(
        args,
        model_specs,
        perturbation_stds,
        num_episodes=100,
    )

    plot_evaluation(
        args,
        results,
        labels,
        perturbation_stds,
        color_map=color_map,
        save=True,
        base_filename="plot_inference/humanoid_cmdp_comparison",
        smooth_window=20,
    )
