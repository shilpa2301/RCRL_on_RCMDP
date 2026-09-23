import os
import argparse
import pickle
import importlib

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from envs.humanoid import (
    HumanoidWithCostPerturbedTest,
    HumanoidCMDPPerturbedTest,
)


# ============================================================
# Fixed SG-style colors
# ============================================================
METHOD_COLORS = {
    "Ours": "#1f77b4",
    "CMDP": "#ff7f0e",
    "RCMDP": "#2ca02c",
    "Baseline": "black",
}


def get_method_color(label):
    return METHOD_COLORS.get(label, "#7f7f7f")


# ============================================================
# Dynamic import
# ============================================================
def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ============================================================
# Environment creation
# ============================================================
def make_eval_env(args, env_type, std, shared_gravity_perturbation):
    """
    Humanoid evaluation env.

    env_type:
        "ours"  -> HumanoidWithCostPerturbedTest
        "cmdp"  -> HumanoidCMDPPerturbedTest
        "rcmdp" -> HumanoidCMDPPerturbedTest
    """

    if env_type == "ours":
        env = HumanoidWithCostPerturbedTest(
            sigma_gravity=std,
            shared_gravity_perturbation=shared_gravity_perturbation,
            seed=args.seed,
        )

    elif env_type in ["cmdp", "rcmdp"]:
        env = HumanoidCMDPPerturbedTest(
            sigma_gravity=std,
            shared_gravity_perturbation=shared_gravity_perturbation,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    return env


# ============================================================
# Set env dimensions
# ============================================================
def set_env_dims_on_args(args, env):
    args.max_action = float(env.action_space.high[0])
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    if hasattr(env, "max_episode_steps"):
        args.max_episode_steps = env.max_episode_steps
    elif hasattr(env, "_max_episode_steps"):
        args.max_episode_steps = env._max_episode_steps
    else:
        args.max_episode_steps = getattr(args, "max_episode_steps", 1000)

    print(
        f"state_dim={args.state_dim}, "
        f"action_dim={args.action_dim}, "
        f"max_action={args.max_action}, "
        f"max_episode_steps={args.max_episode_steps}"
    )

    return args


# ============================================================
# Load agent
# ============================================================
def load_agent(args, spec):
    """
    spec:
        {
            "label": "Ours",
            "module": "code_ipm_rcmdp_rcrl_max_humanoid",
            "agent_class": "Robust_RCAC_NPG",
            "model_path": "./models/HumanoidWithCostPerturbed/run1/Best_RCAC",
            "env_type": "ours"
        }
    """

    AgentClass = import_class(spec["module"], spec["agent_class"])

    agent = AgentClass(args)

    save_path = spec["model_path"]

    actor_path = f"{save_path}_actor"
    rcritic_path = f"{save_path}_Rcritic"
    ccritic_path = f"{save_path}_Ccritic"

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Missing actor file: {actor_path}")

    if not os.path.exists(rcritic_path):
        raise FileNotFoundError(f"Missing Rcritic file: {rcritic_path}")

    if not os.path.exists(ccritic_path):
        raise FileNotFoundError(f"Missing Ccritic file: {ccritic_path}")

    device = getattr(
        agent,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    agent.actor.load(actor_path, device=device)
    agent.Rcritic.load(rcritic_path, device=device)
    agent.Ccritic.load(ccritic_path, device=device)

    state_norm = None
    reward_scaling = None

    if getattr(args, "use_state_norm", False):
        state_norm_path = f"{save_path}_state_norm"

        if os.path.exists(state_norm_path):
            print(f"[Loading state norm] {state_norm_path}")
            with open(state_norm_path, "rb") as f:
                state_norm = pickle.load(f)
        else:
            print(f"[Warning] Missing state norm: {state_norm_path}")

    if getattr(args, "use_reward_scaling", False):
        reward_scaling_path = f"{save_path}_reward_scaling"

        if os.path.exists(reward_scaling_path):
            print(f"[Loading reward scaling] {reward_scaling_path}")
            with open(reward_scaling_path, "rb") as f:
                reward_scaling = pickle.load(f)
        else:
            print(f"[Warning] Missing reward scaling: {reward_scaling_path}")

    print(f"[Loaded Agent] {spec['label']} from {save_path}")

    return agent, state_norm, reward_scaling


# ============================================================
# Step parser
# ============================================================
def parse_step_output(step_out):
    """
    Expected Humanoid format:
        next_state, reward, cost, truncated, terminated, info

    Some Gymnasium envs use:
        next_state, reward, cost, terminated, truncated, info
    """

    if len(step_out) != 6:
        raise ValueError(f"Unexpected step output length: {len(step_out)}")

    next_state, reward, cost, flag1, flag2, info = step_out

    # Your old script used:
    # next_state, reward, cost, truncated, terminated, info
    truncated = flag1
    terminated = flag2

    done = truncated or terminated

    return next_state, reward, cost, done, info


# ============================================================
# Evaluate one model
# ============================================================
def evaluate_one_model(
    args,
    spec,
    std,
    shared_gravity_perturbation,
    num_episodes=100,
):
    label = spec["label"]
    env_type = spec["env_type"]

    env = make_eval_env(
        args=args,
        env_type=env_type,
        std=std,
        shared_gravity_perturbation=shared_gravity_perturbation,
    )

    args = set_env_dims_on_args(args, env)
    args.gravity_std = std

    agent, state_norm, reward_scaling = load_agent(args, spec)

    rewards = []
    total_costs = []
    plotted_costs = []

    for episode in range(num_episodes):
        reset_out = env.reset()

        if isinstance(reset_out, tuple):
            state = reset_out[0]
        else:
            state = reset_out

        state = np.asarray(state)

        if getattr(args, "use_state_norm", False) and state_norm is not None:
            state = state_norm(state, update=False)

        done = False

        episode_reward = 0.0
        episode_total_cost = 0.0

        if env_type == "ours":
            # Ours: max over step costs.
            episode_plotted_cost = float("-inf")
        else:
            # CMDP / RCMDP: sum incremental_max_cost.
            episode_plotted_cost = 0.0

        while not done:
            action = agent.evaluate(state)

            if agent.policy_dist == "Beta":
                action = 2.0 * (action - 0.5) * agent.max_action

            step_out = env.step(action)
            next_state, reward, cost, done, info = parse_step_output(step_out)

            next_state = np.asarray(next_state)

            if getattr(args, "use_state_norm", False) and state_norm is not None:
                next_state = state_norm(next_state, update=False)

            episode_reward += float(reward)

            if env_type == "ours":
                step_cost = float(cost)

                episode_total_cost += step_cost
                episode_plotted_cost = max(episode_plotted_cost, step_cost)

            else:
                incremental_max_cost = float(info.get("incremental_max_cost", 0.0))

                episode_total_cost += incremental_max_cost
                episode_plotted_cost += incremental_max_cost

            state = next_state

        rewards.append(episode_reward)
        total_costs.append(episode_total_cost)
        plotted_costs.append(episode_plotted_cost)

        print(
            f"[{label}] std={std} | Episode {episode + 1}/{num_episodes} | "
            f"Reward={episode_reward:.4f} | "
            f"TotalCost={episode_total_cost:.4f} | "
            f"PlottedCost={episode_plotted_cost:.4f}"
        )

    env.close()

    return {
        "rewards": rewards,
        "costs": total_costs,
        "plotted_costs": plotted_costs,
    }


# ============================================================
# Evaluate all methods across perturbation stds
# ============================================================
def evaluate_all_models(args, model_specs, perturbation_stds, num_episodes=100):
    results = {}

    for std in perturbation_stds:
        rng = np.random.default_rng(args.seed)
        shared_gravity_perturbation = float(rng.normal(0.0, std))

        print("\n" + "=" * 80)
        print(f"Perturbation std = {std}")
        print(f"Shared gravity perturbation = {shared_gravity_perturbation}")
        print("=" * 80)

        for spec in model_specs:
            label = spec["label"]

            if label not in results:
                results[label] = []

            print("\n" + "-" * 80)
            print(f"Evaluating: {label}")
            print(f"Module: {spec['module']}")
            print(f"Model: {spec['model_path']}")
            print(f"Env type: {spec['env_type']}")
            print("-" * 80)

            out = evaluate_one_model(
                args=args,
                spec=spec,
                std=std,
                shared_gravity_perturbation=shared_gravity_perturbation,
                num_episodes=num_episodes,
            )

            results[label].append(out)

    return results


# ============================================================
# Smoothing
# ============================================================
def smooth_curve(data, window_size):
    data = np.asarray(data, dtype=np.float32).reshape(-1)

    if window_size <= 1:
        return data

    if len(data) < window_size:
        return data

    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode="valid",
    )


# ============================================================
# Save separate legends
# ============================================================
def save_separate_legends(labels, base_filename, line_width=15, font_size=90):
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
# Plot evaluation
# ============================================================
def plot_evaluation(
    args,
    results,
    labels,
    perturbation_stds,
    save=True,
    base_filename="plot_inference/humanoid_comparison_cmdp",
    smooth_window=10,
    save_legends=True,
):
    """
    Plots:
        1. Cumulative Reward
        2. Max Cost / CMDP accumulated incremental max cost

    Cost convention:
        Ours:
            max step cost

        CMDP:
            sum incremental_max_cost

        RCMDP:
            sum incremental_max_cost
    """

    plt.rcParams.update({
        "font.size": 100,
        "lines.linewidth": 15,
        "font.weight": "bold",
    })

    fig_size = 28
    label_font = 130

    save_dir = os.path.dirname(base_filename)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    if save_legends:
        save_separate_legends(
            labels=labels,
            base_filename=base_filename,
            line_width=15,
            font_size=90,
        )

    def style_axes():
        for spine in plt.gca().spines.values():
            spine.set_linewidth(15)

        plt.xticks(fontsize=90, fontweight="bold")
        plt.yticks(fontsize=90, fontweight="bold")
        plt.grid(False)

    # ============================================================
    # Reward plot
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for label in labels:
        for i, std in enumerate(perturbation_stds):
            raw_rewards = np.asarray(results[label][i]["rewards"], dtype=np.float32)
            rewards = smooth_curve(raw_rewards, smooth_window)
            x = np.arange(len(rewards))

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} std={std}"

            plt.plot(
                x,
                rewards,
                label=plot_label,
                color=get_method_color(label),
            )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        reward_path = f"{base_filename}_rewards.png"
        plt.savefig(reward_path, bbox_inches="tight")
        print(f"[Saved] {reward_path}")

    plt.close()

    # ============================================================
    # Cost plot
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    all_cost_values = []

    for label in labels:
        for i, std in enumerate(perturbation_stds):
            raw_costs = np.asarray(results[label][i]["plotted_costs"], dtype=np.float32)
            all_cost_values.append(raw_costs)

            costs = smooth_curve(raw_costs, smooth_window)
            x = np.arange(len(costs))

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} std={std}"

            plt.plot(
                x,
                costs,
                label=plot_label,
                color=get_method_color(label),
            )

    all_cost_values = np.concatenate(all_cost_values)

    y_upper = max(float(np.nanmax(all_cost_values)), args.persistent_eps + 1.0)
    y_lower = min(float(np.nanmin(all_cost_values)), -1.0)

    plt.axhspan(
        args.persistent_eps,
        y_upper,
        color="red",
        alpha=0.1,
    )

    plt.axhspan(
        y_lower,
        args.persistent_eps,
        color="blue",
        alpha=0.1,
    )

    plt.axhline(
        y=args.persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
        linewidth=15,
    )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        cost_path = f"{base_filename}_max_costs.png"
        plt.savefig(cost_path, bbox_inches="tight")
        print(f"[Saved] {cost_path}")

    plt.close()


# ============================================================
# Save arrays
# ============================================================
def save_results_np(results, labels, perturbation_stds, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    for label in labels:
        safe_label = label.replace("/", "_").replace(" ", "_")

        for i, std in enumerate(perturbation_stds):
            safe_std = str(std).replace(".", "p")

            np.save(
                f"{base_dir}/{safe_label}_std{safe_std}_rewards.npy",
                np.asarray(results[label][i]["rewards"], dtype=np.float32),
            )

            np.save(
                f"{base_dir}/{safe_label}_std{safe_std}_costs.npy",
                np.asarray(results[label][i]["costs"], dtype=np.float32),
            )

            np.save(
                f"{base_dir}/{safe_label}_std{safe_std}_plotted_costs.npy",
                np.asarray(results[label][i]["plotted_costs"], dtype=np.float32),
            )

    print(f"[Saved raw arrays] {base_dir}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Humanoid 3 Methods")

    # Basic args
    parser.add_argument("--env", type=str, default="Humanoid")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--smooth_window", type=int, default=20)
    parser.add_argument("--sigma_gravity", type=float, default=2.0)

    parser.add_argument(
        "--perturbation_stds",
        type=float,
        nargs="+",
        default=[2.0],
        help="List of gravity perturbation stds.",
    )

    parser.add_argument(
        "--base_filename",
        type=str,
        default="plot_inference/humanoid_comparison_cmdp",
    )

    # Agent constructor args
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
    parser.add_argument("--lr_cost", type=float, default=1e-3)

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
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_tanh", type=bool, default=True)
    parser.add_argument("--adaptive_alpha", type=bool, default=False)
    parser.add_argument("--weight_reg", type=float, default=0.001)

    parser.add_argument("--GAMMA", type=str, default="0")
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=30000.0)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=150)
    parser.add_argument("--dense_cost_weight", type=float, default=0.01)
    parser.add_argument("--cost_scale", type=int, default=100)

    args = parser.parse_args()

    print("=" * 80)
    print("Humanoid evaluation config")
    print("=" * 80)
    print("seed =", args.seed)
    print("persistent_eps =", args.persistent_eps)
    print("perturbation_stds =", args.perturbation_stds)
    print("num_episodes =", args.num_episodes)
    print("=" * 80)

    # ============================================================
    # Three methods, SG-style labels/colors
    #
    # IMPORTANT:
    # Adjust module names and model paths if your filenames differ.
    # ============================================================
    model_specs = [
        {
            "label": "Ours",
            "module": "code_ipm_rcmdp_rcrl_max_humanoid",
            "agent_class": "Robust_RCAC_NPG",
            "model_path": "./models/HumanoidWithCostPerturbed/run1/Best_RCAC",
            "env_type": "ours",
        },
        {
            "label": "CMDP",
            "module": "code_ipm_rcmdp_rcrl_max_humanoid_RPCRL_CMDP",
            "agent_class": "PrimalDual",
            "model_path": "./models/HumanoidCMDP/run1/Best_RCAC",
            "env_type": "cmdp",
        },
        {
            "label": "RCMDP",
            "module": "code_ipm_rcmdp_rcrl_max_humanoid_RPCRL_CMDP_robust",
            "agent_class": "PrimalDual",
            "model_path": "./models/HumanoidCMDPPerturbed/run1/Best_RCAC",
            "env_type": "rcmdp",
        },
    ]

    labels = [spec["label"] for spec in model_specs]

    results = evaluate_all_models(
        args=args,
        model_specs=model_specs,
        perturbation_stds=args.perturbation_stds,
        num_episodes=args.num_episodes,
    )

    save_results_np(
        results=results,
        labels=labels,
        perturbation_stds=args.perturbation_stds,
        base_dir="plot_inference/humanoid_comparison_cmdp",
    )

    plot_evaluation(
        args=args,
        results=results,
        labels=labels,
        perturbation_stds=args.perturbation_stds,
        save=True,
        base_filename=args.base_filename,
        smooth_window=args.smooth_window,
        save_legends=True,
    )
