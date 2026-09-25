import os
import argparse
import pickle
import importlib
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import safety_gymnasium
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

# ============================================================
# Environment builders matching your training scripts
# ============================================================
from safety_gymnasium.safety_envs.safety_circle_margin import (
    make_env as make_env_ours,
)

from safety_gymnasium.safety_envs.safety_circle_margin_cmdp import (
    make_perturbed_env as make_env_cmdp_perturbed,
)


# ============================================================
# Fixed colors
# ============================================================
METHOD_COLORS = {
    "Ours": "#1f77b4",
    "RCRL": "#9467bd",
    "CMDP": "#ff7f0e",
    "RCMDP": "#2ca02c",
    "Baseline": "black",
}


def get_method_color(label):
    return METHOD_COLORS.get(label, "#7f7f7f")


# ============================================================
# YAML config
# ============================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_config_to_args(args, cfg):
    if cfg is None:
        return args

    for key, value in cfg.items():
        setattr(args, key, value)

    return args


# ============================================================
# Environment creation
# ============================================================
def make_task_env(
    env_id,
    env_type,
    terminate_on_collision=False,
    render_mode=None,
    safety_clearance=0.30,
    dense_cost_weight=0.01,
    cost_scale=1.0,
    sigma_gravity=0.7,
):
    """
    Evaluation envs:

    Ours:
        safety_circle_margin.make_env

    CMDP / RCMDP:
        safety_circle_margin_cmdp.make_perturbed_env

    This matches your current robust CMDP wrapper.
    """

    if "Circle" in env_id:
        agent = "Car"

        if env_type == "ours":
            env = make_env_ours(
                agent=agent,
                level=2,
                render_mode=render_mode,
                safety_clearance=safety_clearance,
            )

        elif env_type in ["cmdp", "rcmdp"]:
            env = make_env_cmdp_perturbed(
                agent=agent,
                level=2,
                render_mode=render_mode,
                safety_clearance=safety_clearance,
                dense_cost_weight=dense_cost_weight,
                cost_scale=cost_scale,
                sigma_gravity=sigma_gravity,
            )

        else:
            raise ValueError(f"Unknown env_type: {env_type}")

    else:
        env = safety_gymnasium.make(env_id, render_mode=render_mode)

    if terminate_on_collision:
        env = TerminateOnCollisionWrapper(env)

    return env


# ============================================================
# Dynamic import
# ============================================================
def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ============================================================
# Set env dimensions before agent construction
# ============================================================
def set_env_dims_on_args(args, spec):
    tmp_env = make_task_env(
        env_id=args.env_id,
        env_type=spec["env_type"],
        terminate_on_collision=args.terminate_on_collision,
        render_mode=args.render_mode,
        safety_clearance=args.safety_clearance,
        dense_cost_weight=args.dense_cost_weight,
        cost_scale=args.cost_scale,
        sigma_gravity=args.sigma_gravity,
    )

    tmp_env.reset(seed=args.seed)
    tmp_env.action_space.seed(args.seed)

    args.state_dim = tmp_env.observation_space.shape[0]
    args.action_dim = tmp_env.action_space.shape[0]
    args.max_action = float(tmp_env.action_space.high[0])

    if hasattr(tmp_env, "max_steps"):
        args.max_episode_steps = tmp_env.max_steps
    elif (
        hasattr(tmp_env, "spec")
        and tmp_env.spec is not None
        and tmp_env.spec.max_episode_steps is not None
    ):
        args.max_episode_steps = tmp_env.spec.max_episode_steps
    else:
        args.max_episode_steps = getattr(args, "max_episode_steps", 1000)

    print(
        f"[Env dims] {spec['label']} | "
        f"state_dim={args.state_dim}, "
        f"action_dim={args.action_dim}, "
        f"max_action={args.max_action}, "
        f"max_episode_steps={args.max_episode_steps}"
    )

    tmp_env.close()

    return args


# ============================================================
# Load trained agent
# ============================================================
def load_agent(args, spec):
    args = set_env_dims_on_args(args, spec)

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
            with open(state_norm_path, "rb") as f:
                state_norm = pickle.load(f)

            print(f"[Loaded] state_norm: {state_norm_path}")
        else:
            print(f"[Warning] use_state_norm=True but missing {state_norm_path}")

    if getattr(args, "use_reward_scaling", False):
        reward_scaling_path = f"{save_path}_reward_scaling"

        if os.path.exists(reward_scaling_path):
            with open(reward_scaling_path, "rb") as f:
                reward_scaling = pickle.load(f)

            print(f"[Loaded] reward_scaling: {reward_scaling_path}")
        else:
            print(f"[Warning] use_reward_scaling=True but missing {reward_scaling_path}")

    print(f"[Loaded Agent] {spec['label']} from {save_path}")

    return agent, state_norm, reward_scaling


# ============================================================
# Step parser
# ============================================================
def parse_step_output(step_out):
    if len(step_out) == 6:
        next_state, reward, cost, terminated, truncated, info = step_out
        done = terminated or truncated
        return next_state, reward, cost, done, info

    if len(step_out) == 5:
        next_state, reward, terminated, truncated, info = step_out
        cost = info.get("cost", 0.0)
        done = terminated or truncated
        return next_state, reward, cost, done, info

    raise ValueError(f"Unexpected env.step output length: {len(step_out)}")


# ============================================================
# Cost extractors
# ============================================================
def get_ours_step_cost(cost, info):
    """
    Ours uses:
        max over continuous_cost
    """

    if info is None:
        return float(cost)

    if "continuous_cost" in info:
        return float(info["continuous_cost"])

    if "cost" in info:
        return float(info["cost"])

    return float(cost)


def get_cmdp_incremental_cost(info):
    """
    CMDP / RCMDP use:
        sum over incremental_max_cost
    """

    if info is None:
        return 0.0

    return float(info.get("incremental_max_cost", 0.0))


# ============================================================
# Evaluate one model
# ============================================================
def evaluate_one_model(args, spec, num_episodes=100):
    label = spec["label"]
    env_type = spec["env_type"]

    agent, state_norm, reward_scaling = load_agent(args, spec)

    env = make_task_env(
        env_id=args.env_id,
        env_type=env_type,
        terminate_on_collision=args.terminate_on_collision,
        render_mode=args.render_mode,
        safety_clearance=args.safety_clearance,
        dense_cost_weight=args.dense_cost_weight,
        cost_scale=args.cost_scale,
        sigma_gravity=args.sigma_gravity,
    )

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

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
            episode_plotted_cost = float("-inf")
        else:
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
                step_cost = get_ours_step_cost(cost, info)
                episode_total_cost += step_cost
                episode_plotted_cost = max(episode_plotted_cost, step_cost)

            else:
                inc_cost = get_cmdp_incremental_cost(info)
                episode_total_cost += inc_cost
                episode_plotted_cost += inc_cost

            state = next_state

        rewards.append(episode_reward)
        total_costs.append(episode_total_cost)
        plotted_costs.append(episode_plotted_cost)

        print(
            f"[{label}] Episode {episode + 1}/{num_episodes} | "
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
# Evaluate all
# ============================================================
def evaluate_all_models(args, model_specs, num_episodes=100):
    results = {}

    for spec in model_specs:
        label = spec["label"]

        print("\n" + "=" * 80)
        print(f"Evaluating: {label}")
        print(f"Module: {spec['module']}")
        print(f"Model: {spec['model_path']}")
        print(f"Env type: {spec['env_type']}")
        print("=" * 80)

        results[label] = evaluate_one_model(
            args=args,
            spec=spec,
            num_episodes=num_episodes,
        )

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
# Legend
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
    save=True,
    base_filename="plot_inference/SG_circle_comparison_CMDP",
    smooth_window=10,
    save_legends=True,
):
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

    # ============================================================
    # Reward plot
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for label in labels:
        rewards = smooth_curve(results[label]["rewards"], smooth_window)
        x = np.arange(len(rewards))

        plt.plot(
            x,
            rewards,
            color=get_method_color(label),
            label=label,
        )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight="bold", fontsize=label_font)

    style_axes()

    if save:
        path = f"{base_filename}_rewards.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[Saved] {path}")

    plt.close()

    # ============================================================
    # Mixed cost plot
    #
    # Ours:
    #   max continuous_cost
    #
    # CMDP / RCMDP:
    #   total incremental_max_cost
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    all_cost_values = []

    for label in labels:
        raw_cost = np.asarray(results[label]["plotted_costs"], dtype=np.float32)
        all_cost_values.append(raw_cost)

        cost = smooth_curve(raw_cost, smooth_window)
        x = np.arange(len(cost))

        plt.plot(
            x,
            cost,
            color=get_method_color(label),
            label=label,
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
        path = f"{base_filename}_max_costs.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"[Saved] {path}")

    plt.close()


# ============================================================
# Save arrays
# ============================================================
def save_results_np(results, labels, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    for label in labels:
        safe_label = label.replace("/", "_").replace(" ", "_")

        np.save(
            f"{base_dir}/{safe_label}_rewards.npy",
            np.asarray(results[label]["rewards"], dtype=np.float32),
        )

        np.save(
            f"{base_dir}/{safe_label}_costs.npy",
            np.asarray(results[label]["costs"], dtype=np.float32),
        )

        np.save(
            f"{base_dir}/{safe_label}_plotted_costs.npy",
            np.asarray(results[label]["plotted_costs"], dtype=np.float32),
        )

    print(f"[Saved raw arrays] {base_dir}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate SG Circle CMDP Comparison")

    parser.add_argument(
        "--config",
        type=str,
        default="/project/ag2682/sm3934/RCRL_on_RCMDP/envs/env_configs/safety_gym_circle.yaml",
        help="Path to config YAML file.",
    )

    # Environment
    parser.add_argument("--env_id", type=str, default="SafetyCarCircle2-v0")
    parser.add_argument("--terminate_on_collision", type=bool, default=False)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--safety_clearance", type=float, default=0.30)
    parser.add_argument("--dense_cost_weight", type=float, default=0.01)
    parser.add_argument("--cost_scale", type=float, default=1.0)
    parser.add_argument("--sigma_gravity", type=float, default=1.5)

    # Evaluation
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--smooth_window", type=int, default=30)
    parser.add_argument("--persistent_eps", type=float, default=0.5)

    # Agent constructor args
    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=5e-3)
    parser.add_argument("--lr_cost", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
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
    parser.add_argument("--lambda_", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=3e4)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=1000)

    # Robust CMDP constructor args
    parser.add_argument("--m_lip_weight", type=float, default=0.01)
    parser.add_argument("--use_input_lip_reg", type=bool, default=True)

    parser.add_argument(
        "--base_filename",
        type=str,
        default="plot_inference/SG_circle_comparison_CMDP",
    )

    args = parser.parse_args()
    args.sigma_gravity = 2.0
    cfg = load_config(args.config)
    args = apply_config_to_args(args, cfg)

    print("=" * 80)
    print("Evaluation config")
    print("=" * 80)
    print("env_id =", args.env_id)
    print("seed =", args.seed)
    print("persistent_eps =", args.persistent_eps)
    print("safety_clearance =", args.safety_clearance)
    print("dense_cost_weight =", args.dense_cost_weight)
    print("cost_scale =", args.cost_scale)
    print("sigma_gravity =", args.sigma_gravity)
    print("num_episodes =", args.num_episodes)
    print("=" * 80)

    # ============================================================
    # Model specs
    # ============================================================
    model_specs = [
        {
            "label": "Ours",
            "module": "code_ipm_rcmdp_rcrl_max_safety_gym_rpcrl",
            "agent_class": "RPCRL",
            "model_path": "./models/SafetyCarCircle2-v0/run100/Best_RCAC",
            "env_type": "ours",
        },
        {
            "label": "RCRL",
            "module": "code_ipm_rcmdp_rcrl_max_safety_gym_RCRL",
            "agent_class": "RCRL",
            "model_path": "./models/SafetyCarCircle2-v0/run110/Best_RCAC",
            "env_type": "ours",
        },
        {
            "label": "CMDP",
            "module": "code_ipm_rcmdp_rcrl_max_safety_gym_cmdp",
            "agent_class": "PrimalDual",
            "model_path": "./models/SafetyCarCircle2-v0/run2/Best_RCAC",
            "env_type": "cmdp",
        },
        {
            "label": "RCMDP",
            "module": "code_ipm_rcmdp_rcrl_max_safety_gym_cmdp_robust",
            "agent_class": "PrimalDual",
            "model_path": "./models/SafetyCarCircle2-v0/run10/Best_RCAC",
            "env_type": "rcmdp",
        },
    ]

    labels = [spec["label"] for spec in model_specs]

    results = evaluate_all_models(
        args=args,
        model_specs=model_specs,
        num_episodes=args.num_episodes,
    )

    save_results_np(
        results=results,
        labels=labels,
        base_dir="plot_inference/SG_circle_comparison_CMDP_RCRL",
    )

    plot_evaluation(
        args=args,
        results=results,
        labels=labels,
        save=True,
        base_filename=args.base_filename,
        smooth_window=args.smooth_window,
        save_legends=True,
    )
