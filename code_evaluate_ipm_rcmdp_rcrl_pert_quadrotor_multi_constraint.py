# #shilpa Windows only
# import os

# if os.name == "nt":
#     os.add_dll_directory(r"C:\Users\rinki\.mujoco\mujoco210\bin")
#     os.add_dll_directory(r"C:\Users\rinki\miniconda3\envs\rpcrl_env\Library\bin")

# os.environ["MUJOCO_PY_MUJOCO_PATH"] = r"C:\Users\rinki\.mujoco\mujoco210"



import torch
import numpy as np
from code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint import (
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
import copy
import time

# from envs.cartpole import CartPolePerturbedEnv
# from envs.half_cheetah import HalfCheetahWithPos
# from envs.reacher import ReacherWithCost
# from envs.swimmer import SwimmerWithPos

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


# ============================================================
# Fixed color scheme: same as training plot script
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

# ============================================================
# Quadrotor cost names
# ============================================================
COST_NAMES = [
    "Rotor 1 Low Thrust (C1)",
    "Rotor 2 Low Thrust (C2)",
    "Rotor 1 High Thrust (C3)",
    "Rotor 2 High Thrust (C4)",
]


def get_method_color(label):
    """
    Return fixed color for known methods.
    If label is unknown, return gray.
    """
    return METHOD_COLORS.get(label, "#7f7f7f")


CONFIG_FACTORY = ConfigFactory()
CONFIG_FACTORY.parser.set_defaults(
    overrides=["./envs/env_configs/constrained_tracking_reset.yaml"]
)
config = CONFIG_FACTORY.merge()

CONFIG_FACTORY_EVAL = ConfigFactory()
CONFIG_FACTORY_EVAL.parser.set_defaults(
    overrides=["./envs/env_configs/constrained_tracking_eval.yaml"]
)
config_eval = CONFIG_FACTORY_EVAL.merge()


def load_agent(args, save_path, load_critics=False):
    """
    Load trained agent and optional normalization/scaling objects.

    For evaluation, actor is enough.
    Critics are optional because some checkpoints may have different cost_dim.
    """

    agent = Robust_RCAC_NPG(args)
    state_norm = None
    reward_scaling = None

    actor_path = f"{save_path}_actor"
    rcritic_path = f"{save_path}_Rcritic"
    ccritic_path = f"{save_path}_Ccritic"

    agent.actor.load(actor_path)

    if load_critics:
        try:
            agent.Rcritic.load(rcritic_path)
            agent.Ccritic.load(ccritic_path)
        except RuntimeError as e:
            print(
                f"[WARNING] Could not load critics for {save_path}. "
                f"Continuing with actor-only evaluation.\n{e}"
            )

    if args.use_state_norm:
        norm_path = f"{save_path}_state_norm"
        if os.path.exists(norm_path):
            print("Loading state norm")
            with open(norm_path, "rb") as file1:
                state_norm = pickle.load(file1)
            print(state_norm.running_ms.mean, state_norm.running_ms.std)
        else:
            print(f"[WARNING] State norm not found: {norm_path}")

    if args.use_reward_scaling:
        scaling_path = f"{save_path}_reward_scaling"
        if os.path.exists(scaling_path):
            print("Loading reward scaling")
            with open(scaling_path, "rb") as file2:
                reward_scaling = pickle.load(file2)
        else:
            print(f"[WARNING] Reward scaling not found: {scaling_path}")

    print(f"Agent actor loaded successfully from: {save_path}")
    return agent, state_norm, reward_scaling

def load_agent_specific_model(args, save_path, model_num, load_critics=False):
    """
    Load trained agent from a specific checkpoint number.

    Example:
        save_path = './models/quadrotor/run501/Best_RCAC'
        model_num = 9400

    Loads:
        ./models/quadrotor/run501/Best_RCAC_actor_9400
    """

    agent = Robust_RCAC_NPG(args)
    state_norm = None
    reward_scaling = None

    actor_path = f"{save_path}_actor_{str(model_num)}"
    rcritic_path = f"{save_path}_Rcritic_{str(model_num)}"
    ccritic_path = f"{save_path}_Ccritic_{str(model_num)}"

    agent.actor.load(actor_path)

    if load_critics:
        try:
            agent.Rcritic.load(rcritic_path)
            agent.Ccritic.load(ccritic_path)
        except RuntimeError as e:
            print(
                f"[WARNING] Could not load critics for {save_path}, checkpoint={model_num}. "
                f"Continuing with actor-only evaluation.\n{e}"
            )

    if args.use_state_norm:
        # First try checkpoint-specific normalization.
        norm_path_specific = f"{save_path}_state_norm_{str(model_num)}"
        norm_path_default = f"{save_path}_state_norm"

        if os.path.exists(norm_path_specific):
            norm_path = norm_path_specific
        else:
            norm_path = norm_path_default

        if os.path.exists(norm_path):
            print(f"Loading state norm: {norm_path}")
            with open(norm_path, "rb") as file1:
                state_norm = pickle.load(file1)
            print(state_norm.running_ms.mean, state_norm.running_ms.std)
        else:
            print(
                f"[WARNING] State norm not found: "
                f"{norm_path_specific} or {norm_path_default}"
            )

    if args.use_reward_scaling:
        scaling_path_specific = f"{save_path}_reward_scaling_{str(model_num)}"
        scaling_path_default = f"{save_path}_reward_scaling"

        if os.path.exists(scaling_path_specific):
            scaling_path = scaling_path_specific
        else:
            scaling_path = scaling_path_default

        if os.path.exists(scaling_path):
            print(f"Loading reward scaling: {scaling_path}")
            with open(scaling_path, "rb") as file2:
                reward_scaling = pickle.load(file2)
        else:
            print(
                f"[WARNING] Reward scaling not found: "
                f"{scaling_path_specific} or {scaling_path_default}"
            )

    print(
        f"Agent actor loaded successfully from: {save_path}, "
        f"checkpoint={model_num}"
    )
    return agent, state_norm, reward_scaling

def get_model_num_for_path(save_path, model_num_map=None, default_model_num=None):
    """
    Return checkpoint number for a given save_path.

    model_num_map examples:
        {
            "run501": 9400,
            "run101": 13600,
        }
    """

    if model_num_map is None:
        return default_model_num

    normalized_path = save_path.replace("\\", "/")

    for key, value in model_num_map.items():
        if key in normalized_path:
            return value

    return default_model_num


def extract_multicost(args, info):
    """
    Extract multi-dimensional cost exactly like trainer.

    Trainer logic:
        c = args.cost_scale * np.asarray(info["constraint_values"]).reshape(-1)
        c = np.maximum(c, 0.0)
    """

    if "constraint_values" not in info:
        raise KeyError(
            "Expected info['constraint_values'] from environment, "
            f"but info keys are: {list(info.keys())}"
        )

    c = args.cost_scale * np.asarray(
        info["constraint_values"], dtype=np.float32
    ).reshape(-1)

    c = np.maximum(c, 0.0)

    if c.shape[0] != args.cost_dim:
        raise ValueError(
            f"Expected cost_dim={args.cost_dim}, "
            f"but got cost shape={c.shape}, value={c}"
        )

    return c


def env_reset_compat(env, seed=None):
    """
    Handles both Gymnasium-style reset and custom safe-control-gym reset.
    """

    if seed is not None:
        out = env.reset(seed=seed)
    else:
        out = env.reset()

    if isinstance(out, tuple):
        # Gymnasium: obs, info
        # Some of your older code had env.reset()[0][0].
        # For quadrotor trainer, reset gives s, _.
        obs = out[0]
    else:
        obs = out

    # Handle accidental nested shape from some envs.
    obs = np.asarray(obs)

    if obs.ndim > 1:
        obs = obs.reshape(-1)

    return obs


def env_step_compat(env, action):
    """
    Handles both:
        s_, r, done, info
    and:
        s_, r, cost, truncated, terminated, info
    and Gymnasium:
        s_, r, terminated, truncated, info
    """

    out = env.step(action)

    if len(out) == 4:
        # safe-control-gym style in trainer:
        # s_, r, done, info
        s_, r, done, info = out
        return s_, r, done, info

    elif len(out) == 5:
        # Gymnasium style:
        # s_, r, terminated, truncated, info
        s_, r, terminated, truncated, info = out
        done = terminated or truncated
        return s_, r, done, info

    elif len(out) == 6:
        # Your older env style:
        # next_state, reward, cost, truncated, terminated, info
        s_, r, _unused_cost, truncated, terminated, info = out
        done = terminated or truncated
        return s_, r, done, info

    else:
        raise RuntimeError(f"Unsupported env.step output length: {len(out)}")


def test_agent_multiple_models(
    args,
    save_paths,
    env,
    model_num=None,
    model_num_map=None,
    num_episodes=100,
):

    """
    Test one model or an ensemble of models.

    Multi-constraint outputs:
        rewards: list of scalar episode rewards
        costs: list of vectors, each shape [cost_dim]
        max_costs: list of vectors, each shape [cost_dim]
        total_cost_sums: list of scalar sum over cost dimensions
        max_total_costs: list of scalar max over max-cost dimensions
    """

    rewards = []

    # Multi-constraint arrays per episode
    costs = []
    max_costs = []

    # Scalar summaries per episode
    total_cost_sums = []
    max_total_costs = []

    agents = []

    if isinstance(save_paths, str):
        save_paths = [save_paths]

    # Load all agents first
    #shilpa specific model

    # for save_path in save_paths:

    #         agent, state_norm, reward_scaling = load_agent(args, save_path)
    #         agents.append((agent, state_norm, reward_scaling))
    
   
    for save_path in save_paths:
        this_model_num = get_model_num_for_path(
            save_path,
            model_num_map=model_num_map,
            default_model_num=model_num,
        )

        if this_model_num is not None:
            agent, state_norm, reward_scaling = load_agent_specific_model(
                args,
                save_path,
                this_model_num,
                load_critics=False,
            )
        else:
            agent, state_norm, reward_scaling = load_agent(
                args,
                save_path,
                load_critics=False,
            )

        agents.append((agent, state_norm, reward_scaling))


    # Use first agent's normalization object for state preprocessing.
    # This assumes all ensemble models were trained with the same normalization.
    first_state_norm = agents[0][1]
    first_reward_scaling = agents[0][2]

    for episode in range(num_episodes):
        state = env_reset_compat(env)

        if args.use_state_norm and first_state_norm is not None:
            state = first_state_norm(state, update=False)

        total_reward = 0.0
        total_cost = np.zeros(args.cost_dim, dtype=np.float64)
        max_cost = np.full(args.cost_dim, -np.inf, dtype=np.float64)

        done = False

        while not done:
            actions = []

            for agent, _, _ in agents:
                action = agent.evaluate(state)

                if agent.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * agent.max_action

                actions.append(action)

            mean_action = np.mean(actions, axis=0)

            next_state, reward, done, info = env_step_compat(env, mean_action)
            time.sleep(5.0 / 240.0)

            c = extract_multicost(args, info)

            if args.use_state_norm and first_state_norm is not None:
                next_state = first_state_norm(next_state, update=False)

            if args.use_reward_scaling and first_reward_scaling is not None:
                reward = first_reward_scaling(reward, update=False)
                # In trainer, cost scaling is NOT passed through reward_scaling.
                # Keep cost unnormalized for reporting.

            total_reward += reward
            total_cost += c
            max_cost = np.maximum(max_cost, c)

            state = copy.deepcopy(next_state)

        total_cost_sum = float(np.sum(total_cost))
        max_total_cost = float(np.max(max_cost))

        rewards.append(total_reward)
        costs.append(total_cost.copy())
        max_costs.append(max_cost.copy())
        total_cost_sums.append(total_cost_sum)
        max_total_costs.append(max_total_cost)

        cost_str = " | ".join(
            [f"Total C{i+1}={total_cost[i]:.3f}" for i in range(args.cost_dim)]
        )
        max_cost_str = " | ".join(
            [f"Max C{i+1}={max_cost[i]:.3f}" for i in range(args.cost_dim)]
        )

        safe = np.all(max_cost <= args.persistent_eps)

        print(
            f"Episode {episode + 1}: "
            f"Reward={total_reward:.3f} | "
            f"{cost_str} | "
            f"Total Cost Sum={total_cost_sum:.3f} | "
            f"{max_cost_str} | "
            f"Max Total Cost={max_total_cost:.3f} | "
            f"Safe={safe}"
        )

    return rewards, costs, max_costs, total_cost_sums, max_total_costs


def smooth(data, window_size):
    """
    Smooth data using simple moving average.

    Supports 1D arrays only. For multi-cost arrays, call per dimension.
    """

    data = np.asarray(data)

    if len(data) < window_size:
        return data

    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def test_multiple_dirs(
    args,
    save_paths,
    labels,
    model_num=None,
    model_num_map=None,
    num_episodes=100,
):

    """
    Evaluate each model path or model ensemble.

    save_paths can contain:
        - str: single model
        - list[str]: ensemble averaged action model

    labels:
        method labels with same length as save_paths
    """

    results = {}

    for save_path, label in zip(save_paths, labels):
        print(f"\n============================================================")
        print(f"Testing method: {label}")
        print(f"Model path(s): {save_path}")
        print(f"============================================================\n")

        env = make("quadrotor", **config_eval.quadrotor_config)
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        args.max_action = float(env.action_space.high[0])
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]

        (
            rewards,
            costs,
            max_costs,
            total_cost_sums,
            max_total_costs,
        ) = test_agent_multiple_models(
            args,
            save_path,
            env,
            model_num=model_num,
            model_num_map=model_num_map,
            num_episodes=num_episodes,
        )

        results[label] = [
            {
                "rewards": rewards,
                "costs": costs,
                "max_costs": max_costs,
                "total_cost_sums": total_cost_sums,
                "max_total_costs": max_total_costs,
            }
        ]

        try:
            env.close()
        except Exception:
            pass

    return results



def plot_evaluation(
    args,
    results,
    save_paths,
    labels,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10,
):
    """
    Quadrotor multi-constraint evaluation plots.

    Creates 3 plots similar to training:

        1. {base_filename}_rewards.png
        2. {base_filename}_worst_max_costs.png
        3. {base_filename}_individual_constraints_horizontal.png

    Uses:
        results[label][0]["rewards"]
        results[label][0]["max_costs"]      # shape [episodes, cost_dim]

    Worst-case max cost:
        max over constraint dimensions per episode.
    """

    save_dir = os.path.dirname(base_filename)
    if save and save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 100,
            "lines.linewidth": 15,
            "font.weight": "bold",
        }
    )

    fig_size = 28
    label_font = 130

    legend_elements = []
    legend_labels = []

    # ============================================================
    # 1. Evaluation reward plot
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for save_path, label in zip(save_paths, labels):
        key = label

        rewards = np.asarray(results[key][0]["rewards"], dtype=np.float32)

        smoothed_rewards = smooth(rewards, smooth_window)
        x = range(len(smoothed_rewards))

        color = get_method_color(label)

        line, = plt.plot(
            x,
            smoothed_rewards,
            label=label,
            color=color,
        )

        if label not in legend_labels:
            legend_elements.append(line)
            legend_labels.append(label)

    plt.xlabel(
        "Episode",
        fontweight="bold",
        fontsize=label_font,
    )

    plt.ylabel(
        "Reward",
        fontweight="bold",
        fontsize=label_font,
    )

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    plt.tick_params(width=8, length=20)

    if save:
        plt.savefig(
            f"{base_filename}_rewards.png",
            bbox_inches="tight",
        )

    plt.close()

    # ============================================================
    # 2. Worst-case max cost plot
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    plt.axhspan(
        args.persistent_eps,
        12.5,
        color="red",
        alpha=0.1,
    )

    plt.axhspan(
        -1,
        args.persistent_eps,
        color="blue",
        alpha=0.1,
    )

    for save_path, label in zip(save_paths, labels):
        key = label

        max_costs = np.asarray(
            results[key][0]["max_costs"],
            dtype=np.float32,
        )  # [episodes, cost_dim]

        worst_max_costs = np.max(max_costs, axis=1)

        y = smooth(worst_max_costs, smooth_window)
        x = range(len(y))

        color = get_method_color(label)

        plt.plot(
            x,
            y,
            label=label,
            color=color,
        )

    plt.axhline(
        y=args.persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
    )

    plt.xlabel(
        "Episode",
        fontweight="bold",
        fontsize=label_font,
    )

    plt.ylabel(
        "Max Cost",
        fontweight="bold",
        fontsize=label_font,
    )

    for spine in plt.gca().spines.values():
        spine.set_linewidth(15)

    plt.tick_params(width=8, length=20)

    if save:
        plt.savefig(
            f"{base_filename}_worst_max_costs.png",
            bbox_inches="tight",
        )

    plt.close()

    # ============================================================
    # 3. Individual constraints horizontally
    # ============================================================
    fig, axes = plt.subplots(
        1,
        args.cost_dim,
        figsize=(fig_size * args.cost_dim, fig_size),
        sharey=True,
    )

    if args.cost_dim == 1:
        axes = [axes]

    for ci in range(args.cost_dim):
        ax = axes[ci]

        ax.axhspan(
            args.persistent_eps,
            12.5,
            color="red",
            alpha=0.1,
        )

        ax.axhspan(
            -1,
            args.persistent_eps,
            color="blue",
            alpha=0.1,
        )

        for save_path, label in zip(save_paths, labels):
            key = label

            max_costs = np.asarray(
                results[key][0]["max_costs"],
                dtype=np.float32,
            )  # [episodes, cost_dim]

            y = smooth(max_costs[:, ci], smooth_window)
            x = range(len(y))

            color = get_method_color(label)

            ax.plot(
                x,
                y,
                label=label,
                color=color,
            )

        ax.axhline(
            y=args.persistent_eps,
            color=METHOD_COLORS["Baseline"],
            linestyle="--",
        )

        cost_name = COST_NAMES[ci] if ci < len(COST_NAMES) else f"C{ci + 1}"

        ax.set_title(
            cost_name,
            fontweight="bold",
            fontsize=label_font,
        )

        if ci == 0:
            ax.set_ylabel(
                "Max Cost",
                fontweight="bold",
                fontsize=label_font - 10,
            )

        for spine in ax.spines.values():
            spine.set_linewidth(15)

        ax.tick_params(width=8, length=20)

    # Shared x-axis label placed manually to avoid large gap
    fig.text(
        0.5,
        0.04,
        "Episode",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=label_font - 20,
    )

    plt.subplots_adjust(
        bottom=0.22,
        wspace=0.25,
    )

    if save:
        plt.savefig(
            f"{base_filename}_individual_constraints_horizontal.png",
            bbox_inches="tight",
        )

    plt.close()

    # ============================================================
    # Separate horizontal legend
    # ============================================================
    # if save:
    #     save_legend(
    #         legend_elements,
    #         legend_labels,
    #         f"{base_filename}_legend_horizontal.png",
    #         horizontal=True,
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")

    parser.add_argument("--env", type=str, default="Quadrotor")
    parser.add_argument("--uncer_set", type=str, default="IPM", help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2)
    parser.add_argument("--random_steps", type=int, default=int(25e3))
    parser.add_argument("--max_train_steps", type=int, default=int(16e3))
    parser.add_argument("--evaluate_freq", type=float, default=1e2)
    parser.add_argument("--save_freq", type=int, default=20)

    parser.add_argument("--policy_dist", type=str, default="Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=64)

    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=1e-3)
    parser.add_argument("--lr_cost", type=float, default=5e-4)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)

    parser.add_argument(
        "--persistent_eps",
        type=float,
        default=0.5,
        help="Safety threshold",
    )

    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=False)
    parser.add_argument("--use_reward_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=False)
    parser.add_argument("--entropy_coef", type=float, default=0.007)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=float, default=True)
    parser.add_argument("--use_tanh", type=float, default=True)
    parser.add_argument("--adaptive_alpha", type=float, default=False)
    parser.add_argument("--weight_reg", type=float, default=0.001)

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--GAMMA", type=str, default="0")
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--lambda_", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=1e5)
    parser.add_argument("--run", type=int, default=13)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=1300)
 

    # Important for multi-constraint
    parser.add_argument(
        "--cost_dim",
        type=int,
        default=4,
        help="number of constraints",
    )
    parser.add_argument(
        "--cost_scale",
        type=float,
        default=100.0,
        help="same cost scale used in trainer",
    )

    parser.add_argument("--num_episodes", type=int, default=100)

    args = parser.parse_args()

    # ============================================================
    # Same methods/runs as training plot script
    # ============================================================

    labels = [
        "Ours",
        "Surrogate Obj",        
        "FAC",
        "RCRL",
    ]

    directories = [
        [
            "./models/quadrotor/run16/Best_RCAC",
            # "./models/quadrotor/run17/Best_RCAC",
        ],
        [
            # "./models/quadrotor/run1/Best_RCAC",
            "./models/quadrotor/run17/Best_RCAC",
        ],
        [
            "./models/quadrotor/run501/Best_RCAC",
            # "./models/quadrotor/run502/Best_RCAC",
        ],
        [
            "./models/quadrotor/run101/Best_RCAC",
            # "./models/quadrotor/run102/Best_RCAC",
        ],
    ]

    model_num = None

    model_num_map = {
        "run501": 9400,
        "run101": 13600,
    }



    results = test_multiple_dirs(
        args,
        directories,
        labels=labels,
        model_num=model_num,
        model_num_map=model_num_map,
        num_episodes=args.num_episodes,
    )

    plot_evaluation(
        args,
        results,
        directories,
        labels,
        save=True,
        base_filename="plot_inference/quadrotor_inference_all",
        smooth_window=80,
    )
