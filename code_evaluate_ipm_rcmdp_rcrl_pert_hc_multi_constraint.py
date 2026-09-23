import os

# ============================================================
# If using Windows + mujoco_py, uncomment if needed
# ============================================================
if os.name == "nt":
    os.add_dll_directory(r"C:\Users\rinki\.mujoco\mujoco210\bin")
    os.add_dll_directory(r"C:\Users\rinki\miniconda3\envs\rpcrl_env\Library\bin")

os.environ["MUJOCO_PY_MUJOCO_PATH"] = r"C:\Users\rinki\.mujoco\mujoco210"

import argparse
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from code_ipm_rcmdp_rcrl_max_hc import (
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

from envs.half_cheetah_multi_constraint_cost import (
    # HalfCheetahMultiConstraintCost,
    HalfCheetahMultiConstraintCostPerturbedTest,
)


# ============================================================
# Fixed color scheme
# ============================================================
METHOD_COLORS = {
    "Ours": "#1f77b4",
    "Surrogate Obj": "#ff7f0e",
    "PD": "#2ca02c",
    "FAC": "#d62728",
    "RCRL": "#9467bd",
    "RESPO": "#8c564b",
    "Baseline": "black",
}


def get_method_color(label):
    return METHOD_COLORS.get(label, "#7f7f7f")


# ============================================================
# Cost names for your experiment
# ============================================================
COST_NAMES = [
    "Pitch (C1)",
    "Joint Speed (C2)",
]


# ============================================================
# Smoothing
# ============================================================
def smooth(data, window_size):
    data = np.asarray(data, dtype=np.float32).squeeze()

    if data.ndim != 1:
        raise ValueError(f"smooth() expected 1D data, got shape {data.shape}")

    if window_size <= 1:
        return data

    if len(data) < window_size:
        return data

    return np.convolve(
        data,
        np.ones(window_size, dtype=np.float32) / window_size,
        mode="valid",
    )


# ============================================================
# Agent loading
# ============================================================
def load_agent(args, save_path, load_critics=False):
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
            print(f"Loading state norm: {norm_path}")
            with open(norm_path, "rb") as file1:
                state_norm = pickle.load(file1)
            print(state_norm.running_ms.mean, state_norm.running_ms.std)
        else:
            print(f"[WARNING] State norm not found: {norm_path}")

    if args.use_reward_scaling:
        scaling_path = f"{save_path}_reward_scaling"

        if os.path.exists(scaling_path):
            print(f"Loading reward scaling: {scaling_path}")
            with open(scaling_path, "rb") as file2:
                reward_scaling = pickle.load(file2)
        else:
            print(f"[WARNING] Reward scaling not found: {scaling_path}")

    print(f"Agent actor loaded successfully from: {save_path}")

    return agent, state_norm, reward_scaling


def load_agent_specific_model(args, save_path, model_num, load_critics=False):
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
                f"[WARNING] Could not load critics for {save_path}, "
                f"checkpoint={model_num}. Continuing with actor-only evaluation.\n{e}"
            )

    if args.use_state_norm:
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
    if model_num_map is None:
        return default_model_num

    normalized_path = save_path.replace("\\", "/")

    for key, value in model_num_map.items():
        if key in normalized_path:
            return value

    return default_model_num


# ============================================================
# Environment helpers
# ============================================================
def make_hc_multiconstraint_env(env_type, perturbation_std):
    """
    env_type:
        "nominal"   -> HalfCheetahMultiConstraintCost
        "perturbed" -> HalfCheetahMultiConstraintCostPerturbed
    """

    if env_type == "perturbed":
        return HalfCheetahMultiConstraintCostPerturbedTest(
            sigma_gravity=perturbation_std
        )

    elif env_type == "nominal":
        return HalfCheetahMultiConstraintCostPerturbedTest(
            sigma_gravity=perturbation_std
        )

    else:
        raise ValueError(
            f"Unknown env_type={env_type}. Use 'nominal' or 'perturbed'."
        )


def env_reset_compat(env, seed=None):
    if seed is not None:
        out = env.reset(seed=seed)
    else:
        out = env.reset()

    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out

    obs = np.asarray(obs)

    if obs.ndim > 1:
        obs = obs.reshape(-1)

    return obs


def env_step_compat(env, action):
    """
    Supports:
        next_state, reward, cost, truncated, terminated, info
        next_state, reward, terminated, truncated, info
        next_state, reward, done, info
    """

    out = env.step(action)

    if len(out) == 6:
        next_state, reward, cost, truncated, terminated, info = out
        done = truncated or terminated
        return next_state, reward, cost, done, info

    elif len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = terminated or truncated
        cost = None
        return next_state, reward, cost, done, info

    elif len(out) == 4:
        next_state, reward, done, info = out
        cost = None
        return next_state, reward, cost, done, info

    else:
        raise RuntimeError(f"Unsupported env.step output length: {len(out)}")


# ============================================================
# Cost extraction for your HalfCheetahMultiConstraintCost
# ============================================================
def extract_multicost(args, info, scalar_cost=None):
    """
    Extract 2D cost vector.

    Preferred keys:
        info["constraint_values"]
        info["costs"]
        info["cost"]

    Expected output:
        shape [2]
    """

    if "constraint_values" in info:
        c = np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1)

    elif "costs" in info:
        c = np.asarray(info["costs"], dtype=np.float32).reshape(-1)

    elif "cost" in info:
        c = np.asarray(info["cost"], dtype=np.float32).reshape(-1)

    elif scalar_cost is not None:
        c = np.asarray(scalar_cost, dtype=np.float32).reshape(-1)

    else:
        raise KeyError(
            "Could not find multi-constraint cost. Expected one of "
            "info['constraint_values'], info['costs'], info['cost'], "
            "or scalar cost from env.step(). "
            f"Info keys are: {list(info.keys())}"
        )

    c = args.cost_scale * c
    c = np.maximum(c, 0.0)

    if c.shape[0] != args.cost_dim:
        raise ValueError(
            f"Expected cost_dim={args.cost_dim}, got cost shape={c.shape}, "
            f"value={c}, info keys={list(info.keys())}"
        )

    return c


# ============================================================
# Eval one method/model group
# ============================================================
def test_agent_multiple_models(
    args,
    save_paths,
    env,
    model_num=None,
    model_num_map=None,
    num_episodes=100,
):
    rewards = []
    costs = []
    max_costs = []
    total_cost_sums = []
    max_total_costs = []

    agents = []

    if isinstance(save_paths, str):
        save_paths = [save_paths]

    for save_path in save_paths:
        this_model_num = get_model_num_for_path(
            save_path,
            model_num_map=model_num_map,
            default_model_num=model_num,
        )

        if this_model_num is not None:
            agent, state_norm, reward_scaling = load_agent_specific_model(
                args=args,
                save_path=save_path,
                model_num=this_model_num,
                load_critics=False,
            )
        else:
            agent, state_norm, reward_scaling = load_agent(
                args=args,
                save_path=save_path,
                load_critics=False,
            )

        agents.append((agent, state_norm, reward_scaling))

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

            next_state, reward, scalar_cost, done, info = env_step_compat(
                env,
                mean_action,
            )

            c = extract_multicost(
                args=args,
                info=info,
                scalar_cost=scalar_cost,
            )

            if args.use_state_norm and first_state_norm is not None:
                next_state = first_state_norm(next_state, update=False)

            if args.use_reward_scaling and first_reward_scaling is not None:
                reward = first_reward_scaling(reward, update=False)

            total_reward += reward
            total_cost += c
            max_cost = np.maximum(max_cost, c)

            state = copy.deepcopy(next_state)

        total_cost_sum = float(np.sum(total_cost))
        max_total_cost = float(np.max(max_cost))
        safe = np.all(max_cost <= args.persistent_eps)

        rewards.append(total_reward)
        costs.append(total_cost.copy())
        max_costs.append(max_cost.copy())
        total_cost_sums.append(total_cost_sum)
        max_total_costs.append(max_total_cost)

        cost_str = " | ".join(
            [f"Total C{i + 1}={total_cost[i]:.3f}" for i in range(args.cost_dim)]
        )
        max_cost_str = " | ".join(
            [f"Max C{i + 1}={max_cost[i]:.3f}" for i in range(args.cost_dim)]
        )

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


# ============================================================
# Evaluate all experiment specs
# ============================================================
def test_multiple_dirs(
    args,
    model_specs,
    perturbation_stds,
    model_num=None,
    model_num_map=None,
    num_episodes=100,
):
    results = {}

    for spec in model_specs:
        label = spec["label"]
        model_path = spec["model_path"]
        env_type = spec["env_type"]

        results[label] = []

        for std in perturbation_stds:
            print("\n============================================================")
            print(f"Testing method: {label}")
            print(f"Perturbation std: {std}")
            print(f"Model path(s): {model_path}")
            print(f"Env type: {env_type}")
            print("============================================================\n")

            env = make_hc_multiconstraint_env(
                env_type=env_type,
                perturbation_std=std,
            )

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
                args=args,
                save_paths=model_path,
                env=env,
                model_num=model_num,
                model_num_map=model_num_map,
                num_episodes=num_episodes,
            )

            results[label].append(
                {
                    "rewards": rewards,
                    "costs": costs,
                    "max_costs": max_costs,
                    "total_cost_sums": total_cost_sums,
                    "max_total_costs": max_total_costs,
                }
            )

            try:
                env.close()
            except Exception:
                pass

    return results


# ============================================================
# Legends
# ============================================================
def save_separate_legends(
    labels,
    base_filename,
    line_width=15,
    font_size=90,
):
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

    save_dir = os.path.dirname(base_filename)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

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
# Plot eval same as training plots
# ============================================================
def plot_evaluation_multiconstraint(
    args,
    results,
    labels,
    perturbation_stds,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10,
    save_legends=True,
):
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
            labels=labels,
            base_filename=base_filename,
            line_width=15,
            font_size=90,
        )

    def style_current_axes():
        for spine in plt.gca().spines.values():
            spine.set_linewidth(15)
        plt.tick_params(width=8, length=20)

    def style_axis(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(15)
        ax.tick_params(width=8, length=20)

    # ============================================================
    # 1. Rewards
    # ============================================================
    plt.figure(figsize=(fig_size + 8, fig_size))

    for label in labels:
        color = get_method_color(label)

        for std_index, std in enumerate(perturbation_stds):
            rewards = np.asarray(
                results[label][std_index]["rewards"],
                dtype=np.float32,
            )

            y = smooth(rewards, smooth_window)
            x = range(len(y))

            plt.plot(
                x,
                y,
                color=color,
                label=label,
            )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Reward", fontweight="bold", fontsize=label_font)

    style_current_axes()

    if save:
        plt.savefig(
            f"{base_filename}_rewards.png",
            bbox_inches="tight",
        )

    plt.close()

    # ============================================================
    # 2. Worst-case max cost
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

    for label in labels:
        color = get_method_color(label)

        for std_index, std in enumerate(perturbation_stds):
            max_costs = np.asarray(
                results[label][std_index]["max_costs"],
                dtype=np.float32,
            )

            worst_max_costs = np.max(max_costs, axis=1)

            y = smooth(worst_max_costs, smooth_window)
            x = range(len(y))

            plt.plot(
                x,
                y,
                color=color,
                label=label,
            )

    plt.axhline(
        y=args.persistent_eps,
        color=METHOD_COLORS["Baseline"],
        linestyle="--",
    )

    plt.xlabel("Episode", fontweight="bold", fontsize=label_font)
    plt.ylabel("Max Cost", fontweight="bold", fontsize=label_font)

    style_current_axes()

    if save:
        plt.savefig(
            f"{base_filename}_worst_max_costs.png",
            bbox_inches="tight",
        )

    plt.close()

    # ============================================================
    # 3. Individual constraints horizontal
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

        for label in labels:
            color = get_method_color(label)

            for std_index, std in enumerate(perturbation_stds):
                max_costs = np.asarray(
                    results[label][std_index]["max_costs"],
                    dtype=np.float32,
                )

                y = smooth(max_costs[:, ci], smooth_window)
                x = range(len(y))

                ax.plot(
                    x,
                    y,
                    color=color,
                    label=label,
                )

        ax.axhline(
            y=args.persistent_eps,
            color=METHOD_COLORS["Baseline"],
            linestyle="--",
        )

        cost_name = COST_NAMES[ci] if ci < len(COST_NAMES) else f"C{ci + 1}"

        ax.set_ylabel(
            f"Max {cost_name}",
            fontweight="bold",
            fontsize=label_font - 20,
        )

        style_axis(ax)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")

    parser.add_argument("--env", type=str, default="HalfCheetahMultiConstraintCostPerturbedTest")
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

    parser.add_argument("--persistent_eps", type=float, default=0.1)

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
    parser.add_argument("--sigma_viscosity", type=float, default=0.7)

    # Important for this experiment
    parser.add_argument("--cost_dim", type=int, default=2)
    parser.add_argument("--cost_scale", type=float, default=1.0)
    parser.add_argument("--num_episodes", type=int, default=100)

    args = parser.parse_args()

    # ============================================================
    # Exact experiment folders from your training plot script
    # ============================================================
    HALFCHEETAH_ENV_NAME = "HalfCheetahMultiConstraintCost"
    HALFCHEETAH_ENV_NAME_PERTURBED = "HalfCheetahMultiConstraintCostPerturbed"

    HALFCHEETAH_COST_VERSION = "multi_constraint_pitch_joint_v1"

    HALFCHEETAH_MODEL_DIR = (
        f"./models/{HALFCHEETAH_ENV_NAME}/{HALFCHEETAH_COST_VERSION}"
    )

    HALFCHEETAH_MODEL_DIR_PERTURBED = (
        f"./models/{HALFCHEETAH_ENV_NAME_PERTURBED}/{HALFCHEETAH_COST_VERSION}"
    )

    # ============================================================
    # Same experiments as training plot specs
    #
    # Important:
    # Each model_path list is treated as an action ensemble.
    # If you want single model eval, keep only one run in each list.
    # ============================================================
    model_specs = [
        {
            "label": "Ours",
            "model_path": [
                f"{HALFCHEETAH_MODEL_DIR}/run1/Final_RCAC",
                # f"{HALFCHEETAH_MODEL_DIR}/run2/Best_RCAC",
            ],
            "env_type": "nominal",
        },
        {
            "label": "Surrogate Obj",
            "model_path": [
                # f"{HALFCHEETAH_MODEL_DIR_PERTURBED}/run1/Best_RCAC",
                f"{HALFCHEETAH_MODEL_DIR_PERTURBED}/run2/Final_RCAC",
            ],
            "env_type": "perturbed",
        },
        {
            "label": "FAC",
            "model_path": [
                f"{HALFCHEETAH_MODEL_DIR}/run501/Final_RCAC",
                # f"{HALFCHEETAH_MODEL_DIR}/run502/Best_RCAC",
            ],
            "env_type": "nominal",
        },
        {
            "label": "RCRL",
            "model_path": [
                f"{HALFCHEETAH_MODEL_DIR}/run101/Final_RCAC",
                # f"{HALFCHEETAH_MODEL_DIR}/run102/Best_RCAC",
            ],
            "env_type": "nominal",
        },
    ]

    labels = [spec["label"] for spec in model_specs]

    # Match your eval perturbation choice.
    perturbation_stds = [1.5]

    model_num = None

    model_num_map = {
        # Add only if needed:
        "run1": 8000,
        "run2": 10000,
        "run501": 8000,
        "run101": 8000,
    }

    results = test_multiple_dirs(
        args=args,
        model_specs=model_specs,
        perturbation_stds=perturbation_stds,
        model_num=model_num,
        model_num_map=model_num_map,
        num_episodes=args.num_episodes,
    )

    plot_evaluation_multiconstraint(
        args=args,
        results=results,
        labels=labels,
        perturbation_stds=perturbation_stds,
        save=True,
        base_filename="plot_inference/dummy_halfcheetah_multiconstraint_eval_all",
        smooth_window=20,
        save_legends=True,
    )

