import torch
import numpy as np
from code_ipm_rcmdp_rcrl_max_hc import Actor_Beta, Actor_Gaussian, Actor_Discrete, Critic, CostCritic, Robust_RCAC_NPG, Normalization, RunningMeanStd, RewardScaling
import argparse
import pickle
import matplotlib.pyplot as plt
import os
# from envs.cartpole import CartPolePerturbedEnv
import glob
from envs.half_cheetah import HalfCheetahWithPos, HalfCheetahWithPosPerturbed
# from envs.reacher import ReacherWithCost
# from envs.swimmer import SwimmerWithPos

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
    agent.actor.load(actor_path)
    rcritic_path = f"{save_path}_Rcritic"
    agent.Rcritic.load(rcritic_path)
    ccritic_path = f"{save_path}_Ccritic"
    agent.Ccritic.load(ccritic_path)

    if args.use_state_norm:
        print("Loading state norm")
        with open(f'{save_path}_state_norm', 'rb') as file1:
            state_norm = pickle.load(file1)
        print(state_norm.running_ms.mean,state_norm.running_ms.std)

    if args.use_reward_scaling:
        print("Loading reward scaling") 
        with open(f'{save_path}_reward_scaling', 'rb') as file2:
            reward_scaling = pickle.load(file2)

    print("Agent and normalization objects loaded successfully!")
    return agent, state_norm, reward_scaling


def test_single_models(args, model_specs, perturbation_stds, num_episodes=100):
    """
    Test one model per method across perturbation stds.

    model_specs format:
        [
            {
                "label": "Ours",
                "model_path": "./models/HalfCheetahWithPosPerturbed/run2/Best_RCAC"
            },
            ...
        ]

    Returns:
        results[label][std_index] = {
            'rewards': rewards,
            'costs': costs,
            'max_costs': max_costs
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

            # Evaluation environment
            env = HalfCheetahWithPosPerturbed(sigma_gravity=std)

            # If you want nominal evaluation instead, use:
            # env = HalfCheetahWithPos()

            env.reset(seed=args.seed)
            env.action_space.seed(args.seed)

            args.max_action = float(env.action_space.high[0])
            args.state_dim = env.observation_space.shape[0]
            args.action_dim = env.action_space.shape[0]

            rewards, costs, max_costs = test_agent_multiple_models(
                args,
                model_path,
                env,
                num_episodes=num_episodes
            )

            results[label].append({
                "rewards": rewards,
                "costs": costs,
                "max_costs": max_costs
            })

    return results


def test_multiple_model_groups(args, model_specs, perturbation_stds, num_episodes=100):
    """
    Test multiple model groups across perturbation stds.

    model_specs format:
        [
            {
                "label": "Surrogate Obj(NP)",
                "model_paths": [
                    "./models/HalfCheetahWithPos/run1/Best_RCAC",
                    "./models/HalfCheetahWithPos/run2/Best_RCAC",
                    "./models/HalfCheetahWithPos/run3/Best_RCAC"
                ]
            },
            ...
        ]

    Returns:
        results[label][std_index][run_index] = {
            'rewards': rewards,
            'costs': costs,
            'max_costs': max_costs
        }
    """

    results = {}

    for spec in model_specs:
        label = spec["label"]
        model_paths = spec["model_paths"]

        results[label] = []

        for std in perturbation_stds:
            print(f"\nTesting method: {label}, perturbation std = {std}")

            std_results = []

            for model_path in model_paths:
                print(f"  Loading model: {model_path}")

                # Evaluation environment
                # If you want nominal HalfCheetah evaluation:
                # env = HalfCheetahWithPos()

                # If later you want perturbed evaluation, use this instead:
                env = HalfCheetahWithPosPerturbed(sigma_gravity=std)

                env.reset(seed=args.seed)
                env.action_space.seed(args.seed)

                args.max_action = float(env.action_space.high[0])
                args.state_dim = env.observation_space.shape[0]
                args.action_dim = env.action_space.shape[0]

                rewards, costs, max_costs = test_agent_multiple_models(
                    args,
                    model_path,
                    env,
                    num_episodes=num_episodes
                )

                std_results.append({
                    "rewards": rewards,
                    "costs": costs,
                    "max_costs": max_costs
                })

            results[label].append(std_results)

    return results


def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

def test_agent_multiple_models(args, save_paths, env, num_episodes=100):

    rewards = []
    costs = []
    max_costs = []

    # Load all agents ahead of time
    agents = []
    if isinstance(save_paths, str):
        save_paths = [save_paths]
    for save_path in save_paths:
        agent, state_norm, reward_scaling = load_agent(args, save_path)
        agents.append((agent, state_norm, reward_scaling))

    for episode in range(num_episodes):
        state = env.reset()[0][0]
        if args.use_state_norm:
            state = state_norm(state, update=False)
        total_reward = 0
        total_cost = 0
        max_cost = float('-inf')

        done = False

        while not done:
            actions = []

            # Get actions from all loaded agents
            for agent, _, _ in agents:
                # Get action from the policy
                action = agent.evaluate(state)
                if agent.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * agent.max_action  # Map [0, 1] to [-max_action, max_action]

                actions.append(action)

            # Calculate the mean action
            mean_action = np.mean(actions, axis=0)

            # Step in the environment with the mean action
            next_state, reward, cost, truncated, terminated, _ = env.step(mean_action)
            done = truncated or terminated
            
            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            total_reward += reward
            total_cost += cost
            max_cost = max(max_cost, cost)
            state = next_state

        rewards.append(total_reward)
        costs.append(total_cost)
        max_costs.append(max_cost)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Max Cost= {max_cost}, Total Cost = {total_cost}")

    return rewards, costs, max_costs



# Function to test multiple models across multiple gravity perturbations
def test_multiple_model_groups(args, model_specs, perturbation_stds, num_episodes=100):
    """
    Test multiple model groups across perturbation stds.

    model_specs format:
        [
            {
                "label": "Surrogate Obj(NP)",
                "model_paths": [
                    "./models/HalfCheetahWithPos/run1/Best_RCAC",
                    "./models/HalfCheetahWithPos/run2/Best_RCAC",
                    "./models/HalfCheetahWithPos/run3/Best_RCAC"
                ]
            },
            ...
        ]

    Returns:
        results[label][std_index][run_index] = {
            'rewards': rewards,
            'costs': costs,
            'max_costs': max_costs
        }
    """

    results = {}

    for spec in model_specs:
        label = spec["label"]
        model_paths = spec["model_paths"]

        results[label] = []

        for std in perturbation_stds:
            print(f"\nTesting method: {label}, perturbation std = {std}")

            std_results = []

            for model_path in model_paths:
                print(f"  Loading model: {model_path}")

                # Evaluation environment
                # If you want nominal HalfCheetah evaluation:
                # env = HalfCheetahWithPos()

                # If later you want perturbed evaluation, use this instead:
                env = HalfCheetahWithPosPerturbed(sigma_gravity=std)

                env.reset(seed=args.seed)
                env.action_space.seed(args.seed)

                args.max_action = float(env.action_space.high[0])
                args.state_dim = env.observation_space.shape[0]
                args.action_dim = env.action_space.shape[0]

                rewards, costs, max_costs = test_agent_multiple_models(
                    args,
                    model_path,
                    env,
                    num_episodes=num_episodes
                )

                std_results.append({
                    "rewards": rewards,
                    "costs": costs,
                    "max_costs": max_costs
                })

            results[label].append(std_results)

    return results

def plot_evaluation_single(
    args,
    results,
    labels,
    perturbation_stds,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10
):
    """
    Plot evaluation results for one model per method.
    No mean/std shading.
    """

    plt.rcParams.update({
        'font.size': 100,
        'lines.linewidth': 15,
        'font.weight': 'bold'
    })

    fig_size = 28
    label_font = 130

    def style_axes():
        plt.grid(False)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(15)

    def get_metric(label, std_index, metric_name):
        metric = np.array(results[label][std_index][metric_name])
        smoothed_metric = smooth(metric, smooth_window)
        x = range(len(smoothed_metric))
        return x, smoothed_metric

    legend_elements = []
    legend_labels = []

    # =====================
    # Plot Rewards
    # =====================
    plt.figure(figsize=(fig_size + 16, fig_size))

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, rewards = get_metric(label, std_index, "rewards")

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} (std={std})"

            line, = plt.plot(x, rewards, label=plot_label)
            legend_elements.append(line)
            legend_labels.append(plot_label)

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches="tight")

    plt.close()

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_legend_horizontal.png",
        horizontal=True
    )

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_legend_vertical.png",
        horizontal=False
    )

    # =====================
    # Plot Max Costs
    # =====================
    plt.figure(figsize=(fig_size + 14, fig_size))

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, max_costs = get_metric(label, std_index, "max_costs")

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} (std={std})"

            plt.plot(x, max_costs, label=plot_label)

    y_min, y_max = plt.gca().get_ylim()

    plt.axhline(
        y=args.persistent_eps,
        color='black',
        linestyle='--',
        linewidth=15,
        label="Baseline"
    )

    plt.axhspan(args.persistent_eps, y_max, color='red', alpha=0.1)
    plt.axhspan(y_min, args.persistent_eps, color='blue', alpha=0.1)

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Max Cost", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches="tight")

    plt.close()

    # =====================
    # Plot Total Costs
    # =====================
    plt.figure(figsize=(fig_size, fig_size))

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, costs = get_metric(label, std_index, "costs")

            if len(perturbation_stds) == 1:
                plot_label = label
            else:
                plot_label = f"{label} (std={std})"

            plt.plot(x, costs, label=plot_label)

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Cumulative Cost", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_total_costs.png", bbox_inches="tight")

    plt.close()


def plot_evaluation_grouped(
    args,
    results,
    labels,
    perturbation_stds,
    save=False,
    base_filename="evaluation_plot",
    smooth_window=10
):
    """
    Plot mean ± std evaluation results across multiple trained runs.

    results[label][std_index][run_index] contains:
        rewards, costs, max_costs
    """

    plt.rcParams.update({
        'font.size': 100,
        'lines.linewidth': 15,
        'font.weight': 'bold'
    })

    fig_size = 28
    label_font = 130

    def style_axes():
        plt.grid(False)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(15)

    def aggregate_metric(label, std_index, metric_name):
        runs = results[label][std_index]
        metric_arrays = [np.array(run_result[metric_name]) for run_result in runs]

        min_len = min(len(x) for x in metric_arrays)
        metric_arrays = np.array([x[:min_len] for x in metric_arrays])

        mean_metric = np.mean(metric_arrays, axis=0)
        std_metric = np.std(metric_arrays, axis=0)

        smoothed_mean = smooth(mean_metric, smooth_window)
        smoothed_std = smooth(std_metric, smooth_window)
        x = range(len(smoothed_mean))

        return x, smoothed_mean, smoothed_std

    # =====================
    # Plot Rewards
    # =====================
    plt.figure(figsize=(fig_size + 16, fig_size))

    legend_elements = []
    legend_labels = []

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, mean_rewards, std_rewards = aggregate_metric(
                label,
                std_index,
                "rewards"
            )

            line, = plt.plot(x, mean_rewards, label=label)
            plt.fill_between(
                x,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2
            )

            legend_elements.append(line)
            if len(perturbation_stds) == 1:
                legend_labels.append(label)
            else:
                legend_labels.append(f"{label} (std={std})")

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Cumulative Reward", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_rewards.png", bbox_inches="tight")

    plt.close()

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_legend_horizontal.png",
        horizontal=True
    )

    save_legend(
        legend_elements,
        legend_labels,
        f"{base_filename}_legend_vertical.png",
        horizontal=False
    )

    # =====================
    # Plot Max Costs
    # =====================
    plt.figure(figsize=(fig_size + 14, fig_size))

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, mean_max_costs, std_max_costs = aggregate_metric(
                label,
                std_index,
                "max_costs"
            )

            plt.plot(x, mean_max_costs, label=label)
            plt.fill_between(
                x,
                mean_max_costs - std_max_costs,
                mean_max_costs + std_max_costs,
                alpha=0.2
            )

    y_min, y_max = plt.gca().get_ylim()

    plt.axhline(
        y=args.persistent_eps,
        color='black',
        linestyle='--',
        linewidth=15,
        label="Baseline"
    )

    plt.axhspan(args.persistent_eps, y_max, color='red', alpha=0.1)
    plt.axhspan(y_min, args.persistent_eps, color='blue', alpha=0.1)

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Max Cost", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_max_costs.png", bbox_inches="tight")

    plt.close()

    # =====================
    # Plot Total Costs
    # =====================
    plt.figure(figsize=(fig_size, fig_size))

    for label in labels:
        for std_index, std in enumerate(perturbation_stds):
            x, mean_costs, std_costs = aggregate_metric(
                label,
                std_index,
                "costs"
            )

            plt.plot(x, mean_costs, label=label)
            plt.fill_between(
                x,
                mean_costs - std_costs,
                mean_costs + std_costs,
                alpha=0.2
            )

    plt.xlabel("Episode", fontweight='bold', fontsize=label_font)
    plt.ylabel("Cumulative Cost", fontweight='bold', fontsize=label_font)
    style_axes()

    if save:
        plt.savefig(f"{base_filename}_total_costs.png", bbox_inches="tight")

    plt.close()


def save_legend(legend_elements, labels, filename, horizontal=True):
    """
    Save a separate legend as an image.

    Args:
        legend_elements: List of matplotlib line objects for the legend.
        labels: List of labels corresponding to the legend elements.
        filename: File name to save the legend image.
        horizontal: Whether to save the legend as horizontal or vertical.
    """
    fig = plt.figure(figsize=(20, 5) if horizontal else (5, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(handles=legend_elements, labels=labels, loc='center', ncol=len(legend_elements) if horizontal else 1, frameon=False)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    # Define your arguments (or load them from a config file)
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='SwimmerWithPos',help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(16e3), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian or Discrete")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")

        # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--persistent_eps", type=float, default=0.1, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.001, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0.001, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=4, help="seed 2, 5, 7, 11, 17") 
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")
    parser.add_argument("--baseline",type=int,default=9,help="baseline")
    parser.add_argument("--lambda_",type=int,default=1.0,help="lambda")
    parser.add_argument("--beta",type=float,default=30000.0,help="beta") 
    parser.add_argument("--run",type=int,default=1,help="run_number") 
    parser.add_argument("--warm_start_flag",type=int,default=0,help="warm_start_flag") 
    parser.add_argument("--warm_start_episode",type=int,default=150,help="warm_start_episode")
    parser.add_argument("--sigma_viscosity",type=float,default=0.7,help="sigma of gravity perturbation")
    parser.add_argument("--lr_cost",type=float,default=1e-3,help="learning rate for cost function")

    args = parser.parse_args()

    #################### Multiple Models Evaluation #######################
    # model_specs = [
    #     {
    #         "label": "Surrogate Obj(NP)",
    #         "model_paths": [
    #             "./models/HalfCheetahWithPos/run1/Best_RCAC",
    #             "./models/HalfCheetahWithPos/run2/Best_RCAC",
    #             "./models/HalfCheetahWithPos/run3/Best_RCAC"
    #         ]
    #     },
    #     {
    #         "label": "Ours",
    #         "model_paths": [
    #             "./models/HalfCheetahWithPosPerturbed/run1/Best_RCAC",
    #             "./models/HalfCheetahWithPosPerturbed/run2/Best_RCAC"
    #         ]
    #     },
    #     {
    #         "label": "RCRL",
    #         "model_paths": [
    #             "./models/HalfCheetahWithPos/run101/Best_RCAC",
    #             "./models/HalfCheetahWithPos/run102/Best_RCAC"
    #         ]
    #     },
    #     {
    #         "label": "Primal Dual",
    #         "model_paths": [
    #             "./models/HalfCheetahWithPos/run103/Best_RCAC",
    #             "./models/HalfCheetahWithPos/run104/Best_RCAC"
    #         ]
    #     }
    # ]

    # labels = [spec["label"] for spec in model_specs]

    # perturbation_stds = [1.5]

    # results = test_multiple_model_groups(
    #     args,
    #     model_specs,
    #     perturbation_stds,
    #     num_episodes=100
    # )

    # plot_evaluation_grouped(
    #     args,
    #     results,
    #     labels,
    #     perturbation_stds,
    #     save=True,
    #     base_filename="plot_inference/HC_baseline_comparison_all",
    #     smooth_window=20
    # )


    #################### Single Model Evaluation #######################
    model_specs = [
        {
            "label": "Surrogate Obj(NP)",
            "model_path": "./models/HalfCheetahWithPos/run3/Best_RCAC"
        },
        {
            "label": "Ours",
            "model_path": "./models/HalfCheetahWithPosPerturbed/run1/Best_RCAC"
        },
        {
            "label": "RCRL",
            "model_path": "./models/HalfCheetahWithPos/run102/Best_RCAC"
        },
        {
            "label": "Primal Dual",
            "model_path": "./models/HalfCheetahWithPos/run104/Best_RCAC"
        }
    ]

    labels = [spec["label"] for spec in model_specs]

    perturbation_stds = [1.5]

    results = test_single_models(
        args,
        model_specs,
        perturbation_stds,
        num_episodes=100
    )

    plot_evaluation_single(
        args,
        results,
        labels,
        perturbation_stds,
        save=True,
        base_filename="plot_inference/HC_baseline_comparison_all",
        smooth_window=20
    )

