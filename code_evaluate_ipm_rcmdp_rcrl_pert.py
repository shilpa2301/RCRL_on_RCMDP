import torch
import numpy as np
from code_ipm_rcmdp_rcrl_max import Actor_Beta, Actor_Gaussian, Actor_Discrete, Critic, CostCritic, Robust_RCAC_NPG, Normalization, RunningMeanStd, RewardScaling
import argparse
import pickle
import matplotlib.pyplot as plt
import os
from envs.cartpole import CartPolePerturbedEnv

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

def test_agent(agent, env, num_episodes=100, state_norm=None):
    """
    Test the trained agent in the given environment.

    Args:
        agent: The trained agent.
        env: The environment to test the agent on.
        num_episodes: Number of episodes to test.
        state_norm: State normalization object (optional).

    Returns:
        rewards: List of total rewards for each episode.
        costs: List of total costs for each episode.
    """
    rewards = []
    costs = []
    max_costs = []

    for episode in range(num_episodes):
        state = env.reset()
        if args.use_state_norm:
                state = state_norm(state, update=False)
        total_reward = 0
        total_cost = 0
        max_cost = float('-inf')

        done = False

        while not done:
            # Normalize state if necessary
            

            # Get action from the policy
            action = agent.evaluate(state)
            if agent.policy_dist == "Beta":
                action = 2 * (action - 0.5) * agent.max_action  # Map [0, 1] to [-max_action, max_action]

            # Step in the environment
            # print("taking step")
            next_state, reward, cost, done, _ = env.step(action)

            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            # if args.use_reward_scaling:
            #     reward = reward_scaling(reward)
            #     cost = reward_scaling(cost)

            total_reward += reward
            total_cost += cost
            max_cost = max(max_cost, cost)          
            state = next_state

        rewards.append(total_reward)
        costs.append(total_cost)
        max_costs.append(max_cost)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Max Cost= {max_cost}, Total Cost = {total_cost}")

    return rewards, costs, max_costs


def smooth(data, window_size):
    """Smooth the data using a simple moving average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

# Function to test multiple models across multiple gravity perturbations
def test_multiple_dirs(args, save_paths, gravity_perturbation_stds, num_episodes=100):
    """
    Test multiple models across different gravity perturbation standard deviations.

    Args:
        args: Argument parser with required parameters.
        save_paths: List of directories where models are saved.
        gravity_perturbation_stds: List of gravity perturbation std values to test.
        num_episodes: Number of episodes to test.

    Returns:
        results: Dictionary where keys are model directories, and values are lists of results for each gravity perturbation std.
    """
    results = {}
    for save_path in save_paths:
        model_results = []
        for std in gravity_perturbation_stds:
            print(f"Testing model from {save_path} with gravity_perturbation_std = {std}")
            
            # Create a new environment instance with the specified gravity perturbation std
            env = CartPolePerturbedEnv(gravity_perturbation_std=std)
            env.reset(seed=args.seed)
            env.action_space.seed(args.seed)
            args.max_action = float(env.action_space.high[0])
            args.state_dim = env.observation_space.shape[0]
            args.action_dim = env.action_space.shape[0]
            
            # Load the agent
            agent, state_norm, reward_scaling = load_agent(args, save_path)
            
            # Test the agent in the environment
            rewards, costs, max_costs = test_agent(agent, env, num_episodes, state_norm)
            model_results.append({'rewards': rewards, 'costs': costs, 'max_costs': max_costs})
        
        # Store results for this model
        results[save_path] = model_results
    return results



def plot_evaluation1(data, labels, save=False, base_filename="evaluation_plot", smooth_window=10):
    """
    Plot raw and smoothed evaluation metrics from multiple directories.

    Args:
        data: A list of dictionaries, each containing 'rewards', 'costs', and 'max_costs'.
        labels: A list of labels corresponding to each dataset.
        save: Whether to save the plots to files.
        base_filename: Base file name to save the plots.
        smooth_window: Window size for smoothing.
    """
    assert len(data) == len(labels), "Data and labels must have the same length"

    # Set global font size
    # plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'font.size': 100, 'lines.linewidth': 15})
    fig_size = 30

    # Plot total rewards
    plt.figure(figsize=(fig_size, fig_size))  # Adjust aspect ratio for square plots
    for dataset, label in zip(data, labels):
        rewards = np.array(dataset['rewards'])
        smoothed_rewards = smooth(rewards, smooth_window)
        smoothed_x = range(len(smoothed_rewards))

        # Plot smoothed rewards
        plt.plot(smoothed_x, smoothed_rewards, label=f"{label}")
    # plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # Optional: Add a baseline at y=0
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    # plt.title("Total Reward per Episode")
    plt.title("Trained in Non-Perturbed Env")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_rewards.png")
    plt.close()

    # Plot max costs
    plt.figure(figsize=(fig_size+4, fig_size))  # Adjust aspect ratio for square plots

    # Determine the maximum y-value among all max_costs
    max_y_value = max(max(dataset['max_costs']) for dataset in data)
    y_limit = max(2.5, max_y_value * 1.1)  # Ensure y-limit is at least 2.5

    # Set y-axis limits explicitly
    plt.ylim(-1, y_limit)

    # Add light red background above the threshold and light blue below
    plt.axhspan(2.0, y_limit, color='red', alpha=0.1)
    plt.axhspan(-1, 2.0, color='blue', alpha=0.1)

    for dataset, label in zip(data, labels):
        max_costs = np.array(dataset['max_costs'])
        smoothed_max_costs = smooth(max_costs, smooth_window)
        smoothed_x = range(len(smoothed_max_costs))

        # Plot smoothed max costs
        plt.plot(smoothed_x, smoothed_max_costs, label=f"{label}")

    # Always plot the threshold line at y=2.0
    plt.axhline(y=2.0, color='black', linestyle='--', label="Baseline")

    plt.xlabel("Episode")
    plt.ylabel("Peak Cost")
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_label_coords(-0.1, 0.5)  # Adjust the y-axis label position
    # plt.title("Trained in Non-Perturbed Env")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png")
    plt.close()


    # Plot total costs
    plt.figure(figsize=(fig_size, fig_size))  # Adjust aspect ratio for square plots
    for dataset, label in zip(data, labels):
        costs = np.array(dataset['costs'])
        smoothed_costs = smooth(costs, smooth_window)
        smoothed_x = range(len(smoothed_costs))

        # Plot smoothed total costs
        plt.plot(smoothed_x, smoothed_costs, label=f"{label}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    # plt.title("Total Cost per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_total_costs.png")
    plt.close()

def plot_evaluation2(data, labels, save=False, base_filename="evaluation_plot", smooth_window=10):
    plt.rcParams.update({'font.size': 85, 'lines.linewidth': 7})
    fig_size = 28

    # Plot total rewards
    plt.figure(figsize=(fig_size, fig_size))
    for dataset, label in zip(data, labels):
        rewards = np.array(dataset['rewards'])
        smoothed_rewards = smooth(rewards, smooth_window)
        smoothed_x = range(len(smoothed_rewards))
        plt.plot(smoothed_x, smoothed_rewards, label=f"{label}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_rewards.png")
    plt.close()

    # Plot max costs
    plt.figure(figsize=(fig_size, fig_size))
    for dataset, label in zip(data, labels):
        max_costs = np.array(dataset['max_costs'])
        smoothed_max_costs = smooth(max_costs, smooth_window)
        smoothed_x = range(len(smoothed_max_costs))
        plt.plot(smoothed_x, smoothed_max_costs, label=f"{label}")
    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.title("Max Cost per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png")
    plt.close()

    # Plot total costs
    plt.figure(figsize=(fig_size, fig_size))
    for dataset, label in zip(data, labels):
        costs = np.array(dataset['costs'])
        smoothed_costs = smooth(costs, smooth_window)
        smoothed_x = range(len(smoothed_costs))
        plt.plot(smoothed_x, smoothed_costs, label=f"{label}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    plt.title("Cumulative Cost per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_total_costs.png")
    plt.close()

def plot_evaluation(results, save_paths, gravity_perturbation_stds, labels, save=False, base_filename="evaluation_plot", smooth_window=10):
    """
    Plot evaluation results for multiple models across gravity perturbation std values.

    Args:
        results: Dictionary where keys are model directories and values are lists of results for each gravity perturbation std.
        save_paths: List of model directory paths.
        gravity_perturbation_stds: List of gravity perturbation std values.
        labels: List of labels corresponding to each model directory.
        save: Whether to save the plots.
        base_filename: Base file name to save the plots.
        smooth_window: Window size for smoothing.
    """
    plt.rcParams.update({'font.size': 40, 'lines.linewidth': 6})
    fig_size = 20

    # Prepare legend elements
    legend_elements = []

    # Plot total rewards
    plt.figure(figsize=(fig_size, fig_size))
    for save_path, label in zip(save_paths, labels):
        for i, std in enumerate(gravity_perturbation_stds):
            rewards = np.array(results[save_path][i]['rewards'])
            smoothed_rewards = smooth(rewards, smooth_window)
            smoothed_x = range(len(smoothed_rewards))
            line, = plt.plot(smoothed_x, smoothed_rewards, label=f"{label} (std={std})")
            legend_elements.append(line)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode")
    plt.legend()
    plt.grid(True)

    # Adjust y-axis ticks for better readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    if save:
        plt.savefig(f"{base_filename}_rewards.png")
    plt.close()

    # Save horizontal and vertical legends for rewards plot
    save_legend(legend_elements, labels, f"{base_filename}_rewards_legend_horizontal.png", horizontal=True)
    save_legend(legend_elements, labels, f"{base_filename}_rewards_legend_vertical.png", horizontal=False)

    # Plot max costs
    plt.figure(figsize=(fig_size, fig_size))
    for save_path, label in zip(save_paths, labels):
        for i, std in enumerate(gravity_perturbation_stds):
            max_costs = np.array(results[save_path][i]['max_costs'])
            smoothed_max_costs = smooth(max_costs, smooth_window)
            smoothed_x = range(len(smoothed_max_costs))
            line, = plt.plot(smoothed_x, smoothed_max_costs, label=f"{label} (std={std})")
            legend_elements.append(line)

    # Add dashed baseline and shaded regions
    y_limit = plt.gca().get_ylim()[1]  # Get current y-axis upper limit
    plt.axhline(y=2.0, color='black', linestyle='--', linewidth=2, label="Baseline")
    plt.axhspan(2.0, y_limit, color='red', alpha=0.1)
    plt.axhspan(plt.gca().get_ylim()[0], 2.0, color='blue', alpha=0.1)

    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.title("Max Cost per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_max_costs.png")
    plt.close()

    # Save horizontal and vertical legends for max costs plot
    save_legend(legend_elements, labels, f"{base_filename}_max_costs_legend_horizontal.png", horizontal=True)
    save_legend(legend_elements, labels, f"{base_filename}_max_costs_legend_vertical.png", horizontal=False)

    # Plot total costs
    plt.figure(figsize=(fig_size, fig_size))
    for save_path, label in zip(save_paths, labels):
        for i, std in enumerate(gravity_perturbation_stds):
            costs = np.array(results[save_path][i]['costs'])
            smoothed_costs = smooth(costs, smooth_window)
            smoothed_x = range(len(smoothed_costs))
            line, = plt.plot(smoothed_x, smoothed_costs, label=f"{label} (std={std})")
            legend_elements.append(line)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    plt.title("Cumulative Cost per Episode")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"{base_filename}_total_costs.png")
    plt.close()

    # Save horizontal and vertical legends for total costs plot
    save_legend(legend_elements, labels, f"{base_filename}_total_costs_legend_horizontal.png", horizontal=True)
    save_legend(legend_elements, labels, f"{base_filename}_total_costs_legend_vertical.png", horizontal=False)


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


# Function to run the environment with different gravity perturbation std values
def run_and_plot_comparison(args, save_path, gravity_perturbation_stds, num_episodes=100):
    """
    Run the CartPole environment with different gravity perturbation standard deviations
    and plot the results for comparison.

    Args:
        args: Argument parser with required parameters.
        save_path: Path to the saved agent.
        gravity_perturbation_stds: List of gravity perturbation std values to test.
        num_episodes: Number of episodes to test.
    """
    results = []
    labels = []

    for std in gravity_perturbation_stds:
        print(f"Running simulation with gravity_perturbation_std = {std}")
        
        # Create a new environment instance with the specified gravity perturbation std
        env = CartPolePerturbedEnv(gravity_perturbation_std=std)
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        args.max_action = float(env.action_space.high[0])
        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
        
        agent, state_norm, reward_scaling = load_agent(args, save_path)
        rewards, costs, max_costs = test_agent(agent, env, num_episodes, state_norm)
        results.append({'rewards': rewards, 'costs': costs, 'max_costs': max_costs})
        labels.append(f"Gravity Perturbation Std = {std}")

    # Plot the evaluation results
    plot_evaluation(results, labels, save=True, base_filename="plot_inference/gravity_comparison", smooth_window=10)



if __name__ == "__main__":
    # Define your arguments (or load them from a config file)
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='CartPolePerturbedEnv',help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(5e3), help="Maximum number of training steps")
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
    parser.add_argument("--persistent_eps", type=float, default=2.0, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=4, help="seed 2, 5, 7, 11, 17") 
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")
    parser.add_argument("--baseline",type=int,default=9,help="baseline")
    parser.add_argument("--lambda_",type=int,default=1.0,help="lambda")
    parser.add_argument("--beta",type=float,default=25.0,help="beta") 
    parser.add_argument("--run",type=int,default=1,help="run_number") 
    parser.add_argument("--warm_start_flag",type=int,default=0,help="warm_start_flag") 
    parser.add_argument("--warm_start_episode",type=int,default=300,help="warm_start_episode")

    args = parser.parse_args()

    # # Create the environment
    # env = CartPolePerturbedEnv()
    # args.max_action = float(env.action_space.high[0])
    # args.state_dim = env.observation_space.shape[0]
    # args.action_dim = env.action_space.shape[0]
    
    # env.reset(seed=args.seed)
    # env.action_space.seed(args.seed)

    # Specify the directories for the trained models
    # directories = [
    #     "./models/CartPolePerturbedEnv/run5/Best_RCAC",
    #     "./models/CartPoleCostEnv/run5/Best_RCAC"
    # ]
    # labels = ["CartPolePerturbedEnv", "CartPoleCostEnv"]

    # directories = [
    #     "./models/CartPoleCostEnv/run2/Best_RCAC",
    #     "./models/CartPoleCostEnv_PD_RCRL/run2/Best_RCAC"
    # ]
    # labels = ["Surrogate Objective", "PrimalDual"]

    directories = [
        "./models/CartPoleCostEnv/run2/Best_RCAC",
        "./models/CartPoleCostEnv_PD_RCRL/run3/Best_RCAC",
        "./models/CartPolePerturbedEnv/run3/Best_RCAC"
    ]
    labels = ["Surrogate Obj(NP)", "PD(NP)", "Ours(P)"]

    gravity_perturbation_stds = [1.0, 6.0]

    # Run tests for all models across different gravity perturbations
    results = test_multiple_dirs(args, directories, gravity_perturbation_stds, num_episodes=100)

    # Plot the evaluation results
    plot_evaluation(results, directories, gravity_perturbation_stds, labels, save=True, base_filename="plot_inference/gravity_comparison", smooth_window=30)

    # run_and_plot_comparison(args, directories, gravity_perturbation_stds, num_episodes=100)




    # # Test agents from multiple directories
    # results = test_multiple_dirs(args, env, directories, num_episodes=100)

    # # Plot the evaluation results
    # plot_evaluation(results, labels, save=True, base_filename="plot_inference/New_comparison6", smooth_window=10)
    # # plot_evaluation(results, labels, save=True, base_filename="plot_inference/comparison_plot_fixed5", smooth_window=80)
