
import torch
import numpy as np
from ipm_rcmdp_rcrl import Actor_Beta, Actor_Gaussian, Actor_Discrete, Critic, CostCritic, Robust_RCAC_NPG, CartPoleCostEnv, CartPolePerturbedEnv, Normalization, RunningMeanStd, RewardScaling
import argparse
import pickle
import matplotlib.pyplot as plt
import os


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

def plot_metrics(episode_rewards, episode_costs, max_costs, save=False, filename="test_metrics.png"):
    """
    Plot the metrics (reward and cost) over episodes and optionally save the plot.
    Args:
        episode_rewards: List of total rewards per episode.
        episode_costs: List of total costs per episode.
        save: Whether to save the plot to a file.
        filename: File name to save the plot.
    """
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the current figure to avoid overlapping plots

    # Plot total rewards
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Plot max costs
    plt.subplot(3, 1, 2)
    plt.plot(max_costs, label="Max Cost", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Max Cost")
    plt.title("Max Cost per Episode")
    plt.legend()

    # Plot total costs
    plt.subplot(3, 1, 3)
    plt.plot(episode_costs, label="Total Cost", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Total Cost")
    plt.title("Total Cost per Episode")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Define your arguments (or load them from a config file)
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='CartPolePerturbedEnv',help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian or Discrete")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--max_train_steps", type=int, default=int(4.5e3), help="Maximum number of training steps")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--persistent_eps", type=float, default=2.0, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--lambda_",type=int,default=50,help="lambda")
    #  parser.add_argument("--baseline",type=int,default=200,help="baseline")
    parser.add_argument("--beta",type=float,default=2.0,help="beta")

    
    
    
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--evaluate_freq", type=float, default=2e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")

        # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")

    args = parser.parse_args([])

    # Create the environment
    env = CartPolePerturbedEnv() #CartPoleCostEnv() # CartPolePerturbedEnv()
    args.max_action = float(env.action_space.high[0])
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Specify the save path for the trained models
    run = 1
    save_path = "./models/CartPolePerturbedEnv/run1/RCAC"
    save_path2 = "./models/CartPoleCostEnv/run1/RCAC"
    # save_path = "./models_baseline/RCAC"

    # Load the trained agent and normalization objects
    agent, state_norm, reward_scaling = load_agent(args, save_path)

    

    # Test the agent
    num_episodes = 100
    rewards, costs, max_costs = test_agent(agent, env, num_episodes, state_norm)

    # Analyze the results
    print(f"Average Reward: {np.mean(rewards)}")
    print(f"Average Cost: {np.mean(costs)}")
    print(f"Average Max Cost: {np.mean(max_costs)}")

    # Plot the results
    if not os.path.exists(f"./plot_data/{args.env}"):
        os.makedirs("./plot_data/{args.env}")
    # plot_metrics(rewards, costs, max_costs, save=True, filename=f"./plot_inference/{args.env}/run{run}_cartpole_on_robust.png")
    plot_metrics(rewards, costs, max_costs, save=True, filename=f"./plot_inference/{args.env}/run{run}_cartpole_perturbed.png")


    # Optional: Save the results to a file
    # np.save("test_rewards.npy", rewards)
    # np.save("test_costs.npy", costs)
    # print("Test results saved to 'test_rewards.npy' and 'test_costs.npy'.")
