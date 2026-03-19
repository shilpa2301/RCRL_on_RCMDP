
import torch
import numpy as np
from ipm_rcmdp_rcrl import Actor_Beta, Actor_Gaussian, Actor_Discrete, Critic, CostCritic, Robust_RCAC_NPG, CartPolePerturbedEnv, Normalization, RunningMeanStd, RewardScaling
import argparse
import pickle

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
    # Initialize the agent
    agent = Robust_RCAC_NPG(args)

    # Load actor weights
    actor_path = f"{save_path}_actor"
    agent.actor.load(actor_path)

    # Load reward critic weights
    rcritic_path = f"{save_path}_Rcritic"
    agent.Rcritic.load(rcritic_path)

    # Load cost critic weights
    ccritic_path = f"{save_path}_Ccritic"
    agent.Ccritic.load(ccritic_path)

    # Load state normalization and reward scaling if used
    with open(f'{save_path}_state_norm', 'rb') as file1:
        state_norm = pickle.load(file1)

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
        total_reward = 0
        total_cost = 0
        max_cost = float('-inf')

        done = False

        while not done:
            # Normalize state if necessary
            if state_norm:
                state = state_norm(state, update=False)

            # Get action from the policy
            action = agent.evaluate(state)
            if agent.policy_dist == "Beta":
                action = 2 * (action - 0.5) * agent.max_action  # Map [0, 1] to [-max_action, max_action]

            # Step in the environment
            next_state, reward, cost, done, _ = env.step(action)

            if state_norm:
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
    parser.add_argument("--persistent_eps", type=float, default=200.0, help="Persistent Safety Perturbation")
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
    
    
    
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")

        # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")

    args = parser.parse_args([])

    # args = argparse.Namespace(
    #     state_dim=4,
    #     action_dim=1,
    #     hidden_width=64,
    #     max_action=10.0,
    #     policy_dist="Gaussian",
    #     use_tanh=True,
    #     use_orthogonal_init=True,
    #     lr_a=3e-4,
    #     lr_c=3e-4,
    #     gamma=0.99,
    #     lamda=0.95,
    #     epsilon=0.2,
    #     K_epochs=10,
    #     entropy_coef=0.01,
    #     use_adv_norm=True,
    #     use_grad_clip=True,
    #     set_adam_eps=True,
    #     adaptive_alpha=False,
    #     weight_reg=0,
    #     baseline=200,
    #     lambda_=50,
    #     batch_size=2048,
    #     mini_batch_size=64
    # )

    # Create the environment
    env = CartPolePerturbedEnv()
    args.max_action = float(env.action_space.high[0])
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    # Specify the save path for the trained models
    save_path = "./models/run1/RCAC"

    # Load the trained agent and normalization objects
    agent, state_norm, reward_scaling = load_agent(args, save_path)

    

    # Test the agent
    num_episodes = 100
    rewards, costs, max_costs = test_agent(agent, env, num_episodes, state_norm)

    # Analyze the results
    print(f"Average Reward: {np.mean(rewards)}")
    print(f"Average Cost: {np.mean(costs)}")
    print(f"Average Max Cost: {np.mean(max_costs)}")

    # Optional: Save the results to a file
    # np.save("test_rewards.npy", rewards)
    # np.save("test_costs.npy", costs)
    # print("Test results saved to 'test_rewards.npy' and 'test_costs.npy'.")
