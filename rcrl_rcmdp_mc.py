# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
import matplotlib.pyplot as plt  # Import for plotting

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

log_file = "rcrl_training_log_episode_mc_betaexp_normalizedcost_eps0.5_beta_1_200.log"
plot_file = "training_metrics_episode_mc_betaexp_normalizedcost_eps0.5_beta_1_200.png"

# === Logging Configuration ===
logging.basicConfig(
    filename=log_file,  # Log file name
    filemode="w",                # Overwrite the file each time the script is run
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO           # Log level (INFO, DEBUG, ERROR, etc.)
)

# === Hyperparameters ===
gamma = 0.99
hidden_dim = 128 #256
learning_rate = 1e-3
epochs = 10000  # Number of episodes to train
lambda_fixed = 1.0
# beta = 50 #2 #1000.0
# Parameters for dynamic beta
min_beta = 1  # Starting value of beta
max_beta = 200  # Maximum value of beta
scale = epochs / 3 #5  # Controls the rate of growth (adjust as needed)
perturb_eps = 0.1
epsilon_tolerance = 0.5 #0.1
# CartPole specific
safe_distance = 1.0 #0.5
safe_angle = 0.12

# === Environment ===
env = gym.make("CartPole-v1", render_mode=None)
print("max permitted steps:", env.spec.max_episode_steps)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# === Actor and Critic Networks ===
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)

class ValueCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state)

class CostCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

    def forward(self, state):
        return self.model(state)

# === Utilities ===
def add_uniform_noise(state, eps=0.05):
    """
    Add uniform noise to the state.
    Args:
        state: A PyTorch tensor representing the state.
        eps: The maximum magnitude of the noise.
    Returns:
        A PyTorch tensor with added noise.
    """
    noise = np.random.uniform(-eps, eps, size=state.shape)
    # noise = np.random.uniform(0, eps, size=state.shape)
    return torch.FloatTensor(state.numpy() + noise)

def discount(values, gamma):
    result = []
    G = 0
    for v in reversed(values):
        G = v + gamma * G
        result.insert(0, G)
    return torch.FloatTensor(result)


def robust_value_function(trajectory, actor, gamma):
    """
    Compute the robust value function for a given trajectory and actor using Monte Carlo returns.
    
    Args:
        trajectory (dict): Dictionary containing the trajectory data:
            - 'states': List of states in the trajectory.
            - 'actions': List of actions taken in the trajectory (stored as scalars).
            - 'costs': List of costs incurred at each step in the trajectory.
        actor (nn.Module): The actor network that outputs the action probabilities.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Robust value function estimates (V_h^pi) for each state in the trajectory.
        float: Maximum robust value (V_L^pi) across all states in the trajectory.
    """
    states = trajectory['states']
    actions = trajectory['actions']
    costs = trajectory['costs']

    v_h_pi_values = []
    cost_values = []

    # Compute discounted returns for costs
    discounted_costs = discount(costs, gamma)

    # Iterate through all states in the trajectory
    for t, (state, action) in enumerate(zip(states, actions)):
        # Get the policy probabilities for the current state
        probs = actor(state)
        dist = Categorical(probs)

        # Compute Q-values for each possible action
        q_values = []
        for a in range(len(probs)):  # Iterate through all possible actions
            if a == action.item():  # Compare with the scalar action
                q_value = discounted_costs[t]  # Use the actual discounted cost for the taken action
            else:
                q_value = discounted_costs[t]  # For simplicity, assume the same cost for other actions
            q_values.append(q_value)

        # Compute V_h^pi (expected value of the robust value function)
        q_values = torch.tensor(q_values)
        v_h_pi = torch.sum(probs * q_values)
        v_h_pi_values.append(v_h_pi.item())
        cost_values.append(v_h_pi.item())

    # Compute V_L^pi as the maximum robust value
    vl_pi = max(cost_values)

    return torch.tensor(v_h_pi_values), vl_pi



def compute_cost(state, safe_distance, safe_angle):
    x, x_dot, theta, theta_dot = state

    # Define maximum possible deviations (based on environment constraints)
    max_position_deviation = 2.4  # Maximum cart position (e.g., CartPole limits)
    max_angle_deviation = 0.2094395  # ~12 degrees in radians (CartPole limits)

    # Continuous cost based on how far the state deviates from safe ranges
    position_cost = max(0, (abs(x) - safe_distance)**2)
    angle_cost = max(0, (abs(theta) - safe_angle)**2)

    # Normalize costs
    normalized_position_cost = position_cost / (max_position_deviation - safe_distance)**2
    normalized_angle_cost = angle_cost / (max_angle_deviation - safe_angle)**2

    # Combine normalized costs (weighted sum)
    total_normalized_cost = (normalized_position_cost + normalized_angle_cost) / 2

    return total_normalized_cost






# === Plotting Function ===
# === Plotting Function ===
def plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=False, filename="training_metrics.png"):
    """
    Plot the metrics (reward, cost, and V_L(pi)) over episodes and optionally save the plot.
    Args:
        episode_rewards: List of total rewards per episode.
        episode_costs: List of total costs per episode.
        episode_vl_pi: List of V_L(pi) values per episode.
        save: Whether to save the plot to a file.
        filename: File name to save the plot.
    """
    # Enable interactive mode
    plt.ion()
    
    # Clear the current figure but don't close it
    plt.clf()

    # Plot total rewards
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Plot total costs
    plt.subplot(3, 1, 2)
    plt.plot(episode_costs, label="Total Cost", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("Total Cost per Episode")
    plt.legend()

    # Plot V_L(pi)
    plt.subplot(3, 1, 3)
    plt.plot(episode_vl_pi, label="V_L(pi)", color="green")
    plt.xlabel("Episode")
    plt.ylabel("V_L(pi)")
    plt.title("V_L(pi) per Episode")
    plt.legend()

 


    # Update the plot
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update

    if save:
        plt.savefig(filename)  # Save the plot to a file


def main():
    # === Initialize Networks and Optimizers ===
    actor = Actor()
    reward_critic = ValueCritic()
    cost_critic = CostCritic()

    actor_optim = optim.Adam(actor.parameters(), lr=learning_rate)
    reward_optim = optim.Adam(reward_critic.parameters(), lr=learning_rate)
    cost_optim = optim.Adam(cost_critic.parameters(), lr=learning_rate)

    # === Tracking ===
    # dataF = {'cost': [], 'reward': []}

    # last_50_actor_params = []
    best_reward = float('-inf')
    best_actor_state_dict = None

    # Metrics for plotting
    episode_rewards = []
    episode_costs = []
    episode_vl_pi = []

    # Exploration parameters for epsilon-greedy
    # epsilon = epsilon_start


    # === Training Loop ===
    for ep in range(epochs):
        print("ep=", ep)
        # beta = 1 + (ep / epochs) * (max_beta - min_beta)
        

        # Update beta dynamically
        beta = min(max_beta, min_beta * np.exp(ep / scale))

        state, _ = env.reset()
        state = add_uniform_noise(torch.FloatTensor(state), perturb_eps)
        # state = torch.FloatTensor(state)

        log_probs = []
        rewards = []
        costs = []
        reward_values = []
        cost_values = []

        total_reward = 0
        total_cost = 0
        done = False

        # h_s_values = []

        # Store trajectory data for robust value function
        trajectory = {'states': [], 'actions': [], 'next_states': [], 'costs': []}

        while not done:
            
            # Track safety violations
            # h_s_values.append(h_s)

            probs = actor(state)
            # print(f"Action probabilities: {probs.detach().numpy()}")
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            next_state = add_uniform_noise(torch.FloatTensor(next_state), perturb_eps)
            next_state = torch.FloatTensor(next_state)

            cost = compute_cost(state, safe_distance, safe_angle)

            # Store trajectory data
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['next_states'].append(next_state)
            trajectory['costs'].append(cost)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            costs.append(cost)
            # reward_values.append(reward_critic(state))
            # cost_values.append(cost_critic(state))

            total_reward += reward
            # print("reward, total reward= ", reward, total_reward)
            total_cost += cost
            state = next_state

        # print(f"Episode ended at step {len(rewards)}")
        # print(f"Cart position: {state[0].item()}, Pole angle: {state[2].item()}")

        # Discounted returns
        reward_returns = discount(rewards, gamma)
        cost_returns = discount(costs, gamma)
        # reward_values = torch.cat(reward_values).squeeze()
        # cost_values = torch.cat(cost_values).squeeze()
        # log_probs = torch.stack(log_probs)

        # adv_r = reward_returns - reward_values.detach()
        # adv_c = cost_returns - cost_values.detach()

        # Compute V_L(pi) and RCMDP objective
        # Compute robust value function and vl_pi for the episode
        v_h_pi_values, vl_pi = robust_value_function(trajectory, actor, gamma)
        # vl_pi = max(h_s_values)  # V_L(pi) = max_s V_h^pi(s)
        # avg_reward = reward_returns.mean().item()
        penalty_term = max(0, vl_pi - epsilon_tolerance)  # Apply penalty only if V_L(pi) > epsilon
        beta_penalty = beta * penalty_term

        # Compute advantages
        reward_values = torch.cat([reward_critic(s) for s in trajectory['states']]).squeeze()
        cost_values = torch.cat([cost_critic(s) for s in trajectory['states']]).squeeze()
        log_probs = torch.stack(log_probs)

        adv_r = reward_returns - reward_values.detach()
        adv_c = cost_returns - cost_values.detach()

        chosen_adv = []
        for vr, vc, ar, ac in zip(reward_returns, cost_returns, adv_r, adv_c):
            # if rcmdp_objective > avg_reward / lambda_fixed:
            if vr.item() > lambda_fixed * beta_penalty :
                chosen_adv.append(ar)  # Penalize cost
            else:
                chosen_adv.append(-ac)  # Reward advantage
        chosen_adv = torch.stack(chosen_adv)

        #Losses
        actor_loss = -(log_probs * chosen_adv).mean() # - 0.1 * entropy_loss
        reward_loss = nn.functional.mse_loss(reward_values, reward_returns)
        cost_loss = nn.functional.mse_loss(cost_values, cost_returns)


        # Backprop
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        reward_optim.zero_grad()
        reward_loss.backward()
        reward_optim.step()

        cost_optim.zero_grad()
        cost_loss.backward()
        cost_optim.step()

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_vl_pi.append(vl_pi)

        # Track Best Actor
        # if vl_pi <= epsilon_tolerance and total_reward > best_reward:
        if total_reward > best_reward:
            best_reward = total_reward
            best_actor_state_dict = deepcopy(actor.state_dict())


        if (ep + 1) % 50 == 0:
            print(f"Ep {ep + 1} | Reward: {total_reward:.1f} | Cost: {total_cost:.2f} | V_L(pi): {vl_pi:.3f} | "
                  f"Actor Loss: {actor_loss.item():.3f} | Best Reward (under safety): {best_reward:.1f} | Beta : {beta}")
            plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=True, filename=plot_file)

        # # Save the final plot
    plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=True, filename=plot_file)

    # Save all models and data
    env.close()
    df = pd.DataFrame(dataF)
    df.to_excel('rcmdp_with_robustness_episode_wise.xlsx')

    torch.save(actor.state_dict(), 'actor_episode_wise.pth')
    torch.save(reward_critic.state_dict(), 'reward_critic_episode_wise.pth')
    torch.save(cost_critic.state_dict(), 'cost_critic_episode_wise.pth')


if __name__ == "__main__":
    main()

