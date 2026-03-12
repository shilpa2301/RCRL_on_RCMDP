# import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
import matplotlib.pyplot as plt  # Import for plotting
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

log_file = "rcrl_training_log_td_quadrotor.log"
plot_file = "training_metrics_td_quadrotor.png"

# === Logging Configuration ===
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# === Hyperparameters ===
gamma = 0.99
hidden_dim = 128
learning_rate = 1e-3
epochs = 10000
lambda_fixed = 1.0
min_beta = 1
max_beta = 200
scale = epochs / 3
# beta = 50
perturb_eps = 0.1
epsilon_tolerance = 0.5
safe_distance = 1.0
safe_angle = 0.12
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# === Environment ===
CONFIG_FACTORY = ConfigFactory()
CONFIG_FACTORY.parser.set_defaults(overrides=['/project/ag2682/sm3934/RCRL_on_RMDP/env_configs/constrained_tracking_reset.yaml'])
config = CONFIG_FACTORY.merge()

env = make("quadrotor", **config.quadrotor_config)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# === Actor and Critic Networks ===
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)  # Output mean
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)  # Output log standard deviation

    def forward(self, state):
        x = self.fc(state)
        mean = torch.tanh(self.mean_layer(x))  # Tanh to keep mean in [-1, 1]
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)  # Clamp log std for stability
        # std = torch.exp(log_std)  # Convert log std to std
        return mean, log_std

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
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.model(state)

# === Utilities ===

def select_action(state, actor):
    mean, log_std = actor(state)
    std = log_std.exp()  # Convert log std to std
    normal = torch.distributions.Normal(mean, std)
    z = normal.rsample()  # Use rsample() for reparameterization
    action = torch.tanh(z)  # Ensure actions are within [-1, 1]
    return action


def add_uniform_noise(state, eps=0.05):
    noise = np.random.uniform(-eps, eps, size=state.shape)
    return torch.FloatTensor(state.numpy() + noise)

def discount(values, gamma):
    result = []
    G = 0
    for v in reversed(values):
        G = v + gamma * G
        result.insert(0, G)
    return torch.FloatTensor(result)

# def compute_cost(state, safe_distance, safe_angle):
#     x, x_dot, theta, theta_dot = state

#     # Define maximum possible deviations (based on environment constraints)
#     max_position_deviation = 2.4  # Maximum cart position (e.g., CartPole limits)
#     max_angle_deviation = 0.2094395  # ~12 degrees in radians (CartPole limits)

#     # Continuous cost based on how far the state deviates from safe ranges
#     position_cost = max(0, (abs(x) - safe_distance)**2)
#     angle_cost = max(0, (abs(theta) - safe_angle)**2)

#     # Normalize costs
#     normalized_position_cost = position_cost / (max_position_deviation - safe_distance)**2
#     normalized_angle_cost = angle_cost / (max_angle_deviation - safe_angle)**2

#     # Combine normalized costs (weighted sum)
#     total_normalized_cost = (normalized_position_cost + normalized_angle_cost) / 2

#     return total_normalized_cost


def robust_value_function(trajectory, actor, cost_critic, gamma):
    states = trajectory['states']
    actions = trajectory['actions']
    next_states = trajectory['next_states']
    costs = trajectory['costs']

    v_h_pi_values = []
    cost_values = []

    for state, action, next_state, h_s in zip(states, actions, next_states, costs):
        # Get mean and log_std from the actor for the current state
        mean, log_std = actor(state)
        std = torch.exp(log_std)  # Convert log_std to standard deviation

        # Create a Normal distribution for the action
        dist = torch.distributions.Normal(mean, std)

        # Sample multiple actions to approximate the expectation
        sampled_actions = dist.rsample((100,))  # Reparameterized samples
        sampled_actions = torch.tanh(sampled_actions)  # Ensure actions are within [-1, 1]

        q_values = []
        for sampled_action in sampled_actions:
            # Check if the sampled action is close to the actual action in the trajectory
            if torch.allclose(sampled_action, action, atol=1e-3):
                h_s = h_s  # Use the actual cost from the trajectory
            else:
                h_s = info.get('cost', 0)

            next_value = cost_critic(next_state).item()
            q_value = h_s + gamma * max(h_s, next_value)
            q_values.append(q_value)

        # Compute the expected robust value function
        q_values = torch.tensor(q_values)
        probs = dist.log_prob(sampled_actions).exp()  # Get probabilities for sampled actions
        probs = probs / probs.sum()  # Normalize probabilities to sum to 1
        v_h_pi = torch.sum(probs * q_values)  # Weighted sum of Q-values
        v_h_pi_values.append(v_h_pi.item())
        cost_values.append(v_h_pi.item())

    vl_pi = max(cost_values)
    return torch.tensor(v_h_pi_values), vl_pi

def plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=False, filename="training_metrics.png"):
    plt.ion()
    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(episode_costs, label="Total Cost", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("Total Cost per Episode")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(episode_vl_pi, label="V_L(pi)", color="green")
    plt.xlabel("Episode")
    plt.ylabel("V_L(pi)")
    plt.title("V_L(pi) per Episode")
    plt.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

    if save:
        plt.savefig(filename)

def main():
    actor = Actor()
    reward_critic = ValueCritic()
    cost_critic = CostCritic()

    actor_optim = optim.Adam(actor.parameters(), lr=learning_rate)
    reward_optim = optim.Adam(reward_critic.parameters(), lr=learning_rate)
    cost_optim = optim.Adam(cost_critic.parameters(), lr=learning_rate)

    best_reward = float('-inf')
    best_actor_state_dict = None

    episode_rewards = []
    episode_costs = []
    episode_vl_pi = []

    # Exploration parameters for epsilon-greedy
    epsilon = epsilon_start

    for ep in range(epochs):
        # beta = min(max_beta, min_beta * np.exp(ep / scale))
        beta = 1 + (ep / epochs) * (max_beta - min_beta)

        state, _ = env.reset()
        state = add_uniform_noise(torch.FloatTensor(state), perturb_eps)

        total_reward = 0
        total_cost = 0
        done = False

        trajectory = {'states': [], 'actions': [], 'next_states': [], 'costs': []}

        log_probs = []
        rewards = []
        costs = []
        reward_values = []
        cost_values = []

        while not done:
            # probs = actor(state)
            # dist = Categorical(probs)
            # action = dist.sample()
            # Epsilon-greedy exploration
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            action = select_action(state_tensor, actor).detach().numpy()
            
            # Scale the action according to the environment's action space
            action = action * env.action_space.high

            # Scale the action to match the action space range
            # action = action.detach().numpy() * env.action_space.high

            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy() 
            next_state, reward, done, info = env.step(action.numpy())
            # done = done or truncated
            next_state = add_uniform_noise(torch.FloatTensor(next_state), perturb_eps)

            cost = info.get('cost', 0)

            # Store trajectory data for robust value function
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['next_states'].append(next_state)
            trajectory['costs'].append(cost)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            costs.append(cost)
            reward_values.append(reward_critic(state))
            cost_values.append(cost_critic(state))

            # TD Target for reward critic
            reward_target = reward + gamma * reward_critic(next_state).item() * (1 - done)
            reward_value = reward_critic(state)
            reward_loss = nn.functional.mse_loss(reward_value, torch.tensor([reward_target]))

            # TD Target for cost critic
            cost_target = cost + gamma * cost_critic(next_state).item() * (1 - done)
            cost_value = cost_critic(state)
            cost_loss = nn.functional.mse_loss(cost_value, torch.tensor([cost_target]))

            # Update reward critic
            reward_optim.zero_grad()
            reward_loss.backward()
            reward_optim.step()

            # Update cost critic
            cost_optim.zero_grad()
            cost_loss.backward()
            cost_optim.step()

            total_reward += reward
            total_cost += cost
            state = next_state

        # Compute V_L(pi) using robust value function
        v_h_pi_values, vl_pi = robust_value_function(trajectory, actor, cost_critic, gamma)

        # Decay epsilon (for epsilon-greedy exploration)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Compute penalty term
        penalty_term = max(0, vl_pi - epsilon_tolerance)
        beta_penalty = beta * penalty_term

        # Compute advantages
        reward_returns = discount(rewards, gamma)
        cost_returns = discount(costs, gamma)
        reward_values = torch.cat(reward_values).squeeze()
        cost_values = torch.cat(cost_values).squeeze()
        log_probs = torch.stack(log_probs)

        adv_r = reward_returns - reward_values.detach()
        adv_c = cost_returns - cost_values.detach()

        # Compute chosen_adv
        chosen_adv = []
        for vr, vc, ar, ac in zip(reward_returns, cost_returns, adv_r, adv_c):
            if vr.item() > lambda_fixed * beta_penalty:
                chosen_adv.append(ar)  # Prioritize reward advantage
            else:
                chosen_adv.append(-ac)  # Penalize cost
        chosen_adv = torch.stack(chosen_adv)

        # Actor loss
        # actor_loss = -(log_probs * chosen_adv).mean()
        # Actor loss with entropy regularization
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))  # Compute entropy
        actor_loss = -(log_probs * chosen_adv).mean() - 0.01 * entropy  # Add entropy term

        # Update actor
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Update metrics
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_vl_pi.append(vl_pi)

        # Track Best Actor
        if total_reward > best_reward:
            best_reward = total_reward
            best_actor_state_dict = deepcopy(actor.state_dict())

        if (ep + 1) % 50 == 0:
            print(f"Ep {ep + 1} | Reward: {total_reward:.1f} | Cost: {total_cost:.2f} | V_L(pi): {vl_pi:.3f} | "
                  f"Actor Loss: {actor_loss.item():.3f} | Best Reward (under safety): {best_reward:.1f} | Beta : {beta}")
            plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=True, filename=plot_file)

    plot_metrics(episode_rewards, episode_costs, episode_vl_pi, save=True, filename=plot_file)

    env.close()
    torch.save(actor.state_dict(), 'actor_td.pth')
    torch.save(reward_critic.state_dict(), 'reward_critic_td.pth')
    torch.save(cost_critic.state_dict(), 'cost_critic_td.pth')

if __name__ == "__main__":
    main()