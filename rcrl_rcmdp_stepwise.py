import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from copy import deepcopy
import logging

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# === Logging Configuration ===
logging.basicConfig(
    filename="rcrl_training_log.log",  # Log file name
    filemode="w",                # Overwrite the file each time the script is run
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO           # Log level (INFO, DEBUG, ERROR, etc.)
)

# === Hyperparameters ===
gamma = 0.99
hidden_dim = 256
learning_rate = 1e-3
episodes = 1000
lambda_fixed = 1.0 #20.0
beta = 1000.0
perturb_eps = 0.1
epsilon_tolerance = 1.0
#cartpole specific
safe_distance = 0.5
safe_angle = 0.17

# === Environment ===
env = gym.make("CartPole-v1")
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

# === Utilities ===
def add_uniform_noise(state, eps=0.05):
    noise = np.random.uniform(-eps, eps, size=state.shape)
    return state + noise

def discount(values, gamma):
    result = []
    G = 0
    for v in reversed(values):
        G = v + gamma * G
        result.insert(0, G)
    return torch.FloatTensor(result)

def robust_value_function(state, actor, cost_critic, perturb_eps, safe_distance, safe_angle):
    # Compute robust value function V_h^pi(s)
    state = add_uniform_noise(state, perturb_eps)
    state = state.to(torch.float32)
    probs = actor(state)
    dist = Categorical(probs)
    actions = torch.arange(len(probs))

    # Compute Q_h^pi(s, a) for all actions
    q_values = []
    h_s = 0
    for action in actions:
        action_tensor = action.detach() if action.requires_grad else action

        # cost = abs(state[0].item())  # h(s) = cost at state s
        cost = compute_cost(state, safe_distance, safe_angle)

        # # Check if the current state is unsafe
        # if cost > safe_distance:
        if cost > 0:
            h_s = 1  # Mark as unsafe

        next_state, _, done, _, _ = env.step(action.item())
        next_state = add_uniform_noise(np.array(next_state), perturb_eps)
        next_state = torch.FloatTensor(next_state)

        # Robust Bellman equation
        next_value = cost_critic(next_state).item() if not done else 0
        q_value = (1 - gamma) * cost + gamma * max(cost, next_value)
        q_values.append(q_value)

    # Compute V_h^pi(s) as the expected value of Q_h^pi(s, a) under the policy
    v_h_pi = torch.sum(probs * torch.tensor(q_values))

    return v_h_pi, h_s

#cartpole specific
# def compute_cost(state, safe_distance):
#     # Check if the cart's position is within the safe distance
#     if abs(state[0]) <= safe_distance:
#         return 0  # Safe
#     else:
#         return 1  # Unsafe

# === Updated compute_cost Function ===
def compute_cost(state, safe_distance, safe_angle):
    """
    Compute the safety cost for a given state.
    Args:
        state: The current state (x, x_dot, theta, theta_dot).
        safe_distance: The maximum allowed absolute position of the cart.
        safe_angle: The maximum allowed absolute angle of the pole.
    Returns:
        cost: 1 if the state violates any safety constraint, 0 otherwise.
    """
    x, x_dot, theta, theta_dot = state

    # Check if the cart's position is within the safe distance
    if abs(x) > safe_distance:
        return 1  # Unsafe: cart is out of safe position range

    # Check if the pole's angle is within the safe angle
    if abs(theta) > safe_angle:
        return 1  # Unsafe: pole angle exceeds safe range

    return 0  # Safe state



def main():
    # === Initialize Networks and Optimizers ===
    actor = Actor()
    reward_critic = ValueCritic()
    cost_critic = ValueCritic()

    actor_optim = optim.Adam(actor.parameters(), lr=learning_rate)
    reward_optim = optim.Adam(reward_critic.parameters(), lr=learning_rate)
    cost_optim = optim.Adam(cost_critic.parameters(), lr=learning_rate)

    # === Tracking ===
    dataF = {'cost': [], 'reward': []}
    last_50_actor_params = []

    best_reward = float('-inf')
    best_actor_state_dict = None

    # === Training Parameters ===
    max_steps = 100000  # Total number of steps to train
    current_step = 0    # Step counter

    # === Training Loop ===
    while current_step < max_steps:
        state, _ = env.reset()
        state = add_uniform_noise(np.array(state), perturb_eps)
        state = torch.FloatTensor(state)

        log_probs = []
        rewards = []
        costs = []
        reward_values = []
        cost_values = []

        total_reward = 0
        total_cost = 0
        done = False

        h_s_values = []

        while not done:
            # Compute robust value function and safety indicator
            v_h_pi, h_s = robust_value_function(state, actor, cost_critic, perturb_eps, safe_distance, safe_angle)

            # Track safety violations
            h_s_values.append(h_s)

            probs = actor(state)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            next_state = add_uniform_noise(np.array(next_state), perturb_eps)
            next_state = torch.FloatTensor(next_state)

            cost = compute_cost(state, safe_distance, safe_angle)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            costs.append(cost)
            reward_values.append(reward_critic(state))
            cost_values.append(cost_critic(state))

            total_reward += reward
            total_cost += cost
            state = next_state

            current_step += 1  # Increment step counter

            # Break the loop if the total steps exceed the limit
            if current_step >= max_steps:
                break

        # Discounted returns
        reward_returns = discount(rewards, gamma)
        cost_returns = discount(costs, gamma)

        reward_values = torch.cat(reward_values).squeeze()
        cost_values = torch.cat(cost_values).squeeze()
        log_probs = torch.stack(log_probs)

        adv_r = reward_returns - reward_values.detach()
        adv_c = cost_returns - cost_values.detach()

        # Compute V_L(pi) and RCMDP objective
        vl_pi = max(h_s_values)  # V_L(pi) = max_s V_h^pi(s)
        avg_reward = reward_returns.mean().item()

        penalty_term = max(0, vl_pi - epsilon_tolerance)  # Apply penalty only if V_L(pi) > epsilon
        rcmdp_objective = max(reward_returns.mean().item() / lambda_fixed, beta * penalty_term)

        # Choose advantage based on RCMDP objective
        chosen_adv = []
        for vr, vc, ar, ac in zip(reward_returns, cost_returns, adv_r, adv_c):
            if rcmdp_objective > avg_reward / lambda_fixed:
                chosen_adv.append(-ac)  # Penalize cost
            else:
                chosen_adv.append(ar)  # Reward advantage
        chosen_adv = torch.stack(chosen_adv)

        # Losses
        entropy_loss = dist.entropy().mean()  # Add entropy regularization
        actor_loss = -(log_probs * chosen_adv).mean() - 0.01 * entropy_loss
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

        # Logging
        dataF['cost'].append(total_cost)
        dataF['reward'].append(total_reward)

        # Store for averaging
        if len(last_50_actor_params) >= 50:
            last_50_actor_params.pop(0)
        last_50_actor_params.append(deepcopy(actor.state_dict()))

        # === Track Best Actor ===
        if vl_pi <= epsilon_tolerance and total_reward > best_reward:  # Enforce statewise persistent safety
            best_reward = total_reward
            best_actor_state_dict = deepcopy(actor.state_dict())

        # Log every 5000 steps
        print(current_step)
        if current_step % 100 == 0:
            logging.info(
                f"Step {current_step} | Reward: {total_reward:.1f} | Cost: {total_cost:.2f} | V_L(pi): {vl_pi:.3f} | "
                f"Actor Loss: {actor_loss.item():.3f} | Best Reward (under safety): {best_reward:.1f}"
            )

    # === Save All Models ===
    env.close()
    df = pd.DataFrame(dataF)
    df.to_excel('rcmdp_with_robustness.xlsx')

    torch.save(actor.state_dict(), 'actor.pth')                       # Final actor
    torch.save(reward_critic.state_dict(), 'reward_critic.pth')
    torch.save(cost_critic.state_dict(), 'cost_critic.pth')


if __name__ == "__main__":
    main()
