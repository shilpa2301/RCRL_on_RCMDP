import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from copy import deepcopy

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# === Hyperparameters ===
gamma = 0.99
hidden_dim = 256
learning_rate = 1e-3
episodes = 1000
lambda_fixed = 20.0
b = 200.0
perturb_eps = 1.0

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
    noise = np.random.uniform(0, eps, size=state.shape)
    return state + noise

def discount(values, gamma):
    result = []
    G = 0
    for v in reversed(values):
        G = v + gamma * G
        result.insert(0, G)
    return torch.FloatTensor(result)

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

    # === Training Loop ===
    for ep in range(episodes):
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

        while not done:
            probs = actor(state)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            next_state = add_uniform_noise(np.array(next_state), perturb_eps)
            next_state = torch.FloatTensor(next_state)

            cost = abs(state[0].item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            costs.append(cost)
            reward_values.append(reward_critic(state))
            cost_values.append(cost_critic(state))

            total_reward += reward
            total_cost += cost
            state = next_state

        # Discounted returns
        reward_returns = discount(rewards, gamma)
        cost_returns = discount(costs, gamma)

        reward_values = torch.cat(reward_values).squeeze()
        cost_values = torch.cat(cost_values).squeeze()
        log_probs = torch.stack(log_probs)

        adv_r = reward_returns - reward_values.detach()
        adv_c = cost_returns - cost_values.detach()

        chosen_adv = []
        for vr, vc, ar, ac in zip(reward_returns, cost_returns, adv_r, adv_c):
            if vr.item() > lambda_fixed * (vc.item() - b):
                chosen_adv.append(ar)
            else:
                chosen_adv.append(-ac)
        chosen_adv = torch.stack(chosen_adv)

        # Losses
        actor_loss = -(log_probs * chosen_adv).mean()
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
        if total_cost < b and total_reward > best_reward:
            best_reward = total_reward
            best_actor_state_dict = deepcopy(actor.state_dict())

        # Display
        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Cost: {total_cost:.2f} | Actor Loss: {actor_loss.item():.3f} | Best Reward (under cost): {best_reward:.1f}")

    # === Save Averaged Actor (last 50 episodes) ===
    avg_actor_state_dict = deepcopy(last_50_actor_params[0])
    for key in avg_actor_state_dict:
        for i in range(1, len(last_50_actor_params)):
            avg_actor_state_dict[key] += last_50_actor_params[i][key]
        avg_actor_state_dict[key] /= len(last_50_actor_params)

    avg_actor = Actor()
    avg_actor.load_state_dict(avg_actor_state_dict)

    # === Save All Models ===
    env.close()
    df = pd.DataFrame(dataF)
    df.to_excel('tvf_and_tcf_data_with_uncertainity.xlsx')

    torch.save(actor.state_dict(), 'actor.pth')                       # Final actor
    torch.save(avg_actor.state_dict(), 'actor_avg_last50.pth')       # Averaged actor
    torch.save(reward_critic.state_dict(), 'reward_critic.pth')
    torch.save(cost_critic.state_dict(), 'cost_critic.pth')

if __name__ == "__main__":
    main()
