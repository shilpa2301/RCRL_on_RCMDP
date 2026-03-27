# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:56:54 2026

@author: Sourav
"""

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# ===================== Actor =====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, s):
        logits = self.net(s)
        return torch.distributions.Categorical(logits=logits)


# ===================== Value Critics =====================
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.net(s)

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

    def denormal(self, x, update=False):
        x = x * (self.running_ms.std + 1e-8) + self.running_ms.mean
        return x

# ===================== Plotting Function =====================
def plot_metrics(rewards, costs, filename="training_metrics.png"):
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the current figure to avoid overlapping plots
    # plt.figure(figsize=(10, 6))

    # plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(costs, label="Costs", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Total Cost")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close()



# ===================== Main =====================
def main():
    # Directories for saving models and data
    model_dir = "./models/CartPole-v1_PD/run1"
    data_dir = "./data_train/CartPole-v1_PD/run1"
    plot_dir = "./plot_data/CartPole-v1_PD/run1"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    env = gym.make("CartPole-v1")
    seed = 1
    env.reset(seed=seed)  # Set the seed for the environment's random number generator
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    V_r = ValueNet(state_dim)
    V_c = ValueNet(state_dim)

    # Initialize the state normalization
    state_norm = Normalization(shape=state_dim)  

    optim_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optim_Vr = torch.optim.Adam(V_r.parameters(), lr=1e-3)
    optim_Vc = torch.optim.Adam(V_c.parameters(), lr=1e-3)

    gamma = 0.95 #0.99
    lr_lambda = 1e-4 #3e-4 #1e-3
    lambda_ = 0.0
    beta = 0.01

    # constraint threshold
    baseline = 9.0  

    n_epochs = 10000
    best_reward = float('-inf')
    episode_rewards = []
    episode_costs = []

    for epoch in range(n_epochs):

        s, _ = env.reset()
        s = state_norm(s)
        done = False

        states = []
        actions = []
        rewards = []
        costs = []
        log_probs = []

        total_reward = 0
        total_cost = 0

        while not done:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

            dist = actor(s_tensor)
            a = dist.sample()
            log_prob = dist.log_prob(a)

            s_next, r, term, trunc, _ = env.step(a.item())
            done = term or trunc

            # CartPole fails if position too large
            c = 0.0
            if abs(s[0]) > 1:
                c = abs(s[0])
            # c = float(abs(s[0]) > 1.0)
            if done:
                c +=10.0

            s_next = state_norm(s_next) 

            states.append(s)
            actions.append(a.item())
            rewards.append(r)
            costs.append(c)
            log_probs.append(log_prob)

            s = s_next
            total_reward += r
            total_cost += c

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        costs = torch.tensor(costs, dtype=torch.float32)
        log_probs = torch.stack(log_probs)

        # ================= Returns =================
        returns_r = []
        returns_c = []

        G_r = 0
        G_c = 0

        for r, c in zip(reversed(rewards), reversed(costs)):
            G_r = r + gamma * G_r
            G_c = c + gamma * G_c
            returns_r.insert(0, G_r)
            returns_c.insert(0, G_c)

        returns_r = torch.tensor(returns_r, dtype=torch.float32)
        returns_c = torch.tensor(returns_c, dtype=torch.float32)

        # ================= Value Estimates =================
        V_r_pred = V_r(states).squeeze()
        V_c_pred = V_c(states).squeeze()

        # ================= Advantages =================
        A_r = returns_r - V_r_pred.detach()
        A_c = returns_c - V_c_pred.detach()

        # Normalize advantages (VERY IMPORTANT)
        A_r = (A_r - A_r.mean()) / (A_r.std() + 1e-8)
        A_c = (A_c - A_c.mean()) / (A_c.std() + 1e-8)

        # ================= Actor Update =================
        adv = A_r - lambda_ * A_c
        # actor_loss = -(log_probs * adv).mean()
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * adv).mean() - beta * entropy

        optim_actor.zero_grad()
        actor_loss.backward()
        optim_actor.step()

        # ================= Critic Updates =================
        loss_Vr = F.mse_loss(V_r_pred, returns_r)
        loss_Vc = F.mse_loss(V_c_pred, returns_c)

        optim_Vr.zero_grad()
        loss_Vr.backward()
        optim_Vr.step()

        optim_Vc.zero_grad()
        loss_Vc.backward()
        optim_Vc.step()

        # ================= Dual Update =================
        cost_mean = costs.sum().item()   # episodic cost

        if epoch >500:
            lambda_ = max(0.0, lambda_ + lr_lambda * (cost_mean - baseline))

        # # ================= Logging =================
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch} | Reward: {total_reward:.1f} | Cost: {total_cost:.1f} | Lambda: {lambda_:.3f}")
        
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        
        # Save models periodically
        if epoch % 100 == 0:
        # if best_reward >= total_reward:
            # best_reward = total_reward
            torch.save(actor.state_dict(), os.path.join(model_dir, f"actor_epoch_{epoch}.pth"))
            torch.save(V_r.state_dict(), os.path.join(model_dir, f"value_r_epoch_{epoch}.pth"))
            torch.save(V_c.state_dict(), os.path.join(model_dir, f"value_c_epoch_{epoch}.pth"))

        # Save training metrics periodically
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Reward: {total_reward:.1f} | Cost: {total_cost:.1f} | Lambda: {lambda_:.3f}")
            np.save(os.path.join(data_dir, "episode_rewards.npy"), np.array(episode_rewards))
            np.save(os.path.join(data_dir, "episode_costs.npy"), np.array(episode_costs))

        # Save plots periodically
        if epoch % 1 == 0:
            plot_metrics(episode_rewards, episode_costs, filename=os.path.join(plot_dir, "training_metrics.png"))

if __name__ == "__main__":
    main()
