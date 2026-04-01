import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from torch.distributions import Beta, Normal, Categorical

# from normalization import Normalization, RewardScaling
from torch.distributions import Uniform
import gym
import argparse
import pickle
import math
import random
import copy

# import mujoco
import os
from tqdm import tqdm

# from gymnasium.envs.mujoco import MujocoEnv
# from gym import utils
from typing import Optional, List, Tuple

# from gymnasium import spaces
import matplotlib.pyplot as plt  # Import for plotting

# from envs.cartpole import CartPoleCostEnv, CartPolePerturbedEnv
from envs.pendulum_v1 import PendulumEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = (
            F.softplus(self.alpha_layer(s)) + 1.0
        )  # softplus is a smooth approximation to ReLU function
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device="cpu"):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.action_dim)
        )  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(
            self.mean_layer(s)
        )  # [-1,1]->[-max_action,max_action]
        return mean

    # def get_dist(self, s):
    #     mean = self.forward(s)
    #     log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
    #     std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
    #     dist = Normal(mean, std)  # Get the Gaussian distribution
    #     return dist

    def get_dist(self, s):
        mean = self.forward(s)
        # print(f"mean shape: {mean.shape}")  # Debugging: Print the shape of mean
        # print(f"log_std shape before expand: {self.log_std.shape}")  # Debugging: Print the shape of log_std
        log_std = self.log_std.expand(
            mean.shape[0], -1
        )  # Expand log_std to match the shape of mean
        # print(f"log_std shape after expand: {log_std.shape}")  # Debugging: Print the shape of expanded log_std
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device="cpu"):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class Actor_Discrete(nn.Module):
    def __init__(self, args):
        super(Actor_Discrete, self).__init__()
        self.nA = args.action_dim
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.action_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.action_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        action = self.action_layer(s)
        return action

    def get_dist(self, s):
        action = self.forward(s)
        dist = Categorical(action)
        return dist


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


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x, update=True):
        self.R = self.gamma * self.R + x
        if update:
            self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done, we should reset 'self.R'
        self.R = np.zeros(self.shape)

    def inverse(self, x_normalized):
        return x_normalized * (self.running_ms.std + 1e-8)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        # v_s = torch.sigmoid(self.fc3(s))
        return v_s

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device="cpu"):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class CostCritic(nn.Module):
    def __init__(self, args):
        super(CostCritic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))  # Apply activation to the first layer
        s = self.activate_func(self.fc2(s))  # Apply activation to the second layer
        # v_s = torch.sigmoid(self.fc3(s))  # Apply sigmoid activation to the last layer
        v_s = self.fc3(s)
        return v_s

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device="cpu"):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.c = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, c, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.c[self.count] = c
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        c = torch.tensor(self.c, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, c, s_, dw, done


class PrimalDual:
    def __init__(self, args):
        if args.env == "CartPolePerturbedEnv":
            self.env = (
                CartPolePerturbedEnv()
            )  # CartPolePerturbedEnv() # CartPoleCostEnv()#HopperPerturbedEnv()
        elif args.env == "CartPoleCostEnv":
            self.env = CartPoleCostEnv()
        elif args.env == "HopperPerturbedEnv":
            self.env = HopperPerturbedEnv()
        elif args.env == "PendulumEnv":
            self.env = PendulumEnv()
        else:
            print("No env selected")
        # self.env.seed(args.seed)
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.persistent_eps = args.persistent_eps
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.adaptive_alpha = args.adaptive_alpha
        self.weight_reg = args.weight_reg
        self.lambda_ = args.lambda_
        self.lr_lambda = args.lr_lambda
        self.baseline = args.baseline
        # self.b = args.baseline
        if self.adaptive_alpha:
            self.target_entropy = -args.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimzier = torch.optim.Adam([self.log_alpha], lr=self.lr_a)
        else:
            self.alpha = 0.0

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        elif self.policy_dist == "Gaussian":
            self.actor = Actor_Gaussian(args)
        else:
            self.actor = Actor_Discrete(args)
        self.V_r = Critic(args)
        self.V_c = CostCritic(args)

        self.beta = args.beta
        # self.persistent_eps = 0.0
        self.warm_start_flag = args.warm_start_flag

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5
            )
            self.optimizer_reward_critic = torch.optim.Adam(
                self.V_r.parameters(), lr=self.lr_c, eps=1e-5
            )
            self.optimizer_cost_critic = torch.optim.Adam(
                self.V_c.parameters(), lr=self.lr_c, eps=1e-5
            )
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a
            )
            self.optimizer_reward_critic = torch.optim.Adam(
                self.V_r.parameters(), lr=self.lr_c
            )
            self.optimizer_cost_critic = torch.optim.Adam(
                self.V_c.parameters(), lr=self.lr_c
            )

    def evaluate(
        self, s
    ):  # When evaluating the policy, we only use the mean in Beta and gaussian and simply the action for Discrete
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            elif self.policy_dist == "Gaussian":
                a = self.actor(s).detach().numpy().flatten()
            else:
                a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = (
                    dist.sample()
                )  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(
                    a
                )  # The log probability density of the action
        elif self.policy_dist == "Gaussian":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = (
                    dist.sample()
                )  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(
                    a
                )  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_Rcritic.param_groups:
            p["lr"] = lr_c_now
        for p in self.optimizer_Ccritic.param_groups:
            p["lr"] = lr_c_now

    def softmax_fn(self, a, b, temperature=0.1):
        exp_a = torch.exp(a / temperature)
        exp_b = torch.exp(b / temperature)
        softmax_weighted = (a * exp_a + b * exp_b) / (exp_a + exp_b)
        return softmax_weighted

    def log_sum_exp_fn(self, a, b, eta=0.1):
        # Compute the Log-Sum-Exp smooth approximation of max(a, b)
        # print("a, b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta))= ", a,b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta)))
        # lse = eta * torch.log(torch.exp(a / eta) + torch.exp(b / eta))

        # Find the maximum value between a and b : else exp(10/0.1) becomes infinity
        max_val = torch.max(a, b)
        # Stabilize the log-sum-exp computation
        lse = max_val + eta * torch.log(
            torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta)
        )
        return lse

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob_old, r, c, s_, dw, done = (
            replay_buffer.numpy_to_tensor()
        )  # Get training data

        # ==================== Compute GAE advantages and value targets ====================
        with torch.no_grad():
            V_r_pred = self.V_r(s)
            V_r_next = self.V_r(s_)
            V_c_pred = self.V_c(s)
            V_c_next = self.V_c(s_)

            deltas_r = r + self.gamma * (1 - dw) * V_r_next - V_r_pred
            deltas_c = c + self.gamma * (1 - dw) * V_c_next - V_c_pred

            adv_r = []
            adv_c = []
            gae_r, gae_c = 0, 0
            for delta_r, delta_c, d in zip(
                reversed(deltas_r.flatten().numpy()),
                reversed(deltas_c.flatten().numpy()),
                reversed(done.flatten().numpy()),
            ):
                gae_r = delta_r + self.gamma * self.lamda * gae_r * (1.0 - d)
                adv_r.insert(0, gae_r)
                gae_c = delta_c + self.gamma * self.lamda * gae_c * (1.0 - d)
                adv_c.insert(0, gae_c)

            adv_r = torch.tensor(adv_r, dtype=torch.float32).view(-1, 1)
            adv_c = torch.tensor(adv_c, dtype=torch.float32).view(-1, 1)

            # Critic targets = advantage + baseline value
            v_target_r = adv_r + V_r_pred
            v_target_c = adv_c + V_c_pred

        # Normalize advantages for actor (after computing critic targets)
        if self.use_adv_norm:
            adv_r_norm = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c_norm = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
        else:
            adv_r_norm = adv_r
            adv_c_norm = adv_c

        # Combined advantage for primal-dual
        adv_combined = adv_r_norm - self.lambda_ * adv_c_norm

        # Old log probs (sum over action dims)
        a_logprob_old_sum = a_logprob_old.sum(dim=1, keepdim=True)

        # ==================== PPO mini-batch updates for K epochs ====================
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                drop_last=False,
            ):
                # ----- Actor update with PPO clipped objective -----
                dist = self.actor.get_dist(s[index])
                log_probs = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                entropy = dist.entropy().sum(dim=1, keepdim=True)

                ratio = torch.exp(log_probs - a_logprob_old_sum[index])
                surr1 = ratio * adv_combined[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * adv_combined[index]
                )
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                )

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # ----- Reward critic update -----
                v_r = self.V_r(s[index])
                reward_critic_loss = F.mse_loss(v_r, v_target_r[index])
                self.optimizer_reward_critic.zero_grad()
                reward_critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.V_r.parameters(), 0.5)
                self.optimizer_reward_critic.step()

                # ----- Cost critic update -----
                v_c = self.V_c(s[index])
                cost_critic_loss = F.mse_loss(v_c, v_target_c[index])
                self.optimizer_cost_critic.zero_grad()
                cost_critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.V_c.parameters(), 0.5)
                self.optimizer_cost_critic.step()

        # ==================== Dual Variable Update ====================
        cost_mean = c.sum().item()  # Episodic cost
        self.lambda_ = max(
            0.0, self.lambda_ + self.lr_lambda * (cost_mean - self.baseline)
        )

        print(
            "loss actor, reward critic=", actor_loss.item(), reward_critic_loss.item()
        )


def evaluate_policy(args, env, agent, state_norm=None, reward_scaling=None):
    times = 3
    evaluate_reward = 0
    evaluate_cost = 0
    evaluate_max_cost = float("-inf")
    for _ in range(times):
        s = env.reset()[0]
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        episode_cost = 0
        max_cost = float("-inf")
        while not done:
            a = agent.evaluate(
                s
            )  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, c, truncated, terminated, _ = env.step(action)
            done = truncated or terminated
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            episode_reward += r
            episode_cost += c
            max_cost = max(max_cost, c)

            # if args.use_reward_norm:
            #     r = reward_norm(r, update=False)
            #     c = reward_norm(c, update=False)
            # elif args.use_reward_scaling:
            #     r = reward_scaling(r, update=False)
            #     c = reward_scaling(c, update=False)
            # episode_reward += r
            # episode_cost += c
            # max_cost = max(max_cost, c)
            s = s_
        evaluate_reward += episode_reward
        evaluate_cost += episode_cost
        evaluate_max_cost = max(evaluate_max_cost, max_cost)

    return evaluate_reward / times, evaluate_cost / times, evaluate_max_cost


def save_agent(agent, save_path, state_norm=None, reward_scaling=None):
    agent.actor.save(f"{save_path}_actor")
    agent.V_r.save(f"{save_path}_Rcritic")
    agent.V_c.save(f"{save_path}_Ccritic")
    if state_norm:
        with open(f"{save_path}_state_norm", "wb") as file1:
            pickle.dump(state_norm, file1)
    if reward_scaling:
        with open(f"{save_path}_reward_scaling", "wb") as file2:
            pickle.dump(reward_scaling, file2)


def plot_metrics(
    episode_rewards,
    episode_costs,
    max_costs,
    save=False,
    filename="training_metrics.png",
):
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
    # plt.figure(figsize=(10, 6))

    # Plot total rewards
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Plot total costs
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


def main(args, run_number):
    seed, GAMMA = args.seed, args.GAMMA

    # Create directories for the current run
    model_dir = f"./models/{args.env}_PD/run{run_number}/"
    data_train_dir = f"./data_train/{args.env}_PD/run{run_number}/"
    plot_data_dir = f"./plot_data/{args.env}_PD/run{run_number}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_train_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    if args.env == "CartPolePerturbedEnv":
        env = (
            CartPolePerturbedEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            CartPolePerturbedEnv()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = (
            CartPolePerturbedEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == "CartPoleCostEnv":
        env = (
            CartPoleCostEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            CartPoleCostEnv()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = (
            CartPoleCostEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == "HopperPerturbed":
        env = (
            HopperPerturbed()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            HopperPerturbed()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = (
            HopperPerturbed()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == "PendulumEnv":
        env = (
            PendulumEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            PendulumEnv()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = PendulumEnv()
    # Set random seed
    # env.reset(seed=seed)
    # env.seed(seed)
    # env = gym.make(args.env)
    reward_offset = 0  # 16.2736044

    env.reset(seed=seed)
    env.action_space.seed(seed)

    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)

    env_reset.reset(seed=seed)
    env_reset.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = (
        env.max_steps
    )  # Must match environment's truncation limit (200)
    lambda_ = args.lambda_
    b = args.baseline
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_costs = []  # Record the costs during the evaluating
    total_steps = 0  # Record the total steps during the training
    max_value = -np.inf
    evaluate_max_costs = []

    replay_buffer = ReplayBuffer(args)
    agent = PrimalDual(args)

    # Build a tensorboard
    writer = SummaryWriter(
        log_dir=f"runs/RNAC/env_{args.env}_{args.policy_dist}_run{run_number}_seed_{seed}_GAMMA_{GAMMA}"
    )

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # Tracking metrics for plotting
    episode_rewards = []
    episode_costs = []
    # steps = []
    # vl_pi_values = []
    episode_max_costs = []

    # Initialize variables to track the best performance
    best_reward = float("-inf")  # Start with the minimum possible reward
    best_model_path = None

    for total_steps in tqdm(range(args.max_train_steps)):
        # if total_steps > args.max_train_steps // 2:
        #    agent.gamma = 0.999
        s = env.reset()[0]
        # s_org = copy.deepcopy(s)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        total_reward = 0
        total_cost = 0
        max_cost = float("-inf")

        agent.beta = (
            args.beta
        )  # 50.0 #min(max_beta, min_beta * np.exp(total_steps / scale))
        if total_steps > args.warm_start_episode:
            agent.warm_start_flag = 1
        else:
            agent.warm_start_flag = 0
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            if args.uncer_set == "DS":
                # Multi-run
                v_min, index = torch.tensor(float("inf")), 0
                v_candidate, index_candidate = torch.tensor(float("inf")), 0
                flag = 0
                noise_list, nexts_list, r_list, c_list = [], [], [], []
                for i in range(args.next_steps):
                    obs = env_reset.reset(state=s_org, x_pos=x_pos)
                    s_, r, c, truncated, terminated, info = env_reset.step(action)
                    done = truncated or terminated
                    # total_reward += r
                    # total_cost += c
                    r_list.append(r)
                    c_list.append(c)
                    noise_list.append(info["noise"])
                    if args.use_state_norm:
                        s_ = state_norm(s_, update=False)
                    nexts_list.append(s_)

                    #########################Please check this part if USING Double Sampling ############################################
                    with torch.no_grad():
                        if agent.Rcritic(torch.tensor(s_, dtype=torch.float)) < v_min:
                            v_min = agent.Rcritic(torch.tensor(s_, dtype=torch.float))
                            index = i
                        if (
                            agent.Rcritic(
                                torch.tensor(nexts_list[i], dtype=torch.float)
                            )
                            < v_candidate
                            and lambda_
                            * (agent.Ccritic(torch.tensor(s_, dtype=torch.float)) - b)
                            < 0
                        ):
                            v_candidate = agent.Rcritic(
                                torch.tensor(nexts_list[i], dtype=torch.float)
                            )
                            index_candidate = i
                            flag = 1
                if flag == 1:
                    index = index_candidate
                ############################# UP UNTIL HERE ################################################################
                # pick next state for robust critic update
                ridx = random.randint(0, args.next_steps)
                if ridx == args.next_steps:
                    ridx = index
                s_, r, c, done, info = env.step(
                    np.concatenate((action, noise_list[ridx]))
                )
                total_reward += r
                total_cost += c
            else:
                s_, r, c, truncated, terminated, info = env.step(action)
                # print("reward step =", r+ reward_offset)
                done = truncated or terminated
                total_reward += r
                total_cost += c
                max_cost = max(max_cost, c)
            # x_pos = np.array([info['x_position']])
            if args.use_state_norm:
                # nexts = state_norm(nexts, update=False)
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
                # c = reward_norm(c)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
                # c = reward_scaling(c)

            # total_reward += r
            # total_cost += c
            # max_cost = max(max_cost, c)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r + reward_offset, c, s_, dw, done)
            s = copy.deepcopy(s_)
            # s_org = copy.deepcopy(state_norm.denormal(s_, update=False))

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            if not args.use_reward_scaling:
                reward_scaling = None
            if not args.use_state_norm:
                state_norm = None
            evaluate_reward, evaluate_cost, evaluate_max_cost = evaluate_policy(
                args,
                env_evaluate,
                agent,
                state_norm=state_norm,
                reward_scaling=reward_scaling,
            )
            # evaluate_cost = evaluate_cost_function(args, env_evaluate, agent, state_norm)
            evaluate_rewards.append(evaluate_reward)
            evaluate_costs.append(evaluate_cost)
            evaluate_max_costs.append(evaluate_max_cost)

            print(
                "evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{} \t evaluate_max_cost:{}".format(
                    evaluate_num, evaluate_reward, evaluate_cost, evaluate_max_cost
                )
            )
            writer.add_scalar(
                "step_rewards_{}".format(args.env),
                evaluate_rewards[-1],
                global_step=total_steps,
            )
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            np.save(
                f"{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_rewards.npy",
                np.array(evaluate_rewards),
            )
            np.save(
                f"{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy",
                np.array(evaluate_costs),
            )
            np.save(
                f"{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy",
                np.array(evaluate_max_cost),
            )

            # Check if the current model satisfies the conditions for being the best
            if (
                evaluate_reward > best_reward
            ):  # and evaluate_max_cost <= args.persistent_eps:
                best_reward = evaluate_reward
                best_model_path = f"{model_dir}/Best_RCAC"
                print(
                    f"New best model found! Saving model with reward: {evaluate_reward} and max cost: {evaluate_max_cost}"
                )

                # Save the best model
                if args.use_reward_scaling and args.use_state_norm:
                    save_agent(agent, best_model_path, state_norm, reward_scaling)
                elif args.use_reward_scaling:
                    save_agent(
                        agent,
                        best_model_path,
                        state_norm=None,
                        reward_scaling=reward_scaling,
                    )
                elif args.use_state_norm:
                    save_agent(agent, best_model_path, state_norm)
                else:
                    save_agent(agent, best_model_path)

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)  # Save data for plotting
        np.save(f"{plot_data_dir}/episode_rewards.npy", episode_rewards)
        np.save(f"{plot_data_dir}/episode_max_costs.npy", episode_max_costs)
        plot_metrics(
            episode_rewards,
            episode_costs,
            episode_max_costs,
            save=True,
            filename=f"{plot_data_dir}/training_metrics.png",
        )

    # Save the evaluation rewards and costs for this run
    np.save(f"{data_train_dir}/evaluate_rewards.npy", evaluate_rewards)
    np.save(f"{data_train_dir}/evaluate_costs.npy", evaluate_costs)
    np.save(f"{data_train_dir}/evaluate_max_costs.npy", evaluate_max_costs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument(
        "--env",
        type=str,
        default="PendulumEnv",
        help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv/PendulumEnv",
    )
    parser.add_argument("--uncer_set", type=str, default="IPM", help="DS/IPM")
    parser.add_argument(
        "--next_steps", type=int, default=2, help="Number of next states"
    )
    parser.add_argument(
        "--random_steps",
        type=int,
        default=int(25e3),
        help="Uniformlly sample action within random steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(4.5e3),
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=1e2,
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument(
        "--policy_dist",
        type=str,
        default="Gaussian",
        help="Beta or Gaussian or Discrete",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument(
        "--mini_batch_size", type=int, default=64, help="Minibatch size"
    )
    parser.add_argument(
        "--hidden_width",
        type=int,
        default=64,
        help="The number of neurons in hidden layers of the neural network",
    )
    parser.add_argument(
        "--lr_a", type=float, default=1e-3, help="Learning rate of actor"
    )
    parser.add_argument(
        "--lr_c", type=float, default=1e-3, help="Learning rate of critic"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor 0.99"
    )

    # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument(
        "--persistent_eps",
        type=float,
        default=2.0,
        help="Persistent Safety Perturbation",
    )
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument(
        "--use_adv_norm",
        type=bool,
        default=True,
        help="Trick 1:advantage normalization",
    )
    parser.add_argument(
        "--use_state_norm", type=bool, default=True, help="Trick 2:state normalization"
    )
    parser.add_argument(
        "--use_reward_norm",
        type=bool,
        default=False,
        help="Trick 3:reward normalization",
    )
    parser.add_argument(
        "--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy"
    )
    parser.add_argument(
        "--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay"
    )
    parser.add_argument(
        "--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip"
    )
    parser.add_argument(
        "--use_orthogonal_init",
        type=bool,
        default=True,
        help="Trick 8: orthogonal initialization",
    )
    parser.add_argument(
        "--set_adam_eps",
        type=float,
        default=True,
        help="Trick 9: set Adam epsilon=1e-5",
    )
    parser.add_argument(
        "--use_tanh",
        type=float,
        default=True,
        help="Trick 10: tanh activation function",
    )
    parser.add_argument(
        "--adaptive_alpha",
        type=float,
        default=False,
        help="Trick 11: adaptive entropy regularization",
    )
    parser.add_argument(
        "--weight_reg",
        type=float,
        default=0,
        help="Regularization for weight of critic",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed 2, 5, 7, 11, 17")
    parser.add_argument("--GAMMA", type=str, default="0", help="file name")
    parser.add_argument("--baseline", type=int, default=9, help="baseline")
    parser.add_argument("--lambda_", type=int, default=0.0, help="lambda")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")
    parser.add_argument("--run", type=int, default=1, help="run_number")
    parser.add_argument(
        "--warm_start_flag", type=int, default=0, help="warm_start_flag"
    )
    parser.add_argument(
        "--warm_start_episode", type=int, default=500, help="warm_start_episode"
    )
    parser.add_argument("--lr_lambda", type=int, default=0.0, help="warm_start_episode")

    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    print("run=", args.run, "seed=", args.seed, "env=", args.env)

    main(args, run_number=args.run)