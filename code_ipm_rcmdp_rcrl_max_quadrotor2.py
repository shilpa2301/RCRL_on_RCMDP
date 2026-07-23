import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from torch.distributions import Beta, Normal, Categorical

# from normalization import Normalization, RewardScaling
from torch.distributions import Uniform
import gymnasium as gym
import argparse
import pickle
import math
import random
import copy

# import mujoco
import os
from tqdm import tqdm

# from gymnasium.envs.mujoco import MujocoEnv
from gym import utils
from typing import Optional, List, Tuple
from gymnasium import spaces
import matplotlib.pyplot as plt  # Import for plotting
from envs.cartpole import CartPoleCostEnv, CartPolePerturbedEnv
from envs.pendulum_v1 import PendulumEnv, PendulumCostEnv, PendulumPerturbedEnv
from envs.half_cheetah import HalfCheetahWithPos
from envs.swimmer import SwimmerWithPos
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

CONFIG_FACTORY = ConfigFactory()
CONFIG_FACTORY.parser.set_defaults(overrides=['./envs/env_configs/constrained_tracking_reset.yaml'])
config = CONFIG_FACTORY.merge()

# Evaluation environment (deterministic, fixed starts)
CONFIG_FACTORY_EVAL = ConfigFactory()
CONFIG_FACTORY_EVAL.parser.set_defaults(overrides=['./envs/env_configs/constrained_tracking_eval.yaml'])
config_eval = CONFIG_FACTORY_EVAL.merge()


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
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.action_dim)
        )  # We use 'nn.Parameter' to train log_std automatically
        # self.log_std = nn.Parameter(torch.full((1, args.action_dim), -2.0))

        self.activate_func = [nn.ReLU(), nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
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
        
        # log_std = torch.clamp(log_std, min=-4.0, max=-1.0)

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
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc4 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)


    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        v_s = self.fc4(s)
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
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc4 = nn.Linear(args.hidden_width, 1)
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.activate_func = nn.ReLU()

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))  # Apply activation to the first layer
        s = self.activate_func(self.fc2(s))  # Apply activation to the second layer
        s = self.activate_func(self.fc3(s))
        # v_s = torch.sigmoid(self.fc3(s))  # Apply sigmoid activation to the last layer
        v_s = self.fc4(s)
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
    
class RPCRL_MC:
    def __init__(self, args):
        if args.env == "CartPolePerturbedEnv":
            self.env = CartPolePerturbedEnv(
                args.gravity_std
            )  # CartPolePerturbedEnv() # CartPoleCostEnv()#HopperPerturbedEnv()
        elif args.env == "CartPoleCostEnv":
            self.env = CartPoleCostEnv()
        elif args.env == "PendulumEnv":
            self.env = PendulumEnv()
        elif args.env == "PendulumCostEnv":
            self.env = PendulumCostEnv()
        elif args.env == "PendulumPerturbedEnv":
            self.env = PendulumPerturbedEnv()
        elif args.env == "HopperPerturbedEnv":
            self.env = HopperPerturbedEnv()
        elif args.env == "HalfCheetahWithPos":
            self.env = HalfCheetahWithPos()
        elif args.env == "SwimmerWithPos":
            self.env = SwimmerWithPos(sigma_viscosity=args.sigma_viscosity, max_steps=1000)
        elif args.env == "Quadrotor":
            self.env = make('quadrotor', **config.quadrotor_config)
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
        self.lr_cost = args.lr_cost  # Learning rate of cost critic
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
        self.Rcritic = Critic(args)
        self.Ccritic = CostCritic(args)

        self.beta = args.beta
        # self.persistent_eps = 0.0
        self.warm_start_flag = args.warm_start_flag
        #shilpa quadrotor reward LR set
        self.reward_learning_after_cost =0

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5
            )
            self.optimizer_Rcritic = torch.optim.Adam(
                self.Rcritic.parameters(), lr=self.lr_c, eps=1e-5
            )
            self.optimizer_Ccritic = torch.optim.Adam(
                self.Ccritic.parameters(), lr=self.lr_cost, eps=1e-5
            )
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a
            )
            self.optimizer_Rcritic = torch.optim.Adam(
                self.Rcritic.parameters(), lr=self.lr_c
            )
            self.optimizer_Ccritic = torch.optim.Adam(
                self.Ccritic.parameters(), lr=self.lr_cost
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
        lr_cost_now = self.lr_cost * (1 - total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_Rcritic.param_groups:
            p["lr"] = lr_c_now
        for p in self.optimizer_Ccritic.param_groups:
            p["lr"] = lr_cost_now

    def softmax_fn(self, a, b, temperature=0.1):
        exp_a = torch.exp(a / temperature)
        exp_b = torch.exp(b / temperature)
        softmax_weighted = (a * exp_a + b * exp_b) / (exp_a + exp_b)
        return softmax_weighted

    # def log_sum_exp_fn(self, a, b, eta=0.01): #prev 0.001
    #     # Compute the Log-Sum-Exp smooth approximation of max(a, b)
    #     # print("a, b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta))= ", a,b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta)))
    #     # lse = eta * torch.log(torch.exp(a / eta) + torch.exp(b / eta))

    #     if not isinstance(a, torch.Tensor):
    #         a = torch.tensor(a, dtype=torch.float32)
    #     if not isinstance(b, torch.Tensor):
    #         b = torch.tensor(b, dtype=torch.float32)

    #     # Find the maximum value between a and b : else exp(10/0.1) becomes infinity
    #     max_val = torch.max(a, b)
    #     # Stabilize the log-sum-exp computation
    #     lse = max_val + eta * torch.log(
    #         torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta)
    #     )
    #     return lse

    def log_sum_exp_all(self, x, eta=0.01):
        """
        Smooth approximation of max over all elements in x.

        Numerically stable log-sum-exp:

            max(x) ≈ max_x + eta * log(sum(exp((x - max_x) / eta)))
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        max_x = torch.max(x)

        return max_x + eta * torch.log(
            torch.sum(torch.exp((x - max_x) / eta))
        )
    
    def log_sum_exp_fn(self, a, b, eta=0.01): #prev 0.001
        # Compute the Log-Sum-Exp smooth approximation of max(a, b)
        # print("a, b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta))= ", a,b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta)))
        # lse = eta * torch.log(torch.exp(a / eta) + torch.exp(b / eta))

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)

        # Find the maximum value between a and b : else exp(10/0.1) becomes infinity
        max_val = torch.max(a, b)
        # Stabilize the log-sum-exp computation
        lse = max_val + eta * torch.log(
            torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta)
        )
        return lse

  

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, c, s_, dw, done = (
            replay_buffer.numpy_to_tensor()
        )  # Get training data
       
        # ------------------------------------------------------------
        # Monte Carlo return computation
        # ------------------------------------------------------------
        with torch.no_grad():
            vs = self.Rcritic(s)
            vcs = self.Ccritic(s)

            # ========================================================
            # Reward Monte Carlo returns
            # ========================================================
            mc_returns_r = []
            G_r = 0.0

            for reward, d in zip(
                reversed(r.flatten().numpy()),
                reversed(done.flatten().numpy())
            ):
                if d:
                    G_r = 0.0

                G_r = reward + self.gamma * G_r
                mc_returns_r.insert(0, G_r)

            v_target_r = torch.tensor(
                mc_returns_r,
                dtype=torch.float
            ).view(-1, 1)

            adv_r = v_target_r - vs

            # ========================================================
            # Cost Monte Carlo returns
            # ========================================================
            mc_returns_c = []

            G_c = torch.tensor(0.0, dtype=torch.float)

            for cost, d in zip(
                reversed(c.flatten()),
                reversed(done.flatten())
            ):
                if d.item() == 1.0:
                    G_c = torch.tensor(0.0, dtype=torch.float)

                immediate_cost = (1.0 - self.gamma) * cost
                future_cost = self.gamma * G_c

                G_c = self.log_sum_exp_fn(
                    immediate_cost,
                    future_cost,
                    eta=0.01
                )

                mc_returns_c.insert(0, G_c)

            v_target_c = torch.stack(mc_returns_c).view(-1, 1)

            adv_c = v_target_c - vcs
        
            # ========================================================
            # Robust cost objective:
            # max over all Monte Carlo cost returns,
            # approximated by log-sum-exp
            # ========================================================
            vl_pi = self.log_sum_exp_all(v_target_c, eta=0.01)

            # Cost advantage:
            # Every state compares its current cost critic value against
            # the robust worst-case batch/trajectory-level cost.
            adv_c = vl_pi - vcs

            # ------------------------------------------------------------
            # Reward/cost mode selection
            # ------------------------------------------------------------
            penalty_term = vl_pi - torch.tensor(self.persistent_eps)
            beta_penalty = self.beta * penalty_term

            vs_mean = v_target_r.mean().item()

            if self.warm_start_flag == 1:
                ch = np.argmax([vs_mean, beta_penalty])
                # if ch == 1 and self.reward_learning_after_cost==0:
                #     self.reward_learning_after_cost=1
                #     if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
                #         self.optimizer_actor = torch.optim.Adam(
                #             self.actor.parameters(), lr=self.lr_a, eps=1e-5
                #         )
                #         self.optimizer_Rcritic = torch.optim.Adam(
                #             self.Rcritic.parameters(), lr=self.lr_c, eps=1e-5
                #         )
                #         self.optimizer_Ccritic = torch.optim.Adam(
                #             self.Ccritic.parameters(), lr=self.lr_cost, eps=1e-5
                #         )
                #     else:
                #         self.optimizer_actor = torch.optim.Adam(
                #             self.actor.parameters(), lr=self.lr_a
                #         )
                #         self.optimizer_Rcritic = torch.optim.Adam(
                #             self.Rcritic.parameters(), lr=self.lr_c
                #         )
                #         self.optimizer_Ccritic = torch.optim.Adam(
                #             self.Ccritic.parameters(), lr=self.lr_cost
                #         )

            else:
                ch = 0

            print(
                "RP_CRL_MC | ch, vs_mean, vl_pi, beta penalty=",
                ch,
                vs_mean,
                vl_pi,
                beta_penalty,
            )

    
            # ------------------------------------------------------------
            # Select actor advantage
            # ------------------------------------------------------------
            if ch == 1:
                # Cost mode:
                # minimize robust cost = maximize negative cost advantage
                if self.use_adv_norm:
                    adv_mean = adv_c.mean()
                    adv_std = adv_c.std()

                    adv_c = (adv_c - adv_mean) / (adv_std + 1e-5)
                    adv_c = torch.clamp(adv_c, -3.0, 3.0)

                adv = -adv_c

            else:
                # Reward mode
                adv = adv_r

                if self.use_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # ------------------------------------------------------------
        # PPO-style actor and critic optimization
        # ------------------------------------------------------------
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                False
            ):
                dist_now = self.actor.get_dist(s[index])

                dist_entropy = dist_now.entropy().sum(
                    1,
                    keepdim=True
                )

                a_logprob_now = dist_now.log_prob(a[index])

                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True)
                    - a_logprob[index].sum(1, keepdim=True)
                )

                surr1 = ratios * adv[index]

                surr2 = torch.clamp(
                    ratios,
                    1 - self.epsilon,
                    1 + self.epsilon
                ) * adv[index]

                actor_loss = (
                    -torch.min(surr1, surr2)
                    - self.entropy_coef * dist_entropy
                )

                # ----------------------------------------------------
                # Update actor
                # ----------------------------------------------------
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        0.2
                    )

                self.optimizer_actor.step()

                # ----------------------------------------------------
                # Update reward critic using MC reward target
                # ----------------------------------------------------
                v_s = self.Rcritic(s[index])
                Rcritic_loss = F.mse_loss(v_target_r[index], v_s)

                self.optimizer_Rcritic.zero_grad()
                Rcritic_loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.Rcritic.parameters(),
                        0.2
                    )

                self.optimizer_Rcritic.step()

                # ----------------------------------------------------
                # Update cost critic using MC cost target
                # ----------------------------------------------------
                v_cs = self.Ccritic(s[index])
                Ccritic_loss = F.mse_loss(v_target_c[index], v_cs)

                self.optimizer_Ccritic.zero_grad()
                Ccritic_loss.backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.Ccritic.parameters(),
                        0.2
                    )

                self.optimizer_Ccritic.step()

        # ------------------------------------------------------------
        # Learning-rate decay
        # ------------------------------------------------------------
        if self.use_lr_decay:
            self.lr_decay(total_steps)

        # ------------------------------------------------------------
        # Adaptive entropy coefficient
        # ------------------------------------------------------------
        if self.adaptive_alpha:
            alpha_loss = -(
                self.log_alpha.exp()
                * (
                    a_logprob.sum(dim=1, keepdim=True)
                    + self.target_entropy
                ).detach()
            ).mean()

            self.alpha_optimzier.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzier.step()

            self.alpha = self.log_alpha.exp()


class Robust_RCAC_NPG:
    def __init__(self, args):
        if args.env == "CartPolePerturbedEnv":
            self.env = CartPolePerturbedEnv(
                args.gravity_std
            )  # CartPolePerturbedEnv() # CartPoleCostEnv()#HopperPerturbedEnv()
        elif args.env == "CartPoleCostEnv":
            self.env = CartPoleCostEnv()
        elif args.env == "PendulumEnv":
            self.env = PendulumEnv()
        elif args.env == "PendulumCostEnv":
            self.env = PendulumCostEnv()
        elif args.env == "PendulumPerturbedEnv":
            self.env = PendulumPerturbedEnv()
        elif args.env == "HopperPerturbedEnv":
            self.env = HopperPerturbedEnv()
        elif args.env == "HalfCheetahWithPos":
            self.env = HalfCheetahWithPos()
        elif args.env == "SwimmerWithPos":
            self.env = SwimmerWithPos(sigma_viscosity=args.sigma_viscosity, max_steps=1000)
        elif args.env == "Quadrotor":
            self.env = make('quadrotor', **config.quadrotor_config)
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
        self.lr_cost = args.lr_cost  # Learning rate of cost critic
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
        self.Rcritic = Critic(args)
        self.Ccritic = CostCritic(args)

        #shilpa target critic
        # Initialize target critics
        # self.target_Rcritic = Critic(args)  # Target reward critic
        # self.target_Ccritic = CostCritic(args)  # Target cost critic        
        # # # Copy initial weights from critics to target critics
        # self.target_Rcritic.load_state_dict(self.Rcritic.state_dict())
        # self.target_Ccritic.load_state_dict(self.Ccritic.state_dict())

        self.beta = args.beta
        # self.persistent_eps = 0.0
        self.warm_start_flag = args.warm_start_flag

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a, eps=1e-5
            )
            self.optimizer_Rcritic = torch.optim.Adam(
                self.Rcritic.parameters(), lr=self.lr_c, eps=1e-5
            )
            self.optimizer_Ccritic = torch.optim.Adam(
                self.Ccritic.parameters(), lr=self.lr_cost, eps=1e-5
            )
        else:
            self.optimizer_actor = torch.optim.Adam(
                self.actor.parameters(), lr=self.lr_a
            )
            self.optimizer_Rcritic = torch.optim.Adam(
                self.Rcritic.parameters(), lr=self.lr_c
            )
            self.optimizer_Ccritic = torch.optim.Adam(
                self.Ccritic.parameters(), lr=self.lr_cost
            )

        #shilpa target critic 
        # Target networks are not optimized directly
        # self.target_Rcritic.eval()
        # self.target_Ccritic.eval()
        # for p in self.target_Rcritic.parameters():
        #     p.requires_grad = False
        # for p in self.target_Ccritic.parameters():
        #     p.requires_grad = False
        # self.tau = getattr(args, "tau", 0.005)

        # In __init__, add:
        # self.ch = 0                        # current mode: 0=reward, 1=cost
        # self.enter_cost_threshold  = self.persistent_eps          # enter cost mode if max > this
        # self.exit_cost_threshold   = self.persistent_eps * 0.8   # exit cost mode only if max < this (more lenient)


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
        lr_cost_now = self.lr_cost * (1 - total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_Rcritic.param_groups:
            p["lr"] = lr_c_now
        for p in self.optimizer_Ccritic.param_groups:
            p["lr"] = lr_cost_now

    def softmax_fn(self, a, b, temperature=0.1):
        exp_a = torch.exp(a / temperature)
        exp_b = torch.exp(b / temperature)
        softmax_weighted = (a * exp_a + b * exp_b) / (exp_a + exp_b)
        return softmax_weighted

    def log_sum_exp_fn(self, a, b, eta=0.01): #prev 0.001
        # Compute the Log-Sum-Exp smooth approximation of max(a, b)
        # print("a, b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta))= ", a,b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta)))
        # lse = eta * torch.log(torch.exp(a / eta) + torch.exp(b / eta))

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)

        # Find the maximum value between a and b : else exp(10/0.1) becomes infinity
        max_val = torch.max(a, b)
        # Stabilize the log-sum-exp computation
        lse = max_val + eta * torch.log(
            torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta)
        )
        return lse

  

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, c, s_, dw, done = (
            replay_buffer.numpy_to_tensor()
        )  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
           
            adv_r, adv_c = [], []
            gae = 0
            with torch.no_grad():  # adv and v_target have no gradient
                # shilpa target critic
                vs = self.Rcritic(s)
                vs_ = self.Rcritic(s_)
                vcs = self.Ccritic(s)
                vcs_ = self.Ccritic(s_)
                # vs = self.Rcritic(s)
                # vcs = self.Ccritic(s)
                # vs_ = self.target_Rcritic(s_)
                # vcs_ = self.target_Ccritic(s_)


                # Construct trajectory dynamically from replay buffer

                trajectory = {"states": s, "actions": a, "next_states": s_, "costs": c}

                # with torch.no_grad():
                    # Compute robust value function and V_L^pi
                
                vl_pi = vcs.max()
                # penalty_term = max(0, vl_pi - self.persistent_eps)  # Apply penalty only if V_L(pi) > epsilon_tolerance
                penalty_term = vl_pi - torch.tensor(self.persistent_eps)
                beta_penalty = self.beta * penalty_term
                vs_mean = vs.mean().item()
                if self.warm_start_flag == 1:
                    ch = np.argmax([vs_mean, beta_penalty])
                else:
                    ch = 0
                print(
                    "ch, vs_mean, vl_pi, beta penalty=",
                    ch,
                    vs_mean,
                    vl_pi,
                    beta_penalty,
                )

               
                # vs_mean = vs.mean().item()
                # penalty_normalized = vcs.max() - torch.tensor(self.persistent_eps)
                # if self.warm_start_flag == 1:
                #     # ch = 0 if (vs_mean_normalized > 0.5 and penalty_normalized < 0) else (
                #     #             1 if penalty_normalized > 0 else 0
                #     #         )
                #     ch = 0 if penalty_normalized < 0 else 1
                # else:
                #     ch = 0
                # # ch = 1
                # print(
                #     "ch, vs_mean, vl_pi=",
                #     ch,
                #     vs_mean,
                #     penalty_normalized
                # )
    
                reg_norm, weight_norm, bias_norm = 0, [], []

                linear_layers = [l for l in self.Ccritic.children() if isinstance(l, nn.Linear)]
                if len(linear_layers) == 0:
                    raise ValueError("Ccritic has no nn.Linear layer")
                reg_norm = torch.norm(linear_layers[-1].weight, p=2)

                cost_deltas = (
                    (1 - self.gamma) * c
                    + self.gamma * self.log_sum_exp_fn(c, (1.0 - dw) * vcs_)
                    - vcs
                    # - self.alpha * a_logprob.sum(dim=1, keepdim=True)
                    
                    # + self.weight_reg * reg_norm
                    
                )  
                adv_c = cost_deltas.clone()  
                v_target_c = cost_deltas + vcs + self.weight_reg * reg_norm#- self.weight_reg * reg_norm #+ self.alpha * a_logprob.sum(dim=1, keepdim=True)


                reg_norm, weight_norm, bias_norm = 0, [], []
                # for layer in self.Rcritic.children():
                #         if isinstance(layer, nn.Linear):
                #             weight_norm.append(
                #                 torch.norm(layer.state_dict()["weight"]) ** 2
                #             )
                #             bias_norm.append(torch.norm(layer.state_dict()["bias"]) ** 2)
                # reg_norm = torch.sqrt(
                #     torch.sum(torch.stack(weight_norm))
                #     + torch.sum(torch.stack(bias_norm[0:-1]))
                # )
                linear_layers = [layer for layer in self.Rcritic.children() if isinstance(layer, nn.Linear)]
                if len(linear_layers) == 0:
                    raise ValueError("Rcritic has no nn.Linear layer")
                last_linear = linear_layers[-1]
                reg_norm = torch.norm(last_linear.weight, p=2)
                #ppo
                deltas = (
                    r
                    + self.gamma * (1.0 - dw) * vs_
                    - vs
                    # - self.alpha * a_logprob.sum(dim=1, keepdim=True)
                    # + self.weight_reg * reg_norm
                )
                
                for delta, d in zip(
                    reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())
                ):
                    gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                    adv_r.insert(0, gae)
                adv_r = torch.tensor(adv_r, dtype=torch.float).view(-1, 1)
                v_target_r = adv_r + vs + self.weight_reg * reg_norm#+ self.alpha * a_logprob.sum(dim=1, keepdim=True)

                if ch == 1:
                    
                    # ── Normalize then negate for actor ─────────────────────────────────
                    if self.use_adv_norm:
                        # adv = (adv - adv.mean()) / (adv.std() + 1e-5)
                        adv_mean = adv_c.mean()
                        adv_std  = adv_c.std()
                        # Always normalize, but SCALE DOWN smoothly when signal is small.
                        # This avoids the bang-bang switching at the 1e-2 threshold.
                        adv_c = (adv_c - adv_mean) / (adv_std + 1e-5)
                        adv_c = torch.clamp(adv_c, -3.0, 3.0)
                        # Soft gate: multiply by tanh(std / threshold) so near-zero std → near-zero adv
                        # but there is NO hard discontinuity
                        # soft_gate = torch.tanh(adv_std / 0.05)   # smooth 0→1 around std=0.05
                        # adv = adv * soft_gate
                    adv = -adv_c  # minimize cost = maximize negative cost advantage
                else:
                    
                    if self.use_adv_norm:  # Trick 1:advantage normalization
                        adv =  (adv_r - adv_r.mean()) / (adv_r.std() + 1e-5)

        # # Optimize policy for K epochs:
        # for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(
                    1, keepdim=True
                )  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # print("a_logprob_now shape:", a_logprob_now.shape)  # Debugging: Print the shape of a_logprob_now
                # print("a_logprob shape:", a_logprob[index].shape)  # Debugging: Print the shape of a_logprob


                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True)
                    - a_logprob[index].sum(1, keepdim=True)
                )  # shape(mini_batch_size X 1)

                surr1 = (
                    ratios * adv[index]
                )  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                )
                
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                )  # Trick 5: policy entropy
                
                #policy gradient
                # actor_loss = - (a_logprob_now * adv[index])
                # actor_loss = (-a_logprob_now  * adv[index] - self.entropy_coef * dist_entropy)


                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.2) #shilpa 0.5 to 0.2
                self.optimizer_actor.step()

                # if ch == 0:
                v_s = self.Rcritic(s[index])
                # # Calculate the loss of critic
                Rcritic_loss = F.mse_loss(v_target_r[index], v_s)
                # Update Reward critic
                self.optimizer_Rcritic.zero_grad()
                Rcritic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.Rcritic.parameters(), 0.2) #shilpa
                self.optimizer_Rcritic.step()
                
                v_cs = self.Ccritic(s[index])
                Ccritic_loss = F.mse_loss(v_target_c[index], v_cs)
                # Update Cost critic
                self.optimizer_Ccritic.zero_grad()
                Ccritic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.Ccritic.parameters(), 0.2) #shilpa
                self.optimizer_Ccritic.step()

               
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        #shilpa target critic
        # Soft update target networks
        # self.soft_update_target_networks(self.tau)

        if self.adaptive_alpha:
            alpha_loss = -(
                self.log_alpha.exp()
                * (a_logprob.sum(dim=1, keepdim=True) + self.target_entropy).detach()
            ).mean()
            self.alpha_optimzier.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzier.step()
            self.alpha = self.log_alpha.exp()

    #shilpa target critic
    def soft_update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_Rcritic.parameters(), self.Rcritic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_Ccritic.parameters(), self.Ccritic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def evaluate_policy(args, env, agent, state_norm=None, reward_scaling=None):
    times = 3
    evaluate_reward = 0
    evaluate_cost = 0
    evaluate_max_cost = float("-inf")
    for _ in range(times):
        s, _ = env.reset()#[0][0]
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
            # s_, r, c, truncated, terminated, info = env.step(action)
            s_, r, done, info = env.step(action)
            # print("Rendering environment...")
            env.render()

            # c = np.max(info.get('constraint_values', 0.))
            # c= 0.0
            c = args.omega1*max(0, info["constraint_values"][0]) + args.omega2*max(0, info["constraint_values"][1])  # Assuming info contains constraint values
            #shilpa cost scale
            c = c* args.cost_scale
            # print("eval cost =", c)

 
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            if args.use_reward_scaling:
                r = reward_scaling(r, update=False)
                c = reward_scaling(c, update=False)
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
    agent.Rcritic.save(f"{save_path}_Rcritic")
    agent.Ccritic.save(f"{save_path}_Ccritic")
    if state_norm:
        with open(f"{save_path}_state_norm", "wb") as file1:
            pickle.dump(state_norm, file1)
    if reward_scaling:
        with open(f"{save_path}_reward_scaling", "wb") as file2:
            pickle.dump(reward_scaling, file2)
# ── ADD this new function right after the existing plot_metrics function ──────

def plot_eval_metrics(
    evaluate_rewards,
    evaluate_costs,
    evaluate_max_costs,
    persistent_eps,
    save=False,
    filename="eval_metrics.png",
):
    """
    Plot evaluation metrics (reward, total cost, max cost) over evaluation
    checkpoints and optionally save the plot.
    Args:
        evaluate_rewards:   List of avg rewards per evaluation checkpoint.
        evaluate_costs:     List of avg total costs per evaluation checkpoint.
        evaluate_max_costs: List of max costs per evaluation checkpoint.
        persistent_eps:     Safety threshold — drawn as a horizontal reference line.
        save:               Whether to save the plot to a file.
        filename:           File name to save the plot.
    """
    evals = list(range(1, len(evaluate_rewards) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # ── Subplot 1: Evaluate Reward ────────────────────────────────────────────
    axes[0].plot(evals, evaluate_rewards, color="blue", label="Eval Reward")
    axes[0].set_xlabel("Evaluation #")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Evaluation Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Subplot 2: Evaluate Max Cost (with safety threshold line) ────────────
    axes[1].plot(evals, evaluate_max_costs, color="red", label="Eval Max Cost")
    axes[1].axhline(
        y=persistent_eps,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Safety threshold ({persistent_eps})",
    )
    axes[1].set_xlabel("Evaluation #")
    axes[1].set_ylabel("Max Cost")
    axes[1].set_title("Evaluation Max Cost per Checkpoint")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Subplot 3: Evaluate Total Cost ───────────────────────────────────────
    axes[2].plot(evals, evaluate_costs, color="green", label="Eval Total Cost")
    axes[2].set_xlabel("Evaluation #")
    axes[2].set_ylabel("Total Cost")
    axes[2].set_title("Evaluation Total Cost per Checkpoint")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=150)
    plt.close()

            
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
    # plt.ion()  # Turn on interactive mode
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
    # plt.show()
    plt.close()


def main(args, run_number):
    seed, GAMMA = args.seed, args.GAMMA

    # Create directories for the current run
    model_dir = f"./models/{args.env}/run{run_number}/"
    data_train_dir = f"./data_train/{args.env}/run{run_number}/"
    plot_data_dir = f"./plot_data/{args.env}/run{run_number}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_train_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    if args.env == "CartPolePerturbedEnv":
        env = CartPolePerturbedEnv(
            args.gravity_std
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

    elif args.env == "PendulumCostEnv":
        env = (
            PendulumCostEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            PendulumCostEnv()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = PendulumCostEnv()
    elif args.env == "PendulumPerturbedEnv":
        env = (
            PendulumPerturbedEnv()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            PendulumPerturbedEnv()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = PendulumPerturbedEnv()
    elif args.env == "HalfCheetahWithPos":
        env = (
            HalfCheetahWithPos()
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            HalfCheetahWithPos() #HalfCheetahWithPostest()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = HalfCheetahWithPos()
    elif args.env == "SwimmerWithPos":
        env = (
            SwimmerWithPos(sigma_viscosity=args.sigma_viscosity, max_steps=1000)
        )  # CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = (
            SwimmerWithPos(sigma_viscosity=args.sigma_viscosity, max_steps=1000) #SwimmerWithPostest()
        )  # CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = SwimmerWithPos(sigma_viscosity=args.sigma_viscosity, max_steps=1000)
    elif args.env == "Quadrotor":
        env = make('quadrotor', **config.quadrotor_config)

        env_evaluate = make('quadrotor', **config_eval.quadrotor_config)
        env_reset = make('quadrotor', **config.quadrotor_config)


    # Set random seed
    # env.reset(seed=seed)
    # env.seed(seed)
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
    # args.max_episode_steps = env.max_steps  # Must match environment's truncation limit
    lambda_ = args.lambda_
    b = args.baseline
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    # print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_costs = []  # Record the costs during the evaluating
    total_steps = 0  # Record the total steps during the training
    max_value = -np.inf
    evaluate_max_costs = []

    replay_buffer = ReplayBuffer(args)
    # agent = RPCRL_MC(args) #Robust_RCAC_NPG(args) #shilpa Monte Carlo
    agent = Robust_RCAC_NPG(args) #shilpa Monte Carlo


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

    reward_offset = 0 # 40 #17
    for total_steps in tqdm(range(args.max_train_steps)):
        # if total_steps > args.max_train_steps // 2:
        #    agent.gamma = 0.999
        # if total_steps > args.warm_start_episode:
        #             agent.entropy_coef = 0.0
        s, _ = env.reset()#[0][0]
        # print ("Initial state:", s)  # Debugging: Print the initial state
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

        #shilpa multi constraint debug
        upper_raw_values = []
        upper_cost_values = []
        lower_cost_values = []


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
                    s_, r, c, done, info = env_reset.step(action)
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
                s_, r, done, info = env.step(action)
                # c = np.max(info.get('constraint_values', 0.))
                # print(info['constraint_values'].shape)  
                # print(info['constraint_values'])
                # c = 0.0
                # config[quadrotor_config]["constraints"][0]["lower_bounds"] and config[quadrotor_config]["constraints"][0]["upper_bounds"]
                c  = args.omega1 * max(0, info['constraint_values'][0])+ args.omega2 * max(0, info['constraint_values'][1])
                #shilpa cost scale
                c = c* args.cost_scale
                # if c<=0.0:
                #     c=0.0
                # print("training cost =", c)
                total_reward += r
                total_cost += c
                max_cost = max(max_cost, c)
            # x_pos = np.array([info["x_position"]])
            if args.use_state_norm:
                # nexts = state_norm(nexts, update=False)
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
                # c = reward_norm(c)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
                # c = reward_scaling(c)

            #shilpa multi constraint debug
            upper_raw_values.append(info["constraint_values"][1])
            lower_cost_values.append(max(0, info["constraint_values"][0]))
            upper_cost_values.append(max(0, info["constraint_values"][1]))


            # total_reward += r
            # total_cost += c
            # max_cost = max(max_cost, c)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done :
                dw = True
            else:
                dw = False
            #shilpa hc
            # dw = done

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r + reward_offset, c, s_, dw, done)
            s = copy.deepcopy(s_)
            # s_org = copy.deepcopy(state_norm.denormal(s_, update=False))

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
        #shilpa multi constraint debug
        if total_steps % 100 == 0:
            print("lower cost max:", np.max(lower_cost_values))
            print("upper raw min/max:", np.min(upper_raw_values), np.max(upper_raw_values))
            print("upper cost max:", np.max(upper_cost_values))

        raw_constraints = info["constraint_values"]
        if total_steps % 100 == 0 and episode_steps < 5:
            print(
                "raw constraint_values:",
                raw_constraints,
                "lower term:",
                max(0, raw_constraints[0]),
                "upper term:",
                max(0, raw_constraints[1]),
            )
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
            # ── NEW: save evaluation plot after every checkpoint ──────────────
            plot_eval_metrics(
                evaluate_rewards,
                evaluate_costs,
                evaluate_max_costs,
                persistent_eps=args.persistent_eps,
                save=True,
                filename=f"{plot_data_dir}/eval_metrics.png",
            )
            # ─────────────────────────────────────────────────────────────────

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
                and evaluate_max_cost <= args.persistent_eps
            ):
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
        # default="CartPolePerturbedEnv",
        default="Quadrotor",
        help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv/PendulumEnv/PendulumCostEnv/HalfCheetahWithPos",
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
        "--mini_batch_size", type=int, default=128, help="Minibatch size"
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
        "--lr_cost", type=float, default=5e-4, help="Learning rate of critic"
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
        default=0.3,
        help="Persistent Safety Perturbation 0.17",
    )
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument(
        "--use_adv_norm",
        type=bool,
        default=True,
        help="Trick 1:advantage normalization",
    )
    parser.add_argument(
        "--use_state_norm", type=bool, default=False, help="Trick 2:state normalization"
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
        default=0.001,
        help="Regularization for weight of critic",
    )
    parser.add_argument("--seed", type=int, default=2, help="seed 2, 5, 7, 11, 17")
    parser.add_argument("--GAMMA", type=str, default="0", help="file name")
    parser.add_argument("--baseline", type=int, default=9, help="baseline")
    parser.add_argument("--lambda_", type=int, default=50, help="lambda")
    parser.add_argument("--beta", type=float, default=1e5, help="beta 600")
    parser.add_argument("--run", type=int, default=5, help="run_number")
    parser.add_argument(
        "--warm_start_flag", type=int, default=0, help="warm_start_flag"
    )
    parser.add_argument(
        "--warm_start_episode", type=int, default=1300, help="warm_start_episode"
    )
    parser.add_argument(
        "--gravity_std", type=float, default=0.5, help="gravity perturbation"
    )
    parser.add_argument(
        "--omega1", type=float, default=0.5, help="cost lower bounds weigt"
    )
    parser.add_argument(
        "--omega2", type=float, default=0.5, help="cost upper bounds weigt"
    )
    parser.add_argument(
        "--cost_scale", type=float, default=100.0, help="cost scale"
    )
    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    print("run=", args.run, "seed=", args.seed, "env=", args.env,  "k_epochs", args.K_epochs)

    main(args, run_number=args.run)
