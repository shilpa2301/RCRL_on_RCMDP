# #shilpa Windows only
# import os

# if os.name == "nt":
#     os.add_dll_directory(r"C:\Users\rinki\.mujoco\mujoco210\bin")
#     os.add_dll_directory(r"C:\Users\rinki\miniconda3\envs\rpcrl_env\Library\bin")

# os.environ["MUJOCO_PY_MUJOCO_PATH"] = r"C:\Users\rinki\.mujoco\mujoco210"

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
        #shilpa multi constraint
        self.fc4 = nn.Linear(args.hidden_width, args.cost_dim)
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

class REFCostCritic(nn.Module):
    def __init__(self, args):
        super(REFCostCritic, self).__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, args.hidden_width)

        # IMPORTANT:
        # Multi-constraint REF critic outputs one probability per constraint.
        self.fc4 = nn.Linear(args.hidden_width, args.cost_dim)

        self.activate_func = nn.ReLU()

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

        # Shape: [batch_size, cost_dim]
        v_s = torch.sigmoid(self.fc4(s))

        return v_s

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        #shilpa multi constraint
        # self.c = np.zeros((args.batch_size, 1))
        self.c = np.zeros((args.batch_size, args.cost_dim))  # For multi-constraint
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        #shilpa multi constraint
        self.cost_dim = args.cost_dim

    def store(self, s, a, a_logprob, r, c, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        #shilpa multi constraint
        # self.c[self.count] = c
        # c should be [c1, c2]
        c = np.asarray(c).reshape(-1)
        assert c.shape[0] == self.cost_dim, f"Expected c=[c1, c2], got shape {c.shape}, value {c}"
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


class RESPO:
    def __init__(self, args):
        if args.env == "Quadrotor":
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
        # self.lr_cost = args.lr_cost  # Learning rate of cost critic
        self.lr_p = args.lr_cost
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
        # self.lambda_ = args.lambda_
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

        
        self.V_p = REFCostCritic(args)
        self.lambda_ = torch.tensor(0.0,requires_grad=True).float()

        self.beta = args.beta
        # self.persistent_eps = 0.0
        self. warm_start_flag = args.warm_start_flag

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_reward_critic = torch.optim.Adam(self.V_r.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_cost_critic = torch.optim.Adam(self.V_c.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_p_critic = torch.optim.Adam(self.V_p.parameters(), lr=self.lr_p, eps=1e-5)
            self.lambda_optimizer = torch.optim.Adam([self.lambda_], lr=self.lr_lambda, eps=1e-5)


        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_reward_critic = torch.optim.Adam(self.V_r.parameters(), lr=self.lr_c)
            self.optimizer_cost_critic = torch.optim.Adam(self.V_c.parameters(), lr=self.lr_c)
            self.optimizer_p_critic = torch.optim.Adam(self.V_p.parameters(), lr=self.lr_p)
            self.lambda_optimizer = torch.optim.Adam([self.lambda_], lr=self.lr_lambda)


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
            p['lr'] = lr_a_now
        for p in self.optimizer_Rcritic.param_groups:
            p['lr'] = lr_c_now
        for p in self.optimizer_Ccritic.param_groups:
            p['lr'] = lr_cost_now

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
    
    #shilpa multi constraint
    def log_sum_exp_dim(self, x, dim=1, eta=0.01, keepdim=True):
        """
        Smooth max over a dimension.

        Example:
            x shape: [batch_size, 2]
            output shape with keepdim=True: [batch_size, 1]
        """
        max_val, _ = torch.max(x, dim=dim, keepdim=True)

        lse = max_val + eta * torch.log(
            torch.sum(torch.exp((x - max_val) / eta), dim=dim, keepdim=True)
        )

        if not keepdim:
            lse = lse.squeeze(dim)

        return lse


  

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, c, s_, dw, done = replay_buffer.numpy_to_tensor()

        """
        Multi-constraint version.

        Expected shapes:
            r:          [batch_size, 1]
            c:          [batch_size, cost_dim]
            dw:         [batch_size, 1]
            done:       [batch_size, 1]
            V_r:        [batch_size, 1]
            V_c:        [batch_size, cost_dim]
            V_p:        [batch_size, cost_dim]
            lambda_:    [cost_dim]
            eps:        [cost_dim]
        """

        device = s.device
        batch_size = s.shape[0]
        cost_dim = c.shape[1]

        # ============================================================
        # Critic predictions
        # ============================================================
        V_r_pred = self.V_r(s)
        V_r_next = self.V_r(s_)

        V_c_pred = self.V_c(s)
        V_c_next = self.V_c(s_)

        V_p_pred = self.V_p(s)
        V_p_next = self.V_p(s_)

        # Safety checks. You can remove these after debugging.
        assert r.dim() == 2 and r.shape[1] == 1, f"Expected r shape [B, 1], got {r.shape}"
        assert c.dim() == 2, f"Expected c shape [B, cost_dim], got {c.shape}"
        assert V_c_pred.shape == c.shape, f"V_c_pred shape {V_c_pred.shape} must match c shape {c.shape}"
        assert V_c_next.shape == c.shape, f"V_c_next shape {V_c_next.shape} must match c shape {c.shape}"
        assert V_p_pred.shape == c.shape, f"V_p_pred shape {V_p_pred.shape} must match c shape {c.shape}"
        assert V_p_next.shape == c.shape, f"V_p_next shape {V_p_next.shape} must match c shape {c.shape}"

        # Make sure dw/done can broadcast with reward and cost tensors.
        if dw.dim() == 1:
            dw = dw.view(-1, 1)
        if done.dim() == 1:
            done = done.view(-1, 1)

        # Indicator for immediate constraint violation per constraint.
        # Shape: [batch_size, cost_dim]
        p_indicator = (c > 0.0).float()

        # ============================================================
        # Reward and multi-cost GAE
        # ============================================================
        # deltas_r: [batch_size, 1]
        # deltas_c: [batch_size, cost_dim]
        deltas_r = r + self.gamma * (1.0 - dw) * V_r_next - V_r_pred
        deltas_c = c + self.gamma * (1.0 - dw) * V_c_next - V_c_pred

        # Advantage tensors:
        # adv_r: [batch_size, 1]
        # adv_c: [batch_size, cost_dim]
        adv_r = torch.zeros_like(r, device=device)
        adv_c = torch.zeros_like(c, device=device)

        gae_r = torch.zeros(1, device=device)
        gae_c = torch.zeros(cost_dim, device=device)

        # Important:
        # Use done to reset GAE across episode boundaries.
        # Use dw in TD target to remove bootstrap if no valid next state.
        for t in reversed(range(batch_size)):
            mask = 1.0 - done[t]  # shape [1], broadcasts to [cost_dim]

            gae_r = deltas_r[t] + self.gamma * self.lamda * mask * gae_r
            adv_r[t] = gae_r

            gae_c = deltas_c[t] + self.gamma * self.lamda * mask * gae_c
            adv_c[t] = gae_c

        # ============================================================
        # Value targets for reward and cost critics
        # ============================================================
        with torch.no_grad():
            V_r_target = adv_r + V_r_pred
            V_c_target = adv_c + V_c_pred

            # ========================================================
            # Multi-constraint REF target:
            # p_i(s) = max{1_{c_i > 0}, gamma * p_i(s')}
            # If dw == 1, no next-state contribution.
            #
            # Shapes:
            #   p_indicator: [batch_size, cost_dim]
            #   V_p_next:    [batch_size, cost_dim]
            #   p_target:    [batch_size, cost_dim]
            # ========================================================
            p_target = torch.maximum(
                p_indicator,
                self.gamma * (1.0 - dw) * V_p_next.detach()
            )

        # ============================================================
        # Normalize reward and cost advantages
        # Do NOT normalize p target.
        # ============================================================
        if self.use_adv_norm:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

            # For multi-constraint costs, normalize each cost dimension separately.
            adv_c = (adv_c - adv_c.mean(dim=0, keepdim=True)) / (
                adv_c.std(dim=0, keepdim=True) + 1e-8
            )

        # ============================================================
        # Prepare persistent_eps and lambda shapes
        # ============================================================
        # persistent_eps should be shape [cost_dim].
        # If it is a scalar float/tensor, this expands it to all constraints.
        if not torch.is_tensor(self.persistent_eps):
            persistent_eps = torch.full(
                (cost_dim,),
                float(self.persistent_eps),
                device=device,
                dtype=V_c_pred.dtype
            )
        else:
            persistent_eps = self.persistent_eps.to(device=device, dtype=V_c_pred.dtype)
            if persistent_eps.dim() == 0:
                persistent_eps = persistent_eps.repeat(cost_dim)
            persistent_eps = persistent_eps.view(-1)

        assert persistent_eps.shape[0] == cost_dim, (
            f"persistent_eps must have shape [cost_dim], got {persistent_eps.shape}"
        )

        # lambda_ should be shape [cost_dim].
        # This assumes self.lambda_ is a torch Parameter or tensor of shape [cost_dim].
        if self.lambda_.dim() == 0:
            # This will allow scalar lambda to run, but true multi-constraint
            # behavior requires initializing lambda_ as [cost_dim].
            lambda_vec = self.lambda_.view(1).repeat(cost_dim)
        else:
            lambda_vec = self.lambda_.view(-1)

        assert lambda_vec.shape[0] == cost_dim, (
            f"lambda_ must have shape [cost_dim], got {self.lambda_.shape}"
        )

        # ============================================================
        # Dual Variable / Lambda Update
        # ============================================================
        # Constraint value per constraint:
        #   constraint_value_i = mean_batch V_c_i(s) * (1 - p_i(s))
        #
        # Shapes:
        #   constraint_value:     [cost_dim]
        #   constraint_violation: [cost_dim]
        # ============================================================
        if self.warm_start_flag == 1:
            with torch.no_grad():
                constraint_value = (
                    V_c_pred.detach() * (1.0 - V_p_pred.detach())
                ).mean(dim=0)

                constraint_violation = constraint_value - persistent_eps

            # For vector lambda:
            # maximize lambda * violation, implemented as minimizing negative.
            #
            # loss_lambda is scalar.
            loss_lambda = -(lambda_vec * constraint_violation).sum()

            self.lambda_optimizer.zero_grad()
            loss_lambda.backward()
            self.lambda_optimizer.step()

            with torch.no_grad():
                self.lambda_.clamp_(min=0.0)

        else:
            with torch.no_grad():
                self.lambda_.fill_(0.0)

            constraint_value = torch.zeros(cost_dim, device=device, dtype=V_c_pred.dtype)
            constraint_violation = torch.zeros(cost_dim, device=device, dtype=V_c_pred.dtype)

        # ============================================================
        # Debug print
        # ============================================================
        with torch.no_grad():
            lambda_print = self.lambda_.detach().view(-1).cpu().numpy()
            eps_print = persistent_eps.detach().cpu().numpy()
            pred_cost_print = V_c_pred.mean(dim=0).detach().cpu().numpy()
            p_value_print = V_p_pred.mean(dim=0).detach().cpu().numpy()
            constraint_value_print = constraint_value.detach().cpu().numpy()
            constraint_violation_print = constraint_violation.detach().cpu().numpy()

            print(
                f"lambda={lambda_print}, "
                f"eps={eps_print}, "
                f"pred_cost={pred_cost_print}, "
                f"p_value={p_value_print}, "
                f"constraint_value={constraint_value_print}, "
                f"constraint_violation={constraint_violation_print}"
            )

        # ============================================================
        # Actor Update
        # ============================================================
        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SubsetRandomSampler(range(batch_size)),
                self.mini_batch_size,
                False
            ):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)

                a_logprob_now = dist_now.log_prob(a[index])

                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True)
                    - a_logprob[index].sum(1, keepdim=True)
                )

                # ====================================================
                # Warm start: reward-only PPO
                # ====================================================
                if self.warm_start_flag == 0:
                    surr1 = ratios * adv_r[index]
                    surr2 = (
                        torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
                        * adv_r[index]
                    )

                    actor_loss = (
                        -torch.min(surr1, surr2)
                        - self.entropy_coef * dist_entropy
                    )

                # ====================================================
                # After warm start: multi-constraint RESPO-style actor update
                # ====================================================
                else:
                    # Vp: [mini_batch, cost_dim]
                    Vp = V_p_pred[index].detach()

                    # lam: [1, cost_dim], broadcasts over mini-batch
                    lam = self.lambda_.detach().view(1, -1)

                    feasible_weight = 1.0 - Vp       # [mini_batch, cost_dim]
                    infeasible_weight = Vp           # [mini_batch, cost_dim]

                    # Cost weight from RESPO:
                    # lambda_i * (1 - p_i) + p_i
                    #
                    # Shape: [mini_batch, cost_dim]
                    cost_weight = lam * feasible_weight + infeasible_weight

                    # ------------------------------------------------
                    # Reward term:
                    # maximize reward only in feasible region.
                    #
                    # Since feasible_weight is per constraint, reduce it to
                    # one scalar state weight for the reward term.
                    #
                    # Conservative option:
                    #   use product over dimensions = reward only when all
                    #   constraints are predicted feasible.
                    # Alternative:
                    #   feasible_weight_r = feasible_weight.mean(dim=1, keepdim=True)
                    # ------------------------------------------------
                    feasible_weight_r = feasible_weight.prod(dim=1, keepdim=True)

                    surr1_r = ratios * adv_r[index] * feasible_weight_r
                    surr2_r = (
                        torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
                        * adv_r[index]
                        * feasible_weight_r
                    )

                    loss_rpi = -torch.min(surr1_r, surr2_r)  # [mini_batch, 1]

                    # ------------------------------------------------
                    # Cost term:
                    # minimize each cost constraint.
                    #
                    # adv_c[index]: [mini_batch, cost_dim]
                    # ratios:       [mini_batch, 1], broadcasts
                    # cost_weight:  [mini_batch, cost_dim]
                    # ------------------------------------------------
                    surr1_c = ratios * adv_c[index] * cost_weight
                    surr2_c = (
                        torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
                        * adv_c[index]
                        * cost_weight
                    )

                    # Conservative PPO clipping for minimization: max.
                    # Shape before reduction: [mini_batch, cost_dim]
                    loss_cpi_per_dim = torch.max(surr1_c, surr2_c)

                    # Sum over constraints to get [mini_batch, 1].
                    # This gives each constraint its own penalty.
                    loss_cpi = loss_cpi_per_dim.sum(dim=1, keepdim=True)

                    # ------------------------------------------------
                    # Total actor loss
                    # ------------------------------------------------
                    actor_loss = (
                        loss_rpi
                        + loss_cpi
                        - self.entropy_coef * dist_entropy
                    )

                # ====================================================
                # Actor optimizer step
                # ====================================================
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()

                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.optimizer_actor.step()

        # ============================================================
        # Critic Updates
        # ============================================================
        reward_critic_loss = F.mse_loss(V_r_pred, V_r_target.detach())

        # Multi-constraint cost critic loss.
        # Standard version averages over batch and cost dimensions.
        cost_critic_loss = F.mse_loss(V_c_pred, V_c_target.detach())

        # Optional per-dimension cost critic loss for debugging:
        # cost_critic_loss_per_dim = ((V_c_pred - V_c_target.detach()) ** 2).mean(dim=0)
        # cost_critic_loss = cost_critic_loss_per_dim.mean()

        # ============================================================
        # REF Update
        # ============================================================
        # V_p_pred and p_target both have shape [batch_size, cost_dim].
        # V_p must output values in [0, 1], e.g. through sigmoid.
        # ============================================================
        ref_critic_loss = F.binary_cross_entropy(
            V_p_pred,
            p_target.detach()
        )

        # Alternative:
        # ref_critic_loss = F.mse_loss(V_p_pred, p_target.detach())

        self.optimizer_reward_critic.zero_grad()
        reward_critic_loss.backward()
        self.optimizer_reward_critic.step()

        self.optimizer_cost_critic.zero_grad()
        cost_critic_loss.backward()
        self.optimizer_cost_critic.step()

        self.optimizer_p_critic.zero_grad()
        ref_critic_loss.backward()
        self.optimizer_p_critic.step()

        # ============================================================
        # Optional final debug print for losses
        # ============================================================
        # with torch.no_grad():
        #     print


   
def evaluate_policy(args, env, agent, state_norm=None, reward_scaling=None):
    times = 3
    evaluate_reward = 0
    #shilpa multi constraint
    # evaluate_cost = 0
    evaluate_cost = np.zeros(args.cost_dim)
    evaluate_max_cost = np.full(args.cost_dim, float("-inf"))  # Initialize max cost for each constraint
    # evaluate_max_cost = float("-inf")
    for _ in range(times):
        s = env.reset()[0]#[0]
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0

        #shilpa multi constraint
        # episode_cost = 0
        # max_cost = float("-inf")
        episode_cost = np.zeros(args.cost_dim)
        max_cost = np.full(args.cost_dim, float("-inf"))  # Initialize max cost

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
            # c = args.omega1*max(0, info["constraint_values"][0]) + args.omega2*max(0, info["constraint_values"][1])  # Assuming info contains constraint values
            #shilpa multi constraint
            # c = info["constraint_values"]  # Assuming info contains constraint values
            c = args.cost_scale * (np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1))
            c = np.maximum(c, 0.0)
            # print("eval cost =", c)

 
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            if args.use_reward_scaling:
                r = reward_scaling(r, update=False)
                c = reward_scaling(c, update=False)
            episode_reward += r
            episode_cost += c
            #shilpa multi constraint
            # max_cost = max(max_cost, c)
            max_cost = np.maximum(max_cost, c)  # Update max cost for each constraint
    
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
        #shilpa multi constraint
        # evaluate_max_cost = max(evaluate_max_cost, max_cost)
        evaluate_max_cost = np.maximum(evaluate_max_cost, max_cost)

    #shilpa multi constraint
    # return evaluate_reward / times, evaluate_cost / times, evaluate_max_cost
    
    # Scalar summaries
    # evaluate_total_cost = np.sum(evaluate_cost)
    # evaluate_max_total_cost = np.max(evaluate_max_cost)

    # return (
    #     evaluate_reward,
    #     evaluate_cost,
    #     evaluate_max_cost,
    #     evaluate_total_cost,
    #     evaluate_max_total_cost,
    # )

    evaluate_reward = evaluate_reward / times
    evaluate_cost = evaluate_cost / times

    evaluate_total_cost = np.sum(evaluate_cost)
    evaluate_max_total_cost = np.max(evaluate_max_cost)

    return (
        evaluate_reward,
        evaluate_cost,
        evaluate_max_cost,
        evaluate_total_cost,
        evaluate_max_total_cost,
    )


#shilpa model save
# def save_agent(agent, save_path, state_norm=None, reward_scaling=None):
#     agent.actor.save(f"{save_path}_actor")
#     agent.Rcritic.save(f"{save_path}_Rcritic")
#     agent.Ccritic.save(f"{save_path}_Ccritic")
#     if state_norm:
#         with open(f"{save_path}_state_norm", "wb") as file1:
#             pickle.dump(state_norm, file1)
#     if reward_scaling:
#         with open(f"{save_path}_reward_scaling", "wb") as file2:
#             pickle.dump(reward_scaling, file2)

def save_agent(agent, save_path, ep=0, state_norm=None, reward_scaling=None):
    agent.actor.save(f"{save_path}_actor_{str(ep)}")
    agent.V_r.save(f"{save_path}_Rcritic_{str(ep)}")
    agent.V_c.save(f"{save_path}_Ccritic_{str(ep)}")
    if state_norm:
        with open(f"{save_path}_state_norm_{str(ep)}", "wb") as file1:
            pickle.dump(state_norm, file1)
    if reward_scaling:
        with open(f"{save_path}_reward_scaling_{str(ep)}", "wb") as file2:
            pickle.dump(reward_scaling, file2)

#shilpa multiple constraints
# def plot_eval_metrics(
#     evaluate_rewards,
#     evaluate_costs,
#     evaluate_max_costs,
#     persistent_eps,
#     save=False,
#     filename="eval_metrics.png",
# ):
#     """
#     Plot evaluation metrics (reward, total cost, max cost) over evaluation
#     checkpoints and optionally save the plot.
#     Args:
#         evaluate_rewards:   List of avg rewards per evaluation checkpoint.
#         evaluate_costs:     List of avg total costs per evaluation checkpoint.
#         evaluate_max_costs: List of max costs per evaluation checkpoint.
#         persistent_eps:     Safety threshold — drawn as a horizontal reference line.
#         save:               Whether to save the plot to a file.
#         filename:           File name to save the plot.
#     """
#     evals = list(range(1, len(evaluate_rewards) + 1))

#     fig, axes = plt.subplots(3, 1, figsize=(10, 9))

#     # ── Subplot 1: Evaluate Reward ────────────────────────────────────────────
#     axes[0].plot(evals, evaluate_rewards, color="blue", label="Eval Reward")
#     axes[0].set_xlabel("Evaluation #")
#     axes[0].set_ylabel("Reward")
#     axes[0].set_title("Evaluation Reward")
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     # ── Subplot 2: Evaluate Max Cost (with safety threshold line) ────────────
#     axes[1].plot(evals, evaluate_max_costs, color="red", label="Eval Max Cost")
#     axes[1].axhline(
#         y=persistent_eps,
#         color="black",
#         linestyle="--",
#         linewidth=1.5,
#         label=f"Safety threshold ({persistent_eps})",
#     )
#     axes[1].set_xlabel("Evaluation #")
#     axes[1].set_ylabel("Max Cost")
#     axes[1].set_title("Evaluation Max Cost per Checkpoint")
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)

#     # ── Subplot 3: Evaluate Total Cost ───────────────────────────────────────
#     axes[2].plot(evals, evaluate_costs, color="green", label="Eval Total Cost")
#     axes[2].set_xlabel("Evaluation #")
#     axes[2].set_ylabel("Total Cost")
#     axes[2].set_title("Evaluation Total Cost per Checkpoint")
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3)

#     plt.tight_layout()
#     if save:
#         plt.savefig(filename, dpi=150)
#     plt.close()

def plot_eval_metrics(
    evaluate_rewards,
    evaluate_costs,
    evaluate_max_costs,
    evaluate_total_costs,
    evaluate_max_total_costs,
    persistent_eps,
    cost_dim,
    save=False,
    filename="eval_metrics.png",
):
    """
    Plot evaluation metrics for arbitrary cost dimension.

    evaluate_costs:
        list where each entry has shape [cost_dim]

    evaluate_max_costs:
        list where each entry has shape [cost_dim]

    evaluate_total_costs:
        list of scalar sum of total costs over dimensions

    evaluate_max_total_costs:
        list of scalar max over max costs across dimensions
    """

    evals = list(range(1, len(evaluate_rewards) + 1))

    evaluate_costs = np.asarray(evaluate_costs)                  # [N, cost_dim]
    evaluate_max_costs = np.asarray(evaluate_max_costs)          # [N, cost_dim]
    evaluate_total_costs = np.asarray(evaluate_total_costs)      # [N]
    evaluate_max_total_costs = np.asarray(evaluate_max_total_costs)  # [N]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # ------------------------------------------------------------------
    # Subplot 1: Evaluation Reward
    # ------------------------------------------------------------------
    axes[0].plot(evals, evaluate_rewards, color="blue", label="Eval Reward")
    axes[0].set_xlabel("Evaluation #")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Evaluation Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Subplot 2: Max Cost per dimension
    # ------------------------------------------------------------------
    for i in range(cost_dim):
        axes[1].plot(
            evals,
            evaluate_max_costs[:, i],
            label=f"Max C{i+1}",
        )

    # axes[1].plot(
    #     evals,
    #     evaluate_max_total_costs,
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.8,
    #     label="Max Total Cost",
    # )

    axes[1].axhline(
        y=persistent_eps,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Safety threshold ({persistent_eps})",
    )

    axes[1].set_xlabel("Evaluation #")
    axes[1].set_ylabel("Max Cost")
    axes[1].set_title("Evaluation Max Cost per Constraint Dimension")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Subplot 3: Total Cost per dimension
    # ------------------------------------------------------------------
    for i in range(cost_dim):
        axes[2].plot(
            evals,
            evaluate_costs[:, i],
            label=f"Total C{i+1}",
        )

    # axes[2].plot(
    #     evals,
    #     evaluate_total_costs,
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.8,
    #     label="Total Cost Sum",
    # )

    axes[2].set_xlabel("Evaluation #")
    axes[2].set_ylabel("Total Cost")
    axes[2].set_title("Evaluation Total Cost per Constraint Dimension")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=150)

    plt.close()


 #shilpa multi constraints           
# def plot_metrics(
#     episode_rewards,
#     episode_costs,
#     max_costs,
#     save=False,
#     filename="training_metrics.png",
# ):
#     """
#     Plot the metrics (reward and cost) over episodes and optionally save the plot.
#     Args:
#         episode_rewards: List of total rewards per episode.
#         episode_costs: List of total costs per episode.
#         save: Whether to save the plot to a file.
#         filename: File name to save the plot.
#     """
#     # plt.ion()  # Turn on interactive mode
#     plt.figure(figsize=(10, 6))
#     plt.clf()  # Clear the current figure to avoid overlapping plots
#     # plt.figure(figsize=(10, 6))

#     # Plot total rewards
#     plt.subplot(3, 1, 1)
#     plt.plot(episode_rewards, label="Total Reward", color="blue")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.title("Total Reward per Episode")
#     plt.legend()

#     # Plot total costs
#     plt.subplot(3, 1, 2)
#     plt.plot(max_costs, label="Max Cost", color="red")
#     plt.xlabel("Episode")
#     plt.ylabel("Max Cost")
#     plt.title("Max Cost per Episode")
#     plt.legend()

#     # Plot total costs
#     plt.subplot(3, 1, 3)
#     plt.plot(episode_costs, label="Total Cost", color="green")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Cost")
#     plt.title("Total Cost per Episode")
#     plt.legend()

#     plt.tight_layout()
#     if save:
#         plt.savefig(filename)
#     # plt.show()
#     plt.close()

def plot_metrics(
    episode_rewards,
    episode_costs,
    max_costs,
    cost_dim,
    save=False,
    filename="training_metrics.png",
):
    """
    Plot training metrics for arbitrary cost dimension.

    episode_costs:
        list where each entry has shape [cost_dim]

    max_costs:
        list where each entry has shape [cost_dim]
    """

    episodes = list(range(1, len(episode_rewards) + 1))

    episode_costs = np.asarray(episode_costs)  # [N, cost_dim]
    max_costs = np.asarray(max_costs)          # [N, cost_dim]

    total_cost_sum = np.sum(episode_costs, axis=1)
    max_cost_over_dims = np.max(max_costs, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    axes[0].plot(episodes, episode_rewards, label="Total Reward", color="blue")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Total Reward per Episode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Max costs
    # ------------------------------------------------------------------
    for i in range(cost_dim):
        axes[1].plot(
            episodes,
            max_costs[:, i],
            label=f"Max C{i+1}",
        )

    # axes[1].plot(
    #     episodes,
    #     max_cost_over_dims,
    #     label="Max Total Cost",
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.8,
    # )

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Max Cost")
    axes[1].set_title("Max Cost per Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Total costs
    # ------------------------------------------------------------------
    for i in range(cost_dim):
        axes[2].plot(
            episodes,
            episode_costs[:, i],
            label=f"Total C{i+1}",
        )

    # axes[2].plot(
    #     episodes,
    #     total_cost_sum,
    #     label="Total Cost Sum",
    #     color="black",
    #     linestyle="-.",
    #     linewidth=1.8,
    # )

    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Total Cost")
    axes[2].set_title("Total Cost per Episode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=150)

    plt.close()



def main(args, run_number):
    seed, GAMMA = args.seed, args.GAMMA

    # Create directories for the current run
    model_dir = f"./models/{args.env}_PD_RESPO/run{run_number}/"
    data_train_dir = f"./data_train/{args.env}_PD_RESPO/run{run_number}/"
    plot_data_dir = f"./plot_data/{args.env}_PD_RESPO/run{run_number}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_train_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    if args.env == "Quadrotor":
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
    #shilpa disturbances
    print("DISTURBANCE_MODES:", env.DISTURBANCE_MODES)
    print("disturbances:", env.disturbances)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_costs = []  # Record the costs during the evaluating

    #shilpa multi constraint
    evaluate_max_costs = []  # Record the max costs during the evaluating
    evaluate_total_costs = []  # Record the total costs during the evaluating
    evaluate_max_total_costs = []  # Record the max total costs during the evaluating

    total_steps = 0  # Record the total steps during the training
    max_value = -np.inf
    evaluate_max_costs = []

    replay_buffer = ReplayBuffer(args)
    agent = RESPO(args)

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
        s = env.reset()[0]#[0]
        # print ("Initial state:", s)  # Debugging: Print the initial state
        # s_org = copy.deepcopy(s)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        total_reward = 0
        #shilpa multi constraint
        # total_cost = 0
        # max_cost = float("-inf")
        total_cost = np.zeros(args.cost_dim)
        max_cost = np.full(args.cost_dim, -np.inf)

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

            
            s_, r, done, info = env.step(action)
            # c = np.max(info.get('constraint_values', 0.))
            # print(info['constraint_values'].shape)  
            # print(info['constraint_values'])
            # c = 0.0
            # config[quadrotor_config]["constraints"][0]["lower_bounds"] and config[quadrotor_config]["constraints"][0]["upper_bounds"]
            # c  = args.omega1 * max(0, info['constraint_values'][0])+ args.omega2 * max(0, info['constraint_values'][1])
            c = info["constraint_values"] # Assuming info contains constraint values
            # print("training cost =", c)
            total_reward += r
            #shilps multi constraint
            # total_cost += c
            # max_cost = max(max_cost, c)
            c = args.cost_scale*( np.asarray(info["constraint_values"], dtype=np.float32).reshape(-1))
            c = np.maximum(c, 0.0)
            total_cost += c
            max_cost = np.maximum(max_cost, c)
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

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            if not args.use_reward_scaling:
                reward_scaling = None
            if not args.use_state_norm:
                state_norm = None
            #shilpa multi constraint
            # evaluate_reward, evaluate_cost, evaluate_max_cost = evaluate_policy(
            #     args,
            #     env_evaluate,
            #     agent,
            #     state_norm=state_norm,
            #     reward_scaling=reward_scaling,
            # )
            (
                evaluate_reward,
                evaluate_cost,
                evaluate_max_cost,
                evaluate_total_cost,
                evaluate_max_total_cost,
            ) = evaluate_policy(
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
            #shilpa multi constraint
            evaluate_total_costs.append(evaluate_total_cost)
            evaluate_max_total_costs.append(evaluate_max_total_cost)

            # shilpa multi constraint
            # print(
            #     "evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{} \t evaluate_max_cost:{}".format(
            #         evaluate_num, evaluate_reward, evaluate_cost, evaluate_max_cost
            #     )
            # )
            cost_str = " \t ".join(
                [f"Total C{i+1}:{evaluate_cost[i]:.3f}" for i in range(args.cost_dim)]
            )

            max_cost_str = " \t ".join(
                [f"Max C{i+1}:{evaluate_max_cost[i]:.3f}" for i in range(args.cost_dim)]
            )

            print(
                f"evaluate_num:{evaluate_num} \t "
                f"reward:{evaluate_reward:.3f} \t "
                f"{cost_str} \t "
                f"Total Cost:{evaluate_total_cost:.3f} \t "
                f"{max_cost_str} \t "
                f"Max Total Cost:{evaluate_max_total_cost:.3f}"
            )


            # ── NEW: save evaluation plot after every checkpoint ──────────────
            #shilpa multiple constraint
            # plot_eval_metrics(
            #     evaluate_rewards,
            #     evaluate_costs,
            #     evaluate_max_costs,
            #     persistent_eps=args.persistent_eps,
            #     save=True,
            #     filename=f"{plot_data_dir}/eval_metrics.png",
            # )
            plot_eval_metrics(
                    evaluate_rewards,
                    evaluate_costs,
                    evaluate_max_costs,
                    evaluate_total_costs,
                    evaluate_max_total_costs,
                    persistent_eps=args.persistent_eps,
                    cost_dim=args.cost_dim,
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
                f"{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_max_costs.npy",
                np.array(evaluate_max_costs),
            )

            # Check if the current model satisfies the conditions for being the best
            #shilpa multi constraint
            # if (
            #     evaluate_reward > best_reward
            #     and evaluate_max_cost <= args.persistent_eps
            # ):
            #     best_reward = evaluate_reward
            #     best_model_path = f"{model_dir}/Best_RCAC"
            #     print(
            #         f"New best model found! Saving model with reward: {evaluate_reward} and max cost: {evaluate_max_cost}"
            #     )

            #     # Save the best model
            #     if args.use_reward_scaling and args.use_state_norm:
            #         save_agent(agent, best_model_path, state_norm, reward_scaling)
            #     elif args.use_reward_scaling:
            #         save_agent(
            #             agent,
            #             best_model_path,
            #             state_norm=None,
            #             reward_scaling=reward_scaling,
            #         )
            #     elif args.use_state_norm:
            #         save_agent(agent, best_model_path, state_norm)
            #     else:
            #         save_agent(agent, best_model_path)

            if (
                evaluate_reward >= best_reward
                and np.all(evaluate_max_cost <= args.persistent_eps)
            ):
                best_reward = evaluate_reward
                best_model_path = f"{model_dir}/Best_RCAC"

                max_cost_str = ", ".join(
                    [f"Max C{i+1}: {evaluate_max_cost[i]:.3f}" for i in range(args.cost_dim)]
                )

                print(
                    f"New best model found! Saving model with reward: {evaluate_reward:.3f}, "
                    f"{max_cost_str}, Max Total Cost: {evaluate_max_total_cost:.3f}"
                )

                #shilpa model save
                # if args.use_reward_scaling and args.use_state_norm:
                #     save_agent(agent, best_model_path, state_norm, reward_scaling)
                # elif args.use_reward_scaling:
                #     save_agent(
                #         agent,
                #         best_model_path,
                #         state_norm=None,
                #         reward_scaling=reward_scaling,
                #     )
                # elif args.use_state_norm:
                #     save_agent(agent, best_model_path, state_norm)
                # else:
                #     save_agent(agent, best_model_path)
                if args.use_reward_scaling and args.use_state_norm:
                    save_agent(agent, best_model_path, total_steps, state_norm, reward_scaling)
                elif args.use_reward_scaling:
                    save_agent(
                        agent,
                        best_model_path,
                        total_steps, 
                        state_norm=None,
                        reward_scaling=reward_scaling,
                    )
                elif args.use_state_norm:
                    save_agent(agent, best_model_path, total_steps, state_norm)
                else:
                    save_agent(agent, best_model_path, total_steps)


        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)  # Save data for plotting
        np.save(f"{plot_data_dir}/episode_rewards.npy", episode_rewards)
        np.save(f"{plot_data_dir}/episode_max_costs.npy", episode_max_costs)
        #shilpa multi constraints
        # np.save(f"{plot_data_dir}/episode_costs.npy")
        
        #shilpa multi constraints
        # plot_metrics(
        #     episode_rewards,
        #     episode_costs,
        #     episode_max_costs,
        #     save=True,
        #     filename=f"{plot_data_dir}/training_metrics.png",
        # )
        plot_metrics(
            episode_rewards,
            episode_costs,
            episode_max_costs,
            cost_dim=args.cost_dim,
            save=True,
            filename=f"{plot_data_dir}/training_metrics.png",
        )
        

    # Save the evaluation rewards and costs for this run
    np.save(f"{data_train_dir}/evaluate_rewards.npy", evaluate_rewards)
    np.save(f"{data_train_dir}/evaluate_costs.npy", evaluate_costs)
    np.save(f"{data_train_dir}/evaluate_max_costs.npy", evaluate_max_costs)
    #shilpa multi constraints
    np.save(f"{data_train_dir}/evaluate_max_total_costs.npy", evaluate_max_total_costs)


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
        "--entropy_coef", type=float, default=0.007, help="Trick 5: policy entropy"
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
        "--cost_scale", type=float, default=100.0, help="scale of cost"
        )    
    parser.add_argument(
                "--cost_dim",
                type=int,
                default=4,
                help="Dimension of the cost vector",
    )
    parser.add_argument("--lr_lambda",type=float,default=1e-3,help="warm_start_episode") 



    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    print("run=", args.run, "seed=", args.seed, "env=", args.env,  "k_epochs", args.K_epochs)

    main(args, run_number=args.run)
