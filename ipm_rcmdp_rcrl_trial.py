import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
from torch.distributions import Beta, Normal,Categorical
#from normalization import Normalization, RewardScaling
from torch.distributions import Uniform
import gymnasium as gym
import argparse
import pickle
import math
import random
import copy
import mujoco
import os
from tqdm import tqdm
from gymnasium.envs.mujoco import MujocoEnv
from gym import utils
from typing import Optional, List, Tuple
from gymnasium import spaces
import matplotlib.pyplot as plt  # Import for plotting
# from envs.cartpole import CartPoleCostEnv, CartPolePerturbedEnv
# from envs.hopper_perturbed import HopperPerturbedEnv
from envs.hopper2 import HopperPerturbedEnv



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
        alpha = F.softplus(self.alpha_layer(s)) + 1.0  # softplus is a smooth approximation to ReLU function
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

    def load(self, filename, device='cpu'):
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
            torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        # print(f"mean shape: {mean.shape}")  # Debugging: Print the shape of mean
        # print(f"log_std shape before expand: {self.log_std.shape}")  # Debugging: Print the shape of log_std
        log_std = self.log_std.expand(mean.shape[0], -1)  # Expand log_std to match the shape of mean
        # print(f"log_std shape after expand: {log_std.shape}")  # Debugging: Print the shape of expanded log_std
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist



    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
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


# class Normalization:
#     def __init__(self, shape):
#         self.running_ms = RunningMeanStd(shape=shape)

#     def __call__(self, x, update=True):
#         # Whether to update the mean and std,during the evaluating,update=False
#         if update:
#             self.running_ms.update(x)
#         x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

#         return x

#     def denormal(self, x, update=False):
#         x = x * (self.running_ms.std + 1e-8) + self.running_ms.mean
#         return x

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Handle cases where x is a tuple or list
        if isinstance(x, (tuple, list)):
            x = x[0]  # Extract the actual state
        x = np.array(x)  # Ensure x is a NumPy array
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

    def load(self, filename, device='cpu'):
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

    def load(self, filename, device='cpu'):
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
        self.infos = [] 
        self.count = 0

    def store(self, s, a, a_logprob, r,c, s_, dw, done, info=None):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.c[self.count] = c
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        # Store the info dictionary
        if info is not None:
            self.infos.append(info)
        else:
            self.infos.append({}) 
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

        return s, a, a_logprob, r,c, s_, dw, done,  self.infos

    def reset(self):
        """
        Reset the replay buffer for the next episode.
        """
        self.s.fill(0)
        self.a.fill(0)
        self.a_logprob.fill(0)
        self.r.fill(0)
        self.c.fill(0)
        self.s_.fill(0)
        self.dw.fill(0)
        self.done.fill(0)
        self.infos.clear()
        self.count = 0

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Prioritized Replay Buffer
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            alpha (float): How much prioritization is used (0 = uniform sampling, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition, priority=1.0):
        """
        Add a new transition to the buffer.
        Args:
            transition (tuple): A tuple (s, a, r, s', done) representing a transition.
            priority (float): The priority of the transition.
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max(max_priority, priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with prioritized sampling.
        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): To what degree to use importance sampling (0 - no correction, 1 - full correction).
        Returns:
            transitions (list): A batch of sampled transitions.
            indices (list): Indices of the sampled transitions.
            weights (array): Importance sampling weights for the sampled transitions.
        """
        if len(self.buffer) == 0:
            raise ValueError("The replay buffer is empty!")

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return transitions, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions.
        Args:
            indices (list): Indices of the sampled transitions.
            priorities (array): New priorities for the sampled transitions.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)



class Robust_RCAC_NPG:
  def __init__(self,args):
    if args.env == 'CartPolePerturbedEnv':
        self.env = CartPolePerturbedEnv() #CartPolePerturbedEnv() # CartPoleCostEnv()#HopperPerturbedEnv()
    elif args.env == 'CartPoleCostEnv':
        self.env = CartPoleCostEnv()
    if args.env == 'HopperPerturbedEnv':
        self.env = HopperPerturbedEnv()
    else:
        print("No env selected")
    #self.env.seed(args.seed)
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

    if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_Rcritic = torch.optim.Adam(self.Rcritic.parameters(), lr=self.lr_c, eps=1e-5)
        self.optimizer_Ccritic = torch.optim.Adam(self.Ccritic.parameters(), lr=self.lr_c, eps=1e-5)
    else:
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_Rcritic = torch.optim.Adam(self.Rcritic.parameters(), lr=self.lr_c)
        self.optimizer_Ccritic = torch.optim.Adam(self.Ccritic.parameters(), lr=self.lr_c)

  def evaluate(self, s):  # When evaluating the policy, we only use the mean in Beta and gaussian and simply the action for Discrete
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
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        elif self.policy_dist == "Gaussian":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        #hopper
        # print(f"Chosen action shape: {a.shape}, log_prob shape: {a_logprob.shape}")
        # Flatten the action and log probability tensors to ensure correct shape
        # return a.squeeze(0).numpy(), a_logprob.squeeze(0).numpy()
        return a.numpy().flatten(), a_logprob.numpy().flatten()
  
  def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_Rcritic.param_groups:
            p['lr'] = lr_c_now
        for p in self.optimizer_Ccritic.param_groups:
            p['lr'] = lr_c_now

  def log_sum_exp_fn(self, a, b, eta=0.1):
        # Compute the Log-Sum-Exp smooth approximation of max(a, b)
        # print("a, b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta))= ", a,b, torch.exp(a / eta), torch.exp(b / eta), torch.log(torch.exp(a / eta) + torch.exp(b / eta)))
        # lse = eta * torch.log(torch.exp(a / eta) + torch.exp(b / eta))

        # Find the maximum value between a and b : else exp(10/0.1) becomes infinity
        max_val = torch.max(a, b)    
        # Stabilize the log-sum-exp computation
        lse = max_val + eta * torch.log(torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta))
        return lse

  
  def persistent_safety_function(self, trajectory, actor, cost_critic, gamma):
    states = trajectory['states']
    actions = trajectory['actions']
    next_states = trajectory['next_states']
    costs = trajectory['costs']
    infos = trajectory.get('infos', [{}] * len(states)) 

    v_h_pi_values = []
    cost_values = []

    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        next_state = next_states[i]
        h_s = costs[i]
        info = infos[i]
        #cartpole
        # dist = actor.get_dist(state)
        # Sample multiple actions from the distribution
        # sampled_actions = dist.sample((50,))  # Sample 50 actions for Monte Carlo approximation
        # log_probs = dist.log_prob(sampled_actions)  # Get log probabilities of sampled actions

        #hopper
        dist = actor.get_dist(state.unsqueeze(0))
        sampled_actions = dist.sample((50,))  # Shape: [50, 1, action_dim]
        sampled_actions = sampled_actions.squeeze(1)  # Remove the batch dimension, shape: [50, action_dim]
        log_probs = dist.log_prob(sampled_actions)

        
        # Compute Q_h(s, a) for each sampled action
        q_values = []
        # print("Samples actions=", sampled_actions)
        for sampled_action in sampled_actions:
            #hopper
            # print(f"Action shape after sampled action: {sampled_action.shape}")
            # sampled_action = sampled_action.numpy()  # Flatten the sampled action to (3,)

            #cartpole
            # if torch.allclose(sampled_action, action, atol=1e-4):
            #     current_cost = h_s
            # else:
            #     x_position = state[0].item() 
            #     theta = state[2].item()  
            #     current_cost = self.env.compute_cost(x_position, theta)
            if torch.allclose(sampled_action, action, atol=1e-4):
                current_cost = h_s
            else:
                #hopper
                # Use the info dictionary to get z and angle
                z, angle = self.env.get_z_and_angle_from_state(state, info)
                current_cost = self.env.compute_cost(z, angle)


            #cartpole
            # simulated_next_state = self.env.simulate_next_state(state, sampled_action)
            
            #hopper
            simulated_next_state = self.env.simulate_next_state(state.numpy(), sampled_action.numpy(), info)

            next_value = cost_critic(simulated_next_state)#.item()

            
            # Compute Q_h(s, a)
            current_cost = torch.tensor(current_cost, dtype=torch.float32)
            next_value = torch.tensor(next_value).detach().clone()
            q_value = (1 - gamma) * current_cost + gamma * self.log_sum_exp_fn(current_cost, next_value)
            # q_value = (1-gamma)*current_cost + gamma * max(current_cost, next_value)
            # q_value = current_cost + gamma * max(current_cost, next_value)
            # q_value = max(current_cost, next_value)
            q_values.append(q_value)


            # print("current cost, next value, q value=",current_cost, next_value, q_value)

        # Compute V_h(s) using a weighted average of Q_h(s, a) with the probabilities
        q_values = torch.tensor(q_values)
        v_h_pi = torch.mean(q_values)
        v_h_pi_values.append(v_h_pi.item())
        cost_values.append(v_h_pi.item())

    vl_pi = max(cost_values)

    return vl_pi



  def update(self, replay_buffer, total_steps):
        # s, a, a_logprob, r,c, s_, dw, done, infos = replay_buffer.numpy_to_tensor()  # Get training data
        transitions, indices, weights = replay_buffer.sample(self.mini_batch_size, beta=0.4)
        states, actions, log_probs, rewards, costs, next_states, dw, done, infos = zip(*transitions)

        # # Convert to tensors
        # # s = torch.tensor(states, dtype=torch.float)
        # a = torch.tensor(actions, dtype=torch.float)
        # a_logprob = torch.tensor(log_probs, dtype=torch.float)
        # r = torch.tensor(rewards, dtype=torch.float)
        # c = torch.tensor(costs, dtype=torch.float)
        # # s_ = torch.tensor(next_states, dtype=torch.float)
        # dw = torch.tensor(dw, dtype=torch.float)
        # done = torch.tensor(done, dtype=torch.float)
        # weights = torch.tensor(weights, dtype=torch.float)

        
        # # Ensure states and next_states are tensors
        # s = torch.stack(states)  # Stack the tuple into a single tensor
        # s_ = torch.stack(next_states)  # Stack the tuple into a single tensor

        # Convert states and next_states to tensors
        s = torch.tensor(np.stack(states), dtype=torch.float32)  # Stack the tuple into a single tensor
        s_ = torch.tensor(np.stack(next_states), dtype=torch.float32)  # Stack the tuple into a single tensor
        a = torch.tensor(np.array(actions), dtype=torch.float32)  # Convert actions to tensor
        a_logprob = torch.tensor(np.array(log_probs), dtype=torch.float32)  # Convert log probabilities to tensor
        r = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)  # Convert rewards to tensor and add dimension
        c = torch.tensor(np.array(costs), dtype=torch.float32).unsqueeze(1)  # Convert costs to tensor and add dimension
        dw = torch.tensor(np.array(dw), dtype=torch.float32).unsqueeze(1)  # Convert done-without-penalty flag to tensor
        done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)  # Convert done flag to tensor
        weights = torch.tensor(np.array(weights), dtype=torch.float32).unsqueeze(1)  # Convert weights to tensor
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.Rcritic(s)
            vs_ = self.Rcritic(s_)
            vcs = self.Ccritic(s)
            vcs_ = self.Ccritic(s_)
                       
            trajectory = {
                'states': s,
                'actions': a,
                'next_states': s_,
                'costs': c,
                'infos': infos
            }

            with torch.no_grad():
                # Compute robust value function and V_L^pi
                vl_pi = self.persistent_safety_function(trajectory, self.actor, self.Ccritic, self.gamma)
                penalty_term = max(0, vl_pi - self.persistent_eps)  # Apply penalty only if V_L(pi) > epsilon_tolerance
                beta_penalty = self.beta * penalty_term
                vs_mean = vs.mean().item()
                # ch = np.argmax([vs_mean, beta_penalty]) 
                ch = np.argmax([vs_mean/self.lambda_, beta_penalty])  
                print("ch, vs_mean, vl_pi, beta_penalty =", ch, vs_mean, vl_pi, beta_penalty ) 

            reg_norm, weight_norm, bias_norm = 0, [], []           

           
            if ch==1:
              #print("Cost chosen")
              for layer in self.Ccritic.children():
                  if isinstance(layer, nn.Linear):
                      weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                      bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
              reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
              deltas = c + self.gamma * (1.0 - dw) * vcs_ - vcs - self.alpha * a_logprob.sum(dim=1, keepdim=True) - self.weight_reg * reg_norm
              for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                  gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                  adv.insert(0, gae)
              adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
              v_target = adv + vcs + self.alpha * a_logprob.sum(dim=1, keepdim=True)
              if self.use_adv_norm:  # Trick 1:advantage normalization
                  adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
              #shilpa
              adv = -adv
            else:
              for layer in self.Rcritic.children():
                  if isinstance(layer, nn.Linear):
                      weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                      bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
              reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
              deltas = r + self.gamma * (1.0 - dw) * vs_ - vs - self.alpha * a_logprob.sum(dim=1, keepdim=True) - self.weight_reg * reg_norm
              for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                  gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                  adv.insert(0, gae)
              adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
              v_target = adv + vs + self.alpha * a_logprob.sum(dim=1, keepdim=True)
              if self.use_adv_norm:  # Trick 1:advantage normalization
                  adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
              

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            # for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
            for index in BatchSampler(SubsetRandomSampler(range(self.mini_batch_size)), self.mini_batch_size, False):

                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                             keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.Rcritic(s[index])
                v_cs = self.Ccritic(s[index])
                # Calculate the loss of critic
                Rcritic_loss = F.mse_loss(v_target[index], v_s)
                Ccritic_loss = F.mse_loss(v_target[index], v_cs)
                # Update Reward critic
                self.optimizer_Rcritic.zero_grad()
                Rcritic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.Rcritic.parameters(), 0.5)
                self.optimizer_Rcritic.step()
                #Update Cost critic
                self.optimizer_Ccritic.zero_grad()
                Ccritic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.Ccritic.parameters(), 0.5)
                self.optimizer_Ccritic.step()

        # Update priorities in the replay buffer
        td_errors = torch.abs(r + self.gamma * (1 - dw) * self.Ccritic(s_).detach() - self.Ccritic(s).detach()).numpy()
        replay_buffer.update_priorities(indices, td_errors)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha.exp() * (a_logprob.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
            self.alpha_optimzier.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzier.step()
            self.alpha = self.log_alpha.exp()


def evaluate_policy(args, env, agent, state_norm=None, reward_scaling=None):
    times = 3
    evaluate_reward = 0
    evaluate_cost = 0
    evaluate_max_cost = float('-inf')
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        episode_cost = 0
        max_cost = float('-inf')
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r,c, done, _ = env.step(action)
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

    return evaluate_reward / times,evaluate_cost / times, evaluate_max_cost

def save_agent(agent, save_path, state_norm=None, reward_scaling=None):
    agent.actor.save(f'{save_path}_actor')
    agent.Rcritic.save(f'{save_path}_Rcritic')
    agent.Ccritic.save(f'{save_path}_Ccritic')
    if state_norm:
        with open(f'{save_path}_state_norm', 'wb') as file1:
            pickle.dump(state_norm, file1)
    if reward_scaling:
        with open(f'{save_path}_reward_scaling', 'wb') as file2:
            pickle.dump(reward_scaling, file2)



# def linear_decay_schedule(start_value, end_value, current_step, total_steps):
#     """
#     Linearly decreases a value from start_value to end_value over total_steps.

#     Args:
#         start_value (float): Initial value of the parameter.
#         end_value (float): Final value of the parameter.
#         current_step (int): Current training step.
#         total_steps (int): Total training steps.

#     Returns:
#         float: Updated parameter value.
#     """
#     fraction = min(current_step / total_steps, 1.0)  # Ensure it doesn't go below end_value
#     return start_value - fraction * (start_value - end_value)


def plot_metrics(episode_rewards, episode_costs, max_costs, save=False, filename="training_metrics.png"):
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
    model_dir = f"./models/{args.env}/run{run_number}/"
    data_train_dir = f"./data_train/{args.env}/run{run_number}/"
    plot_data_dir = f"./plot_data/{args.env}/run{run_number}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_train_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    if args.env == 'CartPolePerturbedEnv':
        env = CartPolePerturbedEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = CartPolePerturbedEnv() #CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = CartPolePerturbedEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == 'CartPoleCostEnv':
        env = CartPoleCostEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = CartPoleCostEnv() #CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = CartPoleCostEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == 'HopperPerturbedEnv':
        env = HopperPerturbedEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = HopperPerturbedEnv() #CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = HopperPerturbedEnv() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    
    # Set random seed
    #env.reset(seed=seed)
    #env.seed(seed)
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
    args.max_episode_steps = 1000  # Maximum number of steps per episode
    
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
    # save_path = f"./models/RCAC_{args.env}_{GAMMA}" ###******* TENTATIVE PLEASE CHANGE TO YOUR FOLDER OF SAVING ACCORDINGLY ***********
    evaluate_max_costs = []

    # replay_buffer = ReplayBuffer(args)
    replay_buffer = PrioritizedReplayBuffer(capacity=args.batch_size, alpha=0.6)

    agent = Robust_RCAC_NPG(args)

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/RNAC/env_{}_{}_number_{}_seed_{}_GAMMA_{}'.format(args.env, args.policy_dist, number, seed, GAMMA))
    writer = SummaryWriter(log_dir=f'runs/RNAC/env_{args.env}_{args.policy_dist}_run{run_number}_seed_{seed}_GAMMA_{GAMMA}')

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # Tracking metrics for plotting
    episode_rewards = []
    episode_costs = []
    episode_max_costs = []

    for total_steps in tqdm(range(args.max_train_steps)):
        # if total_steps <= args.max_train_steps-3000 and total_steps%5==0:
        #         agent.persistent_eps = linear_decay_schedule(
        #                         start_value=5.0,  # Start with a high value (relaxed constraints)
        #                         end_value=0.5,    # End with a strict constraint
        #                         current_step=total_steps,
        #                         total_steps=args.max_train_steps-3000)
        #if total_steps > args.max_train_steps // 2:
        #    agent.gamma = 0.999
        s = env.reset()
        # s_org = copy.deepcopy(s)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        total_reward = 0
        total_cost = 0
        max_cost = float('-inf')

        # beta = 1 + (ep / epochs) * (max_beta - min_beta)
        

        # Update beta dynamically
        # max_beta = 200
        # min_beta = 1
        # scale = args.max_train_steps / 5
        agent.beta = args.beta #50.0 #min(max_beta, min_beta * np.exp(total_steps / scale))

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            if args.uncer_set == "DS":
                # Multi-run
                v_min, index = torch.tensor(float('inf')), 0
                v_candidate,index_candidate = torch.tensor(float('inf')), 0
                flag=0
                noise_list, nexts_list, r_list,c_list = [], [], [],[]
                for i in range(args.next_steps):
                    obs = env_reset.reset(state=s_org, x_pos=x_pos)
                    s_, r,c, done, info = env_reset.step(action)
                    # total_reward += r
                    # total_cost += c
                    r_list.append(r)
                    c_list.append(c)
                    noise_list.append(info['noise'])
                    if args.use_state_norm:
                        s_ = state_norm(s_, update=False)
                    nexts_list.append(s_)

                    #########################Please check this part if USING Double Sampling ############################################
                    with torch.no_grad():
                        if agent.Rcritic(torch.tensor(s_, dtype=torch.float)) < v_min:
                            v_min = agent.Rcritic(torch.tensor(s_, dtype=torch.float))
                            index = i
                        if agent.Rcritic(torch.tensor(nexts_list[i], dtype=torch.float)) < v_candidate and lambda_*(agent.Ccritic(torch.tensor(s_,dtype=torch.float))-b)<0:
                            v_candidate = agent.Rcritic(torch.tensor(nexts_list[i], dtype=torch.float))
                            index_candidate = i
                            flag=1
                if flag==1:
                    index = index_candidate
                ############################# UP UNTIL HERE ################################################################
                # pick next state for robust critic update
                ridx = random.randint(0, args.next_steps)
                if ridx == args.next_steps:
                    ridx = index
                s_, r,c, done, info = env.step(np.concatenate((action, noise_list[ridx])))
                total_reward += r
                total_cost += c
            else:
                s_, r,c, done, info = env.step(action)
                total_reward += r
                total_cost += c
                max_cost = max(max_cost, c)
            x_pos = np.array([info['x_position']])
            if args.use_state_norm:
                #nexts = state_norm(nexts, update=False)
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
            # replay_buffer.store(s, a, a_logprob, r,c, s_, dw, done, info)
            td_error = abs(
                    r + args.gamma * agent.Ccritic(torch.tensor(s_, dtype=torch.float, device=agent.Ccritic.fc1.weight.device)).item() -
                    agent.Ccritic(torch.tensor(s, dtype=torch.float, device=agent.Ccritic.fc1.weight.device)).item()
                )
            replay_buffer.add((s, a, a_logprob, r, c, s_, dw, done, info), priority=td_error)


            s = copy.deepcopy(s_)
            # s_org = copy.deepcopy(state_norm.denormal(s_, update=False))

            # When the number of transitions in buffer reaches batch_size,then update
            # if replay_buffer.count == args.batch_size:
            #     agent.update(replay_buffer, total_steps)
            #     replay_buffer.count = 0
            # Update the model when buffer is full
            # if len(replay_buffer) >= args.batch_size and total_steps % 10 == 0:
            #     agent.update(replay_buffer, total_steps)

            # # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % args.evaluate_freq == 0:
            #     evaluate_num += 1
            #     if not args.use_reward_scaling:
            #         reward_scaling = None
            #     if not args.use_state_norm:
            #         state_norm = None
            #     evaluate_reward,evaluate_cost, evaluate_max_cost = evaluate_policy(args, env_evaluate, agent, state_norm=state_norm, reward_scaling=reward_scaling)
            #     #evaluate_cost = evaluate_cost_function(args, env_evaluate, agent, state_norm)
            #     evaluate_rewards.append(evaluate_reward)
            #     evaluate_costs.append(evaluate_cost)
            #     evaluate_max_costs.append(evaluate_max_cost)

            #     print("evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{} \t evaluate_max_cost:{}".format(evaluate_num, evaluate_reward,evaluate_cost, evaluate_max_cost))
            #     writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
            #     # Save the rewards
            #     # if evaluate_num % args.save_freq == 0:
            #     np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_rewards.npy', np.array(evaluate_rewards))
            #     np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy', np.array(evaluate_costs))
            #     np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy', np.array(evaluate_max_cost))

            #     # save actor, critic for evaluation in perturbed environment
            #     # if evaluate_reward >= max_value:
            #     # if total_steps % 10:
            #     #     # save_agent(agent, save_path, state_norm, reward_scaling)
            #     if args.use_reward_scaling and args.use_state_norm:
            #         save_agent(agent, f"{model_dir}/RCAC", state_norm, reward_scaling)
            #     elif args.use_reward_scaling:
            #         save_agent(agent, f"{model_dir}/RCAC", state_norm=None, reward_scaling=reward_scaling)
            #     elif args.use_state_norm:
            #         save_agent(agent, f"{model_dir}/RCAC", state_norm)
            #     else:
            #         save_agent(agent, f"{model_dir}/RCAC")
            #     max_value = evaluate_reward

        if len(replay_buffer) >= args.batch_size and total_steps % 50 == 0:
                for i in range(50):
                    agent.update(replay_buffer, total_steps)
                           
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)        # Save data for plotting
        np.save(f"{plot_data_dir}/episode_rewards.npy", episode_rewards)
        np.save(f"{plot_data_dir}/episode_max_costs.npy", episode_max_costs)
        # print("episode_rewards=", episode_rewards)
        # print("episode_max_costs=", episode_max_costs)
        # print(f"Episode {total_steps}: Total Cost = {total_cost}, Max Cost = {max_cost}")
        plot_metrics(episode_rewards, episode_costs, episode_max_costs, save=True, filename=f"{plot_data_dir}/training_metrics.png")

        # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                if not args.use_reward_scaling:
                    reward_scaling = None
                if not args.use_state_norm:
                    state_norm = None
                evaluate_reward,evaluate_cost, evaluate_max_cost = evaluate_policy(args, env_evaluate, agent, state_norm=state_norm, reward_scaling=reward_scaling)
                #evaluate_cost = evaluate_cost_function(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_costs.append(evaluate_cost)
                evaluate_max_costs.append(evaluate_max_cost)

                print("evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{} \t evaluate_max_cost:{}".format(evaluate_num, evaluate_reward,evaluate_cost, evaluate_max_cost))
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                # if evaluate_num % args.save_freq == 0:
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_rewards.npy', np.array(evaluate_rewards))
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy', np.array(evaluate_costs))
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs.npy', np.array(evaluate_max_cost))

                # save actor, critic for evaluation in perturbed environment
                # if evaluate_reward >= max_value:
                # if total_steps % 10:
                #     # save_agent(agent, save_path, state_norm, reward_scaling)
                if args.use_reward_scaling and args.use_state_norm:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm, reward_scaling)
                elif args.use_reward_scaling:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm=None, reward_scaling=reward_scaling)
                elif args.use_state_norm:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm)
                else:
                    save_agent(agent, f"{model_dir}/RCAC")
                max_value = evaluate_reward

    # Save the evaluation rewards and costs for this run
    np.save(f"{data_train_dir}/evaluate_rewards.npy", evaluate_rewards)
    np.save(f"{data_train_dir}/evaluate_costs.npy", evaluate_costs)
    np.save(f"{data_train_dir}/evaluate_max_costs.npy", evaluate_max_costs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='CartPolePerturbedEnv',help="HopperPerturbedEnv/CartPolePerturbedEnv/CartPoleCostEnv")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(5.5e3), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian or Discrete")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")

        # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--persistent_eps", type=float, default=2.0, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    # parser.add_argument("--use_state_norm", type=lambda x: x.lower() == 'true', default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="seed 2, 5, 7, 11, 17") 
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")
    parser.add_argument("--baseline",type=int,default=9,help="baseline")
    parser.add_argument("--lambda_",type=int,default=1.0,help="lambda")
    parser.add_argument("--beta",type=float,default=50.0,help="beta") 
    parser.add_argument("--run",type=int,default=1,help="run_number") 


    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    print("run=", args.run,"seed=", args.seed, "env=", args.env, "use_state_norm=", args.use_state_norm, "persistent_eps=", args.persistent_eps, "beta=", args.beta)

    main(args, run_number=args.run)