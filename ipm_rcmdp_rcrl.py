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

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

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
        self.count = 0

    def store(self, s, a, a_logprob, r,c, s_, dw, done):
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

        return s, a, a_logprob, r,c, s_, dw, done


class Robust_RCAC_NPG:
  def __init__(self,args):
    self.env = CartPoleCostEnv()#HopperPerturbedEnv()
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
    self.b = args.baseline
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

    self.beta = 0.0
    self.persistent_eps = 0.0

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

  def persistent_safety_function(self, trajectory, actor, cost_critic, gamma):
    """
    Compute the robust value function V_h^pi(s) and vl_pi using bootstrapping.

    Args:
        trajectory (dict): Dictionary containing episode data:
            - 'states': List of states (torch.Tensor).
            - 'actions': List of actions (torch.Tensor).
            - 'next_states': List of next states (torch.Tensor).
            - 'costs': List of costs (torch.Tensor).
        actor (torch.nn.Module): Policy network.
        cost_critic (torch.nn.Module): Cost critic network.
        gamma (float): Discount factor.

    Returns:
        v_h_pi_values (torch.Tensor): Robust value function V_h^pi(s) for all states.
        vl_pi (float): Maximum cost value (max(V_h(s))) over the batch.
    """
    states = trajectory['states']
    actions = trajectory['actions']
    next_states = trajectory['next_states']
    costs = trajectory['costs']

    v_h_pi_values = []
    cost_values = []

    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        next_state = next_states[i]
        h_s = costs[i]

        # Compute action probabilities
        # a, a_logprob = self.choose_action(s)

        #MODIFY this
        # Get the action distribution from the actor
        dist = actor.get_dist(state)

        #PROBLEM1: cost functions critic should be >=0
        #PROBLEM2: sampled actions very similar, so q values very similar, V almost same as any q

        # Sample multiple actions from the distribution
        sampled_actions = dist.sample((50,))  # Sample 100 actions for Monte Carlo approximation
        log_probs = dist.log_prob(sampled_actions)  # Get log probabilities of sampled actions

        # Compute Q_h(s, a) for each sampled action
        q_values = []
        for sampled_action in sampled_actions:
            # Use the actual cost for the action taken
            if torch.allclose(sampled_action, action, atol=1e-4):
                current_cost = h_s
            else:
                # Use the environment's cost calculation logic
                x_position = state[0].item()  # Extract the cart position (x) from the state
                theta = state[2].item()  # Extract the pole angle (theta) from the state
                current_cost = self.env.compute_cost(x_position, theta)

            # Simulate the next state based on the sampled action
            simulated_next_state = self.env.simulate_next_state(state, sampled_action)

            # Get V_h(simulated_next_state) from the cost critic
            next_value = cost_critic(simulated_next_state).item()

            # Compute Q_h(s, a)
            # q_value = (1-gamma)*current_cost + gamma * max(current_cost, next_value)
            q_value = current_cost + gamma * max(current_cost, next_value)
            q_values.append(q_value)

        # Compute V_h(s) using a weighted average of Q_h(s, a) with the probabilities
        q_values = torch.tensor(q_values)
        v_h_pi = torch.mean(q_values)
        v_h_pi_values.append(v_h_pi.item())

        # Append cost value for vl_pi computation
        cost_values.append(v_h_pi.item())

    # Compute vl_pi = max(V_h(s)) over all states
    vl_pi = max(cost_values)

    return vl_pi



  def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r,c, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
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
            # IPM uncertainty set
            # print("VS shape:",vs.shape)
            # print("VCS shape:",vcs.shape)
            #shilpa rcrl
            # Construct trajectory dynamically from replay buffer
            trajectory = {
                'states': s,
                'actions': a,
                'next_states': s_,
                'costs': c
            }

            with torch.no_grad():
                # vs_mean = vs.mean().item()
                # vcs_mean = vcs.mean().item()
                # ch = np.argmax([vs_mean/self.lambda_, (vcs_mean-self.b)])
                #print(vs_mean,vcs_mean,ch)
                #input()
                # Compute robust value function and V_L^pi
                #PROBLEM3: for each state should we calculate a vlpi? but ultimately we are considering aggregated value, so max of all states should be fine
                vl_pi = self.persistent_safety_function(trajectory, self.actor, self.Ccritic, self.gamma)
                # Compute penalty term and beta penalty
                penalty_term = max(0, vl_pi - self.persistent_eps)  # Apply penalty only if V_L(pi) > epsilon_tolerance
                print(self.beta)
                beta_penalty = self.beta * penalty_term
                # Decide whether to prioritize reward or cost based on beta_penalty
                vs_mean = vs.mean().item()
                # vcs_mean = vcs.mean().item()
                ch = np.argmax([vs_mean, beta_penalty])  # Choose between reward and cost prioritization

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
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
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

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha.exp() * (a_logprob.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
            self.alpha_optimzier.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzier.step()
            self.alpha = self.log_alpha.exp()


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    evaluate_cost = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        episode_cost = 0
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
            s = s_
        evaluate_reward += episode_reward
        evaluate_cost += episode_cost

    return evaluate_reward / times,evaluate_cost / times

def save_agent(agent, save_path, state_norm, reward_scaling):
    agent.actor.save(f'{save_path}_actor')
    agent.Rcritic.save(f'{save_path}_Rcritic')
    agent.Ccritic.save(f'{save_path}_Ccritic')
    with open(f'{save_path}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    with open(f'{save_path}_reward_scaling', 'wb') as file2:
        pickle.dump(reward_scaling, file2)

class CartPoleCostEnv(gym.Env):

    def __init__(self):

        # Observation: [cart position, cart velocity, pole angle, pole angular velocity]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # Continuous action force
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32
        )

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.tau = 0.02

        self.state = None
        self.steps = 0
        self.max_episode_steps = 500

        # Define the maximum possible cost for normalization
        self.max_cost = 2.4 + 10.0  # Max cart position + penalty

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0

        return self.state

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = float(action)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        self.steps += 1

        # reward (same idea as hopper)
        reward = 1.0

        # cost = distance from center
        cost = abs(x)

        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        if done and self.steps < 450:
            cost += 10.0   # penalty value (tunable)

        # Normalize cost between 0 and 1
        # normalized_cost = cost / self.max_cost
        # cost = normalized_cost

        info = {
            "x_position": x
        }

        return self.state, reward, cost, done, info

    def compute_cost(self, x, theta):
        """
        Compute the cost based on the current state.

        Args:
            x (float): The cart's position (distance from the center).
            theta (float): The pole's angle (in radians).

        Returns:
            float: The computed cost.
        """
        # Cost is the absolute distance of the cart from the center (x)
        cost = abs(x)

        # # Optional: Add more terms to the cost function based on other state variables
        # # For example, penalize large pole angles or angular velocities
        # angle_limit = 12 * np.pi / 180  # Angle limit in radians (12 degrees)
        # if abs(theta) > angle_limit:
        #     cost += 10.0  # Add a penalty for exceeding the angle limit

        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        if done and self.steps < 450:
            cost += 10.0   # penalty value (tunable)

        # Normalize cost between 0 and 1
        # normalized_cost = cost / self.max_cost
        # cost = normalized_cost

        return cost

    def simulate_next_state(self, state, action):
        """
        Simulate the next state based on the current state and action, without modifying the environment's state.

        Args:
            state (np.array): Current state [x, x_dot, theta, theta_dot].
            action (float): Action to apply.

        Returns:
            torch.Tensor: Simulated next state.
        """
        x, x_dot, theta, theta_dot = state
        force = float(action)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Compute the next state based on the current state and action
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        return torch.tensor(next_state, dtype=torch.float32)


class HopperPerturbedEnv(MujocoEnv, utils.EzPickle):

    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        hindsight_e=0.0,
        hindsight=False
    ):

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float64
        )

        MujocoEnv.__init__(self, xml_file, 4, observation_space)

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        # save base values*
        self.gravity = -9.81

        self.thigh_joint_damping = 1.0
        self.leg_joint_damping = 1.0
        self.foot_joint_damping = 1.0

        self.actuator_ctrlrange = (-1.0, 1.0)
        self.actuator_ctrllimited = int(1)

        # hindsight parameter*
        self.hindsight_e = hindsight_e
        self.hindsight = hindsight

        #MujocoEnv.__init__(self, xml_file, 4)



    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def test(self):
        #sim = self.sim
        model = self.model
        #print(sim.get_state())
        print('body_names: ', model.body_names)
        print('joint_names: ', model.joint_names)
        print('actuator_names: ', model.actuator_names)
        print('model.actuator_forcelimited', model.actuator_forcelimited)
        print('actuator_ctrlrange', model.actuator_ctrlrange)
        print('_actuator_gear', model.actuator_gear)
        print('_jnt_stiffness', model.jnt_stiffness)
        print('_dof_damping', model.dof_damping)
        print('_dof_frictionloss', model.dof_frictionloss)
        print('actuator_ctrllimited', model.actuator_ctrllimited)

    def step(self, action):
        if np.random.binomial(n=1, p=self.hindsight_e):
            action = self.action_space.sample()

        x_position_before = self.data.qpos[0]
        # add noise to action for next state -> stochastic model
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        noise = self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        self.do_simulation(action_noise, self.frame_skip)

        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        cost  = 1
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "noise":noise
        }

        return observation, reward,cost, done, info

    def reset(
        self,
        x_pos: float = 0.0,
        state: Optional[int] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        use_xml: bool = False,
        gravity: float = -9.81,
        thigh_joint_stiffness: float = 0.0,
        leg_joint_stiffness: float = 0.0,
        foot_joint_stiffness: float = 0.0,
        springref: float = 0.0,
        actuator_ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        joint_damping_p: float = 0.0,
        joint_frictionloss: float = 0.0
    ):
        ob, info = super().reset(seed=seed, options=options)
        # hindsight*
        if self.hindsight:
            actuator_ctrlrange = (-0.85, 0.85)
        # grab model
        model = self.model
        # perturb gravity in z (3rd) dimension*
        model.opt.gravity[2] = gravity
        # perturb thigh joint*
        model.jnt_stiffness[3] = thigh_joint_stiffness
        model.qpos_spring[3] = springref
        # perturb leg joint*
        model.jnt_stiffness[4] = leg_joint_stiffness
        model.qpos_spring[4] = springref
        # perturb foot joint*
        model.jnt_stiffness[5] = foot_joint_stiffness
        model.qpos_spring[5] = springref
        # perturb actuator (controller) control range*
        model.actuator_ctrllimited[0] = self.actuator_ctrllimited
        model.actuator_ctrlrange[0] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        model.actuator_ctrllimited[1] = self.actuator_ctrllimited
        model.actuator_ctrlrange[1] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        model.actuator_ctrllimited[2] = self.actuator_ctrllimited
        model.actuator_ctrlrange[2] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        # perturb joint damping in percentage
        model.dof_damping[3] = self.thigh_joint_damping * (1 + joint_damping_p)
        model.dof_damping[4] = self.leg_joint_damping * (1 + joint_damping_p)
        model.dof_damping[5] = self.foot_joint_damping * (1 + joint_damping_p)
        # perturb joint frictionloss
        model.dof_frictionloss[3] = joint_frictionloss
        model.dof_frictionloss[4] = joint_frictionloss
        model.dof_frictionloss[5] = joint_frictionloss
        return ob

    def save_xml(self, savepath):
      mujoco.mj_saveLastXML(savepath, self.model)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


# Define the plot_metrics function
def plot_metrics(episode_rewards, episode_costs, save=False, filename="training_metrics.png"):
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
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, label="Total Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    # Plot total costs
    plt.subplot(2, 1, 2)
    plt.plot(episode_costs, label="Total Cost", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("Total Cost per Episode")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()
    plt.close()


def main(args, number):
    seed, GAMMA = args.seed, args.GAMMA
    env = CartPoleCostEnv()#gym.make(args.env)
    env_evaluate = CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
    env_reset = CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
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
    save_path = f"./models/RCAC_{args.env}_{GAMMA}" ###******* TENTATIVE PLEASE CHANGE TO YOUR FOLDER OF SAVING ACCORDINGLY ***********

    replay_buffer = ReplayBuffer(args)
    agent = Robust_RCAC_NPG(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/RNAC/env_{}_{}_number_{}_seed_{}_GAMMA_{}'.format(args.env, args.policy_dist, number, seed, GAMMA))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # Tracking metrics for plotting
    episode_rewards = []
    episode_costs = []
    # steps = []

    for total_steps in tqdm(range(args.max_train_steps)):
        #if total_steps > args.max_train_steps // 2:
        #    agent.gamma = 0.999
        s = env.reset()
        s_org = copy.deepcopy(s)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        total_reward = 0
        total_cost = 0

        # beta = 1 + (ep / epochs) * (max_beta - min_beta)
        

        # Update beta dynamically
        max_beta = 200
        min_beta = 1
        scale = args.max_train_steps / 5
        agent.beta = 1.0 #50.0 #min(max_beta, min_beta * np.exp(total_steps / scale))

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            #if total_steps < args.random_steps:  # Take the random actions in the beginning for the better exploration
            #    a = env.action_space.sample()
            #    s_tensor = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
            #    with torch.no_grad():
            #        dist = agent.actor.get_dist(s_tensor)
            #        a_logprob = dist.log_prob(torch.Tensor(a)).numpy().flatten()
            #else:
            #    a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
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
            x_pos = np.array([info['x_position']])
            if args.use_state_norm:
                #nexts = state_norm(nexts, update=False)
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
                c = reward_norm(c)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
                c = reward_scaling(c)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r,c, s_, dw, done)
            s = copy.deepcopy(s_)
            s_org = copy.deepcopy(state_norm.denormal(s_, update=False))

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward,evaluate_cost = evaluate_policy(args, env_evaluate, agent, state_norm)
                #evaluate_cost = evaluate_cost_function(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_costs.append(evaluate_cost)
                print("evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{}".format(evaluate_num, evaluate_reward,evaluate_cost))
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/RNAC_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))

                # save actor, critic for evaluation in perturbed environment
                if evaluate_reward > max_value:
                    save_agent(agent, save_path, state_norm, reward_scaling)
                    max_value = evaluate_reward

            # Track rewards and costs for the episode
            # if done:
            #     episode_rewards.append(total_reward)
            #     episode_costs.append(total_cost)
            #     steps.append(total_steps)  # Track the total steps for the x-axis
                
            # Plot metrics every 50 episodes
            #     # Plot metrics every 50 episodes
            # if (total_steps + 1) % 50 == 0:
            #     plot_metrics(total_reward, total_cost, save=True, filename="ipm_training_metrics.png", step=True, steps=steps)
        # Track rewards and costs for the episode
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        plot_metrics(episode_rewards, episode_costs, save=True, filename="ipm_rcrl_training_metrics_trial.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='CartPolePerturbed',help="HopperPerturbed/CartPolePerturbed")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(4.5e3), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
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
    parser.add_argument("--persistent_eps", type=float, default=200.0, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")
    parser.add_argument("--baseline",type=int,default=200,help="baseline")
    parser.add_argument("--lambda_",type=int,default=50,help="lambda")

    args = parser.parse_args([])
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    main(args, number=1)