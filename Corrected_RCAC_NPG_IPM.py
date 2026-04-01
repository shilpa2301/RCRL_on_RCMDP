# Generated from: Corrected_RCAC_NPG_IPM.ipynb
# Converted at: 2026-03-31T22:58:35.906Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#system files import
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
from gymnasium import utils
from typing import Optional, List, Tuple
from gymnasium import spaces
import matplotlib.pyplot as plt  # Import for plotting
#### Self made files
from Actor_Critic import *
#from safety_gym_wrapper import *
#from RCAC_NPG import *
from cartpole import *
from hopper import *
#from pendulum import *
from replay_buffer import *
from general_tweeks import *
import mujoco

class AdversarialPendulum:
    def __init__(self):
        self.env = gym.make("Pendulum-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self,seed=None, options=None):
        s, _ = self.env.reset()
        return s
    def simulate_step(self, state, action):
        """
        Simulate one step from arbitrary state WITHOUT breaking env.
        """
    
        # Save original internal state
        s = self.env.reset()
        orig_state = self.env.unwrapped.state.copy()
    
        # ---- FIX: convert observation → internal state ----
        cos_t, sin_t, theta_dot = state
        theta = np.arctan2(sin_t, cos_t)
    
        self.env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
    
        # Ensure correct action shape
        action = np.array(action, dtype=np.float32).reshape(1,)
    
        # Step
        s_next, reward, terminated, truncated, _ = self.env.step(action)
    
        # Compute cost (your definition)
        cos_t, sin_t, theta_dot = s_next
        theta = np.arctan2(sin_t, cos_t)
        cost = float(abs(theta) > 0.8)
    
        # Restore original state
        self.env.unwrapped.state = state
    
        return torch.tensor(s_next,dtype=torch.float32), torch.tensor(s_next,dtype=torch.float32)
    def step(self, a, adv=[0,0]):
        s_next, _, done, trunc, _ = self.env.step(a)

        fx, fy = adv
        cos_t, sin_t, theta_dot = s_next
        theta = np.arctan2(sin_t, cos_t)

        theta += 0.05 * fx
        theta_dot += 0.05 * fy

        s_next = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

        reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * a.squeeze()**2)/50  #Trick 3: Reward Normalization

        # constraint: keep near upright
        cost = (abs(theta) > 0.8).astype(np.float32)

        return s_next, reward, cost, done or trunc

class Robust_RCAC_NPG:
  def __init__(self,args,mj=None):
    self.env_nm = args.env
    if args.env == 'CartPolePerturbedEnv':
        self.env = CartPolePerturbedEnv() #CartPolePerturbedEnv() # CartPoleCostEnv()#HopperPerturbedEnv()
    elif args.env == 'CartPoleCostEnv':
        self.env = CartPoleCostEnv()
    elif args.env == 'HopperPerturbedEnv':
        self.env = HopperPerturbedEnv()
    elif args.env == 'safetycar':
        self.env = Safety_car(gym.make("SafetyCarGoal1-v0", disable_env_checker=True))
        #self.sim = self.env.get_sim(self.env.env)
        #obs = args.env.reset()
        self.mj = mj
    elif args.env == 'pendulum':
        self.env = AdversarialPendulum()
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
    self. warm_start_flag = args.warm_start_flag

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
    lse = max_val + eta * torch.log(torch.exp((a - max_val) / eta) + torch.exp((b - max_val) / eta))
    return lse



  def persistent_safety_function(self, trajectory, actor, cost_critic, gamma):
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
        state_input = state.unsqueeze(0) if state.dim() == 1 else state
        dist = actor.get_dist(state_input)
        
        # FIX: remove batch AFTER sampling
        sampled_actions = dist.sample((50,)).squeeze(1)

        # Sample multiple actions from the distribution
        sampled_actions = dist.sample((50,))  # Sample 50 actions for Monte Carlo approximation
        log_probs = dist.log_prob(sampled_actions)  # Get log probabilities of sampled actions

        # Compute Q_h(s, a) for each sampled action
        q_values = []
        sampled_actions = dist.sample((50,)).squeeze(1)

        q_values = []
        
        for sampled_action in sampled_actions:
        
            # FIX 2: convert to numpy correctly
            sampled_action_np = sampled_action.detach().cpu().numpy()
        
            # FIX 3: correct call + unpacking
            #print("Final action shape:", sampled_action_np.shape)
            #print("Expected:", self.env.action_space.shape)
            if self.env=='safetycar':
                simulated_next_state, current_cost = self.env.simulate_step(
                    self.env,      # wrapper is fine since using step_up
                    self.mj,
                    sampled_action_np
                )
                simulated_next_state = torch.Tensor(simulated_next_state)
                next_value = cost_critic(simulated_next_state).item()
            
                current_cost = torch.tensor(current_cost, dtype=torch.float32)
                next_value = torch.tensor(next_value).detach().clone()
            
                q_value = (1 - gamma) * current_cost + gamma * self.log_sum_exp_fn(current_cost, next_value)
            
                q_values.append(q_value)
            else:
                simulated_next_state,current_cost = self.env.simulate_step(state,sampled_action_np)
                next_value = cost_critic(simulated_next_state).detach().clone()
                q_value = (1 - gamma) * current_cost + gamma * self.log_sum_exp_fn(current_cost, next_value)
                q_values.append(q_value)
        q_values = torch.stack(q_values)
        v_h_pi = torch.mean(q_values)
        v_h_pi_values.append(v_h_pi.item())
        cost_values.append(v_h_pi.item())

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
        ch = 0
        with torch.no_grad():
            vs = self.Rcritic(s)
            vs_ = self.Rcritic(s_)
            vcs = self.Ccritic(s)
            vcs_ = self.Ccritic(s_)
            trajectory = {
                'states': s,
                'actions': a,
                'next_states': s_,
                'costs': c
            }

            with torch.no_grad():
                vl_pi = self.persistent_safety_function(trajectory, self.actor, self.Ccritic, self.gamma)
                penalty_term = vl_pi - self.persistent_eps
                beta_penalty = self.beta * penalty_term
                vs_mean = vs.mean().item()
                if self.warm_start_flag == 1:
                    if self.env_nm == 'pendulum':
                        ch = np.argmax([-vs_mean, beta_penalty])
                    else:
                        ch = np.argmax([vs_mean, beta_penalty])
                else:
                    ch = 0
                print("ch, vs_mean, vl_pi, beta penalty=", ch, vs_mean, vl_pi, beta_penalty)
            reg_norm, weight_norm, bias_norm = 0, [], []           

           
            if ch==1:
              for layer in self.Ccritic.children():
                  if isinstance(layer, nn.Linear):
                      weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                      bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
              reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
              deltas = (1-self.gamma)*c + self.gamma *self.log_sum_exp_fn(c, ((1.0 - dw) * vcs_)) - vcs - self.alpha * a_logprob.sum(dim=1, keepdim=True) - self.weight_reg * reg_norm
              for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                  gae = max(delta, gae * (1.0 - d))
                  adv.insert(0, gae)
              adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
              v_target = adv  # + vcs + self.alpha * a_logprob.sum(dim=1, keepdim=True)
              if self.use_adv_norm:  # Trick 1:advantage normalization
                  adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
              adv = -adv
            else:
              for layer in self.Rcritic.children():
                  if isinstance(layer, nn.Linear):
                      weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                      bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
              reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
              deltas = r + self.gamma * (1.0 - dw) * vs_ - vs - self.alpha * a_logprob.sum(dim=1, keepdim=True) - self.weight_reg * reg_norm
              for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                    gae = delta +self.gamma * self.lamda * gae * (1.0 - d) 
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
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.optimizer_actor.step()

                v_s = self.Rcritic(s[index])
                v_cs = self.Ccritic(s[index])
                # Calculate the loss of critic
                if ch == 0:
                    # Update Reward critic
                    Rcritic_loss = F.mse_loss(v_target[index], v_s)
                    self.optimizer_Rcritic.zero_grad()
                    Rcritic_loss.backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.Rcritic.parameters(), 1)
                    self.optimizer_Rcritic.step()
                else:
                    Ccritic_loss = F.mse_loss(v_target[index], v_cs) 
                    #Update Cost critic
                    self.optimizer_Ccritic.zero_grad()
                    Ccritic_loss.backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.Ccritic.parameters(), 1)
                    self.optimizer_Ccritic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha.exp() * (a_logprob.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
            self.alpha_optimzier.zero_grad()
            alpha_loss.backward()
            self.alpha_optimzier.step()
            self.alpha = self.log_alpha.exp()

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


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
            if args.env=='safetycar':
                    out = env.step_up(action)
            else:
                out = env.step(action)

            if len(out) == 5:
                #print("5 is where it is")
                s_, r, terminated, truncated, info = out#s_, r,c, done, info
                c = info.get("cost", 0.0)   # fallback
                done = truncated or terminated
            elif len(out) == 6:
                #print("6 is where it is")
                s_, r, c, terminated, truncated, info = out
                done = truncated or terminated
            elif len(out)==4:
                s_,r,c,done = out
            else:
                raise ValueError(f"Unexpected number of outputs: {len(out)}")
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
    elif args.env == 'HopperPerturbed':
        env = HopperPerturbed() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)
        env_evaluate = HopperPerturbed() #CartPolePerturbedEnv() # CartPoleCostEnv()#gym.make(args.env)  # When evaluating the policy, we need to rebuild an environment
        env_reset = HopperPerturbed() #CartPolePerturbedEnv() #CartPoleCostEnv()#gym.make(args.env)  # When sampling multiple next states, we need to return to the current states
    elif args.env == 'safetycar':
        env = gym.make("SafetyCarGoal1-v0", disable_env_checker=True)
        obs, info = env.reset()
        mj = env.unwrapped.task
        env = Safety_car(env)
        env_evaluate = Safety_car(gym.make("SafetyCarGoal1-v0", disable_env_checker=True))
        env_reset = Safety_car(gym.make("SafetyCarGoal1-v0", disable_env_checker=True))
    elif args.env == 'pendulum':
        env = AdversarialPendulum()
        env_evaluate = AdversarialPendulum()
        env_reset = AdversarialPendulum()
    else:
        raise ValueError(f"Unexpected input: {args.env}")
    # Set random seed
    #env.reset(seed=seed)
    #env.seed(seed)
    env.reset()
    #env.action_space.seed(seed)

    #env_evaluate.reset(seed=seed)
    #env_evaluate.action_space.seed(seed)

    #env_reset.reset(seed=seed)
    #env_reset.action_space.seed(seed)
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
    evaluate_max_costs = []

    replay_buffer = ReplayBuffer(args)
    if args.env=='safetycar':
        agent = Robust_RCAC_NPG(args,mj)
    else:
        agent = Robust_RCAC_NPG(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir=f'runs/RNAC/env_{args.env}_{args.policy_dist}_run{run_number}_seed_{seed}_GAMMA_{GAMMA}')

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

    for total_steps in tqdm(range(args.max_train_steps)):
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

        agent.beta = args.beta #50.0 #min(max_beta, min_beta * np.exp(total_steps / scale))
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
                if args.env=='safetycar':
                    out = env.step_up(action)
                else:
                    out = env.step(action)

                if len(out) == 5:
                    s_, r, terminated, truncated, info = out#s_, r,c, done, info
                    c = info.get("cost", 0.0)   # fallback
                    done = truncated or terminated
                elif len(out) == 6:
                    s_, r, c, terminated, truncated, info = out
                    done = truncated or terminated
                else:
                    raise ValueError(f"Unexpected number of outputs: {len(out)}")
                total_reward += r
                total_cost += c
            else:
                if args.env=='safetycar':
                    out = env.step_up(action)
                else:
                    out = env.step(action)

                if len(out) == 5:
                    s_, r, terminated, truncated, info = out#s_, r,c, done, info
                    c = info.get("cost", 0.0)   # fallback
                    done = truncated or terminated
                elif len(out) == 6:
                    s_, r, c, terminated, truncated, info = out
                    done = truncated or terminated
                elif len(out) == 4:
                    s_,r,c,done = out
                else:
                    raise ValueError(f"Unexpected number of outputs: {len(out)}")
                total_reward += r
                total_cost += c
                max_cost = max(max_cost, c)
            #x_pos = np.array([info['x_position']])
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
            replay_buffer.store(s, a, a_logprob, r,c, s_, dw, done)
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
                evaluate_reward,evaluate_cost, evaluate_max_cost = evaluate_policy(args, env_evaluate, agent, state_norm=state_norm, reward_scaling=reward_scaling)
                #evaluate_cost = evaluate_cost_function(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                evaluate_costs.append(evaluate_cost)
                evaluate_max_costs.append(evaluate_max_cost)

                print("evaluate_num:{} \t evaluate_reward:{} \t evaluate_cost:{} \t evaluate_max_cost:{}".format(evaluate_num, evaluate_reward,evaluate_cost, evaluate_max_cost))
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                # if evaluate_num % args.save_freq == 0:
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_rewards_2nd_try.npy', np.array(evaluate_rewards))
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs_2nd_try.npy', np.array(evaluate_costs))
                np.save(f'{data_train_dir}/RNAC_{args.policy_dist}_env_{args.env}_seed_{seed}_GAMMA_{GAMMA}_costs_2nd_try.npy', np.array(evaluate_max_cost))

                if args.use_reward_scaling and args.use_state_norm:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm, reward_scaling)
                elif args.use_reward_scaling:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm=None, reward_scaling=reward_scaling)
                elif args.use_state_norm:
                    save_agent(agent, f"{model_dir}/RCAC", state_norm)
                else:
                    save_agent(agent, f"{model_dir}/RCAC")
                max_value = evaluate_reward

                           
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)        # Save data for plotting
        np.save(f"{plot_data_dir}/episode_rewards_2nd_try.npy", episode_rewards)
        np.save(f"{plot_data_dir}/episode_max_costs_2nd_try.npy", episode_max_costs)
        if (total_steps+1)%500 == 0:
            plot_metrics(episode_rewards, episode_costs, episode_max_costs, save=True, filename=f"{plot_data_dir}/training_metrics_2nd_try.png")

    # Save the evaluation rewards and costs for this run
    np.save(f"{data_train_dir}/evaluate_rewards_2nd_try.npy", evaluate_rewards)
    np.save(f"{data_train_dir}/evaluate_costs_2nd_try.npy", evaluate_costs)
    np.save(f"{data_train_dir}/evaluate_max_costs_2nd_try.npy", evaluate_max_costs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='pendulum',help="HopperPerturbed/CartPolePerturbedEnv/CartPoleCostEnv/safetycar/pendulumv1")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(4.5e3), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian or Discrete")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-2, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")

        # Save the finmma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--persistent_eps", type=float, default=0.05, help="Persistent Safety Perturbation")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
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
    parser.add_argument("--baseline",type=int,default=100,help="baseline")
    parser.add_argument("--lambda_",type=int,default=50,help="lambda")
    parser.add_argument("--beta",type=float,default=50.0,help="beta") 
    parser.add_argument("--run",type=int,default=1,help="run_number")
    parser.add_argument("--warm_start_flag",type=int,default=0,help="warm_start_flag") 
    parser.add_argument("--warm_start_episode",type=int,default=300,help="warm_start_episode") 


    args = parser.parse_args([])
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    print("run=", args.run,"seed=", args.seed, "env=", args.env)

    main(args, run_number=args.run)

env = AdversarialPendulum()

s_ = env.reset()

env.step(env.action_space.sample())

env = gym.make("Pendulum-v1")
env.reset()

print(env.unwrapped.state)