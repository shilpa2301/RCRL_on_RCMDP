import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
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
