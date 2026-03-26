# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:28:19 2026

@author: Sourav
"""

#conda create -n safety_gym_environment python=3.10
#conda activate safety_gym_environment
#pip install pygame==2.5.2
#pip install mujoco
#pip install gymnasium==0.29.1
#pip install safety-gymnasium
import numpy as np

############# TEST SCRIPT ########################
import safety_gymnasium as gym

env = gym.make("SafetyCarGoal1-v0")

obs, info = env.reset()

print("Observation shape:", obs.shape)

############### Environment Wrapper ####################
'''
    To call this wrapper what you need to do is create an object 
        obj = RobustSafetyGym(gym.make("SafetyCarGoal1-v0"))
                    OR   if you are more accustomed to call the object environment
        env = RobustSafetyGym(gym.make("SafetyCarGoal1-v0"))
        
        But Shilpa, remember when you try to access the observation space or action space you have to call it as obj.observation_space or env.observation_space and obj.action_space or 
        env.action_space respectively
        
        Finally I have provided you with the various options for perturbations here
            1) Perturbing Gravity: Gravity is only in the z-dimension so the gravity(downward) can be perturbed by a suitable epsilon 
            2) The mass of the car can be perturbed (Not very recommended) but can capture some real life scenarios like when the car is heavily loaded.
            3) Friction: One of the best ways to introduce randomness can depict tyre conditions and road conditions when watery or snowing
            4) In the step() I have shown that you can perturb the observation space like in Cartpole (RCAC) and action space but do not do this, it is much difficult to learn this way
'''
class RobustSafetyGym:

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        eps =0.1

        model = self.env.unwrapped.model

        # Gravity perturbation
        model.opt.gravity[2] = -9.81 + np.random.uniform(-2, 2)*eps

        # Mass perturbation
        model.body_mass[:] *= (
            1 + np.random.uniform(-0.2, 0.2, size=model.body_mass.shape)*eps
        )

        # Friction perturbation
        model.geom_friction[:] *= (
            1 + np.random.uniform(-0.3, 0.3, size=model.geom_friction.shape)*eps
        )

        return obs

    def step(self, action):
        #Action perturbation
        action = action + np.random.normal(0, 0.1, size=action.shape)

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        
        #Observation space or State perturbation
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)

        done = terminated or truncated
        cost = info.get("cost", 0.0)

        return obs, reward, cost, done, info

base_env = gym.make("SafetyCarGoal1-v0")
env = RobustSafetyGym(base_env)

s = env.reset()
s_, r, c, done, info = env.step(env.action_space.sample())
