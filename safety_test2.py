# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 21 20:28:19 2026

# @author: Sourav
# """

# # Instructions for setting up the environment
# # conda create -n safety_gym_environment python=3.10
# # conda activate safety_gym_environment
# # pip install pygame==2.5.2
# # pip install mujoco
# # pip install gymnasium==0.29.1
# # pip install safety-gymnasium

# import numpy as np
# import safety_gymnasium as gym
# print(gym.__version__)

# class RobustSafetyGym:

#     def __init__(self, env):
#         self.env = env
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space

#     # def reset(self, seed=None):
#     #     obs, info = self.env.reset(seed=seed)
#     #     eps = 0.1

#     #     print("Unwrapped environment type:", type(self.env.unwrapped))
#     #     print("Attributes of unwrapped environment:", dir(self.env.unwrapped))
#     #     print("Does 'sim' exist in unwrapped environment?:", hasattr(self.env.unwrapped, 'sim'))
#     #     if hasattr(self.env.unwrapped, 'sim'):
#     #         print("Attributes of sim:", dir(self.env.unwrapped.sim))

#     #     model = self.env.unwrapped.model

#     #     # Gravity perturbation
#     #     model.opt.gravity[2] = -9.81 + np.random.uniform(-2, 2) * eps

#     #     # Mass perturbation
#     #     model.body_mass[:] *= (
#     #         1 + np.random.uniform(-0.2, 0.2, size=model.body_mass.shape) * eps
#     #     )

#     #     # Friction perturbation
#     #     model.geom_friction[:] *= (
#     #         1 + np.random.uniform(-0.3, 0.3, size=model.geom_friction.shape) * eps
#     #     )

#     #     return obs
#     def reset(self, seed=None):
#         # Access the environment configuration
#         config = self.env.unwrapped.config
#         eps = 0.1

#         print("config=", config)

#         # Apply perturbations to the configuration
#         try:
#             # Perturb gravity
#             if 'gravity' in config:
#                 print("perturbing gravity")
#                 config['gravity'] = -9.81 + np.random.uniform(-2, 2) * eps

#             # Perturb mass (if applicable)
#             if 'body_mass' in config:
#                 print("perturbing body mass")
#                 config['body_mass'] = [
#                     mass * (1 + np.random.uniform(-0.2, 0.2) * eps)
#                     for mass in config['body_mass']
#                 ]

#             # Perturb friction (if applicable)
#             if 'geom_friction' in config:
#                 print("perturbing geom friction")
#                 config['geom_friction'] = [
#                     friction * (1 + np.random.uniform(-0.3, 0.3) * eps)
#                     for friction in config['geom_friction']
#                 ]

#             # Reinitialize the environment with the modified configuration
#             self.env = gym.make(self.env.unwrapped.spec.id, config=config)

#             # Reset the new environment
#             obs, info = self.env.reset(seed=seed)
#         except Exception as e:
#             print("Error while modifying environment configuration:", e)
#             raise e

#         return obs



#     def step(self, action):
#         # Action perturbation
#         action = action + np.random.normal(0, 0.1, size=action.shape)

#         # obs, reward, terminated, truncated, info = self.env.step(action)
#         # result = self.env.step(action)  # Capture all returned values

#         # if len(result) == 5:  # Gymnasium-style (5 values)
#         #     obs, reward, terminated, truncated, info = result
#         #     done = terminated or truncated  # Combine terminated and truncated flags
#         # elif len(result) == 4:  # OpenAI Gym-style (4 values)
#         #     obs, reward, done, info = result
#         #     terminated, truncated = done, False  # Assume done corresponds to terminated
#         # else:
#         #     raise ValueError(f"Unexpected number of values returned by step(): {len(result)}")

#         obs, reward, terminated, truncated, cost, info = self.env.step(action)

#         # Observation space or state perturbation
#         obs = obs + np.random.normal(0, 0.01, size=obs.shape)

#         done = terminated or truncated
#         # cost = info.get("cost", 0.0)

#         return obs, reward, cost, done, info

# def main():
#     # Create the base environment

#     env = gym.make("SafetyCarGoal1-v0")

#     obs, info = env.reset()

#     print("Observation shape:", obs.shape)

#     base_env = gym.make("SafetyCarGoal1-v0")
#     env = RobustSafetyGym(base_env)

#     # Reset the environment
#     print("Resetting the environment...")
#     obs = env.reset()
#     print("Initial observation:", obs)

#     # Take one step in the environment
#     print("Taking one step in the environment...")
#     action = env.action_space.sample()  # Sample a random action
#     obs, reward, cost, done, info = env.step(action)

#     # Print the results
#     print("Next observation:", obs)
#     print("Reward:", reward)
#     print("Cost:", cost)
#     print("Done:", done)
#     print("Info:", info)

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:28:19 2026

@author: Sourav
"""

import numpy as np
import safety_gymnasium as gym
from safety_gym.robot import Robot

class RobustSafetyGym(Robot):

    def __init__(self, env, robot_path):
        super().__init__(robot_path)

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        eps =0.1

        # Access the MuJoCo model and data from the Robot class
        model = self.model
        data = self.data

        # Gravity perturbation
        print("Original gravity:", model.opt.gravity)
        model.opt.gravity[2] = -9.81 + np.random.uniform(-2, 2) * eps
        print("Perturbed gravity:", model.opt.gravity)

        # Mass perturbation
        print("Original body masses:", model.body_mass)
        model.body_mass[:] *= (
            1 + np.random.uniform(-0.2, 0.2, size=model.body_mass.shape) * eps
        )
        print("Perturbed body masses:", model.body_mass)

        # Friction perturbation
        print("Original geom frictions:", model.geom_friction)
        model.geom_friction[:] *= (
            1 + np.random.uniform(-0.3, 0.3, size=model.geom_friction.shape) * eps
        )
        print("Perturbed geom frictions:", model.geom_friction)

        # Forward the simulation to apply changes
        mujoco.mj_forward(model, data)

        return obs



    def step(self, action):
        # Action perturbation
        action = action + np.random.normal(0, 0.1, size=action.shape)

        # Take a step in the environment
        obs, reward, terminated, truncated, cost, info = self.env.step(action)

        # Observation space or state perturbation
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)

        done = terminated or truncated
        return obs, reward, cost, done, info

def main():
    # Create the base environment
    robot_path = '/project/ag2682/sm3934/RCRL_on_RMDP/safety_gym/xmls/car.xml'
    base_env = gym.make("SafetyCarGoal1-v0")
    env = RobustSafetyGym(base_env, robot_path)

    # Reset the environment
    print("Resetting the environment...")
    obs = env.reset()
    print("Initial observation:", obs)

    # Take one step in the environment
    print("Taking one step in the environment...")
    action = env.action_space.sample()  # Sample a random action
    obs, reward, cost, done, info = env.step(action)

    # Print the results
    print("Next observation:", obs)
    print("Reward:", reward)
    print("Cost:", cost)
    print("Done:", done)
    print("Info:", info)

if __name__ == "__main__":
    main()
