import numpy as np
import os
from gym.envs.mujoco.ant_v3 import AntEnv
import gym

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

###############################################################################
# ANT TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class AntTest(AntEnv):
    def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

    def step(self, action):
        next_ob, reward, done, infos = super().step(action)
        # This is to handle the edge case where mujoco_env calls
        # step in __init__ without calling reset with a random
        # action
        try:
            self.current_timestep += 1
            if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
                self.violations += 1
            if self.violations > VIOLATIONS_ALLOWED:
                done = True
                reward = 0
        except:
            pass
        return next_ob, reward, done, infos

###############################################################################
# ANT WALL ENVIRONMENTS
###############################################################################
ACTION_TORQUE_THRESHOLD = 0.5
class AntCost(AntEnv):
    OBS_DIM = 113
    max_steps = 500
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/env_configs/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            max_steps: int = 500
    ):
        super(AntCost, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )
        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._grav_axis = 2
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        self.max_steps  = max_steps
       
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._grav_axis = 2
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        # self.max_steps  = max_steps

        return obs, {}
    
    def step(self, action):
        
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        #shilpa
        rewards = forward_reward + healthy_reward
        # distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        # rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards  - costs
        #shilpa
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # cost = 0.0
        # ── Termination / truncation ────────────────────────────────────────
        truncated = self._elapsed_steps >= self.max_steps
        terminated = self.terminated

        # done = self.done
        observation = self._get_obs()
        # print(observation.shape)
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        # return observation, reward, cost, done, info
        return observation, reward, cost, truncated, terminated, info

    
#shilpa
class AntCostPerturbed(AntEnv):
    OBS_DIM = 113
    max_steps = 500
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/env_configs/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            sigma_gravity: float = 0.0,
            max_steps: int = 500
    ):
        super(AntCostPerturbed, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )
        self.sigma_gravity = sigma_gravity
        self._elapsed_steps = 0
        self.max_steps  = max_steps

        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])


    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        self.model.opt.gravity[self._grav_axis] = self._base_grav


        return obs, {}
    
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        #shilpa
        rewards = forward_reward + healthy_reward
        # distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        # rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        #shilpa
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # cost = 0.0
        # ── Termination / truncation ────────────────────────────────────────
        truncated = self._elapsed_steps >= self.max_steps
        terminated = self.terminated
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        # return observation, reward, done, info
        return observation, reward, cost, truncated, terminated, info



class AntCostTest(AntCost):
    def step(self, action):
        #shilpa robustness
        self.model.dof_damping[:] += np.random.normal(0, 0.5)
        self.model.dof_damping[:] = np.clip(self.model.dof_damping, 0.0, 10.0)

        # self.model.geom_friction[:] += np.random.normal(0, 0.05)
        # self.model.geom_friction[:] = np.clip(self.model.geom_friction, 0.0, 5.0)

        observation, reward, done, info = super().step(action)
        #if observation[0] < -3 or observation[0] > 3:
        if observation[0] < -3:
            done = True
            reward = 0
        return observation, reward, done, info


class AntCostBroken(AntCost):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


class AntCostBrokenTest(AntCostTest):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


