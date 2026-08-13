import numpy as np
import gym
from gym.envs.mujoco import hopper

ACTION_TORQUE_THRESHOLD = 0.5
REWARD_TYPE = "old"


class HopperCostEnv(hopper.HopperEnv):
    OBS_DIM = 11
    max_steps = 500

    def __init__(self, max_steps: int = 500, **kwargs):
        super().__init__(**kwargs)

        self.max_steps = max_steps
        self._elapsed_steps = 0

        # Use whatever the parent HopperEnv defines.
        self.OBS_DIM = self.observation_space.shape[0]

    def reset(self, seed=None, **kwargs):
        """
        Same reset style as your SwimmerWithPos:
            return obs, {}
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(**kwargs)

        self._elapsed_steps = 0

        return obs, {}

    def old_reward(self, xposbefore, xposafter, action):
        """
        Older Gym Hopper-style reward:
            forward velocity + alive bonus - control cost
        """
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward_alive = 1.0

        reward = reward_run + reward_alive + reward_ctrl

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_alive=reward_alive,
            xpos=xposafter,
            x_position=xposafter,
            x_velocity=reward_run,
            reward_forward=reward_run,
            reward_survive=reward_alive,
        )

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0].copy()

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0].copy()
        ob = self._get_obs()

        self._elapsed_steps += 1

        if REWARD_TYPE == "old":
            reward, info = self.old_reward(
                xposbefore,
                xposafter,
                action,
            )
        else:
            raise ValueError(f"Unknown REWARD_TYPE: {REWARD_TYPE}")

        # Continuous excess-torque constraint cost.
        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        truncated = self._elapsed_steps >= self.max_steps

        # Match SwimmerWithPos behavior.
        terminated = False

        info.update({
            "cost": cost,
            "action_torque_cost": cost,
            "max_action_abs": float(np.max(np.abs(action))),
            "action_torque_threshold": ACTION_TORQUE_THRESHOLD,
        })

        return ob, reward, cost, truncated, terminated, info

class HopperPerturbedEnv(hopper.HopperEnv):
    OBS_DIM = 11
    max_steps = 500

    def __init__(self, max_steps: int = 500, sigma_gravity: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.max_steps = max_steps
        self._elapsed_steps = 0

        # Use whatever the parent HopperEnv defines.
        self.OBS_DIM = self.observation_space.shape[0]

        self.sigma_gravity = sigma_gravity
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
            

    def reset(self, seed=None, **kwargs):
        """
        Same reset style as your SwimmerWithPos:
            return obs, {}
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(**kwargs)

        self._elapsed_steps = 0

        self.model.opt.gravity[self._grav_axis] = self._base_grav
        return obs, {}

    def old_reward(self, xposbefore, xposafter, action):
        """
        Older Gym Hopper-style reward:
            forward velocity + alive bonus - control cost
        """
        reward_ctrl = -1e-3 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward_alive = 1.0

        reward = reward_run + reward_alive + reward_ctrl

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_alive=reward_alive,
            xpos=xposafter,
            x_position=xposafter,
            x_velocity=reward_run,
            reward_forward=reward_run,
            reward_survive=reward_alive,
        )

        return reward, info

    def step(self, action):
        if self.sigma_gravity > 0.0:
                    self.model.opt.gravity[self._grav_axis] = (
                        self._base_grav + np.random.normal(0.0, self.sigma_gravity)
                    )
        xposbefore = self.sim.data.qpos[0].copy()

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0].copy()
        ob = self._get_obs()

        self._elapsed_steps += 1

        if REWARD_TYPE == "old":
            reward, info = self.old_reward(
                xposbefore,
                xposafter,
                action,
            )
        else:
            raise ValueError(f"Unknown REWARD_TYPE: {REWARD_TYPE}")

        # Continuous excess-torque constraint cost.
        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        truncated = self._elapsed_steps >= self.max_steps

        # Match SwimmerWithPos behavior.
        terminated = False

        info.update({
            "cost": cost,
            "action_torque_cost": cost,
            "max_action_abs": float(np.max(np.abs(action))),
            "action_torque_threshold": ACTION_TORQUE_THRESHOLD,
        })

        return ob, reward, cost, truncated, terminated, info
