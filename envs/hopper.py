import numpy as np
import gym
from gym.envs.mujoco import hopper

ACTION_TORQUE_THRESHOLD = 0.5
REWARD_TYPE = "old"


class HopperCostEnv(hopper.HopperEnv):
    OBS_DIM = 11
    max_steps = 500

    def __init__(self, max_steps: int = 500, **kwargs):
        #shilpa windows only
        self._mujoco_initializing = True

        self.max_steps = max_steps
        self._elapsed_steps = 0

        super().__init__(**kwargs)
        #shilpa windows only
        self._mujoco_initializing = False

        # self.max_steps = max_steps
        # self._elapsed_steps = 0

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

        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
            return ob, reward, truncated or terminated, info

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

class HopperPerturbedEnvTest(hopper.HopperEnv):
    """
    Test-time perturbed Hopper environment.

    Samples one gravity perturbation once during __init__ and keeps it fixed
    for all episodes and all steps.

    If shared_gravity_perturbation is given, uses that exact perturbation.
    """

    OBS_DIM = 11
    max_steps = 500

    def __init__(
        self,
        max_steps: int = 500,
        sigma_gravity: float = 0.5,
        shared_gravity_perturbation=None,
        seed=None,
        **kwargs
    ):
        # ------------------------------------------------------------
        # CRITICAL:
        # Old Gym MujocoEnv may call self.step(action) inside __init__.
        # During that phase, return old-style 4-tuple and do NOT apply
        # custom perturbation logic.
        # ------------------------------------------------------------
        self._mujoco_initializing = True

        self.max_steps = int(max_steps)
        self._elapsed_steps = 0

        self.sigma_gravity = float(sigma_gravity)
        self._grav_axis = 2
        self._base_grav = -9.81

        self.shared_gravity_perturbation = shared_gravity_perturbation
        self.fixed_gravity_perturbation = 0.0
        self.fixed_gravity_value = -9.81

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        super().__init__(**kwargs)

        # Parent env exists now.
        self.OBS_DIM = self.observation_space.shape[0]
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

        if self.shared_gravity_perturbation is None:
            if self.sigma_gravity > 0.0:
                self.fixed_gravity_perturbation = float(
                    self.np_random.normal(0.0, self.sigma_gravity)
                )
            else:
                self.fixed_gravity_perturbation = 0.0
        else:
            self.fixed_gravity_perturbation = float(
                self.shared_gravity_perturbation
            )

        self.fixed_gravity_value = (
            self._base_grav + self.fixed_gravity_perturbation
        )

        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        self._mujoco_initializing = False

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        print(
            f"[HopperPerturbedEnvTest] "
            f"base_gravity={self._base_grav}, "
            f"fixed_perturbation={self.fixed_gravity_perturbation}, "
            f"fixed_gravity={self.fixed_gravity_value}"
        )

    def reset(self, seed=None, **kwargs):
        """
        Return:
            obs, info
        Same style as your training env.
        """

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(**kwargs)

        if isinstance(obs, tuple):
            observation = obs[0]
            info = obs[1] if len(obs) > 1 else {}
        else:
            observation = obs
            info = {}

        self._elapsed_steps = 0

        # Keep same fixed gravity every episode.
        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        info.update({
            "gravity": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity": float(self._base_grav),
            "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
            "sigma_gravity": float(self.sigma_gravity),
        })

        return observation, info

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
        # ------------------------------------------------------------
        # CRITICAL:
        # Old Gym MujocoEnv may call self.step(action) inside __init__.
        # During that phase, return old-style 4-tuple.
        # ------------------------------------------------------------
        if getattr(self, "_mujoco_initializing", False):
            xposbefore = self.sim.data.qpos[0].copy()

            self.do_simulation(action, self.frame_skip)

            xposafter = self.sim.data.qpos[0].copy()
            ob = self._get_obs()

            reward, info = self.old_reward(
                xposbefore,
                xposafter,
                action,
            )

            done = False

            return ob, reward, done, info

        # Keep same gravity every step.
        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

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

        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            "cost": cost,
            "action_torque_cost": cost,
            "max_action_abs": float(np.max(np.abs(action))),
            "action_torque_threshold": ACTION_TORQUE_THRESHOLD,

            "gravity": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity": float(self._base_grav),
            "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
            "sigma_gravity": float(self.sigma_gravity),
        })

        return ob, reward, cost, truncated, terminated, info

