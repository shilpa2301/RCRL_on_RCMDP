import numpy as np
import gym

from gym.envs.mujoco import pusher_v4
from gym.spaces import Box


ACTION_TORQUE_THRESHOLD = 1.0
VIOLATIONS_ALLOWED = 100


class PusherTest(pusher_v4.PusherEnv):
    def reset(self, seed=None, **kwargs):
        out = super().reset(seed=seed, **kwargs)

        self.current_timestep = 0
        self.violations = 0

        return out

    def step(self, action):
        out = super().step(action)

        if len(out) == 5:
            next_ob, reward, terminated, truncated, info = out
        else:
            next_ob, reward, done, info = out
            terminated = done
            truncated = False

        try:
            self.current_timestep += 1

            if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
                self.violations += 1

            if self.violations > VIOLATIONS_ALLOWED:
                terminated = True
                reward = 0.0

        except Exception:
            pass

        return next_ob, reward, terminated, truncated, info


class PusherWithCostBase(pusher_v4.PusherEnv):
    OBS_DIM = 23
    max_steps = 100

    def __init__(
        self,
        sigma_viscosity: float = 0.0,
        max_steps: int = 100,
        perturb_viscosity: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sigma_viscosity = sigma_viscosity
        self.max_steps = max_steps
        self.perturb_viscosity = perturb_viscosity
        self._elapsed_steps = 0

        self._base_viscosity = float(self.model.opt.viscosity)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float64,
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )

    def reset(self, seed=None, **kwargs):
        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs = out
            info = {}

        self._elapsed_steps = 0
        self.model.opt.viscosity = self._base_viscosity

        return obs, info

    def compute_reward(self, action):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(action).sum()

        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        info = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            reward_near=reward_near,
        )

        return reward, info

    def step(self, action):
        if self.perturb_viscosity and self.sigma_viscosity > 0.0:
            perturbed = self._base_viscosity + self.np_random.normal(
                0.0,
                self.sigma_viscosity,
            )
            self.model.opt.viscosity = max(0.0, perturbed)

        reward, info = self.compute_reward(action)

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        self._elapsed_steps += 1

        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )
        # cost = 0.0

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        return ob, reward, cost, truncated, terminated, info


class PusherWithCost(PusherWithCostBase):
    def __init__(self, sigma_viscosity: float = 0.0, max_steps: int = 100, **kwargs):
        super().__init__(
            sigma_viscosity=sigma_viscosity,
            max_steps=max_steps,
            perturb_viscosity=False,
            **kwargs
        )


class PusherWithCostPerturbed(PusherWithCostBase):
    def __init__(self, sigma_viscosity: float = 0.0, max_steps: int = 100, **kwargs):
        super().__init__(
            sigma_viscosity=sigma_viscosity,
            max_steps=max_steps,
            perturb_viscosity=True,
            **kwargs
        )
