import numpy as np
import gym

from gym.envs.mujoco import inverted_pendulum_v4
from gym.spaces import Box


ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100


class InvertedPendulumTest(inverted_pendulum_v4.InvertedPendulumEnv):
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


class InvertedPendulumWithCostBase(
    inverted_pendulum_v4.InvertedPendulumEnv
):
    OBS_DIM = 4
    max_steps = 1000

    def __init__(
        self,
        sigma_gravity: float = 0.0,
        max_steps: int = 1000,
        perturb_gravity: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sigma_gravity = sigma_gravity
        self.max_steps = max_steps
        self.perturb_gravity = perturb_gravity
        self._elapsed_steps = 0

        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float64,
        )

    def reset(self, seed=None, **kwargs):
        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs = out
            info = {}

        self._elapsed_steps = 0

        # Reset gravity to base value at episode start
        self.model.opt.gravity[self._grav_axis] = self._base_grav

        return obs, info

    def compute_reward(self, action):
        """
        Original InvertedPendulum reward is usually a constant alive reward.
        """
        reward = 1.0

        info = {}

        return reward, info

    def compute_terminated(self, observation):
        """
        Original InvertedPendulum termination condition.

        observation:
            observation[0] = cart position
            observation[1] = pole angle
            observation[2] = cart velocity
            observation[3] = pole angular velocity
        """
        terminated = bool(
            not np.isfinite(observation).all()
            or np.abs(observation[1]) > 0.2
        )

        return terminated

    def step(self, action):
        if self.perturb_gravity and self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav
                + self.np_random.normal(
                    0.0,
                    self.sigma_gravity,
                )
            )

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()

        reward, info = self.compute_reward(action)

        self._elapsed_steps += 1

        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )
        # cost = 0.0

        truncated = self._elapsed_steps >= self.max_steps
        terminated = self.compute_terminated(observation)

        info.update(
            {
                "x_position": float(observation[0]),
                "theta": float(observation[1]),
            }
        )

        return observation, reward, cost, truncated, terminated, info


class InvertedPendulumWithCost(InvertedPendulumWithCostBase):
    def __init__(
        self,
        sigma_gravity: float = 0.0,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            sigma_gravity=sigma_gravity,
            max_steps=max_steps,
            perturb_gravity=False,
            **kwargs
        )


class InvertedPendulumWithCostPerturbed(InvertedPendulumWithCostBase):
    def __init__(
        self,
        sigma_gravity: float = 0.7,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__(
            sigma_gravity=sigma_gravity,
            max_steps=max_steps,
            perturb_gravity=True,
            **kwargs
        )
