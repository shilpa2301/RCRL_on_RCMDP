import numpy as np
from gym.envs.mujoco import walker2d_v3
import gym
from gym.spaces import Box

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

VIOLATIONS_ALLOWED = 100

##############################################################################
REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                   Swimmer With Global Postion Coordinates                   #
# =========================================================================== #
ACTION_TORQUE_THRESHOLD = 0.5 #action ranges from -0.4 to 0.4


class Walker2dWithCostPerturbed(walker2d_v3.Walker2dEnv):
    OBS_DIM   = 17  # qpos(5) + qvel(5)
    max_steps = 500

    def __init__(self, sigma_gravity: float = 0.7, max_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.sigma_gravity = sigma_gravity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        
        # Perturb gravity
        self.model.opt.gravity[self._grav_axis] = self._base_grav
        return obs, {}
  
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        #shilpa 
        self._elapsed_steps += 1

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # cost = 0.0

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False
        # terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        # if self.render_mode == "human":
        #     self.render()

        return observation, reward, cost, truncated, terminated, info
    
class Walker2dWithCost(walker2d_v3.Walker2dEnv):
    OBS_DIM   = 17  # qpos(5) + qvel(5)
    max_steps = 500

    def __init__(self, sigma_gravity: float = 0.0, max_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.sigma_gravity = sigma_gravity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        
        # Perturb gravity
        self.model.opt.gravity[self._grav_axis] = self._base_grav
        return obs, {}
  
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        #shilpa 
        self._elapsed_steps += 1

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        # cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        cost = 0.0

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False
        # terminated = self.terminated
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
        }

        # if self.render_mode == "human":
        #     self.render()

        return observation, reward, cost, truncated, terminated, info