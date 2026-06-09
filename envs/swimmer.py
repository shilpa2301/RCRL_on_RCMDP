import numpy as np
from gym.envs.mujoco import swimmer
import gym
from gym.spaces import Box

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class SwimmerTest(swimmer.SwimmerEnv):
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


##############################################################################
REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                   Swimmer With Global Postion Coordinates                   #
# =========================================================================== #
ACTION_TORQUE_THRESHOLD = 0.5
class SwimmerWithPos(swimmer.SwimmerEnv):
    OBS_DIM   = 10  # qpos(5) + qvel(5)
    max_steps = 1000

    def __init__(self, sigma_viscosity: float = 0.0, max_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.sigma_viscosity = sigma_viscosity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        # Store the base viscosity from the XML (0.1 for swimmer.xml)
        self._base_viscosity = float(self.model.opt.viscosity)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        self.model.opt.viscosity = self._base_viscosity
        return obs, {}

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info


    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run  = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                xpos=xposafter
                )

        return reward, info


    def step(self, action):
        # ── 1. Optional viscosity perturbation ─────────────────────────────
        # if self.sigma_viscosity > 0.0:
        #     perturbed = self._base_viscosity + self.np_random.normal(0.0, self.sigma_viscosity)
        #     self.model.opt.viscosity = max(0.0, perturbed)  # clamp non-negative
        
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        self._elapsed_steps += 1
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        # done = False
        # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False

        return ob, reward, cost, truncated, terminated, info

# class SwimmerWithPosTest(SwimmerWithPos):
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat,
#             self.sim.data.qvel.flat,
#         ])

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]
#         ob = self._get_obs()
#         if REWARD_TYPE == 'new':
#             reward, info = self.new_reward(xposbefore,
#                                            xposafter,
#                                            action)
#         elif REWARD_TYPE == 'old':
#             reward, info = self.old_reward(xposbefore,
#                                            xposafter,
#                                            action)
#         done = False

#         # If agent violates constraint, terminate the episode
#         if xposafter <= -3:
#             print("Violated constraint in the test environment; terminating episode")
#             done = True
#             reward = 0

#         return ob, reward, done, info


class SwimmerWithPosPerturbed(swimmer.SwimmerEnv):
    OBS_DIM   = 10  # qpos(5) + qvel(5)
    max_steps = 1000

    def __init__(self, sigma_viscosity: float = 0.0, max_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.sigma_viscosity = sigma_viscosity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        # Store the base viscosity from the XML (0.1 for swimmer.xml)
        self._base_viscosity = float(self.model.opt.viscosity)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        self.model.opt.viscosity = self._base_viscosity
        return obs, {}

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info


    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run  = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                xpos=xposafter
                )

        return reward, info


    def step(self, action):
        # ── 1. Optional viscosity perturbation ─────────────────────────────
        if self.sigma_viscosity > 0.0:
            perturbed = self._base_viscosity + self.np_random.normal(0.0, self.sigma_viscosity)
            self.model.opt.viscosity = max(0.0, perturbed)  # clamp non-negative
        
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        self._elapsed_steps += 1
        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif REWARD_TYPE == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        # done = False
        # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False

        return ob, reward, cost, truncated, terminated, info

# class SwimmerWithPosTest(SwimmerWithPos):
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat,
#             self.sim.data.qvel.flat,
#         ])

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]
#         ob = self._get_obs()
#         if REWARD_TYPE == 'new':
#             reward, info = self.new_reward(xposbefore,
#                                            xposafter,
#                                            action)
#         elif REWARD_TYPE == 'old':
#             reward, info = self.old_reward(xposbefore,
#                                            xposafter,
#                                            action)
#         done = False

#         # If agent violates constraint, terminate the episode
#         if xposafter <= -3:
#             print("Violated constraint in the test environment; terminating episode")
#             done = True
#             reward = 0

#         return ob, reward, done, info


