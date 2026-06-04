import os

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import numpy as np



# ========================================================================== #
# CHEETAH WITH TORQUE CONSTRAINT
# ========================================================================== #

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100

class HalfCheetahTest(HalfCheetahEnv):
   def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

   def step(self, action):
        
        next_ob, reward, done, info = super().step(action)
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
        return next_ob, reward, done, info


# ========================================================================== #
# ========================================================================== #

REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

# =========================================================================== #
#                           Cheetah With Wall Infront                         #
# =========================================================================== #

class HalfCheetahWithObstacle(HalfCheetahEnv):
    """Variant of half-cheetah that includes an obstacle."""
    def __init__(self, xml_file=ABS_PATH+"/xmls/half_cheetah_obstacle.xml"):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
        self.observation_space = gym.spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                dtype=np.float32
        )

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
                reward_run=reward_run, reward_ctrl=reward_ctrl)

    def camera_setup(self):
        super(HalfCheetahDirectionEnv, self).camera_setup()
        self.camera._render_camera.distance = 5.0  # pylint: disable=protected-access

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

# =========================================================================== #
#            Cheetah With Equal Reward of Moving Forwards and Backwards       #
# =========================================================================== #

class HalfCheetahEqual(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
                reward_run=reward_run, reward_ctrl=reward_ctrl)

# =========================================================================== #
#                               Cheetah Backward                              #
# =========================================================================== #

class HalfCheetahBackward(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = -(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(
                reward_run=reward_run, reward_ctrl=reward_ctrl)

# =========================================================================== #
#                   Cheetah With Global Postion Coordinates                   #
# =========================================================================== #

class HalfCheetahWithPos(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""
      # ── FIX 1: actual obs size is 18 (qpos=9 + qvel=9) ──────────────────────
    OBS_DIM = 18
    # ── FIX 2: episode length (matches MuJoCo HalfCheetah default) ───────────
    max_steps = 1000
    def __init__(self):
        super().__init__()
        # Override observation_space to match the 18-dim obs we actually return
        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self._elapsed_steps = 0
    
    def reset(self, seed=None, **kwargs):
            # Handle seed for reproducibility
            # FIX: self.seed(seed) does not exist in old gym MuJoCo envs.
            # Directly set self.np_random using gym's seeding utility instead.
            if seed is not None:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            obs = super().reset()
            self._elapsed_steps = 0
            return obs, {}   # (obs, info) tuple expected by your training loop

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run  = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                xpos=xposafter
                )

        return reward, info



    def step(self, action):
        # Perturb gravity each step (or each episode in reset)
        # self.model.opt.gravity[2] = -9.81 + np.random.normal(0, 0.05)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter  = self.sim.data.qpos[0]
        ob         = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        self._elapsed_steps += 1

       
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))

        # ── FIX 4: truncation / termination flags ────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False                  # HalfCheetah never "dies"


    
        # cost = 0
        # Return 6-tuple: obs, reward, cost, truncated, terminated, info
        return ob, reward, cost, truncated, terminated, info
    

class HalfCheetahWithPosPerturbed(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""
      # ── FIX 1: actual obs size is 18 (qpos=9 + qvel=9) ──────────────────────
    OBS_DIM = 18
    # ── FIX 2: episode length (matches MuJoCo HalfCheetah default) ───────────
    max_steps = 1000
    def __init__(self):
        super().__init__()
        # Override observation_space to match the 18-dim obs we actually return
        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self._elapsed_steps = 0
    
    def reset(self, seed=None, **kwargs):
            # Handle seed for reproducibility
            # FIX: self.seed(seed) does not exist in old gym MuJoCo envs.
            # Directly set self.np_random using gym's seeding utility instead.
            if seed is not None:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            obs = super().reset()
            self._elapsed_steps = 0
            return obs, {}   # (obs, info) tuple expected by your training loop

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                xpos=xposafter
                )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run  = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                xpos=xposafter
                )

        return reward, info



    def step(self, action):
        # Perturb gravity each step (or each episode in reset)
        self.model.opt.gravity[2] = -9.81 + np.random.normal(0, 0.05)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter  = self.sim.data.qpos[0]
        ob         = self._get_obs()

        if REWARD_TYPE == 'new':
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        self._elapsed_steps += 1

       
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))

        # ── FIX 4: truncation / termination flags ────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False                  # HalfCheetah never "dies"


    
        # cost = 0
        # Return 6-tuple: obs, reward, cost, truncated, terminated, info
        return ob, reward, cost, truncated, terminated, info




# class HalfCheetahWithPosTest(HalfCheetahWithPos):
#     """Environment to test the agent trained in CheetahWithPos using
#        constraints."""

#     # def step(self, action):
#     #     xposbefore = self.sim.data.qpos[0]
#     #     self.do_simulation(action, self.frame_skip)
#     #     xposafter = self.sim.data.qpos[0]
#     #     ob = self._get_obs()
#     #     if REWARD_TYPE == 'new':
#     #         reward, info = self.new_reward(xposbefore,
#     #                                        xposafter,
#     #                                        action)
#     #     elif REWARD_TYPE == 'old':
#     #         reward, info = self.old_reward(xposbefore,
#     #                                        xposafter,
#     #                                        action)
#     #     done = False

#     #     # If agent violates constraints, terminate the episode
#     #     if xposafter <= -3:
#     #         print("Violated constraint in the test environment, terminating the episode.", flush=True)
#     #         done = True
#     #         reward = 0

#     #     return ob, reward, done, info

#     def step(self, action):
#         # Perturb gravity each step (or each episode in reset)
#         self.model.opt.gravity[2] = -9.81 + np.random.normal(0, 0.05)
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter  = self.sim.data.qpos[0]
#         ob         = self._get_obs()

#         if REWARD_TYPE == 'new':
#             reward, info = self.new_reward(xposbefore, xposafter, action)
#         else:
#             reward, info = self.old_reward(xposbefore, xposafter, action)

#         self._elapsed_steps += 1
#         # cost       = float(np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD))
#         cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
#         truncated  = self._elapsed_steps >= self.max_steps
#         terminated = False

#         # If agent violates position constraint, terminate the episode
#         if xposafter <= -3:
#             print("Violated constraint in test env, terminating.", flush=True)
#             terminated = True
#             reward     = 0

#         return ob, reward, cost, truncated, terminated, info
