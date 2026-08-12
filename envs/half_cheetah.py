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
    def __init__(self, sigma_gravity: float = 0.7, max_steps: int = 1000):
        super().__init__()
        # Override observation_space to match the 18-dim obs we actually return
        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self._elapsed_steps = 0
        self.sigma_gravity = sigma_gravity
        self.max_steps       = max_steps
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
    
    def reset(self, seed=None, **kwargs):
            # Handle seed for reproducibility
            # FIX: self.seed(seed) does not exist in old gym MuJoCo envs.
            # Directly set self.np_random using gym's seeding utility instead.
            if seed is not None:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            obs = super().reset()
            self._elapsed_steps = 0
            self.model.opt.gravity[self._grav_axis] = self._base_grav
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
        # self.model.opt.gravity[2] = -9.81 + np.random.normal(0, 2.0)
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
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

ACTION_TORQUE_THRESHOLD = 0.5
REWARD_TYPE = "old"   # or "new"


class HalfCheetahCMDP(HalfCheetahEnv):
    """
    HalfCheetah CMDP with:
      1. Observation includes full qpos and qvel.
      2. Observation is augmented with max_cost observed so far.
      3. Returned cost is:
             cost_t = c_t - max_cost_{t-1}
         where:
             c_t = max(max(abs(action)) - ACTION_TORQUE_THRESHOLD, 0)
      4. max_cost is reset to 0 at episode reset.
    """

    # Original custom observation:
    # qpos = 9, qvel = 9, total = 18.
    # Augmented with max_cost, so total = 19.
    OBS_DIM = 19

    max_steps = 1000
    

    def __init__(self):
        self._elapsed_steps = 0
        self.max_cost = 0.0
        super().__init__()

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )


    def _get_base_obs(self):
        """
        Original 18-dimensional observation:
            [qpos, qvel]
        """
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _augment_obs(self, obs):
        """
        Append max_cost observed so far:
            obs_aug = [obs, max_cost]
        """
        return np.concatenate([
            np.asarray(obs, dtype=np.float32),
            np.array([self.max_cost], dtype=np.float32),
        ]).astype(np.float32)

    def _get_obs(self):
        """
        Return augmented 19-dimensional observation.
        """
        base_obs = self._get_base_obs()
        return self._augment_obs(base_obs)

    def reset(self, seed=None, **kwargs):
        """
        Reset environment.

        Important:
            max_cost is reset to 0.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # For old Gym MuJoCo envs, super().reset() calls reset_model().
        obs = super().reset()

        self._elapsed_steps = 0

        # 1. Reset max_cost.
        self.max_cost = 0.0

        # Because reset_model() may call _get_obs(), make sure final obs
        # uses max_cost = 0.
        obs = self._get_obs()

        return obs, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )

        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        return self._get_obs()

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            xpos=xposafter,
        )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run = reward_dist / self.dt

        reward = reward_dist + reward_ctrl

        info = dict(
            reward_run=reward_run,
            reward_ctrl=reward_ctrl,
            reward_dist=reward_dist,
            xpos=xposafter,
        )

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]

        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        self._elapsed_steps += 1

        # ============================================================
        # Raw instantaneous constraint value c_t
        # ============================================================
        current_c = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        # ============================================================
        # Returned CMDP cost:
        #
        #     cost_t = c_t - max_cost_{t-1}
        #
        # Use previous max_cost before updating it.
        # ============================================================
        previous_max_cost = self.max_cost

        cost = float(current_c - previous_max_cost)

        # Update max_cost after computing returned cost.
        self.max_cost = float(max(previous_max_cost, current_c))

        # State augmented with max_cost observed up to this time step.
        ob = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            # Returned transformed CMDP cost.
            "cost": cost,

            # Raw instantaneous cost c_t.
            "current_c": current_c,

            # max_cost before this step.
            "previous_max_cost": previous_max_cost,

            # max_cost after this step.
            "max_cost": self.max_cost,

            "max_action_abs": float(np.max(np.abs(action))),
            "action_torque_threshold": ACTION_TORQUE_THRESHOLD,
        })

        # Return 6-tuple expected by your training loop:
        # obs, reward, cost, truncated, terminated, info
        return ob, reward, cost, truncated, terminated, info

