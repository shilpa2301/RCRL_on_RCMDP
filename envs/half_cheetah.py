from cmath import tau
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
        self.last_cost = 0.0
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
            np.array([100.0*self.max_cost], dtype=np.float32),
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
        self.last_cost = 0.0

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

        qvel = self.init_qvel + self.np_random.normal(self.model.nv) * 0.1

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
        # previous_max_cost = self.max_cost
        # raw_cost = np.maximum(float(current_c - previous_max_cost), 0.0)
        # tau = 5e-2  # choose your temperature
        # cost = float(tau * np.log1p(np.exp(raw_cost / tau)))
        # # Update max_cost after computing returned cost.
        # self.max_cost = float(max(previous_max_cost, current_c))

        #working version
        previous_max_cost = self.max_cost
        incremental_max_cost = max(current_c - previous_max_cost, 0.0)
        dense_cost = current_c
        beta = 0.01 #0.1
        alpha = max((self._elapsed_steps-1), 0)/(self._elapsed_steps) #-0.99)#1.0
        # print(dense_cost, incremental_max_cost, alpha, beta)
        cost = beta * dense_cost + alpha * incremental_max_cost
        self.max_cost = float(max(previous_max_cost, current_c))
        self.last_cost = cost
        # print("cost=",cost)


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
            "incremental_max_cost": incremental_max_cost,
        })

        # Return 6-tuple expected by your training loop:
        # obs, reward, cost, truncated, terminated, info
        return ob, reward, 100.0*cost, truncated, terminated, info




class HalfCheetahForwardObstacleCMDP(HalfCheetahEnv):
    """
    HalfCheetah CMDP with one forward obstacle/state constraint.

    Constraint:
        Hard unsafe obstacle region:
            x in [obstacle_x_min, obstacle_x_max]
            default: [2.0, 4.0]

        Dense warning region:
            x in [warning_x_min, obstacle_x_max]
            default: [1.0, 4.0]

    Reward:
        Forward velocity reward:
            reward_run = (xposafter - xposbefore) / dt

        This encourages moving forward, not backward.

    Observation:
        [qpos, qvel, max_cost]
        qpos = 9
        qvel = 9
        max_cost = 1
        total = 19

    Cost:
        current_c:
            hard obstacle violation.

        dense_cost:
            dense proxy cost that starts before the obstacle.

        incremental_max_cost:
            max(current_c - previous_max_cost, 0)

        returned cost:
            cost = beta * dense_cost + alpha * incremental_max_cost

    Step return:
        obs, reward, cost, truncated, terminated, info

    This matches your custom 6-tuple training loop.
    """

    OBS_DIM = 18 #19

    def __init__(
        self,
        obstacle_x_min=2.0,
        obstacle_x_max=4.0,
        warning_x_min=1.0,
        max_steps=1000,
        beta=1.0,
        squared_dense=False,
    ):
        # Needed before super because reset_model can call _get_obs
        self._elapsed_steps = 0

        self.obstacle_x_min = float(obstacle_x_min)
        self.obstacle_x_max = float(obstacle_x_max)
        self.warning_x_min = float(warning_x_min)

        self.max_steps = int(max_steps)


        self.beta = float(beta)
        self.squared_dense = bool(squared_dense)

        if self.warning_x_min > self.obstacle_x_min:
            raise ValueError("warning_x_min must be <= obstacle_x_min.")

        if self.obstacle_x_min >= self.obstacle_x_max:
            raise ValueError("obstacle_x_min must be < obstacle_x_max.")

        super().__init__()

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_base_obs(self):
        """
        Full HalfCheetah state:
            qpos = 9
            qvel = 9
            total = 18
        """
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _get_obs(self):
        """
        Augmented observation:
            [qpos, qvel, max_cost]
        """
        base_obs = self._get_base_obs()
        # print(np.array(base_obs.shape))

        # return np.concatenate([
        #     base_obs,
        #     np.array([self.max_cost], dtype=np.float32),
        # ]).astype(np.float32)
        return base_obs

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, **kwargs):
        """
        Old Gym compatibility, but returns:
            obs, info
        because your training loop expects that.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        super().reset()

        self._elapsed_steps = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )

        qvel = self.init_qvel + self.np_random.normal(
            loc=0.0,
            scale=0.1,
            size=self.model.nv,
        )

        self.set_state(qpos, qvel)

        return self._get_obs()

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _forward_reward(self, xposbefore, xposafter, action):
        """
        Forward-only reward.

        Unlike your old reward:
            abs(xposafter - xposbefore) / dt

        this rewards only forward movement:
            (xposafter - xposbefore) / dt
        """
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt

        reward = reward_run + reward_ctrl

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return float(reward), info

    # ------------------------------------------------------------------ #
    # Constraint
    # ------------------------------------------------------------------ #

    def _hard_obstacle_violation(self, x):
        """
        Hard unsafe-state violation for obstacle region:
            x in [obstacle_x_min, obstacle_x_max]

        Returns:
            0 outside obstacle.
            Positive inside obstacle.

        The cost is highest near the center of the obstacle.
        """
        x = float(x)

        if not (self.obstacle_x_min <= x <= self.obstacle_x_max):
            return 0.0

        center = 0.5 * (self.obstacle_x_min + self.obstacle_x_max)
        half_width = 0.5 * (self.obstacle_x_max - self.obstacle_x_min)

        # Ranges from 0 at edges to 1 at center.
        violation = 1.0 - abs(x - center) / max(half_width, 1e-6)

        return float(max(violation, 0.0))

    def _dense_obstacle_cost(self, x):
        """
        Dense proxy cost.

        Regions:
            x < warning_x_min:
                cost = 0

            warning_x_min <= x < obstacle_x_min:
                warning cost ramps from 0 to 1

            obstacle_x_min <= x <= obstacle_x_max:
                unsafe cost is >= 1 and highest near center

            x > obstacle_x_max:
                cost = 0

        With defaults:
            warning region: [1, 2)
            obstacle region: [2, 4]
            total dense-cost-active region: [1, 4]
        """
        x = float(x)

        # Region 1: before warning starts
        if x < self.warning_x_min:
            return 0.0

        # Region 2: warning ramp, [warning_x_min, obstacle_x_min)
        if self.warning_x_min <= x < self.obstacle_x_min:
            denom = max(self.obstacle_x_min - self.warning_x_min, 1e-6)
            dense = (x - self.warning_x_min) / denom

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        # Region 3: inside obstacle, [obstacle_x_min, obstacle_x_max]
        if self.obstacle_x_min <= x <= self.obstacle_x_max:
            hard = self._hard_obstacle_violation(x)

            # At edges: 1
            # At center: 2
            dense = 1.0 + hard

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        # Region 4: after obstacle
        return 0.0

    def _compute_constraint_cost(self, x):
        """
        Computes:
            current_c:
                hard obstacle violation.

            dense_cost:
                proxy dense cost.

            incremental_max_cost:
                increase in trajectory max violation.

            returned cost:
                beta * dense_cost + alpha * incremental_max_cost
        """

        dense_cost = self._dense_obstacle_cost(x)

        cost = float(self.beta * dense_cost)
        tau = 5e-2  # choose your temperature
        cost = 1.0 *float(tau * np.log1p(np.exp(cost / tau)))


        info = {
            "cost": cost,
            "dense_cost": dense_cost,
            # "max_cost": self.max_cost,
            "obstacle_x_min": self.obstacle_x_min,
            "obstacle_x_max": self.obstacle_x_max,
            "warning_x_min": self.warning_x_min,
            "beta": self.beta,
        }

        return cost, info

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action):
        xposbefore = float(self.sim.data.qpos[0])

        self.do_simulation(action, self.frame_skip)

        xposafter = float(self.sim.data.qpos[0])

        reward, reward_info = self._forward_reward(
            xposbefore,
            xposafter,
            action,
        )

        self._elapsed_steps += 1

        cost, constraint_info = self._compute_constraint_cost(xposafter)

        obs = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps

        # HalfCheetah usually has no terminal death condition.
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)
        info.update({
            "xposbefore": xposbefore,
            "xposafter": xposafter,
            "in_warning_region": bool(
                self.warning_x_min <= xposafter < self.obstacle_x_min
            ),
            "in_obstacle_region": bool(
                self.obstacle_x_min <= xposafter <= self.obstacle_x_max
            ),
        })

        # Your custom 6-tuple format:
        # obs, reward, cost, truncated, terminated, info
        return obs, reward, cost, truncated, terminated, info


#Sparse cost perturbed
class HalfCheetahForwardObstaclePerturbed(HalfCheetahEnv):
    """
    HalfCheetah CMDP with one forward obstacle/state constraint.

    Constraint:
        Hard unsafe obstacle region:
            x in [obstacle_x_min, obstacle_x_max]
            default: [2.0, 4.0]

        Dense warning region:
            x in [warning_x_min, obstacle_x_max]
            default: [1.0, 4.0]

    Reward:
        Forward velocity reward:
            reward_run = (xposafter - xposbefore) / dt

        This encourages moving forward, not backward.

    Observation:
        [qpos, qvel, max_cost]
        qpos = 9
        qvel = 9
        max_cost = 1
        total = 19

    Cost:
        current_c:
            hard obstacle violation.

        dense_cost:
            dense proxy cost that starts before the obstacle.

        incremental_max_cost:
            max(current_c - previous_max_cost, 0)

        returned cost:
            cost = beta * dense_cost + alpha * incremental_max_cost

    Step return:
        obs, reward, cost, truncated, terminated, info

    This matches your custom 6-tuple training loop.
    """

    OBS_DIM = 18 #19

    def __init__(
        self,
        obstacle_x_min=2.0,
        obstacle_x_max=4.0,
        warning_x_min=1.0,
        max_steps=1000,
        beta=1.0,
        squared_dense=False,
        sigma_gravity=0.7,
    ):
        # Needed before super because reset_model can call _get_obs
        self._elapsed_steps = 0

        self.obstacle_x_min = float(obstacle_x_min)
        self.obstacle_x_max = float(obstacle_x_max)
        self.warning_x_min = float(warning_x_min)

        self.max_steps = int(max_steps)


        self.beta = float(beta)
        self.squared_dense = bool(squared_dense)
        self.sigma_gravity = float(sigma_gravity)
        if self.warning_x_min > self.obstacle_x_min:
            raise ValueError("warning_x_min must be <= obstacle_x_min.")

        if self.obstacle_x_min >= self.obstacle_x_max:
            raise ValueError("obstacle_x_min must be < obstacle_x_max.")

        super().__init__()

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
            
    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_base_obs(self):
        """
        Full HalfCheetah state:
            qpos = 9
            qvel = 9
            total = 18
        """
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _get_obs(self):
        """
        Augmented observation:
            [qpos, qvel, max_cost]
        """
        base_obs = self._get_base_obs()
        # print(np.array(base_obs.shape))

        # return np.concatenate([
        #     base_obs,
        #     np.array([self.max_cost], dtype=np.float32),
        # ]).astype(np.float32)
        return base_obs

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, **kwargs):
        """
        Old Gym compatibility, but returns:
            obs, info
        because your training loop expects that.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        super().reset()

        self._elapsed_steps = 0


        obs = self._get_obs()
        self.model.opt.gravity[self._grav_axis] = self._base_grav

        info = {}

        return obs, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )

        qvel = self.init_qvel + self.np_random.normal(
            loc=0.0,
            scale=0.1,
            size=self.model.nv,
        )

        self.set_state(qpos, qvel)

        return self._get_obs()

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _forward_reward(self, xposbefore, xposafter, action):
        """
        Forward-only reward.

        Unlike your old reward:
            abs(xposafter - xposbefore) / dt

        this rewards only forward movement:
            (xposafter - xposbefore) / dt
        """
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt

        reward = reward_run + reward_ctrl

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return float(reward), info

    # ------------------------------------------------------------------ #
    # Constraint
    # ------------------------------------------------------------------ #

    def _hard_obstacle_violation(self, x):
        """
        Hard unsafe-state violation for obstacle region:
            x in [obstacle_x_min, obstacle_x_max]

        Returns:
            0 outside obstacle.
            Positive inside obstacle.

        The cost is highest near the center of the obstacle.
        """
        x = float(x)

        if not (self.obstacle_x_min <= x <= self.obstacle_x_max):
            return 0.0

        center = 0.5 * (self.obstacle_x_min + self.obstacle_x_max)
        half_width = 0.5 * (self.obstacle_x_max - self.obstacle_x_min)

        # Ranges from 0 at edges to 1 at center.
        violation = 1.0 - abs(x - center) / max(half_width, 1e-6)

        return float(max(violation, 0.0))

    def _dense_obstacle_cost(self, x):
        """
        Dense proxy cost.

        Regions:
            x < warning_x_min:
                cost = 0

            warning_x_min <= x < obstacle_x_min:
                warning cost ramps from 0 to 1

            obstacle_x_min <= x <= obstacle_x_max:
                unsafe cost is >= 1 and highest near center

            x > obstacle_x_max:
                cost = 0

        With defaults:
            warning region: [1, 2)
            obstacle region: [2, 4]
            total dense-cost-active region: [1, 4]
        """
        x = float(x)

        # Region 1: before warning starts
        if x < self.warning_x_min:
            return 0.0

        # Region 2: warning ramp, [warning_x_min, obstacle_x_min)
        if self.warning_x_min <= x < self.obstacle_x_min:
            denom = max(self.obstacle_x_min - self.warning_x_min, 1e-6)
            dense = (x - self.warning_x_min) / denom

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        # Region 3: inside obstacle, [obstacle_x_min, obstacle_x_max]
        if self.obstacle_x_min <= x <= self.obstacle_x_max:
            hard = self._hard_obstacle_violation(x)

            # At edges: 1
            # At center: 2
            dense = 1.0 + hard

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        # Region 4: after obstacle
        return 0.0

    def _compute_constraint_cost(self, x):
        """
        Computes:
            current_c:
                hard obstacle violation.

            dense_cost:
                proxy dense cost.

            incremental_max_cost:
                increase in trajectory max violation.

            returned cost:
                beta * dense_cost + alpha * incremental_max_cost
        """

        dense_cost = self._dense_obstacle_cost(x)

        cost = float(self.beta * dense_cost)
        tau = 5e-2  # choose your temperature
        cost = 1.0 *float(tau * np.log1p(np.exp(cost / tau)))


        info = {
            "cost": cost,
            "dense_cost": dense_cost,
            # "max_cost": self.max_cost,
            "obstacle_x_min": self.obstacle_x_min,
            "obstacle_x_max": self.obstacle_x_max,
            "warning_x_min": self.warning_x_min,
            "beta": self.beta,
        }

        return cost, info

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action):
        # Perturb gravity each step (or each episode in reset)
        # self.model.opt.gravity[2] = -9.81 + np.random.normal(0, 2.0)
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                        self._base_grav + np.random.normal(0.0, self.sigma_gravity)
                    )
        xposbefore = float(self.sim.data.qpos[0])

        self.do_simulation(action, self.frame_skip)

        xposafter = float(self.sim.data.qpos[0])

        reward, reward_info = self._forward_reward(
            xposbefore,
            xposafter,
            action,
        )

        self._elapsed_steps += 1

        cost, constraint_info = self._compute_constraint_cost(xposafter)

        obs = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps

        # HalfCheetah usually has no terminal death condition.
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)
        info.update({
            "xposbefore": xposbefore,
            "xposafter": xposafter,
            "in_warning_region": bool(
                self.warning_x_min <= xposafter < self.obstacle_x_min
            ),
            "in_obstacle_region": bool(
                self.obstacle_x_min <= xposafter <= self.obstacle_x_max
            ),
        })

        # Your custom 6-tuple format:
        # obs, reward, cost, truncated, terminated, info
        return obs, reward, cost, truncated, terminated, info

