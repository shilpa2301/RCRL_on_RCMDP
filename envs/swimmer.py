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
        self._mujoco_initializing = True
        self.sigma_viscosity = sigma_viscosity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        # Store the base viscosity from the XML (0.1 for swimmer.xml)
        self._base_viscosity = 0.0 #float(self.model.opt.viscosity)

        super().__init__(**kwargs)
        self._mujoco_initializing = False
        # self.sigma_viscosity = sigma_viscosity
        # self.max_steps       = max_steps
        # self._elapsed_steps  = 0

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

        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
            return ob, reward, truncated or terminated, info


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

REWARD_TYPE = "old"
ACTION_TORQUE_THRESHOLD = 0.5


class SwimmerCMDP(swimmer.SwimmerEnv):
    """
    Swimmer CMDP with max-cost augmented observation.

    Observation:
        qpos + qvel + scaled max_cost

    Returns:
        observation, reward, scaled_cost, truncated, terminated, info
    """

    BASE_OBS_DIM = 10
    OBS_DIM = 11
    max_steps = 1000

    def __init__(
        self,
        max_steps: int = 1000,
        cost_scale: float = 100.0,
        obs_cost_scale: float = 100.0,
        **kwargs
    ):
        # Old Gym MuJoCo may call step during __init__.
        self._mujoco_initializing = True

        self._elapsed_steps = 0
        self.max_cost = 0.0
        self.last_cost = 0.0

        self.cost_scale = float(cost_scale)
        self.obs_cost_scale = float(obs_cost_scale)
        self.max_steps = int(max_steps)

        self._base_viscosity = 0.0

        super().__init__(**kwargs)

        self._base_viscosity = float(self.model.opt.viscosity)

        self._mujoco_initializing = False

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

    def _get_base_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _augment_obs(self, obs):
        return np.concatenate([
            np.asarray(obs, dtype=np.float32),
            np.array([self.obs_cost_scale * self.max_cost], dtype=np.float32),
        ]).astype(np.float32)

    def _get_obs(self):
        base_obs = self._get_base_obs()
        return self._augment_obs(base_obs)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self._elapsed_steps = 0
        self.model.opt.viscosity = self._base_viscosity

        self.max_cost = 0.0
        self.last_cost = 0.0

        obs = self._get_obs()

        return obs, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0.0

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]

        self._elapsed_steps += 1

        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        # Raw instantaneous cost.
        current_c = float(np.sum(np.square(action)))

        previous_max_cost = self.max_cost
        incremental_max_cost = max(current_c - previous_max_cost, 0.0)

        dense_cost = current_c
        beta = 0.01
        alpha = max((self._elapsed_steps - 1), 0) / self._elapsed_steps

        cost = beta * dense_cost + alpha * incremental_max_cost

        self.max_cost = float(max(previous_max_cost, current_c))
        self.last_cost = float(cost)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            "cost": cost,
            "scaled_cost": self.cost_scale * cost,
            "current_c": current_c,
            "previous_max_cost": previous_max_cost,
            "max_cost": self.max_cost,
            "incremental_max_cost": incremental_max_cost,
            "dense_cost": dense_cost,
            "action_l2_cost": current_c,
            "max_action_abs": float(np.max(np.abs(action))),
            "cost_scale": self.cost_scale,
            "obs_cost_scale": self.obs_cost_scale,
            "viscosity": float(self.model.opt.viscosity),
            "base_viscosity": float(self._base_viscosity),
        })

        # Windows / old Gym MuJoCo init guard.
        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, truncated or terminated, info

        return observation, reward, self.cost_scale * cost, truncated, terminated, info


class SwimmerCMDPPerturbed(swimmer.SwimmerEnv):
    """
    Swimmer CMDP with per-step random viscosity perturbation.

    Observation:
        qpos + qvel + scaled max_cost

    Returns:
        observation, reward, scaled_cost, truncated, terminated, info
    """

    BASE_OBS_DIM = 10
    OBS_DIM = 11
    max_steps = 1000

    def __init__(
        self,
        max_steps: int = 1000,
        cost_scale: float = 100.0,
        obs_cost_scale: float = 100.0,
        sigma_viscosity: float = 0.1,
        **kwargs
    ):
        self._mujoco_initializing = True

        self._elapsed_steps = 0
        self.max_cost = 0.0
        self.last_cost = 0.0

        self.cost_scale = float(cost_scale)
        self.obs_cost_scale = float(obs_cost_scale)

        self.sigma_viscosity = float(sigma_viscosity)
        self.max_steps = int(max_steps)

        self._base_viscosity = 0.0

        super().__init__(**kwargs)

        self._base_viscosity = float(self.model.opt.viscosity)

        self._mujoco_initializing = False

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

    def _get_base_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _augment_obs(self, obs):
        return np.concatenate([
            np.asarray(obs, dtype=np.float32),
            np.array([self.obs_cost_scale * self.max_cost], dtype=np.float32),
        ]).astype(np.float32)

    def _get_obs(self):
        base_obs = self._get_base_obs()
        return self._augment_obs(base_obs)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self._elapsed_steps = 0
        self.model.opt.viscosity = self._base_viscosity

        self.max_cost = 0.0
        self.last_cost = 0.0

        obs = self._get_obs()

        return obs, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0.0

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }

        return reward, info

    def step(self, action):
        if self.sigma_viscosity > 0.0:
            perturbed = self._base_viscosity + self.np_random.normal(
                0.0,
                self.sigma_viscosity,
            )
            self.model.opt.viscosity = max(0.0, float(perturbed))

        xposbefore = self.sim.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]

        self._elapsed_steps += 1

        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        current_c = float(np.sum(np.square(action)))

        previous_max_cost = self.max_cost
        incremental_max_cost = max(current_c - previous_max_cost, 0.0)

        dense_cost = current_c
        beta = 0.01
        alpha = max((self._elapsed_steps - 1), 0) / self._elapsed_steps

        cost = beta * dense_cost + alpha * incremental_max_cost

        self.max_cost = float(max(previous_max_cost, current_c))
        self.last_cost = float(cost)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            "cost": cost,
            "scaled_cost": self.cost_scale * cost,
            "current_c": current_c,
            "previous_max_cost": previous_max_cost,
            "max_cost": self.max_cost,
            "incremental_max_cost": incremental_max_cost,
            "dense_cost": dense_cost,
            "action_l2_cost": current_c,
            "max_action_abs": float(np.max(np.abs(action))),
            "cost_scale": self.cost_scale,
            "obs_cost_scale": self.obs_cost_scale,
            "viscosity": float(self.model.opt.viscosity),
            "base_viscosity": float(self._base_viscosity),
            "sigma_viscosity": float(self.sigma_viscosity),
        })

        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, truncated or terminated, info

        return observation, reward, self.cost_scale * cost, truncated, terminated, info


class SwimmerWithCostPerturbedTest(swimmer.SwimmerEnv):
    """
    Test-time perturbed Swimmer with cost.

    Samples one viscosity perturbation once during __init__ and keeps it fixed
    for all episodes and all steps.

    If shared_viscosity_perturbation is given, uses that exact perturbation.

    Observation:
        qpos + qvel

    Returns:
        observation, reward, cost, truncated, terminated, info
    """

    OBS_DIM = 10
    max_steps = 1000

    def __init__(
        self,
        sigma_viscosity: float = 0.0,
        max_steps: int = 1000,
        shared_viscosity_perturbation=None,
        seed=None,
        **kwargs
    ):
        self._mujoco_initializing = True

        self.sigma_viscosity = float(sigma_viscosity)
        self.max_steps = int(max_steps)
        self._elapsed_steps = 0

        self._base_viscosity = 0.0

        self.shared_viscosity_perturbation = shared_viscosity_perturbation
        self.fixed_viscosity_perturbation = 0.0
        self.fixed_viscosity_value = 0.0

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        super().__init__(**kwargs)

        self._base_viscosity = float(self.model.opt.viscosity)

        if self.shared_viscosity_perturbation is None:
            if self.sigma_viscosity > 0.0:
                self.fixed_viscosity_perturbation = float(
                    self.np_random.normal(0.0, self.sigma_viscosity)
                )
            else:
                self.fixed_viscosity_perturbation = 0.0
        else:
            self.fixed_viscosity_perturbation = float(
                self.shared_viscosity_perturbation
            )

        self.fixed_viscosity_value = max(
            0.0,
            self._base_viscosity + self.fixed_viscosity_perturbation,
        )

        self.model.opt.viscosity = self.fixed_viscosity_value

        self._mujoco_initializing = False

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        print(
            f"[SwimmerWithCostPerturbedTest] "
            f"base_viscosity={self._base_viscosity}, "
            f"fixed_perturbation={self.fixed_viscosity_perturbation}, "
            f"fixed_viscosity={self.fixed_viscosity_value}"
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self._elapsed_steps = 0

        # Keep same fixed viscosity every episode.
        self.model.opt.viscosity = self.fixed_viscosity_value

        obs = self._get_obs()

        return obs, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0.0

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }

        return reward, info

    def step(self, action):
        if getattr(self, "_mujoco_initializing", False):
            xposbefore = self.sim.data.qpos[0]

            self.do_simulation(action, self.frame_skip)

            xposafter = self.sim.data.qpos[0]

            observation = self._get_obs()

            if REWARD_TYPE == "new":
                reward, info = self.new_reward(xposbefore, xposafter, action)
            else:
                reward, info = self.old_reward(xposbefore, xposafter, action)

            done = False
            return observation, reward, done, info

        # Keep same fixed viscosity every step.
        self.model.opt.viscosity = self.fixed_viscosity_value

        xposbefore = self.sim.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]

        self._elapsed_steps += 1

        observation = self._get_obs()

        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        cost = float(np.sum(np.square(action)))

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            "action_l2_cost": cost,
            "max_action_abs": float(np.max(np.abs(action))),
            "viscosity": float(self.model.opt.viscosity),
            "base_viscosity": float(self._base_viscosity),
            "fixed_viscosity_perturbation": float(self.fixed_viscosity_perturbation),
            "sigma_viscosity": float(self.sigma_viscosity),
        })

        return observation, reward, cost, truncated, terminated, info


class SwimmerCMDPPerturbedTest(swimmer.SwimmerEnv):
    """
    Test-time perturbed Swimmer CMDP.

    Samples one viscosity perturbation once during __init__ and keeps it fixed
    for all episodes and all steps.

    If shared_viscosity_perturbation is given, uses that exact perturbation.

    Observation:
        qpos + qvel + scaled max_cost

    Returns:
        observation, reward, scaled_cost, truncated, terminated, info
    """

    BASE_OBS_DIM = 10
    OBS_DIM = 11
    max_steps = 1000

    def __init__(
        self,
        max_steps: int = 1000,
        cost_scale: float = 100.0,
        obs_cost_scale: float = 100.0,
        sigma_viscosity: float = 0.1,
        shared_viscosity_perturbation=None,
        seed=None,
        **kwargs
    ):
        self._mujoco_initializing = True

        self._elapsed_steps = 0
        self.max_cost = 0.0
        self.last_cost = 0.0

        self.cost_scale = float(cost_scale)
        self.obs_cost_scale = float(obs_cost_scale)

        self.sigma_viscosity = float(sigma_viscosity)
        self.max_steps = int(max_steps)

        self._base_viscosity = 0.0

        self.shared_viscosity_perturbation = shared_viscosity_perturbation
        self.fixed_viscosity_perturbation = 0.0
        self.fixed_viscosity_value = 0.0

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        super().__init__(**kwargs)

        self._base_viscosity = float(self.model.opt.viscosity)

        if self.shared_viscosity_perturbation is None:
            if self.sigma_viscosity > 0.0:
                self.fixed_viscosity_perturbation = float(
                    self.np_random.normal(0.0, self.sigma_viscosity)
                )
            else:
                self.fixed_viscosity_perturbation = 0.0
        else:
            self.fixed_viscosity_perturbation = float(
                self.shared_viscosity_perturbation
            )

        self.fixed_viscosity_value = max(
            0.0,
            self._base_viscosity + self.fixed_viscosity_perturbation,
        )

        self.model.opt.viscosity = self.fixed_viscosity_value

        self._mujoco_initializing = False

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        print(
            f"[SwimmerCMDPPerturbedTest] "
            f"base_viscosity={self._base_viscosity}, "
            f"fixed_perturbation={self.fixed_viscosity_perturbation}, "
            f"fixed_viscosity={self.fixed_viscosity_value}"
        )

    def _get_base_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32)

    def _augment_obs(self, obs):
        return np.concatenate([
            np.asarray(obs, dtype=np.float32),
            np.array([self.obs_cost_scale * self.max_cost], dtype=np.float32),
        ]).astype(np.float32)

    def _get_obs(self):
        base_obs = self._get_base_obs()
        return self._augment_obs(base_obs)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        out = super().reset(seed=seed, **kwargs)

        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self._elapsed_steps = 0
        self.max_cost = 0.0
        self.last_cost = 0.0

        # Keep same fixed viscosity every episode.
        self.model.opt.viscosity = self.fixed_viscosity_value

        obs = self._get_obs()

        return obs, info

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "xpos": xposafter,
        }

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -1e-4 * np.square(action).sum()
        reward_dist = abs(xposafter) - abs(xposbefore)
        reward_run = reward_dist / self.dt

        if np.sign(xposafter) == np.sign(xposbefore):
            reward = reward_ctrl + reward_run
        else:
            reward = 0.0

        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_dist": reward_dist,
            "xpos": xposafter,
        }

        return reward, info

    def step(self, action):
        if getattr(self, "_mujoco_initializing", False):
            xposbefore = self.sim.data.qpos[0]

            self.do_simulation(action, self.frame_skip)

            xposafter = self.sim.data.qpos[0]

            observation = self._get_base_obs()

            if REWARD_TYPE == "new":
                reward, info = self.new_reward(xposbefore, xposafter, action)
            else:
                reward, info = self.old_reward(xposbefore, xposafter, action)

            done = False
            return observation, reward, done, info

        # Keep same fixed viscosity every step.
        self.model.opt.viscosity = self.fixed_viscosity_value

        xposbefore = self.sim.data.qpos[0]

        self.do_simulation(action, self.frame_skip)

        xposafter = self.sim.data.qpos[0]

        self._elapsed_steps += 1

        if REWARD_TYPE == "new":
            reward, info = self.new_reward(xposbefore, xposafter, action)
        else:
            reward, info = self.old_reward(xposbefore, xposafter, action)

        current_c = float(np.sum(np.square(action)))

        previous_max_cost = self.max_cost
        incremental_max_cost = max(current_c - previous_max_cost, 0.0)

        dense_cost = current_c
        beta = 0.01
        alpha = max((self._elapsed_steps - 1), 0) / self._elapsed_steps

        cost = beta * dense_cost + alpha * incremental_max_cost

        self.max_cost = float(max(previous_max_cost, current_c))
        self.last_cost = float(cost)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info.update({
            "cost": cost,
            "scaled_cost": self.cost_scale * cost,
            "current_c": current_c,
            "previous_max_cost": previous_max_cost,
            "max_cost": self.max_cost,
            "incremental_max_cost": incremental_max_cost,
            "dense_cost": dense_cost,
            "action_l2_cost": current_c,
            "max_action_abs": float(np.max(np.abs(action))),
            "cost_scale": self.cost_scale,
            "obs_cost_scale": self.obs_cost_scale,
            "viscosity": float(self.model.opt.viscosity),
            "base_viscosity": float(self._base_viscosity),
            "fixed_viscosity_perturbation": float(self.fixed_viscosity_perturbation),
            "sigma_viscosity": float(self.sigma_viscosity),
        })

        return observation, reward, self.cost_scale * cost, truncated, terminated, info