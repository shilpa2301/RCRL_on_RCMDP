import numpy as np
from gym.envs.mujoco import humanoid_v3
import gym
from gym.spaces import Box

###############################################################################
# TORQUE CONSTRAINTS
###############################################################################

# ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100

##############################################################################
REWARD_TYPE = 'old'         # Which reward to use, traditional or new one?

# =========================================================================== #
#                   Swimmer With Global Postion Coordinates                   #
# =========================================================================== #
ACTION_TORQUE_THRESHOLD = 0.1 #action ranges from -0.4 to 0.4

def mass_center(model, sim):
        mass = np.expand_dims(model.body_mass, axis=1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class HumanoidWithCost(humanoid_v3.HumanoidEnv):
    OBS_DIM   = 348  # qpos(5) + qvel(5)
    max_steps = 1000

    # def __init__(self, floor_friction: float = 0.0, max_steps: int = 1000, **kwargs):
    def __init__(self, sigma_gravity: float = 0.0, max_steps: int = 1000, **kwargs):
    
        super().__init__(**kwargs)
        self.sigma_gravity = sigma_gravity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        # Store the base friction from the XML
        # self.floor_geom_id = self.model.geom_name2id("floor")  # Get the ID of the floor geom
        # self._base_friction = np.copy(self.model.geom_friction[self.floor_geom_id])
    
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
    
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        
        # Perturb floor friction
        # if self.floor_friction !=0.0:
        #     perturbation = self.np_random.uniform(-self.floor_friction, self.floor_friction)  # Add small random perturbation
        #     new_friction = self._base_friction[0] + perturbation # np.clip(self._base_friction[0] + perturbation, 0.0, 1.0)  # Keep friction in valid range
        #     self.model.geom_friction[self.floor_geom_id][0] = new_friction  # Update sliding friction
        #     self.model.geom_friction[self.floor_geom_id][1] = self._base_friction[1]  # Keep torsional friction same
        #     self.model.geom_friction[self.floor_geom_id][2] = self._base_friction[2]  # Keep rolling friction same

        self.model.opt.gravity[self._grav_axis] = self._base_grav

        return obs, {}
  
    
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        # done = False
        # # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        # cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # # cost = 0.0
        # Cost is the sum of squared actions (energy consumption)
        cost = float(np.sum(np.square(action)))

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False
        # terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, cost, truncated, terminated, info
    

class HumanoidWithCostPerturbed(humanoid_v3.HumanoidEnv):
    OBS_DIM   = 348  # qpos(5) + qvel(5)
    max_steps = 1000

    # def __init__(self, floor_friction: float = 0.0, max_steps: int = 1000, **kwargs):
    def __init__(self, sigma_gravity: float = 0.0, max_steps: int = 1000, **kwargs):
    
        super().__init__(**kwargs)
        self.sigma_gravity = sigma_gravity
        self.max_steps       = max_steps
        self._elapsed_steps  = 0

        # Store the base friction from the XML
        # self.floor_geom_id = self.model.geom_name2id("floor")  # Get the ID of the floor geom
        # self._base_friction = np.copy(self.model.geom_friction[self.floor_geom_id])
    
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
    
    
    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        
        # Perturb floor friction
        # if self.floor_friction !=0.0:
        #     perturbation = self.np_random.uniform(-self.floor_friction, self.floor_friction)  # Add small random perturbation
        #     new_friction = self._base_friction[0] + perturbation # np.clip(self._base_friction[0] + perturbation, 0.0, 1.0)  # Keep friction in valid range
        #     self.model.geom_friction[self.floor_geom_id][0] = new_friction  # Update sliding friction
        #     self.model.geom_friction[self.floor_geom_id][1] = self._base_friction[1]  # Keep torsional friction same
        #     self.model.geom_friction[self.floor_geom_id][2] = self._base_friction[2]  # Keep rolling friction same

        self.model.opt.gravity[self._grav_axis] = self._base_grav
        # print(f"Resetting gravity to {self._base_grav}")
        return obs, {}
  
    
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        # print(f"step Perturbing gravity to {self.model.opt.gravity[self._grav_axis]}")
        xy_position_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.sim)

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        observation = self._get_obs()
        reward = rewards - costs
        # done = False
        # # ── 3. Cost — continuous excess-torque penalty ──────────────────────
        # cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # # cost = 0.0
        # Cost is the sum of squared actions (energy consumption)
        cost = float(np.sum(np.square(action)))

        # ── 4. Termination / truncation ─────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False
        # terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, cost, truncated, terminated, info

####################################  SPARSE COST #############################################
class HumanoidForwardObstacleCMDP(humanoid_v3.HumanoidEnv):
    """
    Humanoid CMDP with one forward obstacle/state constraint.
    """

    OBS_DIM = 376
    max_steps = 1000

    def __init__(
        self,
        obstacle_x_min=2.0,
        obstacle_x_max=4.0,
        warning_x_min=1.0,
        max_steps=1000,
        beta=1.0,
        squared_dense=False,
        **kwargs,
    ):
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

        super().__init__(**kwargs)

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        try:
            obs = super().reset(seed=seed, **kwargs)
        except TypeError:
            obs = super().reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        self._elapsed_steps = 0

        # obs = self._get_obs()

        return obs, {}

    def _humanoid_forward_reward(self, xy_position_before, xy_position_after, action):
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return float(reward), info

    def _hard_obstacle_violation(self, x):
        x = float(x)

        if not (self.obstacle_x_min <= x <= self.obstacle_x_max):
            return 0.0

        center = 0.5 * (self.obstacle_x_min + self.obstacle_x_max)
        half_width = 0.5 * (self.obstacle_x_max - self.obstacle_x_min)

        violation = 1.0 - abs(x - center) / max(half_width, 1e-6)

        return float(max(violation, 0.0))

    def _dense_obstacle_cost(self, x):
        x = float(x)

        if x < self.warning_x_min:
            return 0.0

        if self.warning_x_min <= x < self.obstacle_x_min:
            denom = max(self.obstacle_x_min - self.warning_x_min, 1e-6)
            dense = (x - self.warning_x_min) / denom

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        if self.obstacle_x_min <= x <= self.obstacle_x_max:
            hard = self._hard_obstacle_violation(x)

            dense = 1.0 + hard

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        return 0.0

    def _compute_constraint_cost(self, x):
        dense_cost = self._dense_obstacle_cost(x)

        raw_cost = float(self.beta * dense_cost)

        tau = 5e-2
        cost = float(tau * np.log1p(np.exp(raw_cost / tau)))

        info = {
            "cost": cost,
            "dense_cost": dense_cost,
            "obstacle_x_min": self.obstacle_x_min,
            "obstacle_x_max": self.obstacle_x_max,
            "warning_x_min": self.warning_x_min,
            "beta": self.beta,
        }

        return cost, info

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)

        self.do_simulation(action, self.frame_skip)

        xy_position_after = mass_center(self.model, self.sim)

        self._elapsed_steps += 1

        reward, reward_info = self._humanoid_forward_reward(
            xy_position_before,
            xy_position_after,
            action,
        )

        xposafter = float(xy_position_after[0])

        cost, constraint_info = self._compute_constraint_cost(xposafter)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)
        info.update({
            "xposafter": xposafter,
            "in_warning_region": bool(
                self.warning_x_min <= xposafter < self.obstacle_x_min
            ),
            "in_obstacle_region": bool(
                self.obstacle_x_min <= xposafter <= self.obstacle_x_max
            ),
        })

        return observation, reward, cost, truncated, terminated, info

class HumanoidForwardObstaclePerturbed(humanoid_v3.HumanoidEnv):
    """
    Perturbed Humanoid CMDP with one forward obstacle/state constraint.

    Same as HumanoidForwardObstacleCMDP, but gravity is perturbed during step.

    Step return:
        obs, reward, cost, truncated, terminated, info
    """

    OBS_DIM = 376
    max_steps = 1000

    def __init__(
        self,
        obstacle_x_min=2.0,
        obstacle_x_max=4.0,
        warning_x_min=1.0,
        max_steps=1000,
        beta=1.0,
        squared_dense=False,
        sigma_gravity=0.7,
        **kwargs,
    ):
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

        super().__init__(**kwargs)

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        try:
            obs = super().reset(seed=seed, **kwargs)
        except TypeError:
            obs = super().reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        self._elapsed_steps = 0

        # Reset gravity at the beginning of every episode.
        self.model.opt.gravity[self._grav_axis] = self._base_grav

        obs = self._get_obs()

        return obs, {}

    def _humanoid_forward_reward(self, xy_position_before, xy_position_after, action):
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return float(reward), info

    def _hard_obstacle_violation(self, x):
        x = float(x)

        if not (self.obstacle_x_min <= x <= self.obstacle_x_max):
            return 0.0

        center = 0.5 * (self.obstacle_x_min + self.obstacle_x_max)
        half_width = 0.5 * (self.obstacle_x_max - self.obstacle_x_min)

        violation = 1.0 - abs(x - center) / max(half_width, 1e-6)

        return float(max(violation, 0.0))

    def _dense_obstacle_cost(self, x):
        x = float(x)

        if x < self.warning_x_min:
            return 0.0

        if self.warning_x_min <= x < self.obstacle_x_min:
            denom = max(self.obstacle_x_min - self.warning_x_min, 1e-6)
            dense = (x - self.warning_x_min) / denom

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        if self.obstacle_x_min <= x <= self.obstacle_x_max:
            hard = self._hard_obstacle_violation(x)

            dense = 1.0 + hard

            if self.squared_dense:
                dense = dense ** 2

            return float(dense)

        return 0.0

    def _compute_constraint_cost(self, x):
        dense_cost = self._dense_obstacle_cost(x)

        raw_cost = float(self.beta * dense_cost)

        tau = 5e-2
        cost = float(tau * np.log1p(np.exp(raw_cost / tau)))

        info = {
            "cost": cost,
            "dense_cost": dense_cost,
            "obstacle_x_min": self.obstacle_x_min,
            "obstacle_x_max": self.obstacle_x_max,
            "warning_x_min": self.warning_x_min,
            "beta": self.beta,
        }

        return cost, info

    def step(self, action):
        # Gravity perturbation.
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav
                + self.np_random.normal(
                    loc=0.0,
                    scale=self.sigma_gravity,
                )
            )

        xy_position_before = mass_center(self.model, self.sim)

        self.do_simulation(action, self.frame_skip)

        xy_position_after = mass_center(self.model, self.sim)

        self._elapsed_steps += 1

        reward, reward_info = self._humanoid_forward_reward(
            xy_position_before,
            xy_position_after,
            action,
        )

        xposafter = float(xy_position_after[0])

        cost, constraint_info = self._compute_constraint_cost(xposafter)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)
        info.update({
            "xposafter": xposafter,
            "gravity_z": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity_z": float(self._base_grav),
            "sigma_gravity": float(self.sigma_gravity),
            "in_warning_region": bool(
                self.warning_x_min <= xposafter < self.obstacle_x_min
            ),
            "in_obstacle_region": bool(
                self.obstacle_x_min <= xposafter <= self.obstacle_x_max
            ),
        })

        return observation, reward, cost, truncated, terminated, info

class HumanoidActionMaxCMDP(humanoid_v3.HumanoidEnv):
    """
    Humanoid CMDP with action torque max-cost constraint.

    Motivation follows HalfCheetahCMDP:

      1. Observation is augmented with max_cost observed so far.

      2. Raw instantaneous constraint value:

             c_t = max(max(abs(action)) - ACTION_TORQUE_THRESHOLD, 0)

      3. Incremental max-cost:

             incremental_max_cost_t = max(c_t - max_cost_{t-1}, 0)

      4. Returned CMDP cost:

             cost_t = beta * c_t + alpha * incremental_max_cost_t

      5. max_cost is reset to 0 at episode reset and updated after each step:

             max_cost_t = max(max_cost_{t-1}, c_t)

    Step return:
        obs, reward, cost, truncated, terminated, info
    """

    # Gym Humanoid-v3 default obs is commonly 376.
    # Augmented with max_cost, so 377.
    OBS_DIM = 377

    max_steps = 1000

    def __init__(
        self,
        max_steps=1000,
        action_torque_threshold=0.1,
        alpha=0.1,
        beta=1.0,
        **kwargs,
    ):
        self._elapsed_steps = 0
        self.max_cost = 0.0

        self.max_steps = int(max_steps)
        self.action_torque_threshold = float(action_torque_threshold)
        self.alpha = float(alpha)
        self.beta = float(beta)

        super().__init__(**kwargs)

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

    def _get_base_obs(self):
        """
        Humanoid base observation from parent environment.

        For Humanoid-v3 this is usually 376-dimensional depending on
        exclude_current_positions_from_observation and Gym version.
        """
        return np.asarray(super()._get_obs(), dtype=np.float32)

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
        Return augmented Humanoid observation.
        """
        base_obs = self._get_base_obs()
        return self._augment_obs(base_obs)

    def reset(self, seed=None, **kwargs):
        """
        Reset environment and max_cost.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        try:
            obs = super().reset(seed=seed, **kwargs)
        except TypeError:
            obs = super().reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        self._elapsed_steps = 0
        self.max_cost = 0.0

        # Ensure returned observation has max_cost = 0.
        obs = self._get_obs()

        return obs, {}

    def _humanoid_forward_reward(self, xy_position_before, xy_position_after, action):
        """
        Same Humanoid forward reward logic used in your other Humanoid CMDPs.
        """
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return float(reward), info

    def _compute_action_max_cost(self, action):
        """
        Compute dense + incremental max action constraint cost.

        current_c:
            instantaneous violation over torque threshold.

        incremental_max_cost:
            positive increase over previous episode max.

        returned cost:
            beta * dense_cost + alpha * incremental_max_cost
        """
        action = np.asarray(action, dtype=np.float32)

        max_action_abs = float(np.max(np.abs(action)))

        current_c = float(
            np.maximum(
                max_action_abs - self.action_torque_threshold,
                0.0,
            )
        )

        previous_max_cost = float(self.max_cost)

        incremental_max_cost = float(
            max(current_c - previous_max_cost, 0.0)
        )

        dense_cost = float(current_c)

        cost = float(
            self.beta * dense_cost
            + self.alpha * incremental_max_cost
        )

        self.max_cost = float(max(previous_max_cost, current_c))

        info = {
            "cost": cost,
            "current_c": current_c,
            "dense_cost": dense_cost,
            "previous_max_cost": previous_max_cost,
            "max_cost": self.max_cost,
            "incremental_max_cost": incremental_max_cost,
            "max_action_abs": max_action_abs,
            "action_torque_threshold": self.action_torque_threshold,
            "alpha": self.alpha,
            "beta": self.beta,
        }

        return cost, info

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)

        self.do_simulation(action, self.frame_skip)

        xy_position_after = mass_center(self.model, self.sim)

        self._elapsed_steps += 1

        reward, reward_info = self._humanoid_forward_reward(
            xy_position_before,
            xy_position_after,
            action,
        )

        cost, cost_info = self._compute_action_max_cost(action)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(cost_info)

        return observation, reward, cost, truncated, terminated, info


class HumanoidActionCostCMDP(humanoid_v3.HumanoidEnv):
    """
    Humanoid CMDP with action-based constraint cost.

    This is analogous to HumanoidForwardObstacleCMDP, but the constraint
    is on actions instead of forward x-position.

    Cost options:
        1. dense:
            cost = beta * sum(action^2)

        2. incremental:
            cost = beta * sum(max(abs(action) - action_threshold, 0)^2)

    Step return:
        obs, reward, cost, truncated, terminated, info
    """

    OBS_DIM = 376
    max_steps = 1000

    def __init__(
        self,
        max_steps=1000,
        beta=1.0,
        action_threshold=0.1,
        cost_type="dense",
        squared_incremental=True,
        use_softplus=True,
        softplus_tau=5e-2,
        **kwargs,
    ):
        self._elapsed_steps = 0

        self.max_steps = int(max_steps)
        self.beta = float(beta)
        self.action_threshold = float(action_threshold)
        self.cost_type = str(cost_type)
        self.squared_incremental = bool(squared_incremental)
        self.use_softplus = bool(use_softplus)
        self.softplus_tau = float(softplus_tau)

        valid_cost_types = ["dense", "incremental"]
        if self.cost_type not in valid_cost_types:
            raise ValueError(
                f"cost_type must be one of {valid_cost_types}, got {self.cost_type}"
            )

        if self.action_threshold < 0.0:
            raise ValueError("action_threshold must be non-negative.")

        super().__init__(**kwargs)

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        try:
            obs = super().reset(seed=seed, **kwargs)
        except TypeError:
            obs = super().reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        self._elapsed_steps = 0

        return obs, {}

    def _humanoid_forward_reward(self, xy_position_before, xy_position_after, action):
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return float(reward), info

    def _dense_action_cost(self, action):
        """
        Dense action cost.

        Penalizes total action energy.
        """
        action = np.asarray(action, dtype=np.float32)
        dense_cost = float(np.sum(np.square(action)))
        return dense_cost

    def _incremental_action_cost(self, action):
        """
        Incremental action cost.

        Penalizes only action magnitude above action_threshold.
        """
        action = np.asarray(action, dtype=np.float32)

        excess = np.maximum(np.abs(action) - self.action_threshold, 0.0)

        if self.squared_incremental:
            incremental_cost = float(np.sum(np.square(excess)))
        else:
            incremental_cost = float(np.sum(excess))

        return incremental_cost

    def _compute_constraint_cost(self, action):
        dense_action_cost = self._dense_action_cost(action)
        incremental_action_cost = self._incremental_action_cost(action)

        if self.cost_type == "dense":
            selected_cost = dense_action_cost
        elif self.cost_type == "incremental":
            selected_cost = incremental_action_cost
        else:
            raise RuntimeError(f"Unknown cost_type: {self.cost_type}")

        raw_cost = float(self.beta * selected_cost)

        if self.use_softplus:
            tau = max(self.softplus_tau, 1e-8)
            cost = float(tau * np.log1p(np.exp(raw_cost / tau)))
        else:
            cost = raw_cost

        info = {
            "cost": cost,
            "raw_action_cost": raw_cost,
            "dense_action_cost": dense_action_cost,
            "incremental_action_cost": incremental_action_cost,
            "selected_action_cost": selected_cost,
            "cost_type": self.cost_type,
            "action_threshold": self.action_threshold,
            "beta": self.beta,
            "squared_incremental": self.squared_incremental,
            "use_softplus": self.use_softplus,
            "softplus_tau": self.softplus_tau,
            "action_l2": float(np.linalg.norm(action, ord=2)),
            "action_l1": float(np.linalg.norm(action, ord=1)),
            "action_linf": float(np.max(np.abs(action))),
            "num_action_dims_above_threshold": int(
                np.sum(np.abs(action) > self.action_threshold)
            ),
        }

        return cost, info

    def step(self, action):
        xy_position_before = mass_center(self.model, self.sim)

        self.do_simulation(action, self.frame_skip)

        xy_position_after = mass_center(self.model, self.sim)

        self._elapsed_steps += 1

        reward, reward_info = self._humanoid_forward_reward(
            xy_position_before,
            xy_position_after,
            action,
        )

        cost, constraint_info = self._compute_constraint_cost(action)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)

        return observation, reward, cost, truncated, terminated, info

class HumanoidActionCostCMDPPerturbed(humanoid_v3.HumanoidEnv):
    """
    Perturbed Humanoid CMDP with action-based constraint cost.

    Same as HumanoidActionCostCMDP, but gravity is perturbed during step.

    Cost options:
        1. dense:
            cost = beta * sum(action^2)

        2. incremental:
            cost = beta * sum(max(abs(action) - action_threshold, 0)^2)

    Step return:
        obs, reward, cost, truncated, terminated, info
    """

    OBS_DIM = 376
    max_steps = 1000

    def __init__(
        self,
        max_steps=1000,
        beta=1.0,
        action_threshold=0.1,
        cost_type="dense",
        squared_incremental=True,
        use_softplus=True,
        softplus_tau=5e-2,
        sigma_gravity=0.7,
        **kwargs,
    ):
        self._elapsed_steps = 0

        self.max_steps = int(max_steps)
        self.beta = float(beta)
        self.action_threshold = float(action_threshold)
        self.cost_type = str(cost_type)
        self.squared_incremental = bool(squared_incremental)
        self.use_softplus = bool(use_softplus)
        self.softplus_tau = float(softplus_tau)

        self.sigma_gravity = float(sigma_gravity)

        valid_cost_types = ["dense", "incremental"]
        if self.cost_type not in valid_cost_types:
            raise ValueError(
                f"cost_type must be one of {valid_cost_types}, got {self.cost_type}"
            )

        if self.action_threshold < 0.0:
            raise ValueError("action_threshold must be non-negative.")

        super().__init__(**kwargs)

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        try:
            obs = super().reset(seed=seed, **kwargs)
        except TypeError:
            obs = super().reset()

        if isinstance(obs, tuple):
            obs = obs[0]

        self._elapsed_steps = 0

        self.model.opt.gravity[self._grav_axis] = self._base_grav

        obs = self._get_obs()

        return obs, {}

    def _humanoid_forward_reward(self, xy_position_before, xy_position_after, action):
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "reward_impact": -contact_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return float(reward), info

    def _dense_action_cost(self, action):
        action = np.asarray(action, dtype=np.float32)
        dense_cost = float(np.sum(np.square(action)))
        return dense_cost

    def _incremental_action_cost(self, action):
        action = np.asarray(action, dtype=np.float32)

        excess = np.maximum(np.abs(action) - self.action_threshold, 0.0)

        if self.squared_incremental:
            incremental_cost = float(np.sum(np.square(excess)))
        else:
            incremental_cost = float(np.sum(excess))

        return incremental_cost

    def _compute_constraint_cost(self, action):
        dense_action_cost = self._dense_action_cost(action)
        incremental_action_cost = self._incremental_action_cost(action)

        if self.cost_type == "dense":
            selected_cost = dense_action_cost
        elif self.cost_type == "incremental":
            selected_cost = incremental_action_cost
        else:
            raise RuntimeError(f"Unknown cost_type: {self.cost_type}")

        raw_cost = float(self.beta * selected_cost)

        if self.use_softplus:
            tau = max(self.softplus_tau, 1e-8)
            cost = float(tau * np.log1p(np.exp(raw_cost / tau)))
        else:
            cost = raw_cost

        info = {
            "cost": cost,
            "raw_action_cost": raw_cost,
            "dense_action_cost": dense_action_cost,
            "incremental_action_cost": incremental_action_cost,
            "selected_action_cost": selected_cost,
            "cost_type": self.cost_type,
            "action_threshold": self.action_threshold,
            "beta": self.beta,
            "squared_incremental": self.squared_incremental,
            "use_softplus": self.use_softplus,
            "softplus_tau": self.softplus_tau,
            "action_l2": float(np.linalg.norm(action, ord=2)),
            "action_l1": float(np.linalg.norm(action, ord=1)),
            "action_linf": float(np.max(np.abs(action))),
            "num_action_dims_above_threshold": int(
                np.sum(np.abs(action) > self.action_threshold)
            ),
        }

        return cost, info

    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav
                + self.np_random.normal(
                    loc=0.0,
                    scale=self.sigma_gravity,
                )
            )

        xy_position_before = mass_center(self.model, self.sim)

        self.do_simulation(action, self.frame_skip)

        xy_position_after = mass_center(self.model, self.sim)

        self._elapsed_steps += 1

        reward, reward_info = self._humanoid_forward_reward(
            xy_position_before,
            xy_position_after,
            action,
        )

        cost, constraint_info = self._compute_constraint_cost(action)

        observation = self._get_obs()

        truncated = self._elapsed_steps >= self.max_steps
        terminated = False

        info = {}
        info.update(reward_info)
        info.update(constraint_info)
        info.update({
            "gravity_z": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity_z": float(self._base_grav),
            "sigma_gravity": float(self.sigma_gravity),
        })

        return observation, reward, cost, truncated, terminated, info

