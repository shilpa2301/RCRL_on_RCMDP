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