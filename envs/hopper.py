import numpy as np
import gymnasium as gym
from gym import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import torch

class HopperPerturbedEnv(MujocoEnv, EzPickle):
    def __init__(
        self,
        xml_file="hopper.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        gravity_perturbation_std=0.5,
        joint_damping_perturbation_std=0.1,
        max_episode_steps=1000,
    ):
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float64
        )

        MujocoEnv.__init__(self, xml_file, 4, observation_space)
        EzPickle.__init__(self)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # Perturbation parameters
        self.default_gravity = -9.81
        self.gravity = self.default_gravity  # Perturbed gravity
        self.gravity_perturbation_std = gravity_perturbation_std

        self.thigh_joint_damping = 1.0
        self.leg_joint_damping = 1.0
        self.foot_joint_damping = 1.0
        self.joint_damping_perturbation_std = joint_damping_perturbation_std

        self.steps = 0
        self.max_episode_steps = max_episode_steps

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]

        # Apply perturbations to gravity and joint damping
        self.model.opt.gravity[2] = self.default_gravity + np.random.normal(0, self.gravity_perturbation_std)
        self.model.dof_damping[3] = self.thigh_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)
        self.model.dof_damping[4] = self.leg_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)
        self.model.dof_damping[5] = self.foot_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)

        # Add noise to action
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        self.do_simulation(action_noise, self.frame_skip)

        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        # Reward and cost calculations
        z, angle = self.data.qpos[1:3]
        reward = rewards - costs

        cost = 0.0
        if z < self._healthy_z_range[0]:
            cost += 1.0  # Cost for falling below height threshold
        if abs(angle) > self._healthy_angle_range[1]:
            cost += 0.01  # Cost for exceeding angle threshold
        if not self.is_healthy and self.steps < self.max_episode_steps:
            cost += 20.0  # Additional cost for premature termination

        observation = self._get_obs()
        done = self.done

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "perturbed_gravity": self.model.opt.gravity[2],
            "perturbed_thigh_damping": self.model.dof_damping[3],
            "perturbed_leg_damping": self.model.dof_damping[4],
            "perturbed_foot_damping": self.model.dof_damping[5],
        }

        self.steps += 1

        return observation, reward, cost, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        # Reset gravity to its default value
        self.model.opt.gravity[2] = self.default_gravity

        observation = self._get_obs()
        return observation

    def compute_cost(self, z, angle):
        """
        Compute the cost based on the current state.

        Args:
            z (float): The hopper's vertical height.
            angle (float): The hopper's angle.

        Returns:
            float: The computed cost.
        """
        cost = 0.0

        # Add cost if the hopper falls below the height threshold
        if z < self._healthy_z_range[0]:
            cost += 1.0  # Cost for falling below height threshold

        # Add cost if the hopper exceeds the angle threshold
        if abs(angle) > self._healthy_angle_range[1]:
            cost += 0.01  # Cost for exceeding angle threshold

        # Add penalty if the hopper becomes unhealthy prematurely
        done = not (self.is_healthy and self.steps < self.max_episode_steps)
        if done:
            cost += 20.0  # Penalty for premature termination

        return cost

    def simulate_next_state(self, state, action):
        """
        Simulate the next state based on the current state and action, without modifying the environment's internal state.

        Args:
            state (np.array): Current state [qpos, qvel] or observation.
            action (np.array): Action to apply.

        Returns:
            torch.Tensor: Simulated next state.
        """
        # Check if the state is the full state or observation
        if len(state) == self.model.nq + self.model.nv:  # Full state
            qpos = state[:self.model.nq]
            qvel = state[self.model.nq:]
        elif len(state) == self.observation_space.shape[0]:  # Observation
            qpos = np.concatenate(([0.0], state[:self.model.nq - 1]))  # Add back the excluded position
            qvel = state[self.model.nq - 1:]
        else:
            raise ValueError(f"Invalid state size: {len(state)}. Expected {self.model.nq + self.model.nv} or {self.observation_space.shape[0]}.")

        # Save the current state
        current_qpos = self.data.qpos.copy()
        current_qvel = self.data.qvel.copy()

        # Set the state to the provided state
        self.set_state(qpos, qvel)

        # Apply perturbations to gravity and joint damping
        self.model.opt.gravity[2] = self.default_gravity + np.random.normal(0, self.gravity_perturbation_std)
        self.model.dof_damping[3] = self.thigh_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)
        self.model.dof_damping[4] = self.leg_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)
        self.model.dof_damping[5] = self.foot_joint_damping + np.random.normal(0, self.joint_damping_perturbation_std)

        # Add noise to action
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)

        # Simulate the next state
        # Ensure action_noise is a 1D array
        action_noise = action_noise.squeeze()
        # print(f"Action shape: {action.shape}")
        # print(f"Action noise shape: {action_noise.shape}")

        self.do_simulation(action_noise, self.frame_skip)

        # Get the next state
        next_qpos = self.data.qpos.copy()
        next_qvel = self.data.qvel.copy()

        # Restore the original state
        self.set_state(current_qpos, current_qvel)

        # Combine position and velocity for the next state
        next_state = np.concatenate([next_qpos.flat, np.clip(next_qvel.flat, -10, 10)])
        # Ensure the next state matches the observation space (11 dimensions)
        if self._exclude_current_positions_from_observation:
            next_state = next_state[1:]  # Exclude the first dimension (position)
        return torch.tensor(next_state, dtype=torch.float32)
