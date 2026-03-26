import numpy as np
import gymnasium as gym
from gym import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import torch
from typing import Optional, Tuple


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
        hindsight_e=0.0,
        hindsight=False
    ):

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float64
        )

        MujocoEnv.__init__(self, xml_file, 4, observation_space)

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        # save base values*
        self.gravity = -9.81

        self.thigh_joint_damping = 1.0
        self.leg_joint_damping = 1.0
        self.foot_joint_damping = 1.0

        self.actuator_ctrlrange = (-1.0, 1.0)
        self.actuator_ctrllimited = int(1)

        # hindsight parameter*
        self.hindsight_e = hindsight_e
        self.hindsight = hindsight

        #MujocoEnv.__init__(self, xml_file, 4)
        self.steps = 0
        self.max_episode_steps = 1000



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
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

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

    def test(self):
        #sim = self.sim
        model = self.model
        #print(sim.get_state())
        print('body_names: ', model.body_names)
        print('joint_names: ', model.joint_names)
        print('actuator_names: ', model.actuator_names)
        print('model.actuator_forcelimited', model.actuator_forcelimited)
        print('actuator_ctrlrange', model.actuator_ctrlrange)
        print('_actuator_gear', model.actuator_gear)
        print('_jnt_stiffness', model.jnt_stiffness)
        print('_dof_damping', model.dof_damping)
        print('_dof_frictionloss', model.dof_frictionloss)
        print('actuator_ctrllimited', model.actuator_ctrllimited)

    def step(self, action):
        if np.random.binomial(n=1, p=self.hindsight_e):
            action = self.action_space.sample()

        x_position_before = self.data.qpos[0]
        # add noise to action for next state -> stochastic model
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        noise = self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        self.do_simulation(action_noise, self.frame_skip)

        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        # cost  = 1
        cost = 0.0
        z, angle = self.data.qpos[1:3]

        # Add cost if the hopper falls below the height threshold
        if z < self._healthy_z_range[0]:
            cost += 2.0 #1.0  # Cost for falling below height threshold

        # Add cost if the hopper exceeds the angle threshold
        if abs(angle) > self._healthy_angle_range[1]:
            cost += 1.0 #0.01  # Cost for exceeding angle threshold

        # Add penalty if the hopper becomes unhealthy prematurely
        # done = not (self.is_healthy and self.steps < (self.max_episode_steps - 100))
        done = not self.is_healthy 

        if done:
            cost += 0.0 #20.0  # Penalty for premature termination

        self.steps +=1
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "noise":noise,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy()
        }

        return observation, reward,cost, done, info

    def reset(
        self,
        x_pos: float = 0.0,
        state: Optional[int] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        use_xml: bool = False,
        gravity: float = -9.81,
        thigh_joint_stiffness: float = 0.0,
        leg_joint_stiffness: float = 0.0,
        foot_joint_stiffness: float = 0.0,
        springref: float = 0.0,
        actuator_ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        joint_damping_p: float = 0.0,
        joint_frictionloss: float = 0.0
    ):
        ob, info = super().reset(seed=seed, options=options)
        # hindsight*
        if self.hindsight:
            actuator_ctrlrange = (-0.85, 0.85)
        # grab model
        model = self.model
        # perturb gravity in z (3rd) dimension*
        model.opt.gravity[2] = gravity
        # perturb thigh joint*
        model.jnt_stiffness[3] = thigh_joint_stiffness
        model.qpos_spring[3] = springref
        # perturb leg joint*
        model.jnt_stiffness[4] = leg_joint_stiffness
        model.qpos_spring[4] = springref
        # perturb foot joint*
        model.jnt_stiffness[5] = foot_joint_stiffness
        model.qpos_spring[5] = springref
        # perturb actuator (controller) control range*
        model.actuator_ctrllimited[0] = self.actuator_ctrllimited
        model.actuator_ctrlrange[0] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        model.actuator_ctrllimited[1] = self.actuator_ctrllimited
        model.actuator_ctrlrange[1] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        model.actuator_ctrllimited[2] = self.actuator_ctrllimited
        model.actuator_ctrlrange[2] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        # perturb joint damping in percentage
        model.dof_damping[3] = self.thigh_joint_damping * (1 + joint_damping_p)
        model.dof_damping[4] = self.leg_joint_damping * (1 + joint_damping_p)
        model.dof_damping[5] = self.foot_joint_damping * (1 + joint_damping_p)
        # perturb joint frictionloss
        model.dof_frictionloss[3] = joint_frictionloss
        model.dof_frictionloss[4] = joint_frictionloss
        model.dof_frictionloss[5] = joint_frictionloss
        self.steps = 0
        return ob

    def save_xml(self, savepath):
      mujoco.mj_saveLastXML(savepath, self.model)

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

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def compute_cost(self, z, angle):
        """
        Compute the cost based on the current state.

        Args:
            z (float): The hopper's vertical height.
            angle (float): The hopper's angle.

        Returns:
            float: The computed cost.
        """
        # cost = 0.0
        # z, angle = self.data.qpos[1:3]
        cost = 0.0

        # Add cost if the hopper falls below the height threshold
        if z < self._healthy_z_range[0]:
            cost += 2.0 #1.0  # Cost for falling below height threshold

        # Add cost if the hopper exceeds the angle threshold
        if abs(angle) > self._healthy_angle_range[1]:
            cost += 1.0 #0.01  # Cost for exceeding angle threshold

        # Add penalty if the hopper becomes unhealthy prematurely
        # done = not (self.is_healthy and self.steps < (self.max_episode_steps-100))
        done = not self.is_healthy
        if done:
            cost += 20.0  # Penalty for premature termination

        # cost = 1.0

        return cost

    def simulate_next_state(self, state, action, info):
        """
        Simulate the next state based on the current state and action, without modifying the environment's internal state.

        Args:
            state (np.array): Current state [qpos, qvel] or observation.
            action (np.array): Action to apply.

        Returns:
            torch.Tensor: Simulated next state.
        """
        # # Check if the state is the full state or observation
       
        #------------------------------------------------------------------------------------
        # # Add noise to the action
        # # add noise to action for next state -> stochastic model
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        # action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)

        # # Simulate the environment's dynamics using the perturbed action
        # self.do_simulation(action_noise, 1)

        # # Get the next observation based on the updated state
        # observation = self._get_obs()
        # observation = torch.tensor(observation, dtype=torch.float32)
        # return observation


        #-----------------------------------------------------------------------------------
         # Save the current state of the environment
        current_qpos = self.data.qpos.copy()
        current_qvel = self.data.qvel.copy()

        # Use the full state from the info dictionary if available
        if "qpos" in info and "qvel" in info:
            qpos = info["qpos"]
            qvel = info["qvel"]
        else:
            # Fallback to reconstructing the full state from the observation
            if len(state) == self.model.nq + self.model.nv:  # Full state
                qpos = state[:self.model.nq]
                qvel = state[self.model.nq:]
            elif len(state) == self.observation_space.shape[0]:  # Observation
                qpos = np.concatenate(([0.0], state[:self.model.nq - 1]))  # Add back the excluded position
                qvel = state[self.model.nq - 1:]
            else:
                raise ValueError(f"Invalid state size: {len(state)}. Expected {self.model.nq + self.model.nv} or {self.observation_space.shape[0]}.")

        self.set_state(qpos, qvel)

        # Add noise to the action
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        action_noise = action + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)

        # Simulate the environment's dynamics using the perturbed action
        self.do_simulation(action_noise, 1)

        # Get the next observation based on the updated state
        next_qpos = self.data.qpos.copy()
        next_qvel = self.data.qvel.copy()

        # Restore the original state
        self.set_state(current_qpos, current_qvel)

        # Combine position and velocity for the next state
        next_state = np.concatenate([next_qpos.flat, np.clip(next_qvel.flat, -10, 10)])
        if self._exclude_current_positions_from_observation:
            next_state = next_state[1:]  # Exclude the first dimension (position)

        return torch.tensor(next_state, dtype=torch.float32)

    def get_z_and_angle_from_state(self, state, info):
        """
        Extract z (height) and angle from a given state.

        Args:
            env (gym.Env): The environment object (e.g., HopperPerturbedEnv).
            state (np.array or torch.Tensor): The state sampled from the replay buffer.

        Returns:
            z (float): The vertical height of the hopper.
            angle (float): The angle of the hopper.
        """
        # If state is a torch tensor, convert it to numpy array
        if isinstance(state, torch.Tensor):
            state = state.numpy()

        # Prefer using qpos from info if available
        if info and "qpos" in info:
            qpos = info["qpos"]
            z = qpos[1]  # z is the second element of qpos
            angle = qpos[2]  # angle is the third element of qpos
            return z, angle

        # If info is not provided, reconstruct qpos from the state
        if len(state) == self.model.nq + self.model.nv:  # Full state
            qpos = state[:self.model.nq]
            z = qpos[1]  # z is the second element of qpos
            angle = qpos[2]  # angle is the third element of qpos
        elif len(state) == self.observation_space.shape[0]:  # Observation
            qpos = np.concatenate(([0.0], state[:self.model.nq - 1]))  # Reconstruct qpos
            z = qpos[1]  # z is the second element of qpos
            angle = qpos[2]  # angle is the third element of qpos
        else:
            raise ValueError(f"Invalid state size: {len(state)}. Expected {self.model.nq + self.model.nv} or {self.observation_space.shape[0]}.")

        return z, angle


