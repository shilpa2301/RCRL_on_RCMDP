import numpy as np
import gymnasium as gym
from gym import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gym.utils import EzPickle
import torch
from typing import Optional, Tuple, Dict


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
        self.gravity = -9.81
        self.thigh_joint_damping = 1.0
        self.leg_joint_damping = 1.0
        self.foot_joint_damping = 1.0
        self.actuator_ctrlrange = (-1.0, 1.0)
        self.actuator_ctrllimited = int(1)
        self.hindsight_e = hindsight_e
        self.hindsight = hindsight
        self.steps = 0
        self.max_episode_steps = 1000
        self.frame_skip = 1 #1

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

    def compute_cost(self, obs: np.ndarray, prev_obs: Optional[np.ndarray]) -> Dict[str, float]:
        """
        Compute the spike-like cost based on the current and previous observations.

        Args:
            obs (np.ndarray): Current observation.
            prev_obs (Optional[np.ndarray]): Previous observation.

        Returns:
            Dict[str, float]: A dictionary containing the total cost, collapse cost, and impact cost.
        """
        z = float(obs[0])
        angle = abs(float(obs[1]))

        z_safe, z_fail = self._healthy_z_range[0], self._healthy_z_range[0] - 0.02
        angle_safe, angle_fail = self._healthy_angle_range[1], self._healthy_angle_range[1] + 0.02
        drop_safe, drop_fail = 0.010, 0.050

        c_low = max(0.0, (-1)*(z_safe - z) / max(1e-6, (z_safe - z_fail)))
        c_tilt = max(0.0, (angle - angle_safe) / max(1e-6, (angle_fail - angle_safe)))
        collapse = min(1.0, max(c_low, c_tilt))

        if prev_obs is None:
            impact = 0.0
        else:
            prev_z = float(prev_obs[0])
            dz_down = max(0.0, prev_z - z)
            raw_drop = max(0.0, (dz_down - drop_safe) / max(1e-6, (drop_fail - drop_safe)))
            gate = max(
                0.0,
                min(1.0, (z_safe - z) / max(1e-6, (z_safe - z_fail))),
                min(1.0, (angle - angle_safe) / max(1e-6, (angle_fail - angle_safe))),
            )
            impact = min(1.0, raw_drop * (0.35 + 0.65 * gate))

        total = min(1.0, max(collapse, impact))
        return {"total": total, "collapse": collapse, "impact": impact}

    def step(self, action, prev_obs=None):
        if np.random.binomial(n=1, p=self.hindsight_e):
            action = self.action_space.sample()

        # Compute the cost using the new compute_cost function
        curr_observation = self._get_obs()
        cost_dict = self.compute_cost(curr_observation, prev_obs)
        cost = cost_dict["total"]

        x_position_before = self.data.qpos[0]
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        action_noise = action # + self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
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

        # Compute the cost using the new compute_cost function
        # cost_dict = self.compute_cost(observation, prev_obs)
        # cost = cost_dict["total"]

        self.steps += 1
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "noise": action_noise - action,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "cost_details": cost_dict
        }

        return observation, reward, cost, done, info

    # def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    #     # Call the parent class's reset method
    #     observation, info = super().reset(seed=seed, options=options)

    #     # Reset any custom variables
    #     self.steps = 0

    #     # Return the initial observation and optional info
    #     return observation, info

    def save_xml(self, savepath):
      mujoco.mj_saveLastXML(savepath, self.model)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos # + self.np_random.uniform(
           # low=noise_low, high=noise_high, size=self.model.nq
        #)
        qvel = self.init_qvel # + self.np_random.uniform(
          #  low=noise_low, high=noise_high, size=self.model.nv
        #)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


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


