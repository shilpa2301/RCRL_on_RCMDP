import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from gym.envs.mujoco.hopper_v4 import HopperEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

ABS_PATH = os.getcwd()

ACTION_TORQUE_THRESHOLD = 0.5


class HopperCostEnv(HopperEnv):
    OBS_DIM = 11
    max_steps = 500

    def __init__(
        self,
        forward_reward_weight=1.0,             # default Hopper-v4: 1.0
        ctrl_cost_weight=1e-3,                 # default Hopper-v4: 1e-3
        healthy_reward=1.0,                    # default Hopper-v4: 1.0
        terminate_when_unhealthy=False,        # default Hopper-v4: True
        healthy_state_range=(-100.0, 100.0),   # default Hopper-v4
        healthy_z_range=(0.7, float("inf")),   # default Hopper-v4
        healthy_angle_range=(-0.2, 0.2),       # default Hopper-v4
        reset_noise_scale=5e-3,                # default Hopper-v4
        exclude_current_positions_from_observation=True,  # default Hopper-v4
        xml_file=None,
        max_steps: int = 500,
    ):
        if xml_file is None:
            # Use Gymnasium's default Hopper XML.
            # You can replace this with your custom XML path if needed.
            xml_file = "hopper.xml"

        super(HopperCostEnv, self).__init__(
            xml_file=xml_file,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_state_range=healthy_state_range,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )

        self._elapsed_steps = 0
        self.max_steps = max_steps

    def reset(self, seed=None, **kwargs):
        """
        Return (obs, info) tuple expected by Gymnasium-style reset.

        Your RCRL code may expect:
            obs, info = env.reset()
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs, info = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0

        return obs, info

    def step(self, action):
        x_position_before = self.data.qpos[0].copy()

        self.do_simulation(action, self.frame_skip)

        x_position_after = self.data.qpos[0].copy()
        x_velocity = (x_position_after - x_position_before) / self.dt

        self._elapsed_steps += 1

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        reward = rewards - costs

        # ------------------------------------------------------------
        # Constraint cost
        # Continuous positive-part action torque violation.
        #
        # cost = 0 if all action magnitudes are <= threshold
        # cost > 0 if max torque exceeds threshold
        # ------------------------------------------------------------
        cost = 0.0 #float(
        #     np.maximum(
        #         np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
        #         0.0
        #     )
        # )

        observation = self._get_obs()

        terminated = self.terminated
        truncated = self._elapsed_steps >= self.max_steps

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,

            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,

            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,

            "cost": cost,
            "action_torque_cost": cost,
            "max_action_abs": float(np.max(np.abs(action))),
            "action_torque_threshold": ACTION_TORQUE_THRESHOLD,
        }

        # Same style as your AntCost:
        # return observation, reward, cost, truncated, terminated, info
        return observation, reward, cost, truncated, terminated, info