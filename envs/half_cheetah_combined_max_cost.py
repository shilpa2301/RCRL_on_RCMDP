import numpy as np

from envs.half_cheetah import HalfCheetahWithPos
from envs.env_configs.half_cheetah_combined_safety import (
    PITCH_THRESHOLD,
    PITCH_SCALE,
    JOINT_NAMES,
    JOINT_SPEED_THRESHOLDS,
    JOINT_SPEED_SCALES,
    COMBINED_COST_VERSION,
)


class HalfCheetahCombinedMaxCost(HalfCheetahWithPos):
    def __init__(self):

        self._mujoco_initializing = True
        self.joint_names = list(JOINT_NAMES)
        self.joint_speed_thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64).copy()
        self.joint_speed_scales = np.asarray(JOINT_SPEED_SCALES, dtype=np.float64).copy()
        if self.joint_speed_thresholds.shape != (6,):
            raise ValueError("JOINT_SPEED_THRESHOLDS must contain six values")
        if self.joint_speed_scales.shape != (6,) or np.any(self.joint_speed_scales <= 0):
            raise ValueError("JOINT_SPEED_SCALES must contain six positive values")
        if not np.isclose(float(PITCH_THRESHOLD), np.pi / 2.0, rtol=0.0, atol=1e-12):
            raise ValueError("PITCH_THRESHOLD must equal pi/2")
        if not np.isclose(float(PITCH_SCALE), np.pi / 2.0, rtol=0.0, atol=1e-12):
            raise ValueError("PITCH_SCALE must equal pi/2")
        super().__init__()
        self._mujoco_initializing = False

    def step(self, action):
        # observation, reward, _, truncated, terminated, info = super().step(action)
        step_result = super().step(action)
        if len(step_result) == 4:
            observation, reward, done, info = step_result
            truncated = done
            terminated = False
        elif len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        elif len(step_result) == 6:
            observation, reward, _, truncated, terminated, info = step_result
        else:
            raise RuntimeError(f"Unexpected number of values from super().step(): {len(step_result)}")

        pitch_raw = float(self.sim.data.qpos[2])
        pitch = float(np.arctan2(np.sin(pitch_raw), np.cos(pitch_raw)))
        pitch_abs = abs(pitch)
        pitch_excess = max(pitch_abs - float(PITCH_THRESHOLD), 0.0)
        pitch_cost = float(pitch_excess / float(PITCH_SCALE))

        joint_velocities = np.asarray(self.sim.data.qvel[3:9], dtype=np.float64).copy()
        joint_speeds = np.abs(joint_velocities)
        joint_excess = np.maximum(joint_speeds - self.joint_speed_thresholds, 0.0)
        joint_normalized_excess = joint_excess / self.joint_speed_scales
        joint_speed_cost = float(np.max(joint_normalized_excess))
        max_joint_index = int(np.argmax(joint_normalized_excess))

        all_normalized_costs = np.concatenate(
            [np.asarray([pitch_cost], dtype=np.float64), joint_normalized_excess]
        )
        combined_cost = float(np.max(all_normalized_costs))
        active_index = int(np.argmax(all_normalized_costs))

        if combined_cost <= 0.0:
            active_violation = "safe"
            active_branch = "none"
        elif active_index == 0:
            active_violation = "pitch"
            active_branch = "pitch"
        else:
            active_violation = self.joint_names[active_index - 1]
            active_branch = "joint_speed"

        info = dict(info)
        info.update(
            {
                "torso_pitch_raw": pitch_raw,
                "torso_pitch": pitch,
                "torso_pitch_abs": pitch_abs,
                "pitch_threshold": float(PITCH_THRESHOLD),
                "pitch_excess": float(pitch_excess),
                "pitch_scale": float(PITCH_SCALE),
                "pitch_cost": pitch_cost,
                "joint_velocities": joint_velocities,
                "joint_speeds": joint_speeds,
                "joint_speed_thresholds": self.joint_speed_thresholds.copy(),
                "joint_speed_excess": joint_excess,
                "joint_speed_normalized_excess": joint_normalized_excess,
                "joint_speed_scales": self.joint_speed_scales.copy(),
                "joint_speed_cost": joint_speed_cost,
                "maximum_joint_speed": float(np.max(joint_speeds)),
                "worst_joint_index": max_joint_index,
                "worst_joint_name": self.joint_names[max_joint_index],
                "combined_cost": combined_cost,
                "active_violation": active_violation,
                "active_branch": active_branch,
                "cost_version": COMBINED_COST_VERSION,
            }
        )

        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
                return observation, reward, truncated or terminated, info
        

        return observation, reward, combined_cost, truncated, terminated, info