import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class SkydioTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path,
        render_mode=None,
        trajectory_type="circle",
        frame_skip=2,
        episode_seconds=20.0,
        radius=1.0,
        z_ref=1.0,
        trajectory_period=10.0,
        action_scale=2.0,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "x2"
        )
        if self.body_id < 0:
            raise RuntimeError("Could not find body named 'x2'.")

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip

        self.episode_seconds = episode_seconds
        self.max_steps = int(self.episode_seconds / self.dt)

        self.trajectory_type = trajectory_type
        self.radius = radius
        self.z_ref = z_ref
        self.trajectory_period = trajectory_period
        self.omega = 2.0 * np.pi / self.trajectory_period

        self.hover_ctrl = np.array(
            [3.2495625, 3.2495625, 3.2495625, 3.2495625],
            dtype=np.float64
        )

        self.action_scale = action_scale

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation:
        # pos_error      3
        # vel_error      3
        # quat           4
        # angular vel    3
        # ref_pos        3
        # ref_vel        3
        # prev_ctrl      4
        # time features  2 : sin(omega t), cos(omega t)
        # total = 25
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

        self.prev_ctrl = self.hover_ctrl.copy()
        self.step_count = 0
        self.t = 0.0

    def _reference(self, t):
        if self.trajectory_type == "hover":
            pos_ref = np.array([0.0, 0.0, self.z_ref], dtype=np.float64)
            vel_ref = np.zeros(3, dtype=np.float64)

        elif self.trajectory_type == "circle":
            R = self.radius
            w = self.omega

            pos_ref = np.array([
                R * np.cos(w * t),
                R * np.sin(w * t),
                self.z_ref,
            ], dtype=np.float64)

            vel_ref = np.array([
                -R * w * np.sin(w * t),
                R * w * np.cos(w * t),
                0.0,
            ], dtype=np.float64)

        elif self.trajectory_type == "figure8":
            R = self.radius
            w = self.omega

            pos_ref = np.array([
                R * np.sin(w * t),
                R * np.sin(w * t) * np.cos(w * t),
                self.z_ref,
            ], dtype=np.float64)

            vel_ref = np.array([
                R * w * np.cos(w * t),
                R * w * (np.cos(w * t) ** 2 - np.sin(w * t) ** 2),
                0.0,
            ], dtype=np.float64)

        elif self.trajectory_type == "line":
            # Smooth back-and-forth line.
            R = self.radius
            w = self.omega

            pos_ref = np.array([
                R * np.sin(w * t),
                0.0,
                self.z_ref,
            ], dtype=np.float64)

            vel_ref = np.array([
                R * w * np.cos(w * t),
                0.0,
                0.0,
            ], dtype=np.float64)

        else:
            raise ValueError(f"Unknown trajectory_type: {self.trajectory_type}")

        return pos_ref, vel_ref

    def _get_obs(self):
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        pos_ref, vel_ref = self._reference(self.t)

        pos_error = pos - pos_ref
        vel_error = lin_vel - vel_ref

        time_features = np.array([
            np.sin(self.omega * self.t),
            np.cos(self.omega * self.t),
        ], dtype=np.float64)

        obs = np.concatenate([
            pos_error,
            vel_error,
            quat,
            ang_vel,
            pos_ref,
            vel_ref,
            self.prev_ctrl / 13.0,
            time_features,
        ])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.t = 0.0

        pos_ref, vel_ref = self._reference(self.t)

        # Start near initial reference.
        pos_noise = self.np_random.uniform(low=-0.05, high=0.05, size=3)
        vel_noise = self.np_random.normal(loc=0.0, scale=0.02, size=3)

        self.data.qpos[0:3] = pos_ref + pos_noise
        self.data.qpos[2] = max(0.2, self.data.qpos[2])

        # Identity quaternion.
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        self.data.qvel[0:3] = vel_ref + vel_noise
        self.data.qvel[3:6] = self.np_random.normal(loc=0.0, scale=0.01, size=3)

        self.prev_ctrl = self.hover_ctrl.copy()
        self.data.ctrl[:] = self.hover_ctrl

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        ctrl = self.hover_ctrl + self.action_scale * action
        ctrl = np.clip(ctrl, 0.0, 13.0)

        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.prev_ctrl = ctrl.copy()
        self.step_count += 1
        self.t = self.step_count * self.dt

        obs = self._get_obs()

        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        pos_ref, vel_ref = self._reference(self.t)

        pos_error = pos - pos_ref
        vel_error = lin_vel - vel_ref

        quat_error = np.array([
            1.0 - abs(quat[0]),
            quat[1],
            quat[2],
            quat[3],
        ])

        action_deviation = (ctrl - self.hover_ctrl) / self.action_scale
        action_smoothness = (ctrl - self.prev_ctrl) / self.action_scale

        # Main tracking reward.
        reward = (
            - 15.0 * np.sum(pos_error ** 2)
            - 2.0 * np.sum(vel_error ** 2)
            - 1.0 * np.sum(quat_error ** 2)
            - 0.1 * np.sum(ang_vel ** 2)
            - 0.01 * np.sum(action_deviation ** 2)
            - 0.005 * np.sum(action_smoothness ** 2)
        )

        dist = np.linalg.norm(pos_error)

        # Optional survival/tracking bonus.
        if dist < 0.1:
            reward += 2.0
        elif dist < 0.25:
            reward += 1.0

        too_low = pos[2] < 0.05
        too_high = pos[2] > 5.0
        too_far = dist > 4.0
        bad_tilt = abs(quat[0]) < 0.35
        timeout = self.step_count >= self.max_steps

        terminated = bool(too_low or too_high or too_far or bad_tilt)
        truncated = bool(timeout)

        info = {
            "t": self.t,
            "pos": pos.copy(),
            "vel": lin_vel.copy(),
            "pos_ref": pos_ref.copy(),
            "vel_ref": vel_ref.copy(),
            "pos_error": pos_error.copy(),
            "vel_error": vel_error.copy(),
            "tracking_error": dist,
            "ctrl": ctrl.copy(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

        # Slow down visualization.
        time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


#============================================
# env = SkydioTrackingEnv(
#     xml_path="mujoco_menagerie/skydio_x2/scene.xml",
#     render_mode=None,
#     trajectory_type="circle",
#     radius=0.75,
#     z_ref=1.0,
#     trajectory_period=12.0,
#     frame_skip=2,
#     episode_seconds=20.0,
#     action_scale=2.0,
#     tracking_safe_radius=0.5,
#     min_altitude=0.2,
#     max_altitude=3.0,
#     tilt_threshold_w=0.65,
# )
