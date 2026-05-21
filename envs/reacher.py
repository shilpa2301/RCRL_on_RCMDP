import numpy as np
import gym                          # FIX 1: add gym import (needed for gym.utils.seeding)
from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box


# =========================================================================== #
#                          Base Reacher Environment                           #
# =========================================================================== #

class ReacherEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "reacher.xml", 2, observation_space=observation_space, **kwargs
        )

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        if getattr(self, "render_mode", None) == "human":   # FIX 2: guard render_mode (mujoco-py may not have it)
            self.render()

        ob = self._get_obs()
        return (
            ob,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )


# =========================================================================== #
#                    Reacher With Cost (for RCMDP / RCRL)                     #
# =========================================================================== #

ACTION_TORQUE_THRESHOLD = 0.5


class ReacherWithCost(ReacherEnv):
    """
    Extends ReacherEnv with:

    1. **Cost signal** — continuous excess-torque cost, mirroring HalfCheetahWithPos:
            cost = max(max|action| - ACTION_TORQUE_THRESHOLD, 0)
       Zero when every joint torque is within the safe range; grows linearly
       with the worst-case violation.

    2. **Gravity perturbation** — optional zero-mean Gaussian noise added to
       gravity every step (sigma_gravity > 0 to enable).
       NOTE: standard reacher.xml has gravity disabled (all zeros). Set
       sigma_gravity=0.0 (default) unless you have a custom XML with gravity.

    3. **Return signature** — 6-tuple
            (obs, reward, cost, truncated, terminated, info)
       matching HalfCheetahWithPos / the RCRL training loop.

    Parameters
    ----------
    sigma_gravity : float
        Std-dev of per-step Gaussian gravity perturbation (m/s²). Default 0.0.
    max_steps : int
        Episode length before truncation. Default 50 (standard Reacher).
    """

    OBS_DIM   = 11
    max_steps = 50

    def __init__(self, sigma_gravity: float = 0.0, max_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.sigma_gravity  = sigma_gravity
        self.max_steps      = max_steps
        self._elapsed_steps = 0

        # FIX 3: reacher.xml gravity is all-zero (planar arm, no gravity).
        # np.argmin on an all-zero vector returns 0 arbitrarily — wrong.
        # Hardcode axis=2 (Z) as the perturbation axis; store whatever
        # base value is in the XML (likely 0.0) so perturbation is relative.
        self._grav_axis = 2
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            # FIX 4: was utils.seeding.np_random — AttributeError at runtime.
            # Must be gym.utils.seeding.np_random (requires gym import above).
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # FIX 5: was super(ReacherEnv, self).reset() which skips ReacherEnv
        # entirely and jumps straight to MuJocoPyEnv, bypassing reset_model()
        # so the goal is never re-sampled. Use super().reset() instead so the
        # MRO goes ReacherWithCost → ReacherEnv → MuJocoPyEnv → reset_model().
        obs = super().reset()

        self._elapsed_steps = 0
        # Restore nominal gravity at episode start
        self.model.opt.gravity[self._grav_axis] = self._base_grav
        return obs, {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, a):
        # ── 1. Optional gravity perturbation ─────────────────────────────────
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )

        # ── 2. Reward (computed before simulation, matches original ordering) ─
        vec         = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward      = reward_dist + reward_ctrl

        # ── 3. Simulate ───────────────────────────────────────────────────────
        self.do_simulation(a, self.frame_skip)
        if getattr(self, "render_mode", None) == "human":  # FIX 2 (same guard as base)
            self.render()

        ob = self._get_obs()
        self._elapsed_steps += 1

        # ── 4. Cost — continuous excess-torque penalty ────────────────────────
        cost = float(
            np.maximum(np.max(np.abs(a)) - ACTION_TORQUE_THRESHOLD, 0.0)
        )

        # ── 5. Termination / truncation ───────────────────────────────────────
        truncated  = self._elapsed_steps >= self.max_steps
        terminated = False

        info = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            cost=cost,
            dist_to_target=float(np.linalg.norm(vec)),
        )

        # ── 6-tuple: obs, reward, cost, truncated, terminated, info ──────────
        return ob, reward, cost, truncated, terminated, info


# =========================================================================== #
#                  Reacher With Cost — Test variant                           #
# =========================================================================== #

class ReacherWithCostTest(ReacherWithCost):
    """
    Test-time variant: terminates the episode and zeroes the reward if the
    fingertip strays too far from the target (hard constraint violation).
    Mirrors HalfCheetahWithPosTest.

    dist_threshold : float
        Maximum allowed distance from fingertip to target (metres).
        Default 0.1 m — tune to match your constraint budget.
    """

    def __init__(self, dist_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dist_threshold = dist_threshold

    def step(self, a):
        ob, reward, cost, truncated, terminated, info = super().step(a)

        if info["dist_to_target"] > self.dist_threshold:
            print(
                f"Violated distance constraint in test env "
                f"(dist={info['dist_to_target']:.4f} > {self.dist_threshold}), "
                f"terminating.",
                flush=True,
            )
            terminated = True
            reward     = 0.0

        return ob, reward, cost, truncated, terminated, info