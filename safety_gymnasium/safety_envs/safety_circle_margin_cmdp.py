# safety_circle_margin_cmdp.py

import numpy as np
import gymnasium as gym
import safety_gymnasium
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper


class SafetyCircleMarginCMDP(gym.Wrapper):
    """
    Safety-Gymnasium Circle wrapper with peak-cost CMDP reformulation.

    This wrapper computes a dense safety cost from the distance to the nearest
    wall/pillar, then converts the peak-cost constraint into an additive CMDP
    cost using a running maximum.

    Original instantaneous safety signal:
        current_c = continuous_cost(s_t, a_t, s_{t+1})

    Running maximum:
        max_cost_t = max_{k < t} current_c_k

    Incremental peak cost:
        incremental_max_cost_t = max(current_c_t - max_cost_t, 0)

    Dense cost:
        dense_cost_t = current_c_t

    Returned CMDP cost:
        cost_t = beta * dense_cost_t + alpha_t * incremental_max_cost_t

    Running maximum update:
        max_cost_{t+1} = max(max_cost_t, current_c_t)

    Observation augmentation:
        obs_aug_t = [obs_t, obs_cost_scale * max_cost_t]

    This makes the transformed cost Markovian because the cost depends on the
    previous maximum cost observed so far.
    """

    def __init__(
        self,
        env: gym.Env,
        safety_clearance: float = 0.40,
        max_steps: int = 1000,
        beta: float = 0.01,
        cost_scale: float = 100.0,
        obs_cost_scale: float = 100.0,
        terminate_on_margin_failure: bool = False,
        dense_cost_weight: float = 0.01,
    ):
        super().__init__(env)

        self.safety_clearance = float(safety_clearance)
        self.max_steps = int(max_steps)

        # Dense + incremental cost weights.
        self.beta = float(dense_cost_weight)
        self.cost_scale = float(cost_scale)
        self.obs_cost_scale = float(obs_cost_scale)

        self.terminate_on_margin_failure = bool(terminate_on_margin_failure)

        self._elapsed_steps = 0
        self.max_cost = 0.0
        self.last_cost = 0.0

        self._log_original = True

        self._render_mode = (
            getattr(env, "render_mode", None)
            or getattr(env, "_render_mode", None)
        )

        self._lidar_slice = None
        self._lidar_max_range = 6.0

        # ------------------------------------------------------------------
        # Observation-space augmentation.
        # Assumes original observation space is Box.
        # New observation is original obs plus one scalar: max_cost.
        # ------------------------------------------------------------------
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise TypeError(
                "SafetyCircleMarginCMDP currently assumes Box observation_space. "
                f"Got {type(self.env.observation_space)}."
            )

        old_low = np.asarray(self.env.observation_space.low, dtype=np.float32)
        old_high = np.asarray(self.env.observation_space.high, dtype=np.float32)

        new_low = np.concatenate([
            old_low.reshape(-1),
            np.array([-np.inf], dtype=np.float32),
        ]).astype(np.float32)

        new_high = np.concatenate([
            old_high.reshape(-1),
            np.array([np.inf], dtype=np.float32),
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=np.float32,
        )

    @property
    def render_mode(self):
        """Expose render_mode attribute for stable-baselines3 compatibility."""
        return self._render_mode

    # ======================================================================
    # Observation augmentation
    # ======================================================================

    def _augment_obs(self, obs):
        """
        Append running maximum cost to observation:

            obs_aug = [obs, obs_cost_scale * max_cost]

        The scaling is useful because max_cost may be numerically small or large
        compared to the original observation features.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        obs_aug = np.concatenate([
            obs,
            np.array(
                [self.obs_cost_scale * self.max_cost],
                dtype=np.float32,
            ),
        ]).astype(np.float32)

        return obs_aug

    # ======================================================================
    # Margin computation
    # ======================================================================

    def _compute_lidar_slice(self):
        d = self.env.obs_space_dict
        start = 0

        iterator = d.spaces.items() if hasattr(d, "spaces") else d.items()

        for name, space in iterator:
            size = int(np.prod(space.shape))

            if name == "pillars_lidar":
                self._lidar_slice = slice(start, start + size)
                return

            start += size

        raise RuntimeError("pillars_lidar not found in obs_space_dict.")

    def _margin_from_obs(self, obs: np.ndarray, use_lidar=False):
        """
        Compute safety margin using absolute positions.

        In this version:
            margin = min_distance_to_wall

        If you want the strict margin relative to safety clearance, use:
            margin = min_distance_to_wall - safety_clearance
        """
        if use_lidar:
            if self._lidar_slice is None:
                self._compute_lidar_slice()

            beams = obs[self._lidar_slice]
            pillar_dists = (1.0 - beams) * self._lidar_max_range

        robot_pos = self.env.unwrapped.task.agent.pos[:2]

        sigwalls = None
        for geom_name, geom_obj in self.env.unwrapped.task._geoms.items():
            if geom_name == "sigwalls":
                sigwalls = geom_obj
                break

        if sigwalls is None:
            raise RuntimeError("sigwalls geometry not found in environment.")

        wall_positions = sigwalls.pos
        wall_size = sigwalls.size

        min_distance_sigwall = float("inf")

        for wall_pos in wall_positions:
            wall_xy = wall_pos[:2]

            # Vertical wall: fixed x, extends in y.
            if abs(wall_xy[0]) > abs(wall_xy[1]):
                wall_x = wall_xy[0]
                distance_to_wall = abs(robot_pos[0] - wall_x)

                if abs(robot_pos[1]) > wall_size:
                    wall_y_edge = np.sign(robot_pos[1]) * wall_size
                    distance_to_wall = np.sqrt(
                        (robot_pos[0] - wall_x) ** 2
                        + (robot_pos[1] - wall_y_edge) ** 2
                    )

            # Horizontal wall: fixed y, extends in x.
            else:
                wall_y = wall_xy[1]
                distance_to_wall = abs(robot_pos[1] - wall_y)

                if abs(robot_pos[0]) > wall_size:
                    wall_x_edge = np.sign(robot_pos[0]) * wall_size
                    distance_to_wall = np.sqrt(
                        (robot_pos[0] - wall_x_edge) ** 2
                        + (robot_pos[1] - wall_y) ** 2
                    )

            min_distance_sigwall = min(
                min_distance_sigwall,
                float(distance_to_wall),
            )

        if use_lidar:
            assert pillar_dists is not None
            margin = min(
                float(min_distance_sigwall),
                float(np.min(pillar_dists)),
            )
        else:
            margin = float(min_distance_sigwall)

        return margin, float(min_distance_sigwall)

    # ======================================================================
    # Dense instantaneous safety cost
    # ======================================================================

    def _calculate_continuous_cost(self, min_distance_sigwall: float) -> float:
        """
        Dense instantaneous cost based on distance to nearest wall.

        Let:
            d = min_distance_sigwall
            c = safety_clearance

        Regions:
            1. Safe region:
                d > c
                cost = 0

            2. Warning / violation region:
                0 < d <= c
                cost ramps from 0 to 1 as d approaches 0

            3. Collision / penetration region:
                d <= 0
                cost > 1 and grows as d decreases
        """
        d = float(min_distance_sigwall)
        c = float(self.safety_clearance)

        if d > c:
            return 0.0

        elif 0.0 < d <= c:
            cost = (c - d) / max(c, 1e-6)
            return float(max(0.0, min(cost, 1.0)))

        else:
            return float(1.0 + 10.0 * (-d))

    # ======================================================================
    # Peak-cost CMDP transformation
    # ======================================================================

    def _compute_cmdp_cost(self, current_c: float):
        """
        Convert instantaneous dense safety cost into dense + incremental
        peak-CMDP cost.

        current_c:
            Raw instantaneous safety cost.

        previous_max_cost:
            Maximum safety cost observed before this transition.

        incremental_max_cost:
            Increase in the trajectory peak cost caused by this transition.

        dense_cost:
            Small dense shaping cost used to reduce sparsity.

        returned unscaled cost:
            cost = beta * dense_cost + alpha * incremental_max_cost

        Then update:
            max_cost = max(previous_max_cost, current_c)
        """
        current_c = float(current_c)

        previous_max_cost = float(self.max_cost)

        incremental_max_cost = float(
            max(current_c - previous_max_cost, 0.0)
        )

        dense_cost = float(current_c)

        # Same style as your HumanoidCMDP:
        # after self._elapsed_steps is incremented, alpha = (N - 1) / N.
        alpha = float(
            max((self._elapsed_steps - 1), 0)
            / max(self._elapsed_steps, 1)
        )

        cost = float(
            self.beta * dense_cost
            + alpha * incremental_max_cost
        )
        
        self.max_cost = float(max(previous_max_cost, current_c))
        self.last_cost = float(cost)

        info = {
            "cost": cost,
            "scaled_cost": self.cost_scale * cost,
            "current_c": current_c,
            "previous_max_cost": previous_max_cost,
            "max_cost": self.max_cost,
            "incremental_max_cost": incremental_max_cost,
            "dense_cost": dense_cost,
            "beta": self.beta,
            "alpha": alpha,
            "cost_scale": self.cost_scale,
            "obs_cost_scale": self.obs_cost_scale,
        }
        # print("cost, scaled_cost=", cost, self.cost_scale * cost)

        return self.cost_scale * cost, info

    # ======================================================================
    # Gymnasium API
    # ======================================================================

    def step(self, action):
        """
        Safety-Gymnasium step returns:

            obs, reward, cost, terminated, truncated, info

        This wrapper returns:

            augmented_obs, reward, transformed_cmdp_cost, terminated, truncated, info
        """
        obs, reward, orig_cost, terminated, truncated, info = self.env.step(action)

        self._elapsed_steps += 1

        margin, min_distance_sigwall = self._margin_from_obs(obs)

        continuous_cost = self._calculate_continuous_cost(min_distance_sigwall)

        transformed_cost, cost_info = self._compute_cmdp_cost(continuous_cost)

        if self.terminate_on_margin_failure:
            # Here margin is distance-to-wall.
            # Safety failure means inside the clearance region.
            if min_distance_sigwall < self.safety_clearance:
                terminated = True

        truncated = bool(truncated or self._elapsed_steps >= self.max_steps)

        obs_aug = self._augment_obs(obs)

        if self._log_original:
            info = dict(info or {})
            info.update({
                "orig_reward": float(reward),
                "orig_cost": float(orig_cost),
                "margin_g": float(margin),
                "min_distance_sigwall": float(min_distance_sigwall),
                "safety_clearance": float(self.safety_clearance),
                "safe": float(min_distance_sigwall >= self.safety_clearance),
                "continuous_cost": float(continuous_cost),
            })
            info.update(cost_info)

        return obs_aug, reward, transformed_cost, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset environment and running maximum cost.

        The initial augmented observation contains max_cost = 0.
        """
        while True:
            obs, info = self.env.reset(**kwargs)

            self._elapsed_steps = 0
            self.max_cost = 0.0
            self.last_cost = 0.0

            margin, min_distance_sigwall = self._margin_from_obs(obs)

            # Rejection sampling: start outside the safety-clearance region.
            if min_distance_sigwall >= self.safety_clearance:
                obs_aug = self._augment_obs(obs)
                return obs_aug, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)


def make_env(
    agent: str = "Car",
    level: int = 2,
    render_mode=None,
    safety_clearance: float = 0.20,
    max_steps: int = 1000,
    beta: float = 0.01,
    cost_scale: float = 1000.0,
    terminate_on_margin_failure: bool = False,
    dense_cost_weight: float = 0.01,
    **kwargs,
) -> gym.Env:
    assert agent in {"Point", "Car", "Racecar", "Doggo", "Ant"}

    task_id = f"Safety{agent}Circle{level}-v0"

    base = safety_gymnasium.make(
        task_id,
        render_mode=render_mode,
        **kwargs,
    )
    obs_cost_scale = cost_scale

    base = TerminateOnCollisionWrapper(base)
    print(f"SafetyCircleMarginCMDP: Using safety_clearance={safety_clearance:.4f} with cost_scale={cost_scale:.4f}, obs_cost_scale={obs_cost_scale:.4f}, beta={beta:.4f}")

    return SafetyCircleMarginCMDP(
        base,
        safety_clearance=safety_clearance,
        max_steps=max_steps,
        beta=beta,
        cost_scale=cost_scale,
        obs_cost_scale=obs_cost_scale,
        terminate_on_margin_failure=terminate_on_margin_failure,
        dense_cost_weight=dense_cost_weight
    )
