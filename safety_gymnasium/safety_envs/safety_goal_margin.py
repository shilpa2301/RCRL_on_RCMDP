# safety_goal_margin.py
import numpy as np
import gymnasium as gym
import safety_gymnasium
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

class SafetyGoalMargin(gym.Wrapper):
    """
    Margin wrapper for Safety[Agent]Goal{0,1,2}-v0 tasks.
    g(s) = min_distance_to_hazards_and_vases - safety_clearance
    Uses LiDAR observations for hazards and vases (ignores goal).
    If g(s) < 0, episode terminates (safety failure).
    """
    def __init__(self, env: gym.Env, safety_clearance: float = 0.20, lidar_max_range: float = 3.0):
        super().__init__(env)
        self.safety_clearance = float(safety_clearance)
        self.lidar_max_range = float(lidar_max_range)
        self._log_original = True
        
        self._render_mode = getattr(env, 'render_mode', None) or getattr(env, '_render_mode', None)
        
        # Initialize LiDAR slices - will be computed on first use
        # self._hazards_lidar_slice = None
        # self._vases_lidar_slice = None
        self._pillars_lidar_slice = None
        self._sigwalls_lidar_slice = None

    @property
    def render_mode(self):
        """Expose render_mode attribute for stable-baselines3 compatibility."""
        return self._render_mode
    
    def _compute_lidar_slices(self):
        """Compute the observation slices for hazards and vases LiDAR."""
        d = self.env.obs_space_dict  # Dict(name -> Box)
        start = 0
        
        # hazards_found = False
        # vases_found = False
        sigwalls_found = False
        pillars_found = False

        for name, space in d.spaces.items() if hasattr(d, "spaces") else d.items():
            size = int(np.prod(space.shape))
            
            # if name == "hazards_lidar":
            #     self._hazards_lidar_slice = slice(start, start + size)
            #     hazards_found = True
            # elif name == "vases_lidar":
            #     self._vases_lidar_slice = slice(start, start + size)
            #     vases_found = True
            if name == "sigwalls_lidar":
                self._sigwalls_lidar_slice = slice(start, start + size)
                sigwalls_found = True
            elif name == "pillars_lidar":
                self._pillars_lidar_slice = slice(start, start + size)
                pillars_found = True
                
            start += size
        
        # if not hazards_found:
        #     raise RuntimeError("hazards_lidar not found in obs_space_dict.")
        # if not vases_found:
        #     raise RuntimeError("vases_lidar not found in obs_space_dict.")
        if not sigwalls_found:
            raise RuntimeError("sigwalls_lidar not found in obs_space_dict.")
        if not pillars_found:
            raise RuntimeError("pillars_lidar not found in obs_space_dict.")

    def _margin_from_obs(self, obs: np.ndarray) -> float:
        """
        Compute safety margin using LiDAR observations for hazards, vases, and sigwalls.
        g(s) = min_distance_to_safety_critical_objects - safety_clearance
        
        LiDAR values are in [0,1] where:
        - 0 means no object detected (far away)
        - 1 means object is very close
        - We convert to distances: distance = (1 - lidar_value) * max_range
        """
        # if self._hazards_lidar_slice is None or self._vases_lidar_slice is None or self._sigwalls_lidar_slice is None:
        if self._pillars_lidar_slice is None or self._sigwalls_lidar_slice is None:
            self._compute_lidar_slices()
        
        # Extract LiDAR readings for hazards, vases, and sigwalls
        # hazards_beams = obs[self._hazards_lidar_slice]  # shape (16,), values in [0,1]
        # vases_beams = obs[self._vases_lidar_slice]      # shape (16,), values in [0,1]
        sigwalls_beams = obs[self._sigwalls_lidar_slice]  # shape (16,), values in [0,1]
        pillars_beams = obs[self._pillars_lidar_slice]  # shape (16,), values in [0,1]

        # Convert LiDAR closeness values to actual distances
        # lidar=0 (no object) -> dist=max_range
        # lidar=1 (very close) -> dist=0
        # hazards_dists = (1.0 - hazards_beams) * self.lidar_max_range
        # vases_dists = (1.0 - vases_beams) * self.lidar_max_range
        pillars_dists = (1.0 - pillars_beams) * self.lidar_max_range
        sigwalls_dists = (1.0 - sigwalls_beams) * self.lidar_max_range

        # Find minimum distance to any safety-critical object (hazards, vases, or sigwalls)
        all_safety_dists = np.concatenate([pillars_dists, sigwalls_dists])
        min_safety_distance = float(np.min(all_safety_dists))
        
        # Compute safety margin
        g = min_safety_distance #- self.safety_clearance
        return g

    def _calculate_continuous_cost(self, min_safety_distance: float) -> float:
        """
        Calculate dense continuous cost based on distance to nearest safety object.

        Regions:

            1. Safe region:
                min_safety_distance > safety_clearance
                cost = 0

            2. Warning / violation region:
                safety_clearance >= min_safety_distance > 0
                cost ramps from 0 to 1

            3. Collision / invalid region:
                min_safety_distance <= 0
                cost is high
        """

        d = float(min_safety_distance)
        c = float(self.safety_clearance)

        # Region 1: Safe
        if d > c:
            return 0.0

        # Region 2: Inside safety clearance
        elif 0.0 < d <= c:
            cost = (c - d) / max(c, 1e-6)
            return 1.0 * float(max(0.0, min(cost, 1.0)))

        # Region 3: Collision / penetration / invalid
        else:
            return 1.0 * float(1.0 + 10.0 * (-d))

    def step(self, action):
        # Safety-Gymnasium step returns: obs, reward, cost, terminated, truncated, info
        obs, _reward, cost, terminated, truncated, info = self.env.step(action)
        min_safety_distance = self._margin_from_obs(obs)

        continuous_cost = self._calculate_continuous_cost(min_safety_distance)

        # Terminate on safety failure (negative margin)
        # if g < 0.0:
        #     terminated = True

        if self._log_original:
            info = dict(info or {})
            info.update(
                {
                    "orig_reward": float(_reward),
                    "orig_cost": float(cost),
                    "margin_g": float(margin),
                    "safe": float(margin >= self.safety_clearance),
                    "min_safety_distance": float(min_safety_distance),
                    "continuous_cost": float(continuous_cost),
                }
            )

        return obs, _reward, continuous_cost, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment with rejection sampling to avoid starting in unsafe states."""
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            obs, info = self.env.reset(**kwargs)
            g = self._margin_from_obs(obs)
            
            if g >= 0.0:  # Safe initial state
                return obs, info
                
            attempts += 1
            
        # If we can't find a safe initial state, just return the last attempt
        # This shouldn't happen often in Goal environments as they have large spaces
        print(f"Warning: Could not find safe initial state after {max_attempts} attempts. "
              f"Starting with margin g={g:.3f}")
        return obs, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)


def make_env(agent: str = "Car", level: int = 2, render_mode=None,
             safety_clearance: float = 0.20, lidar_max_range: float = 3.0, **kwargs) -> gym.Env:
    """
    Create a SafetyGoalMargin environment.
    
    Args:
        agent: Agent type ("Point", "Car", "Racecar", "Doggo", "Ant")
        level: Goal level (0, 1, or 2)
        render_mode: Rendering mode
        safety_clearance: Safety clearance distance in meters
        lidar_max_range: Maximum range of LiDAR sensors in meters
        **kwargs: Additional arguments passed to safety_gymnasium.make()
    
    Returns:
        Wrapped environment with safety margin as reward
    """
    assert agent in {"Point", "Car", "Racecar", "Doggo", "Ant"}
    assert level in {0, 1, 2}
    
    task_id = f"Safety{agent}Goal{level}-v0"
    base = safety_gymnasium.make(task_id, render_mode=render_mode, **kwargs)
    base = TerminateOnCollisionWrapper(base)  # optional: end episode on collision
    return SafetyGoalMargin(base, safety_clearance=safety_clearance, lidar_max_range=lidar_max_range)