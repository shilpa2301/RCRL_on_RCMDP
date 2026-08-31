# safety_circle_margin.py
import numpy as np
import gymnasium as gym
import safety_gymnasium
from safety_gymnasium.safety_envs.terminate_on_collision import TerminateOnCollisionWrapper

class SafetyCircleMargin(gym.Wrapper):
    """
    Margin wrapper for Safety[Agent]Circle{0,1,2}-v0 tasks.
    g(s) = min_distance_to_sigwalls - safety_clearance
    Uses absolute robot and wall positions for accurate distance calculation.
    If g(s) < 0, episode terminates (safety failure).
    """
    def __init__(self, env: gym.Env, safety_clearance: float = 0.40):
        super().__init__(env)
        self.safety_clearance = float(safety_clearance)
        self._log_original = True
        
        self._render_mode = getattr(env, 'render_mode', None) or getattr(env, '_render_mode', None)

        self._lidar_slice = None
        self._lidar_max_range = 6.0  # Safety-Gymnasium default for circle environments

    @property
    def render_mode(self):
        """Expose render_mode attribute for stable-baselines3 compatibility."""
        return self._render_mode
    
    def _compute_lidar_slice(self):
        d = self.env.obs_space_dict  # Dict(name -> Box)
        start = 0
        for name, space in d.spaces.items() if hasattr(d, "spaces") else d.items():
            size = int(np.prod(space.shape))
            if name == "pillars_lidar":
                self._lidar_slice = slice(start, start + size)
                return
            start += size
        raise RuntimeError("pillars_lidar not found in obs_space_dict.")

    def _margin_from_obs(self, obs: np.ndarray, use_lidar=False) -> float:
        """
        Compute safety margin using absolute positions instead of noisy lidar.
        g(s) = min_distance_to_sigwalls - safety_clearance
        """
        if use_lidar:
            # LiDAR distance to pillars
            if self._lidar_slice is None:
                self._compute_lidar_slice()
            
            beams = obs[self._lidar_slice]              # shape (16,), values in [0,1]
            # Convert closeness values to actual distances
            pillar_dists = (1.0 - beams) * self._lidar_max_range
            
        # Get robot's absolute position
        robot_pos = self.env.unwrapped.task.agent.pos[:2]  # [x, y] position
        
        # Get sigwalls positions (boundary walls)
        sigwalls = None
        for geom_name, geom_obj in self.env.unwrapped.task._geoms.items():
            if geom_name == 'sigwalls':
                sigwalls = geom_obj
                break
        
        if sigwalls is None:
            raise RuntimeError("sigwalls geometry not found in environment")
        
        # Get wall positions - sigwalls.pos returns list of wall positions
        wall_positions = sigwalls.pos
        
        # Calculate minimum distance to any wall
        min_distance_sigwall = float('inf')
        
        for wall_pos in wall_positions:
            wall_xy = wall_pos[:2]  # Extract x, y coordinates
            
            # Get the wall geometry parameters
            wall_size = sigwalls.size  # This is the half-length of the wall
            
                        # Determine wall orientation and calculate distance to wall surface
            if abs(wall_xy[0]) > abs(wall_xy[1]):  # Vertical wall (left/right)
                # Wall extends in Y direction, fixed X position
                wall_x = wall_xy[0]
                # Distance to vertical wall is absolute difference in X
                distance_to_wall = abs(robot_pos[0] - wall_x)
                
                # Check if robot is within wall's Y extent
                if abs(robot_pos[1]) <= wall_size:
                    # Robot is directly facing the wall
                    pass  # distance_to_wall is correct
                else:
                    # Robot is beyond wall's Y extent, calculate distance to wall corner
                    wall_y_edge = np.sign(robot_pos[1]) * wall_size
                    distance_to_wall = np.sqrt((robot_pos[0] - wall_x)**2 + 
                                             (robot_pos[1] - wall_y_edge)**2)
                    
            else:  # Horizontal wall (top/bottom)
                # Wall extends in X direction, fixed Y position  
                wall_y = wall_xy[1]
                # Distance to horizontal wall is absolute difference in Y
                distance_to_wall = abs(robot_pos[1] - wall_y)
                
                # Check if robot is within wall's X extent
                if abs(robot_pos[0]) <= wall_size:
                    # Robot is directly facing the wall
                    pass  # distance_to_wall is correct
                else:
                    # Robot is beyond wall's X extent, calculate distance to wall corner
                    wall_x_edge = np.sign(robot_pos[0]) * wall_size
                    distance_to_wall = np.sqrt((robot_pos[0] - wall_x_edge)**2 + 
                                             (robot_pos[1] - wall_y)**2)


            min_distance_sigwall = min(min_distance_sigwall, distance_to_wall)
        
        # Compute margin: distance to closest wall minus safety clearance
        # if use_lidar:
        #     assert pillar_dists is not None, RuntimeError("pillar_dists should be computed when use_lidar is True")

        #     g = min(
        #         float(min_distance_sigwall - self.safety_clearance),
        #         float(np.min(pillar_dists) - self.safety_clearance)
        #     )
        # else:
        #     g = float(min_distance_sigwall - self.safety_clearance)

        if use_lidar:
                    assert pillar_dists is not None, RuntimeError("pillar_dists should be computed when use_lidar is True")
        
                    g = min(
                        float(min_distance_sigwall ),
                        float(np.min(pillar_dists) )
                    )
        else:
                    g = float(min_distance_sigwall )
        
        return g, min_distance_sigwall

    # def _calculate_continuous_cost(self, margin: float) -> float:
    #     """Calculate a continuous cost based on the margin."""
    #     if margin < 0:  # If the agent is within the safety clearance
    #         return 10*(-margin) #float('inf')  # High cost for being unsafe
    #     else:
    #         # Normalize the cost based on how far the agent is from the walls
    #         # For example, you can use a linear scaling
    #         # return max(0, 1 - (margin / self.safety_clearance))
    #         return max(0, 1/(margin + 1e-5))  # Inverse of margin for continuous cost

    def _calculate_continuous_cost(self, min_distance_sigwall: float) -> float:
        """
        Calculate a dense continuous cost based on the distance to the nearest wall.

        Regions:
            1. Safe Region:
                min_distance_sigwall > safety_clearance
                cost = 0

            2. Safety-Violation / Warning Region:
                safety_clearance >= min_distance_sigwall > 0
                cost ramps from 0 to 1 as the agent approaches the wall.

            3. Collision / Penetration Region:
                min_distance_sigwall <= 0
                cost is high.
        """
        d = float(min_distance_sigwall)
        c = float(self.safety_clearance)

        # Region 1: Safe region
        if d > c:
            return 0.0

        # Region 2: Within safety clearance but distance is still positive
        elif 0.0 < d <= c:
            # cost = 0 when d == c
            # cost = 1 when d == 0
            cost = (c - d) / max(c, 1e-6)
            return 1.0*float(max(0.0, min(cost, 1.0)))

        # Region 3: Collision / penetration / invalid distance
        else:  # d <= 0
            # cost starts above 1 and grows as d becomes more negative
            return 1.0*float(1.0 + 10.0 * (-d))




    def step(self, action):
        # Safety-Gymnasium step returns: obs, reward, cost, terminated, truncated, info
        obs, _reward, cost, terminated, truncated, info = self.env.step(action)
        margin, min_distance_sigwall = self._margin_from_obs(obs)
        # print(f"Step: margin={margin:.4f}, min_distance_sigwall={min_distance_sigwall:.4f}, safety_clearance={self.safety_clearance:.4f}")

        # Calculate continuous cost based on margin
        continuous_cost = self._calculate_continuous_cost(min_distance_sigwall)

        # Terminate on safety failure (negative margin)
        if margin < 0.0:
            terminated = True

        if self._log_original:
            info = dict(info or {})
            info.update({
                "orig_reward": float(_reward),
                "orig_cost": float(cost),
                "margin_g": float(margin),
                "safe": float(margin >= 0.0),
                "continuous_cost": float(continuous_cost),
            })
        # Replace reward with the margin
        return obs, _reward, continuous_cost, terminated, truncated, info

    def reset(self, **kwargs):
        # rejection sampling, outside failure set
        while True:
            obs, info = self.env.reset(**kwargs)
            margin, _ = self._margin_from_obs(obs)
            if margin >= 0.0:
                return obs, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

def make_env(agent: str = "Car", level: int = 2, render_mode=None,
             safety_clearance: float = 0.20, **kwargs) -> gym.Env:
    assert agent in {"Point", "Car", "Racecar", "Doggo", "Ant"}
    task_id = f"Safety{agent}Circle{level}-v0"
    base = safety_gymnasium.make(task_id, render_mode=render_mode, **kwargs)
    base = TerminateOnCollisionWrapper(base)  # optional: end episode on collision
    return SafetyCircleMargin(base, safety_clearance=safety_clearance)
