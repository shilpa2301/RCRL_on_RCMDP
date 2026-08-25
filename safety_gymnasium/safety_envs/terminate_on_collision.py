import gymnasium as gym
from gymnasium import logger

class TerminateOnCollisionWrapper(gym.Wrapper):
    """Wrapper that terminates episodes when agent collides with walls.
    
    This wrapper monitors the 'cost_out_of_boundary' value in the info dict
    and terminates the episode if it's greater than 0, indicating a wall collision.
    
    Args:
        env: The environment to wrap
        
    Example:
        >>> import safety_gymnasium
        >>> from safety_sb3 import TerminateOnCollisionWrapper
        >>> env = safety_gymnasium.make('SafetyCarCircle2-v0')
        >>> env = TerminateOnCollisionWrapper(env)
        >>> # Now episodes will terminate when agent hits walls
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._collision_occurred = False
        
    def reset(self, **kwargs):
        """Reset the environment and collision flag."""
        self._collision_occurred = False
        return self.env.reset(**kwargs)

    def soft_reset(self, **kwargs):
        """A soft reset that does not reset the environment state."""
        if hasattr(self.env, 'soft_reset'):
            self._collision_occurred = False
            return self.env.soft_reset(**kwargs)
        else:
            logger.warn(
                'The environment does not support soft reset. '
                'Falling back to hard reset.',
            )
            return self.reset(**kwargs)
        
    def step(self, action):
        """Step the environment and check for collisions."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        
        # Check for wall collision
        # collision_cost = info.get('cost_out_of_boundary', 0.0)
        collision_cost = info.get('cost_sum', 0.0)
        if collision_cost > 0:
            self._collision_occurred = True
            terminated = True
            # Optionally add collision info to the info dict
            info['terminated_by_collision'] = True
            
        return obs, reward, cost, terminated, truncated, info
        
    @property
    def collision_occurred(self):
        """Check if a collision occurred in the current episode."""
        return self._collision_occurred