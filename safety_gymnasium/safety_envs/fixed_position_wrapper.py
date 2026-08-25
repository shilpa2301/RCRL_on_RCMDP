# fixed_position_wrapper.py
import numpy as np
import gymnasium as gym


class FixedPositionWrapper(gym.Wrapper):
    """
    Wrapper to enforce a fixed starting position for the agent.
    Useful for testing safety filters with challenging initial states.
    """
    
    def __init__(self, env: gym.Env, fixed_position: tuple = None, fixed_rotation: float = None):
        """
        Args:
            env: The environment to wrap
            fixed_position: (x, y) tuple for fixed starting position. If None, uses random positions.
            fixed_rotation: Fixed starting rotation in radians. If None, uses random rotation.
        """
        super().__init__(env)
        self.fixed_position = fixed_position
        self.fixed_rotation = fixed_rotation
        
    def reset(self, **kwargs):
        """Reset with fixed position if specified."""
        obs, info = self.env.reset(**kwargs)
        
        # Override agent position if specified
        if self.fixed_position is not None:
            self._set_agent_position(self.fixed_position, self.fixed_rotation)
            
            # Get new observation after position change
            obs = self._get_observation()
            
        return obs, info
    
    def _set_agent_position(self, position: tuple, rotation: float = None):
        """Set the agent's position and rotation directly in the MuJoCo simulation."""
        # Access the MuJoCo data through the environment
        env = self.env.unwrapped
        
        # Get current agent position to preserve z-coordinate
        current_pos = env.task.agent.pos
        
        # Set new position (x, y, z) - keep original z height
        new_pos = np.array([position[0], position[1], current_pos[2]])
        
        # Method 1: Try to set position through agent's engine data
        try:
            # For safety-gymnasium, the agent position is typically at index 0-2 in qpos
            env.task.agent.engine.data.qpos[0] = position[0]  # x
            env.task.agent.engine.data.qpos[1] = position[1]  # y
            # Keep z coordinate unchanged: env.task.agent.engine.data.qpos[2] stays the same
            
            # Set rotation if specified (quaternion at indices 3-6)
            if rotation is not None:
                # Convert rotation around z-axis to quaternion
                quat = np.array([np.cos(rotation/2), 0, 0, np.sin(rotation/2)])
                env.task.agent.engine.data.qpos[3:7] = quat
            
            # Reset velocities to zero (typically indices 0-5 in qvel for 6DOF)
            env.task.agent.engine.data.qvel[0:6] = 0
            
            # Forward the simulation to apply changes
            import mujoco
            mujoco.mj_forward(env.task.agent.engine.model, env.task.agent.engine.data)
            
        except Exception as e:
            print(f"Warning: Failed to set agent position directly: {e}")
            
            # Method 2: Alternative approach - try setting through task
            try:
                env.task.agent.engine.data.qpos[:3] = new_pos
                if rotation is not None:
                    quat = np.array([np.cos(rotation/2), 0, 0, np.sin(rotation/2)])
                    env.task.agent.engine.data.qpos[3:7] = quat
                env.task.agent.engine.data.qvel[:6] = 0
                
                import mujoco
                mujoco.mj_forward(env.task.agent.engine.model, env.task.agent.engine.data)
                
            except Exception as e2:
                print(f"Warning: Both position setting methods failed: {e2}")
                print("Fixed position may not work correctly.")
    
    def _get_observation(self):
        """Get the current observation from the environment."""
        return self.env.unwrapped.task.obs()


def make_circle_env_with_fixed_position(agent="Car", level=2, position=None, rotation=None, 
                                       safety_clearance=0.2, render_mode=None):
    """
    Create a Circle environment with fixed starting position.
    
    Args:
        agent: Agent type ("Car", "Point", etc.)
        level: Circle level (0, 1, 2)
        position: (x, y) starting position. If None, uses default random positions.
        rotation: Starting rotation in radians. If None, uses random rotation.
        safety_clearance: Safety margin for the margin wrapper
        render_mode: Render mode ("human", None)
        
    Returns:
        Environment with fixed position wrapper
    """
    from safety_gymnasium.safety_envs.safety_circle_margin import make_env
    
    # Create base environment
    env = make_env(
        agent=agent,
        level=level,
        render_mode=render_mode,
        safety_clearance=safety_clearance
    )
    
    # Add fixed position wrapper if position specified
    if position is not None:
        env = FixedPositionWrapper(env, fixed_position=position, fixed_rotation=rotation)
    
    return env


# Predefined challenging positions for different circle levels
CHALLENGING_POSITIONS = {
    "near_border": {
        0: [(0.7, 0.0), (-0.7, 0.0), (0.0, 0.7), (0.0, -0.7)],  # Level 0: near circle boundary
        1: [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],  # Level 1: near sigwalls
        2: [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],  # Level 2: near sigwalls
    },
    "very_close_to_border": {
        0: [(0.9, 0.0), (-0.9, 0.0), (0.0, 0.9), (0.0, -0.9)],  # Very close to circle
        1: [(1.1, 0.0), (-1.1, 0.0), (0.0, 1.1), (0.0, -1.1)],  # Very close to sigwalls
        2: [(1.1, 0.0), (-1.1, 0.0), (0.0, 1.1), (0.0, -1.1)],  # Very close to sigwalls
    },
    "almost_unsafe": {
        0: [(1.4, 0.0), (-1.4, 0.0), (0.0, 1.4), (0.0, -1.4)],  # Almost at circle boundary
        1: [(1.12, 0.0), (-1.12, 0.0), (0.0, 1.12), (0.0, -1.12)],  # Almost at sigwall
        2: [(1.12, 0.0), (-1.12, 0.0), (0.0, 1.12), (0.0, -1.12)],  # Almost at sigwall
    }
}


def get_challenging_position(difficulty="near_border", level=2, index=0):
    """
    Get a predefined challenging position.
    
    Args:
        difficulty: "near_border", "very_close_to_border", or "almost_unsafe"
        level: Circle level (0, 1, 2)
        index: Position index (0-3 for different sides)
        
    Returns:
        (x, y) position tuple
    """
    return CHALLENGING_POSITIONS[difficulty][level][index % 4]