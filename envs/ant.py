import numpy as np
import os
from gym.envs.mujoco.ant_v3 import AntEnv
import gym

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

###############################################################################
# ANT TORQUE CONSTRAINTS
###############################################################################

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100
class AntTest(AntEnv):
    def reset(self):
        ob = super().reset()
        self.current_timestep = 0
        self.violations = 0
        return ob

    def step(self, action):
        next_ob, reward, done, infos = super().step(action)
        # This is to handle the edge case where mujoco_env calls
        # step in __init__ without calling reset with a random
        # action
        try:
            self.current_timestep += 1
            if np.any(np.abs(action) > ACTION_TORQUE_THRESHOLD):
                self.violations += 1
            if self.violations > VIOLATIONS_ALLOWED:
                done = True
                reward = 0
        except:
            pass
        return next_ob, reward, done, infos

###############################################################################
# ANT WALL ENVIRONMENTS
###############################################################################
ACTION_TORQUE_THRESHOLD = 0.5
class AntCost(AntEnv):
    OBS_DIM = 113
    max_steps = 500
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/env_configs/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            max_steps: int = 500
    ):
        self._mujoco_initializing = True
        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._grav_axis = 2
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        self.max_steps  = max_steps
        super(AntCost, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )

        self._mujoco_initializing = False
        
       
    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._grav_axis = 2
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        # self.max_steps  = max_steps

        return obs, {}
    
    def step(self, action):
        
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        #shilpa
        rewards = forward_reward + healthy_reward
        # distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        # rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards  - costs
        #shilpa
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # cost = 0.0
        # ── Termination / truncation ────────────────────────────────────────
        truncated = self._elapsed_steps >= self.max_steps
        terminated = False #self.terminated

        # done = self.done
        observation = self._get_obs()
        # print(observation.shape)
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        # return observation, reward, cost, done, info
        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, truncated or terminated, info


        return observation, reward, cost, truncated, terminated, info

    
#shilpa
class AntCostPerturbed(AntEnv):
    OBS_DIM = 113
    max_steps = 500
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/env_configs/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            sigma_gravity: float = 0.0,
            max_steps: int = 500
    ):
        self._mujoco_initializing = True
        self._mujoco_initializing = False
        self.sigma_gravity = sigma_gravity
        self._elapsed_steps = 0
        self.max_steps  = max_steps

        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        self._grav_axis = 2
        self._base_grav = -9.81
        super(AntCostPerturbed, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )
        self._mujoco_initializing = False
        
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])


    def reset(self, seed=None, **kwargs):
        """Return (obs, info) tuple expected by the RCRL training loop."""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        self._elapsed_steps = 0
        # Restore nominal viscosity at episode start
        # self.model.opt.viscosity = self._base_viscosity
        # self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        self.model.opt.gravity[self._grav_axis] = self._base_grav


        return obs, {}
    
    def step(self, action):
        if self.sigma_gravity > 0.0:
            self.model.opt.gravity[self._grav_axis] = (
                self._base_grav + np.random.normal(0.0, self.sigma_gravity)
            )
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        #shilpa 
        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        #shilpa
        rewards = forward_reward + healthy_reward
        # distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        # rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        #shilpa
        cost = float(np.maximum(np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD, 0.0))
        # cost = 0.0
        # ── Termination / truncation ────────────────────────────────────────
        truncated = self._elapsed_steps >= self.max_steps
        terminated = False
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }
        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
                    return observation, reward, truncated or terminated, info
        
        # return observation, reward, done, info
        return observation, reward, cost, truncated, terminated, info

class AntCostPerturbedTest(AntCost):
    """
    Test-time perturbed AntCost.

    Samples one gravity perturbation once during __init__ and keeps it fixed
    for all episodes and all steps.

    If shared_gravity_perturbation is given, uses that exact perturbation.
    """

    OBS_DIM = 113
    max_steps = 500

    def __init__(
        self,
        healthy_reward=1.0,
        terminate_when_unhealthy=False,
        xml_file=ABS_PATH + "/env_configs/ant_circle.xml",
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        sigma_gravity: float = 0.0,
        max_steps: int = 500,
        shared_gravity_perturbation=None,
        seed=None,
    ):
        # ------------------------------------------------------------
        # CRITICAL:
        # Old Gym MujocoEnv may call self.step(action) inside __init__.
        # During that phase, we must return old-style 4-tuple from step
        # and avoid custom perturbation logic.
        # ------------------------------------------------------------
        self._mujoco_initializing = True

        self._elapsed_steps = 0
        self.sigma_gravity = float(sigma_gravity)
        self.max_steps = int(max_steps)

        self._grav_axis = 2
        self._base_grav = -9.81

        self.shared_gravity_perturbation = shared_gravity_perturbation
        self.fixed_gravity_perturbation = 0.0
        self.fixed_gravity_value = -9.81

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        super(AntCostPerturbedTest, self).__init__(
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            xml_file=xml_file,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )

        # Now MuJoCo model exists.
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

        if self.shared_gravity_perturbation is None:
            if self.sigma_gravity > 0.0:
                self.fixed_gravity_perturbation = float(
                    self.np_random.normal(0.0, self.sigma_gravity)
                )
            else:
                self.fixed_gravity_perturbation = 0.0
        else:
            self.fixed_gravity_perturbation = float(
                self.shared_gravity_perturbation
            )

        self.fixed_gravity_value = (
            self._base_grav + self.fixed_gravity_perturbation
        )

        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        self._mujoco_initializing = False

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        print(
            f"[AntCostPerturbedTest] "
            f"base_gravity={self._base_grav}, "
            f"fixed_perturbation={self.fixed_gravity_perturbation}, "
            f"fixed_gravity={self.fixed_gravity_value}"
        )

    def reset(self, seed=None, **kwargs):
        """
        Return (obs, info) tuple expected by the RCRL/eval loop.
        Keep the same fixed gravity every episode.
        """

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        # Your AntCost / AntCostPerturbed reset may return either:
        #   obs
        # or
        #   obs, info
        if isinstance(obs, tuple):
            observation = obs[0]
            info = obs[1] if len(obs) > 1 else {}
        else:
            observation = obs
            info = {}

        self._elapsed_steps = 0

        # Keep same fixed gravity every episode.
        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        info.update({
            "gravity": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity": float(self._base_grav),
            "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
            "sigma_gravity": float(self.sigma_gravity),
        })

        return observation, info

    def step(self, action):
        # ------------------------------------------------------------
        # CRITICAL:
        # Old Gym MujocoEnv may call self.step(action) inside __init__.
        # During that phase, return old-style 4-tuple and do NOT apply
        # custom perturbation logic.
        # ------------------------------------------------------------
        if getattr(self, "_mujoco_initializing", False):
            xy_position_before = self.get_body_com("torso")[:2].copy()

            self.do_simulation(action, self.frame_skip)

            xy_position_after = self.get_body_com("torso")[:2].copy()

            xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
            x_velocity, y_velocity = xy_velocity

            ctrl_cost = self.control_cost(action)
            contact_cost = self.contact_cost

            forward_reward = x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward + healthy_reward
            costs = ctrl_cost + contact_cost
            reward = rewards - costs

            observation = self._get_obs()
            done = False

            info = {
                "reward_forward": forward_reward,
                "reward_ctrl": -ctrl_cost,
                "reward_contact": -contact_cost,
                "reward_survive": healthy_reward,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(
                    xy_position_after,
                    ord=2,
                ),
                "x_velocity": x_velocity,
                "y_velocity": y_velocity,
                "forward_reward": forward_reward,
            }

            return observation, reward, done, info

        # Keep same gravity every step.
        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        xy_position_before = self.get_body_com("torso")[:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_position_after = self.get_body_com("torso")[:2].copy()

        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        truncated = self._elapsed_steps >= self.max_steps
        terminated = self.terminated

        observation = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,

            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(
                xy_position_after,
                ord=2,
            ),

            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,

            "gravity": float(self.model.opt.gravity[self._grav_axis]),
            "base_gravity": float(self._base_grav),
            "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
            "sigma_gravity": float(self.sigma_gravity),
        }

        return observation, reward, cost, truncated, terminated, info

class AntCostDampingPerturbedTest(AntCost):
    """
    Test-time perturbed AntCost using fixed joint damping perturbation.

    Samples one damping multiplier once during __init__ and keeps it fixed
    for all episodes and all steps.

    If shared_damping_multiplier is given, uses that exact multiplier.
    """

    OBS_DIM = 113
    max_steps = 500

    def __init__(
        self,
        healthy_reward=1.0,
        terminate_when_unhealthy=False,
        xml_file=ABS_PATH + "/env_configs/ant_circle.xml",
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        sigma_damping: float = 0.0,
        max_steps: int = 500,
        shared_damping_multiplier=None,
        seed=None,
    ):
        # ------------------------------------------------------------
        # Old Gym MujocoEnv may call self.step(action) inside __init__.
        # During that phase, return old-style 4-tuple from step.
        # ------------------------------------------------------------
        self._mujoco_initializing = True

        self._elapsed_steps = 0
        self.sigma_damping = float(sigma_damping)
        self.max_steps = int(max_steps)

        self.shared_damping_multiplier = shared_damping_multiplier
        self.fixed_damping_multiplier = 1.0

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random(None)

        super(AntCostDampingPerturbedTest, self).__init__(
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            xml_file=xml_file,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )

        # ------------------------------------------------------------
        # Store nominal damping after MuJoCo model exists.
        # ------------------------------------------------------------
        self._base_dof_damping = self.model.dof_damping.copy()

        # Sample and apply initial damping perturbation.
        # A new one will be sampled again at every reset().
        self._sample_and_set_damping()


        self._mujoco_initializing = False

        obs_high = np.inf * np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        print(
            f"[AntCostDampingPerturbedTest] "
            f"sigma_damping={self.sigma_damping}, "
            f"fixed_damping_multiplier={self.fixed_damping_multiplier}"
        )

    def reset(self, seed=None, **kwargs):
        """
        Return (obs, info) tuple expected by the RCRL/eval loop.
        Keep the same fixed damping every episode.
        """

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        obs = super().reset(seed=seed, **kwargs)

        if isinstance(obs, tuple):
            observation = obs[0]
            info = obs[1] if len(obs) > 1 else {}
        else:
            observation = obs
            info = {}

        self._elapsed_steps = 0

        # Sample new damping perturbation every episode.
        self._sample_and_set_damping()


        info.update(
            {
                "sigma_damping": float(self.sigma_damping),
                "fixed_damping_multiplier": float(self.fixed_damping_multiplier),
                "mean_base_dof_damping": float(np.mean(self._base_dof_damping)),
                "mean_fixed_dof_damping": float(np.mean(self.fixed_dof_damping)),
            }
        )

        return observation, info

    def _sample_and_set_damping(self):
        """
        Sample a new damping multiplier and apply it to MuJoCo dof damping.

        If shared_damping_multiplier is provided, use that value instead of sampling.
        """

        if self.shared_damping_multiplier is None:
            if self.sigma_damping > 0.0:
                self.fixed_damping_multiplier = float(
                    np.clip(
                        self.np_random.normal(1.0, self.sigma_damping),
                        0.05,
                        5.0,
                    )
                )
            else:
                self.fixed_damping_multiplier = 1.0
        else:
            self.fixed_damping_multiplier = float(self.shared_damping_multiplier)

        self.fixed_dof_damping = (
            self._base_dof_damping * self.fixed_damping_multiplier
        )

        self.model.dof_damping[:] = self.fixed_dof_damping


    def step(self, action):
        # ------------------------------------------------------------
        # During MuJoCo init, return old-style 4-tuple.
        # ------------------------------------------------------------
        if getattr(self, "_mujoco_initializing", False):
            xy_position_before = self.get_body_com("torso")[:2].copy()

            self.do_simulation(action, self.frame_skip)

            xy_position_after = self.get_body_com("torso")[:2].copy()

            xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
            x_velocity, y_velocity = xy_velocity

            ctrl_cost = self.control_cost(action)
            contact_cost = self.contact_cost

            forward_reward = x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward + healthy_reward
            costs = ctrl_cost + contact_cost
            reward = rewards - costs

            observation = self._get_obs()
            done = False

            info = {
                "reward_forward": forward_reward,
                "reward_ctrl": -ctrl_cost,
                "reward_contact": -contact_cost,
                "reward_survive": healthy_reward,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(
                    xy_position_after,
                    ord=2,
                ),
                "x_velocity": x_velocity,
                "y_velocity": y_velocity,
                "forward_reward": forward_reward,
            }

            return observation, reward, done, info

        # Keep same damping every step.
        self.model.dof_damping[:] = self.fixed_dof_damping

        xy_position_before = self.get_body_com("torso")[:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_position_after = self.get_body_com("torso")[:2].copy()

        self._elapsed_steps += 1

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        cost = float(
            np.maximum(
                np.max(np.abs(action)) - ACTION_TORQUE_THRESHOLD,
                0.0,
            )
        )

        truncated = self._elapsed_steps >= self.max_steps
        terminated = self.terminated

        observation = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,

            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(
                xy_position_after,
                ord=2,
            ),

            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,

            "sigma_damping": float(self.sigma_damping),
            "fixed_damping_multiplier": float(self.fixed_damping_multiplier),
            "mean_base_dof_damping": float(np.mean(self._base_dof_damping)),
            "mean_fixed_dof_damping": float(np.mean(self.fixed_dof_damping)),
        }

        return observation, reward, cost, truncated, terminated, info


class AntCostTest(AntCost):
    def step(self, action):
        #shilpa robustness
        self.model.dof_damping[:] += np.random.normal(0, 0.5)
        self.model.dof_damping[:] = np.clip(self.model.dof_damping, 0.0, 10.0)

        # self.model.geom_friction[:] += np.random.normal(0, 0.05)
        # self.model.geom_friction[:] = np.clip(self.model.geom_friction, 0.0, 5.0)

        observation, reward, done, info = super().step(action)
        #if observation[0] < -3 or observation[0] > 3:
        if observation[0] < -3:
            done = True
            reward = 0
        return observation, reward, done, info


class AntCostBroken(AntCost):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


class AntCostBrokenTest(AntCostTest):
    def step(self, action):
        action[4:] = 0
        return super().step(action)


