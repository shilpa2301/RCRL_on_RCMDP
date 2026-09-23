import numpy as np

from envs.half_cheetah_combined_max_cost import HalfCheetahCombinedMaxCost


MULTI_CONSTRAINT_COST_VERSION = "multi_constraint_pitch_joint_v1"
COST_NAMES = ("pitch", "joint_speed")


class HalfCheetahMultiConstraintCost(HalfCheetahCombinedMaxCost):
    cost_dim = 2

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

        cost = np.asarray(
            [info["pitch_cost"], info["joint_speed_cost"]],
            dtype=np.float32,
        )
        info = dict(info)
        info["cost_vector"] = cost.copy()
        info["cost_names"] = COST_NAMES
        info["cost_version"] = MULTI_CONSTRAINT_COST_VERSION

        #shilpa windows only
        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, truncated or terminated, info
                
        return observation, reward, cost, truncated, terminated, info


PERTURBED_MULTI_CONSTRAINT_COST_VERSION = "multi_constraint_pitch_joint_gravity_perturbed_v1"


class HalfCheetahMultiConstraintCostPerturbed(HalfCheetahMultiConstraintCost):
    """
    Training-time gravity-perturbed version of HalfCheetahMultiConstraintCost.

    Inherits cost calculation from HalfCheetahMultiConstraintCost.

    Cost vector:
        cost[0] = pitch_cost
        cost[1] = joint_speed_cost

    Gravity:
        gravity_z = base_gravity + Normal(0, sigma_gravity)

    By default, gravity is perturbed every step.
    If perturb_each_step=False, gravity is perturbed once per episode.
    """

    cost_dim = 2

    def __init__(
        self,
        sigma_gravity: float = 0.7,
        max_steps: int = 1000,
        perturb_each_step: bool = True,
    ):
        self.sigma_gravity = float(sigma_gravity)
        self.max_steps = int(max_steps)
        self.perturb_each_step = bool(perturb_each_step)

        self._elapsed_steps = 0

        self._grav_axis = 2
        self._base_grav = -9.81

        self.current_gravity_perturbation = 0.0
        self.current_gravity_value = -9.81

        super().__init__()

        # MuJoCo model is available after super().__init__()
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])
        self.current_gravity_value = float(self._base_grav)

    def _sample_gravity_perturbation(self):
        if self.sigma_gravity > 0.0:
            return float(np.random.normal(0.0, self.sigma_gravity))
        return 0.0

    def _apply_gravity_perturbation(self, perturbation):
        self.current_gravity_perturbation = float(perturbation)
        self.current_gravity_value = float(
            self._base_grav + self.current_gravity_perturbation
        )
        self.model.opt.gravity[self._grav_axis] = self.current_gravity_value

    def reset(self, seed=None, **kwargs):
        """
        Reset environment and gravity.

        Returns:
            obs, info
        """

        if seed is not None:
            try:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            except Exception:
                pass

        try:
            result = super().reset(seed=seed, **kwargs)
        except TypeError:
            result = super().reset(**kwargs)

        if isinstance(result, tuple):
            obs = result[0]
            info = result[1] if len(result) > 1 else {}
        else:
            obs = result
            info = {}

        self._elapsed_steps = 0

        if self.perturb_each_step:
            # Start episode from nominal gravity.
            self._apply_gravity_perturbation(0.0)
        else:
            # Sample one gravity perturbation per episode.
            gravity_perturbation = self._sample_gravity_perturbation()
            self._apply_gravity_perturbation(gravity_perturbation)

        info = dict(info)
        info.update(
            {
                "gravity": float(self.model.opt.gravity[self._grav_axis]),
                "base_gravity": float(self._base_grav),
                "gravity_perturbation": float(self.current_gravity_perturbation),
                "sigma_gravity": float(self.sigma_gravity),
                "perturb_each_step": bool(self.perturb_each_step),
                "perturbed_env": True,
                "cost_version": PERTURBED_MULTI_CONSTRAINT_COST_VERSION,
            }
        )

        return obs, info

    def step(self, action):
        """
        Apply gravity perturbation, then call parent multi-constraint step.
        """

        # During MuJoCo initialization, old Gym calls self.step().
        # Do not perturb gravity during this phase.
        if not getattr(self, "_mujoco_initializing", False):
            if self.perturb_each_step:
                gravity_perturbation = self._sample_gravity_perturbation()
                self._apply_gravity_perturbation(gravity_perturbation)

        step_result = super().step(action)

        if len(step_result) == 4:
            observation, reward, done, info = step_result
            truncated = done
            terminated = False
            cost = np.asarray(
                info.get("cost_vector", np.zeros(self.cost_dim)),
                dtype=np.float32,
            ).reshape(-1)

        elif len(step_result) == 6:
            observation, reward, cost, truncated, terminated, info = step_result
            cost = np.asarray(cost, dtype=np.float32).reshape(-1)

        else:
            raise RuntimeError(
                f"Unexpected step return length in HalfCheetahMultiConstraintCostPerturbed: "
                f"{len(step_result)}"
            )

        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_steps:
            truncated = True

        info = dict(info)
        info.update(
            {
                "gravity": float(self.model.opt.gravity[self._grav_axis]),
                "base_gravity": float(self._base_grav),
                "gravity_perturbation": float(self.current_gravity_perturbation),
                "sigma_gravity": float(self.sigma_gravity),
                "perturb_each_step": bool(self.perturb_each_step),
                "perturbed_env": True,
                "cost_version": PERTURBED_MULTI_CONSTRAINT_COST_VERSION,
            }
        )

        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, bool(truncated or terminated), info

        return observation, reward, cost, truncated, terminated, info


class HalfCheetahMultiConstraintCostPerturbedTest(HalfCheetahMultiConstraintCost):
    """
    Test-time gravity-perturbed version of HalfCheetahMultiConstraintCost.

    Samples one gravity perturbation once during __init__, then keeps it fixed
    for all episodes and all steps.

    If shared_gravity_perturbation is provided, that exact perturbation is used.
    """

    cost_dim = 2

    def __init__(
        self,
        sigma_gravity: float = 0.7,
        max_steps: int = 1000,
        shared_gravity_perturbation=None,
        seed=None,
        render_mode="human"  #None
    ):
        self.sigma_gravity = float(sigma_gravity)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self._elapsed_steps = 0

        self._grav_axis = 2
        self._base_grav = -9.81

        self.shared_gravity_perturbation = shared_gravity_perturbation
        self.fixed_gravity_perturbation = 0.0
        self.fixed_gravity_value = -9.81

        if seed is not None:
            try:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            except Exception:
                pass

        try:
            super().__init__(render_mode=render_mode)
        except TypeError:
            super().__init__()

        # MuJoCo model is available after super().__init__()
        self._base_grav = float(self.model.opt.gravity[self._grav_axis])

        if self.shared_gravity_perturbation is None:
            if self.sigma_gravity > 0.0:
                self.fixed_gravity_perturbation = float(
                    np.random.normal(0.0, self.sigma_gravity)
                )
            else:
                self.fixed_gravity_perturbation = 0.0
        else:
            self.fixed_gravity_perturbation = float(
                self.shared_gravity_perturbation
            )

        self.fixed_gravity_value = float(
            self._base_grav + self.fixed_gravity_perturbation
        )

        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        print(
            f"[HalfCheetahMultiConstraintCostPerturbedTest] "
            f"base_gravity={self._base_grav}, "
            f"fixed_perturbation={self.fixed_gravity_perturbation}, "
            f"fixed_gravity={self.fixed_gravity_value}"
        )

    def render(self, mode="human"):
        """
        Render environment safely across old Gym / mujoco_py and newer Gym APIs.
        """
        try:
            if self.render_mode is not None:
                return super().render()
            else:
                return super().render(mode=mode)
        except TypeError:
            return super().render()

    def reset(self, seed=None, **kwargs):
        """
        Reset environment while keeping the same fixed gravity.

        Returns:
            obs, info
        """

        if seed is not None:
            try:
                self.np_random, _ = gym.utils.seeding.np_random(seed)
            except Exception:
                pass

        try:
            result = super().reset(seed=seed, **kwargs)
        except TypeError:
            result = super().reset(**kwargs)

        if isinstance(result, tuple):
            obs = result[0]
            info = result[1] if len(result) > 1 else {}
        else:
            obs = result
            info = {}

        self._elapsed_steps = 0

        # Keep fixed gravity for every evaluation episode.
        self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        info = dict(info)
        info.update(
            {
                "gravity": float(self.model.opt.gravity[self._grav_axis]),
                "base_gravity": float(self._base_grav),
                "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
                "fixed_gravity_value": float(self.fixed_gravity_value),
                "sigma_gravity": float(self.sigma_gravity),
                "perturbed_env": True,
                "fixed_perturbation_eval": True,
                "cost_version": PERTURBED_MULTI_CONSTRAINT_COST_VERSION,
            }
        )

        return obs, info

    def step(self, action):
        """
        Keep fixed gravity, then call parent multi-constraint step.
        """

        # During MuJoCo initialization, old Gym calls self.step().
        # Do not force perturbation during this phase.
        if not getattr(self, "_mujoco_initializing", False):
            self.model.opt.gravity[self._grav_axis] = self.fixed_gravity_value

        step_result = super().step(action)

        if len(step_result) == 4:
            observation, reward, done, info = step_result
            truncated = done
            terminated = False
            cost = np.asarray(
                info.get("cost_vector", np.zeros(self.cost_dim)),
                dtype=np.float32,
            ).reshape(-1)

        elif len(step_result) == 6:
            observation, reward, cost, truncated, terminated, info = step_result
            cost = np.asarray(cost, dtype=np.float32).reshape(-1)

        else:
            raise RuntimeError(
                f"Unexpected step return length in HalfCheetahMultiConstraintCostPerturbedTest: "
                f"{len(step_result)}"
            )

        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_steps:
            truncated = True

        info = dict(info)
        info.update(
            {
                "gravity": float(self.model.opt.gravity[self._grav_axis]),
                "base_gravity": float(self._base_grav),
                "fixed_gravity_perturbation": float(self.fixed_gravity_perturbation),
                "fixed_gravity_value": float(self.fixed_gravity_value),
                "sigma_gravity": float(self.sigma_gravity),
                "perturbed_env": True,
                "fixed_perturbation_eval": True,
                "cost_version": PERTURBED_MULTI_CONSTRAINT_COST_VERSION,
            }
        )

        if getattr(self, "_mujoco_initializing", False):
            return observation, reward, bool(truncated or terminated), info

        return observation, reward, cost, truncated, terminated, info