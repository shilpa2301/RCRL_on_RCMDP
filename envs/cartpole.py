import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch

class CartPoleCostEnv(gym.Env):

    def __init__(self):

        # Observation: [cart position, cart velocity, pole angle, pole angular velocity]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # Continuous action force
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32
        )

        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.tau = 0.02

        self.state = None
        self.steps = 0
        self.max_episode_steps = 500

        # Define the maximum possible cost for normalization
        # self.max_cost = 2.4 + 10.0  # Max cart position + penalty

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0

        return self.state

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state

        force = float(action)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        self.steps += 1

        # reward (same idea as hopper)
        reward = 1.0

        # cost = distance from center
        cost = self.compute_cost(x, theta)
       
        # Normalize cost between 0 and 1
        # normalized_cost = cost / self.max_cost
        # cost = normalized_cost
        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        info = {
            "x_position": x
        }

        return self.state, reward, cost, done, info

    def compute_cost(self, x, theta):
        """
        Compute the cost based on the current state.

        Args:
            x (float): The cart's position (distance from the center).
            theta (float): The pole's angle (in radians).

        Returns:
            float: The computed cost.
        """
        # Cost is the absolute distance of the cart from the center (x)
        if abs(x) > 1:
            cost = abs(x)
        else:
            cost = 0

        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        if done and self.steps < 450:
            cost += 10.0   # penalty value (tunable)

        # Normalize cost between 0 and 1
        # normalized_cost = cost / self.max_cost
        # cost = normalized_cost

        return cost

    def simulate_next_state(self, state, action):
        """
        Simulate the next state based on the current state and action, without modifying the environment's state.

        Args:
            state (np.array): Current state [x, x_dot, theta, theta_dot].
            action (float): Action to apply.

        Returns:
            torch.Tensor: Simulated next state.
        """
        x, x_dot, theta, theta_dot = state
        force = float(action)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Compute the next state based on the current state and action
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        return torch.tensor(next_state, dtype=torch.float32)

class CartPolePerturbedEnv(gym.Env):
    def __init__(self, gravity_perturbation_std=0.5):
        # Observation: [cart position, cart velocity, pole angle, pole angular velocity]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        # Continuous action force
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32
        )

        # Physics parameters
        self.default_gravity = 9.8
        self.gravity = 9.8  # Perturbed gravity (will change at each step)
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.tau = 0.02

        # Perturbation parameters
        self.theta_perturbation_std = 0.05  # Noise for theta
        # self.gravity_perturbation_std = 6.0 # #0.5 #0.5  # Noise for gravity
        self.gravity_perturbation_std = gravity_perturbation_std # #0.5 #0.5  # Noise for gravity

        self.state = None
        self.steps = 0
        self.max_episode_steps = 500

        # Define the maximum possible cost for normalization
        # self.max_cost = 2.4 + 10.0  # Max cart position + penalty

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for reproducibility.
            options (dict): Additional options for resetting the environment.

        Returns:
            np.array: Initial state of the environment.
        """
        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0

        # Reset gravity to its default value
        self.gravity = self.default_gravity

        return self.state

    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (float): The action to apply.

        Returns:
            tuple: A tuple containing:
                - state (np.array): The next state.
                - reward (float): The reward for the current step.
                - cost (float): The cost for the current step.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information about the step.
        """
        x, x_dot, theta, theta_dot = self.state

        # Apply dynamic perturbations
        # theta += np.random.normal(0, self.theta_perturbation_std)  # Add noise to theta
        self.gravity = self.default_gravity + np.abs(np.random.normal(0, self.gravity_perturbation_std))  # Add noise to gravity
        # print("perturbed gravity=", self.gravity)

        force = float(action)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        # Use perturbed gravity
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        self.steps += 1

        # reward (same idea as hopper)
        reward = 1.0

        # cost = distance from center
        cost = self.compute_cost(x, theta)
        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        info = {
            "x_position": x,
            "perturbed_gravity": self.gravity,
            "perturbed_theta": theta
        }

        return self.state, reward, cost, done, info

    def compute_cost(self, x, theta):
        """
        Compute the cost based on the current state.

        Args:
            x (float): The cart's position (distance from the center).
            theta (float): The pole's angle (in radians).

        Returns:
            float: The computed cost.
        """
        # Cost is the absolute distance of the cart from the center (x)
        if abs(x) > 1:
            cost = abs(x)
        else:
            cost = 0

        done = (
            abs(x) > 2.4
            or abs(theta) > 12 * np.pi / 180
            or self.steps >= self.max_episode_steps
        )

        if done and self.steps < 450:
            cost += 10.0   # penalty value (tunable)

        return cost

    def compute_cost_2(self, x, theta):
        """
        Compute the combined cost based on the cart's position and pole's angle.

        Args:
            x (float): The cart's position (distance from the center).
            theta (float): The pole's angle (in radians).

        Returns:
            float: The computed combined cost.
        """
        # Define thresholds and warnings for position and angle
        self.x_warning = 1.0  # Warning threshold for cart position
        self.x_threshold = 2.4  # Termination threshold for cart position
        self.theta_warning = 6 * np.pi / 180  # Warning threshold for pole angle (in radians)
        self.theta_threshold_radians = 12 * np.pi / 180  # Termination threshold for pole angle (in radians)

        # Compute the peak margin cost
        peak_margin_cost = min(
            1.0,
            max(
                max(0.0, (abs(x) - self.x_warning) / (self.x_threshold - self.x_warning)),
                max(0.0, (abs(theta) - self.theta_warning) / (self.theta_threshold_radians - self.theta_warning)),
            ),
        )

        return peak_margin_cost


    def simulate_next_state(self, state, action):
        """
        Simulate the next state based on the current state and action, without modifying the environment's state.

        Args:
            state (np.array): Current state [x, x_dot, theta, theta_dot].
            action (float): Action to apply.

        Returns:
            torch.Tensor: Simulated next state.
        """
        x, x_dot, theta, theta_dot = state
        force = float(action)

        # Apply dynamic perturbations
        # theta += np.random.normal(0, self.theta_perturbation_std)  # Add noise to theta
        # perturbed_gravity = self.default_gravity
        perturbed_gravity = self.default_gravity + np.random.normal(0, self.gravity_perturbation_std)  # Perturb gravity

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass

        # Use perturbed gravity
        thetaacc = (
            perturbed_gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Compute the next state based on the current state and action
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        return torch.tensor(next_state, dtype=torch.float32)

