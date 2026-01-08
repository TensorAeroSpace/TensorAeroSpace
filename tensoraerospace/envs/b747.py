"""Boeing 747 Gymnasium environments.

This module provides Gymnasium-compatible environments based on a B747
longitudinal dynamics model, including a classic linear environment and an
improved normalized variant.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import LongitudinalB747


class LinearLongitudinalB747(gym.Env):
    """Simulation of LongitudinalB747 control object in Gym for training.

    Args:
        initial_state (np.ndarray): Initial state.
        reference_signal (np.ndarray): Reference signal.
        number_time_steps (int): Number of simulation steps.
        tracking_states (list[str] | None): Tracked states.
        state_space (list[str] | None): State space.
        control_space (list[str] | None): Control space.
        output_space (list[str] | None): Full output space (including noise).
        reward_func (Callable | None): Reward function (WIP status).
        use_reward (bool): Whether to use reward.
        dt (float): Discretization frequency.

    Notes:
        - Action units expected by the environment are degrees (deg).
        - Actions are converted to radians (rad) before being passed to
          the underlying model.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        tracking_states: Optional[list[str]] = None,
        state_space: Optional[list[str]] = None,
        control_space: Optional[list[str]] = None,
        output_space: Optional[list[str]] = None,
        reward_func: Optional[Callable] = None,
        use_reward: bool = True,
        dt: float = 0.01,
    ) -> None:
        """Initialize LinearLongitudinalB747 environment.

        Args:
            initial_state (np.ndarray): Initial state.
            reference_signal (np.ndarray): Reference signal.
            number_time_steps (int): Number of simulation steps.
            tracking_states (list[str] | None): Tracked states. Defaults to
                ["theta", "q"].
            state_space (list[str] | None): State space. Defaults to
                ["theta", "q"].
            control_space (list[str] | None): Control space. Defaults to
                ["stab"].
            output_space (list[str] | None): Full output space. Defaults to
                ["theta", "q"].
            reward_func (Callable | None): Reward function. Defaults to None.
            use_reward (bool): Whether to use reward. Defaults to True.
            dt (float): Discretization frequency. Defaults to 0.01.
        """
        self.max_action_value = 25.0
        self.dt = dt
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.selected_state_output = (
            output_space
            if output_space is not None
            else [
                "theta",
                "q",
            ]
        )
        self.tracking_states = (
            tracking_states
            if tracking_states is not None
            else [
                "theta",
            ]
        )
        self.state_space = (
            state_space
            if state_space is not None
            else [
                "theta",
                "q",
            ]
        )
        self.control_space = (
            control_space
            if control_space is not None
            else [
                "stab",
            ]
        )
        self.output_space = (
            output_space
            if output_space is not None
            else [
                "theta",
                "q",
            ]
        )
        self.use_reward = use_reward
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = LongitudinalB747(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-self.max_action_value,
            high=self.max_action_value,
            shape=(len(self.control_space),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(len(self.state_space),),
            dtype=np.float32,
        )

        self.ref_signal = reference_signal
        self.model.initialise_system(
            x0=initial_state, number_time_steps=number_time_steps
        )
        self.number_time_steps = number_time_steps
        self.current_step = 0
        self.done = False

    @staticmethod
    def reward(
        state: np.ndarray,
        ref_signal: np.ndarray,
        ts: int,
        action: Optional[np.ndarray] = None,
    ) -> float:
        """Control evaluation.

        Args:
            state (np.ndarray): Current state.
            ref_signal (np.ndarray): Reference state.
            ts (int): Time step.

        Returns:
            float: Control evaluation (negative MSE).
        """
        # Negative mean squared error across all tracked states
        # (higher is better)
        if ref_signal.ndim == 2 and ref_signal.shape[1] > ts:
            ref_at_ts = ref_signal[:, ts].flatten()
        else:
            ref_at_ts = ref_signal.flatten()
        error = np.mean((state.flatten() - ref_at_ts) ** 2)
        return float(-error)

    def _get_info(self):
        """Return additional information about environment state.

        Returns:
            dict: Empty dictionary with additional information.
        """
        return {}

    def step(self, action: np.ndarray):
        """Execute simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators
                in degrees.

        Returns:
            next_state (np.ndarray): Next state of control object.
            reward (np.ndarray): Evaluation of control algorithm actions.
            done (bool): Simulation status, completed or not.
            logging (any): Additional information (not used).
        """
        # Ensure action is a 1D numpy array
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Clamp all control inputs to [-25, 25]
        action = np.clip(action, -self.max_action_value, self.max_action_value)
        # Convert degrees to radians for the model
        action_rad = np.deg2rad(action)
        self.current_step += 1
        # Get next state from the model (SI units: u,w in m/s; q,theta in rad)
        raw_next_state = np.array(self.model.run_step(action_rad)).reshape(-1)
        # Convert only angular states to degrees for observations (leave linear
        # states in SI)
        next_state = raw_next_state.copy()
        try:
            if "q" in self.output_space:
                qi = self.output_space.index("q")
                next_state[qi] = np.rad2deg(next_state[qi])
            if "theta" in self.output_space:
                ti = self.output_space.index("theta")
                next_state[ti] = np.rad2deg(next_state[ti])
        except (ValueError, IndexError):
            # Fallback: assume [u, w, q, theta] order
            if next_state.shape[0] >= 3:
                next_state[2] = np.rad2deg(next_state[2])
            if next_state.shape[0] >= 4:
                next_state[3] = np.rad2deg(next_state[3])
        reward = 1
        if self.use_reward:
            try:
                reward = self.reward_func(
                    next_state,
                    self.reference_signal,
                    self.current_step,
                    action=np.array(action),
                )
            except TypeError:
                reward = self.reward_func(
                    next_state,
                    self.reference_signal,
                    self.current_step,
                )
        self.done = self.current_step >= self.number_time_steps - 2

        return (
            np.array(next_state, dtype=np.float32).reshape(-1, 1),
            reward,
            self.done,
            False,
            {
                "action": action,
                "action_rad": action_rad,
            },
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional initialization options.

        Returns:
            tuple: Observation array and info dictionary.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.model = LongitudinalB747(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.ref_signal = self.reference_signal
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )
        # Build initial observation with angular components in degrees
        init_state = np.array(self.initial_state, dtype=np.float32).reshape(-1)
        next_state = init_state[self.model.selected_state_index].astype(float)
        try:
            if "q" in self.output_space:
                qi = self.output_space.index("q")
                next_state[qi] = np.rad2deg(next_state[qi])
            if "theta" in self.output_space:
                ti = self.output_space.index("theta")
                next_state[ti] = np.rad2deg(next_state[ti])
        except (ValueError, IndexError):
            if next_state.shape[0] >= 3:
                next_state[2] = np.rad2deg(next_state[2])
            if next_state.shape[0] >= 4:
                next_state[3] = np.rad2deg(next_state[3])
        observation = next_state.astype(np.float32).reshape(-1, 1)
        return observation, self._get_info()

    def render(self):
        """Visual display of actions in the environment.

        Note:
            Work in progress (WIP status).

        Raises:
            NotImplementedError: Rendering is not yet implemented.
        """
        raise NotImplementedError()


class ImprovedB747Env(gym.Env):
    """Improved Boeing 747 longitudinal channel control environment.

    Features normalized action/observation spaces and enhanced reward function.

    Key features:
        - Normalized action and observation spaces in range [-1, 1]
        - Extended observation: [pitch_error, pitch_rate, pitch,
          previous_action]
        - Comprehensive reward function: accuracy, stability, energy cost,
          smoothness and elevator jitter suppression
        - Realistic termination conditions based on exceeding
          flight envelope

    Attributes:
        action_space (spaces.Box): Normalized action space [-1, 1].
        observation_space (spaces.Box): Normalized observation space [-1, 1].
        max_pitch_rad (float): Maximum pitch angle in radians.
        max_pitch_rate_rad_s (float): Maximum pitch rate in rad/s.
        max_stabilizer_angle_deg (float): Maximum stabilizer angle in degrees.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_elevator_deg: float = 0.0,
        use_initial_action_on_first_step: bool = True,
        reward_mode: str = "step_response",
        survival_bonus: float = 0.0,
        completion_bonus: float = 0.0,
        early_termination_penalty: float = 0.0,
        early_termination_penalty_per_step: float = 0.0,
        include_reference_in_obs: bool = False,
    ):
        """Initialize ImprovedB747Env environment.

        Args:
            initial_state (np.ndarray): Initial state vector [u, w, q, theta]
                in SI units (m/s, m/s, rad, rad).
            reference_signal (np.ndarray): Reference pitch angle
                trajectory in radians. Shape: (1, number_time_steps).
            number_time_steps (int): Total number of simulation time
                steps.
            dt (float): Simulation time step in seconds. Defaults to 0.01.
            initial_elevator_deg (float): Initial elevator deflection
                in degrees. Used to avoid control jumps at the start.
                Defaults to 0.0.
            use_initial_action_on_first_step (bool): If True, applies the
                initial_elevator_deg value on the first step instead of
                the agent's action to ensure smooth control
                initialization. Defaults to True.
            reward_mode (str): Reward calculation mode. Options:
                - "tracking": Universal reward for any signal type.
                  Uses only base quadratic cost (pitch error, rate error,
                  control effort, smoothness, jerk) plus progress shaping.
                  Step-response metrics are computed but NOT included
                  in reward — only returned in info dict for evaluation.
                - "step_response": Full reward with step-specific penalties
                  (overshoot, settling time, oscillations). Best for
                  training on step references. Default.
            survival_bonus (float): Additive reward per step **only if**
                the episode is not terminated. This can encourage the agent
                to keep the system within constraints for the full horizon.
                Defaults to 0.0 (disabled).
            completion_bonus (float): Additive reward on the final step when
                the episode ends by time limit (``truncated=True``) and was
                **not** terminated. Defaults to 0.0 (disabled).
            early_termination_penalty (float): Extra penalty added when the
                episode terminates early (in addition to the base -100.0).
                Defaults to 0.0 (disabled).
            early_termination_penalty_per_step (float): Extra penalty per
                remaining time step when the episode terminates early.
                This strongly discourages the agent from \"ending\" the episode
                quickly to avoid accumulating negative rewards.
                Defaults to 0.0 (disabled).
        """
        if reward_mode not in ("tracking", "step_response"):
            raise ValueError(
                f"reward_mode must be 'tracking' or 'step_response', got {reward_mode!r}"
            )
        super().__init__()

        # Normalization parameters and physical constraints
        self.max_pitch_rad = np.deg2rad(20.0)  # |theta| <= 20 deg
        self.max_pitch_rate_rad_s = np.deg2rad(5.0)  # |q| <= 5 deg/s
        self.max_stabilizer_angle_deg = 25.0  # |ele| <= 25 deg

        # Gymnasium spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.include_reference_in_obs = bool(include_reference_in_obs)
        obs_dim = 6 if self.include_reference_in_obs else 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Simulation parameters
        self.dt = dt
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        self.reference_signal = np.array(reference_signal, dtype=float)
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0
        self.state = np.array(self.initial_state, dtype=float).reshape(
            -1
        )  # Full state vector [u, w, q, theta] in SI units
        # Initial elevator value, degrees -> normalized value
        self.initial_elevator_deg = float(initial_elevator_deg)
        self.initial_action_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_stabilizer_angle_deg,
                -1.0,
                1.0,
            )
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = 0.0
        self._last_reward = 0.0
        # Reward mode: "tracking" (universal) or "step_response" (with step-specific penalties)
        self.reward_mode = str(reward_mode)
        self.survival_bonus = float(survival_bonus)
        self.completion_bonus = float(completion_bonus)
        self.early_termination_penalty = float(early_termination_penalty)
        self.early_termination_penalty_per_step = float(
            early_termination_penalty_per_step
        )
        if self.survival_bonus < 0:
            raise ValueError("survival_bonus must be >= 0")
        if self.completion_bonus < 0:
            raise ValueError("completion_bonus must be >= 0")
        if self.early_termination_penalty < 0:
            raise ValueError("early_termination_penalty must be >= 0")
        if self.early_termination_penalty_per_step < 0:
            raise ValueError("early_termination_penalty_per_step must be >= 0")
        # Reward scale for Q-value range stability
        self.reward_scale = 0.1

        # Cost function weights (tune for your task)
        self.w_pitch = 5.0  # Pitch angle accuracy (increased)
        self.w_q = 0.2  # Angular velocity damping (decreased)
        self.w_cross = 0.0  # Disable cross-term
        self.w_action = 0.003  # Energy cost (|u|)
        self.w_smooth = 0.01  # Smoothness (|Δu|)
        self.w_jerk = 0.001  # Jitter suppression (|Δ²u|)

        # ------------------------------------------------------------------
        # Step-response reward shaping (settling time, steady-state error,
        # oscillations, overshoot <= 10%).
        #
        # The base LQR-like cost remains, but we add:
        # - time-in-band incentive (short transient)
        # - linear (absolute) error penalty (reduce steady-state error)
        # - oscillation penalty (repeat sign-changes of error)
        # - overshoot penalty above 10% of the commanded step
        # ------------------------------------------------------------------
        self.k_progress = 0.05  # error-reduction shaping (dense)

        # Reference change detection (treat as a new "step" segment)
        self.ref_change_threshold_rad = float(np.deg2rad(0.1))
        self.min_step_amp_rad = float(
            np.deg2rad(0.5)
        )  # ignore tiny steps for overshoot metrics

        # Settling band: max(1% of step amplitude, 0.05 deg)
        # (tighter band -> near-zero steady-state error)
        self.settle_band_ratio = 0.01
        self.settle_band_min_rad = float(np.deg2rad(0.05))
        self.q_settle_rad_s = float(
            np.deg2rad(0.25)
        )  # small pitch-rate requirement for "no oscillations"
        # Require staying inside the band for ~1.0 s
        self.settle_steps_required = int(max(1, np.ceil(1.0 / float(self.dt))))
        self.settle_time_target_s = 1.5  # rewarded if settled faster than this

        # Overshoot constraint (<= 5% of commanded step amplitude)
        self.overshoot_limit_ratio = 0.05

        # Clip unrealistic reference derivatives (e.g. step jumps) used for damping term
        # in the base LQR-like cost. Keeps training stable for step references.
        self.ref_theta_dot_clip_rad_s = float(self.max_pitch_rate_rad_s)

        # Additional shaping weights (added to cost)
        self.w_abs = 0.6  # |e_theta| (helps reduce steady-state error)
        self.w_time = 0.6  # penalty while outside settling band (reduces settling time)
        self.w_osc = 1.0  # penalty for repeated crossings (oscillations)
        self.w_overshoot = 300.0  # penalty for overshoot beyond limit
        self.w_settle_bonus = 4.0  # one-time bonus when settled, scaled by how fast

        # Internal shaping state (reset in reset())
        self._prev_e_theta = 0.0
        self._prev_e_q_rel = 0.0
        self._prev_target_theta = 0.0
        self._seg_start_step = 0
        self._seg_amp = 0.0
        self._seg_sign = 0.0
        self._seg_max_err_dir = 0.0
        self._settle_count = 0
        self._is_settled = False
        self._settle_time_s: Optional[float] = None
        self._prev_error_sign = 0
        self._sign_changes = 0

        # Store initialization arguments for serialization
        self.init_args = locals()

        # Model
        # Important: keep full state output to unambiguously address q/theta
        self.model = LongitudinalB747(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
        )
        self.model.initialise_system(
            x0=self.initial_state,
            number_time_steps=self.number_time_steps,
        )

        # Visualization parameters (lazy pygame initialization)
        self._pygame_initialized = False
        self._pygame_closed = False
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._pg: Any = None
        self._plane_img: Any = None
        self._plane_img_scaled: Any = None
        # Plot buffers (signal history)
        self._history_len = 600
        self._hist_theta_deg: list[float] = []
        self._hist_theta_target_deg: list[float] = []
        self._hist_elev_deg: list[float] = []

    # Helper indices for convenience
    @property
    def _idx_q(self) -> int:
        """Index of pitch rate q in state vector.

        Returns:
            int: Index 2 (model state order: [u, w, q, theta]).
        """
        return 2  # Model state order: [u, w, q, theta]

    @property
    def _idx_theta(self) -> int:
        """Index of pitch angle theta in state vector.

        Returns:
            int: Index 3 (model state order: [u, w, q, theta]).
        """
        return 3

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation.

        Returns:
            np.ndarray: dtype float32.
                If include_reference_in_obs=False (default), shape (4,):
                    [norm_pitch_error, norm_q, norm_theta, norm_prev_action]
                If include_reference_in_obs=True, shape (6,):
                    [..., norm_theta_ref, norm_theta_ref_dot]
        """
        theta = float(self.state[self._idx_theta])
        q = float(self.state[self._idx_q])

        # Target pitch value for current step (in rad)
        # Safe access to reference signal (last available index)
        idx = int(np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1))
        target_theta = float(self.reference_signal[0, idx])

        # 1) Pitch error (normalized)
        pitch_error = target_theta - theta
        norm_pitch_error = float(np.clip(pitch_error / self.max_pitch_rad, -1.0, 1.0))

        # 2) Pitch rate (normalized)
        norm_q = float(np.clip(q / self.max_pitch_rate_rad_s, -1.0, 1.0))

        # 3) Pitch angle (normalized)
        norm_theta = float(np.clip(theta / self.max_pitch_rad, -1.0, 1.0))

        # 4) Previous action (already in [-1, 1])
        norm_prev_action = float(self.previous_action)

        if not self.include_reference_in_obs:
            return np.array(
                [norm_pitch_error, norm_q, norm_theta, norm_prev_action],
                dtype=np.float32,
            )

        # Reference theta and reference theta-dot (normalized)
        idx_prev = int(
            np.clip(self.current_step - 1, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta_prev = float(self.reference_signal[0, idx_prev])
        ref_theta_dot = float((target_theta - target_theta_prev) / float(self.dt))
        ref_theta_dot = float(
            np.clip(
                ref_theta_dot,
                -float(self.ref_theta_dot_clip_rad_s),
                float(self.ref_theta_dot_clip_rad_s),
            )
        )
        norm_target_theta = float(
            np.clip(target_theta / float(self.max_pitch_rad), -1.0, 1.0)
        )
        norm_ref_theta_dot = float(
            np.clip(ref_theta_dot / float(self.max_pitch_rate_rad_s), -1.0, 1.0)
        )
        return np.array(
            [
                norm_pitch_error,
                norm_q,
                norm_theta,
                norm_prev_action,
                norm_target_theta,
                norm_ref_theta_dot,
            ],
            dtype=np.float32,
        )

    def get_init_args(self):
        """Get initialization arguments as a dictionary.

        Returns:
            dict: Dictionary of initialization arguments, excluding 'self'
                and '__class__'.
        """
        init_args = self.init_args.copy()
        # Remove reference to current object from arguments dict
        init_args.pop("self", None)
        # Remove reference to class from arguments dict
        init_args.pop("__class__", None)
        return init_args

    def reset(self, seed=None, options=None):
        """Reset environment to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional reset options.

        Returns:
            tuple: Initial observation and empty info dict.
        """
        super().reset(seed=seed)
        self.model.initialise_system(self.initial_state, self.number_time_steps)
        # Initial state as full vector [u, w, q, theta]
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        # Reset action history to specified initial elevator value
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0

        # Reset step-response tracking for shaped reward
        try:
            idx0 = int(np.clip(0, 0, self.reference_signal.shape[1] - 1))
            target0 = float(self.reference_signal[0, idx0])
        except Exception:  # noqa: BLE001
            ref_flat = np.asarray(self.reference_signal, dtype=float).reshape(-1)
            target0 = float(ref_flat[0]) if ref_flat.size > 0 else 0.0

        self._prev_e_theta = 0.0
        self._prev_e_q_rel = 0.0
        self._prev_target_theta = float(target0)
        self._seg_start_step = 0
        self._seg_amp = 0.0
        self._seg_sign = 0.0
        self._seg_max_err_dir = 0.0
        self._settle_count = 0
        self._is_settled = False
        self._settle_time_s = None
        self._prev_error_sign = 0
        self._sign_changes = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Execute one simulation step.

        Args:
            action (np.ndarray): Normalized action in range [-1, 1].

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        # Convert action to shape (1,) and clip to [-1, 1]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        # Scale from [-1, 1] -> degrees. On first step, can apply
        # fixed initial elevator value (to avoid jump).
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_action_deg = np.array([self.initial_elevator_deg], dtype=np.float32)
        else:
            scaled_action_deg = action * self.max_stabilizer_angle_deg
        scaled_action_rad = np.deg2rad(scaled_action_deg)

        # Simulation step
        self.state = self.model.run_step(scaled_action_rad).reshape(-1)
        self.current_step += 1

        # Reward calculation
        theta = float(self.state[self._idx_theta])
        q = float(self.state[self._idx_q])
        idx_safe = int(
            np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta = float(self.reference_signal[0, idx_safe])

        # Reference θ derivative for velocity damping
        if self.current_step > 0:
            idx_prev = int(
                np.clip(
                    self.current_step - 1,
                    0,
                    self.reference_signal.shape[1] - 1,
                )
            )
            ref_theta_prev = float(self.reference_signal[0, idx_prev])
        else:
            ref_theta_prev = target_theta
        ref_theta_dot = float((target_theta - ref_theta_prev) / self.dt)
        # For discontinuous references (steps), the raw derivative can be extremely
        # large and destabilize learning. Clip it to a physically meaningful range.
        ref_theta_dot = float(
            np.clip(
                ref_theta_dot,
                -float(self.ref_theta_dot_clip_rad_s),
                float(self.ref_theta_dot_clip_rad_s),
            )
        )

        # Base quadratic cost (LQR-like)
        e_theta = float((theta - target_theta) / self.max_pitch_rad)
        e_q_rel = float((q - ref_theta_dot) / self.max_pitch_rate_rad_s)
        # Normalized actually applied action
        u_applied_norm = float(
            np.asarray(scaled_action_deg).reshape(-1)[0] / self.max_stabilizer_angle_deg
        )
        u = u_applied_norm
        du = u_applied_norm - float(self.previous_action)
        ddu = (
            u_applied_norm
            - 2.0 * float(self.previous_action)
            + float(self.pre_previous_action)
        )

        cost_base = float(
            self.w_pitch * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action * (u**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )

        # Progress shaping (encourage faster decay of tracking errors)
        progress = float(
            self.k_progress
            * (
                (self._prev_e_theta**2 + self._prev_e_q_rel**2)
                - (e_theta**2 + e_q_rel**2)
            )
        )
        self._prev_e_theta = float(e_theta)
        self._prev_e_q_rel = float(e_q_rel)

        # Detect a new reference "segment" (step) to measure overshoot/settling
        ref_delta = float(target_theta - float(self._prev_target_theta))
        if abs(ref_delta) > float(self.ref_change_threshold_rad):
            self._seg_start_step = int(self.current_step)
            self._seg_amp = float(ref_delta)
            self._seg_sign = float(np.sign(ref_delta))
            self._seg_max_err_dir = 0.0
            self._settle_count = 0
            self._is_settled = False
            self._settle_time_s = None
            self._prev_error_sign = 0
            self._sign_changes = 0
        self._prev_target_theta = float(target_theta)

        # Step-response metrics
        amp_abs = abs(float(self._seg_amp))
        if amp_abs >= float(self.min_step_amp_rad):
            band_rad = float(
                max(
                    float(self.settle_band_ratio) * amp_abs,
                    float(self.settle_band_min_rad),
                )
            )
        else:
            band_rad = float(self.settle_band_min_rad)

        err_theta = float(theta - target_theta)  # rad
        inside_band = bool(
            (abs(err_theta) <= band_rad) and (abs(q) <= float(self.q_settle_rad_s))
        )

        if inside_band:
            self._settle_count += 1
        else:
            self._settle_count = 0

        just_settled = False
        if (not bool(self._is_settled)) and (
            int(self._settle_count) >= int(self.settle_steps_required)
        ):
            self._is_settled = True
            just_settled = True
            self._settle_time_s = float(
                max(0, int(self.current_step) - int(self._seg_start_step))
                * float(self.dt)
            )

        # Overshoot ratio (only meaningful for step-like changes)
        overshoot_ratio = 0.0
        overshoot_excess = 0.0
        if amp_abs >= float(self.min_step_amp_rad) and float(self._seg_sign) != 0.0:
            err_dir = float(err_theta * float(self._seg_sign))
            self._seg_max_err_dir = float(max(float(self._seg_max_err_dir), err_dir))
            overshoot = float(max(0.0, float(self._seg_max_err_dir)))
            overshoot_ratio = float(overshoot / amp_abs) if amp_abs > 0.0 else 0.0
            overshoot_excess = float(
                max(0.0, overshoot_ratio - float(self.overshoot_limit_ratio))
            )

        # Oscillations: penalize repeat crossings (more than 1 sign change)
        osc_event = 0.0
        if abs(err_theta) > band_rad:
            sgn = int(np.sign(err_theta))
            if sgn != 0:
                if int(self._prev_error_sign) != 0 and sgn != int(
                    self._prev_error_sign
                ):
                    self._sign_changes += 1
                    if int(self._sign_changes) > 1:
                        osc_event = 1.0
                self._prev_error_sign = int(sgn)

        # Step-response specific costs (always computed for metrics, but only
        # added to reward in "step_response" mode)
        cost_abs = float(self.w_abs * abs(e_theta))
        cost_time = float(self.w_time * (0.0 if inside_band else 1.0))
        cost_osc = float(self.w_osc * float(osc_event))
        cost_overshoot = float(self.w_overshoot * (overshoot_excess**2))

        # One-time settling bonus (larger if settled sooner than target time)
        settle_bonus = 0.0
        if (
            just_settled
            and self._settle_time_s is not None
            and float(self.settle_time_target_s) > 0.0
        ):
            speed_factor = float(
                max(
                    0.0,
                    1.0 - float(self._settle_time_s) / float(self.settle_time_target_s),
                )
            )
            settle_bonus = float(self.w_settle_bonus * speed_factor)

        # Compute reward based on mode
        if self.reward_mode == "tracking":
            # Universal tracking reward: only base LQR-like cost + progress shaping
            # Step-response metrics are NOT included in reward (but returned in info)
            cost_total = float(cost_base)
            reward = float(-cost_total) * float(self.reward_scale) + float(progress)
        else:  # "step_response"
            # Full reward with step-specific penalties
            cost_total = float(
                cost_base + cost_abs + cost_time + cost_osc + cost_overshoot
            )
            reward = (
                float(-cost_total) * float(self.reward_scale)
                + float(progress)
                + float(settle_bonus)
            )

        self.pre_previous_action = float(self.previous_action)
        self.previous_action = float(u_applied_norm)

        # Termination conditions
        terminated = bool(abs(theta) > self.max_pitch_rad)
        truncated = self.current_step >= self.number_time_steps - 2

        # --------------------------------------------------------------
        # Survival shaping (optional):
        # - discourage early termination
        # - encourage finishing the full time horizon
        # --------------------------------------------------------------
        if terminated:
            # Extra penalty proportional to remaining steps (prevents \"terminate early\" hacks)
            remaining_steps = float(
                max(
                    0,
                    int(self.number_time_steps - 2) - int(self.current_step),
                )
            )
            reward = float(-100.0 - self.early_termination_penalty) - float(
                self.early_termination_penalty_per_step * remaining_steps
            )
        else:
            # per-step alive bonus
            reward = float(reward) + float(self.survival_bonus)
            # completion bonus only when finishing by time limit (not terminated)
            if truncated:
                reward = float(reward) + float(self.completion_bonus)

        self._last_reward = float(reward)

        info: dict[str, Any] = {
            # Current state
            "theta_rad": float(theta),
            "target_theta_rad": float(target_theta),
            "err_theta_rad": float(err_theta),
            "u_norm": float(u_applied_norm),
            # Step-response metrics (always computed for evaluation)
            "band_rad": float(band_rad),
            "inside_band": bool(inside_band),
            "settled": bool(self._is_settled),
            "settle_time_s": (
                float(self._settle_time_s) if self._settle_time_s is not None else -1.0
            ),
            "overshoot_ratio": float(overshoot_ratio),
            "overshoot_excess": float(overshoot_excess),
            "sign_changes": int(self._sign_changes),
            # Cost breakdown (for debugging/analysis)
            "cost_base": float(cost_base),
            "cost_abs": float(cost_abs),
            "cost_time": float(cost_time),
            "cost_osc": float(cost_osc),
            "cost_overshoot": float(cost_overshoot),
            "cost_total": float(cost_total),
            "progress": float(progress),
            "settle_bonus": float(settle_bonus),
            # Reward mode info
            "reward_mode": str(self.reward_mode),
            # Survival shaping (debug)
            "survival_bonus": float(self.survival_bonus),
            "completion_bonus": float(self.completion_bonus),
            "early_termination_penalty": float(self.early_termination_penalty),
            "early_termination_penalty_per_step": float(
                self.early_termination_penalty_per_step
            ),
        }

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _push_history(
        self, theta_deg: float, target_deg: float, elev_deg: float
    ) -> None:
        """Append values to history buffers and enforce fixed length.

        Args:
            theta_deg (float): Current pitch angle in degrees.
            target_deg (float): Target pitch angle in degrees.
            elev_deg (float): Elevator deflection in degrees.
        """
        self._hist_theta_deg.append(theta_deg)
        self._hist_theta_target_deg.append(target_deg)
        self._hist_elev_deg.append(elev_deg)
        if len(self._hist_theta_deg) > self._history_len:
            self._hist_theta_deg = self._hist_theta_deg[-self._history_len :]
        if len(self._hist_theta_target_deg) > self._history_len:
            self._hist_theta_target_deg = self._hist_theta_target_deg[
                -self._history_len :
            ]
        if len(self._hist_elev_deg) > self._history_len:
            self._hist_elev_deg = self._hist_elev_deg[-self._history_len :]

    def _draw_timeseries(self) -> None:
        """Draw two separate time-series plots below the aircraft.

        Plot 1: theta_ref (yellow) and theta (cyan)
        Plot 2: elevator (orange)
        """
        assert self._screen is not None
        assert self._pg is not None
        assert self._font is not None

        # Geometry of two plots (lower screen area)
        # Stretch across full window width with small margins
        screen_w = int(self._screen.get_width())
        base_x = 10
        base_w = max(0, screen_w - 2 * base_x)
        pad = 10
        # Position below aircraft: starting around 430 px
        top_plot_y = 430
        plot_h = 70
        gap = 16
        bottom_plot_y = top_plot_y + plot_h + gap

        def draw_frame(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
            """Draw panel frame and background.

            Args:
                x (int): X position of top-left corner.
                y (int): Y position of top-left corner.
                w (int): Width of panel.
                h (int): Height of panel.

            Returns:
                tuple[int, int, int, int]: Inner area coordinates
                    (x, y, width, height) after padding.
            """
            # Panel frame and background
            self._pg.draw.rect(
                self._screen, (18, 22, 28), (x, y, w, h), border_radius=8
            )
            self._pg.draw.rect(
                self._screen,
                (80, 90, 100),
                (x, y, w, h),
                width=1,
                border_radius=8,
            )
            return x + pad, y + pad, w - 2 * pad, h - 2 * pad

        # First panel (theta_ref & theta)
        plot1_x, plot1_y, plot1_w, plot1_h = draw_frame(
            base_x, top_plot_y, base_w, plot_h
        )
        for i in range(3):
            y = plot1_y + int(i * plot1_h / 2)
            self._pg.draw.line(
                self._screen,
                (50, 58, 66),
                (plot1_x, y),
                (plot1_x + plot1_w, y),
                1,
            )

        # Y-axis range labels for first panel (pitch)
        pitch_lim = float(np.rad2deg(self.max_pitch_rad))
        p1_min, p1_max = -pitch_lim, pitch_lim
        label_x = plot1_x - 60
        # Ticks and labels (smoothed fonts and light shadows for readability)
        self._pg.draw.line(
            self._screen,
            (140, 140, 150),
            (plot1_x - 6, plot1_y),
            (plot1_x, plot1_y),
            1,
        )
        txt = self._small_font.render(f"{p1_max:+.0f} deg", True, (240, 240, 245))
        self._screen.blit(txt, (label_x, plot1_y - 10))
        self._pg.draw.line(
            self._screen,
            (140, 140, 150),
            (plot1_x - 6, plot1_y + plot1_h),
            (plot1_x, plot1_y + plot1_h),
            1,
        )
        txt = self._small_font.render(f"{p1_min:+.0f} deg", True, (240, 240, 245))
        self._screen.blit(txt, (label_x, plot1_y + plot1_h - 10))

        # Normalize data by ranges and draw lines

        def draw_series(
            data: list[float],
            color: tuple[int, int, int],
            y_min: float,
            y_max: float,
            px: int,
            py: int,
            pw: int,
            ph: int,
        ) -> None:
            """Draw time series as a line plot.

            Args:
                data (list[float]): Time series data to plot.
                color (tuple[int, int, int]): RGB color for the line.
                y_min (float): Minimum value for y-axis normalization.
                y_max (float): Maximum value for y-axis normalization.
                px (int): Panel x position.
                py (int): Panel y position.
                pw (int): Panel width.
                ph (int): Panel height.
            """
            if len(data) < 2:
                return
            # Take all available history (capped at _history_len),
            # and evenly stretch across full width pw
            series = data[-self._history_len :]
            n = len(series)
            # Transform values to pixels

            def to_y(v: float) -> int:
                """Convert data value to screen y-coordinate.

                Args:
                    v (float): Data value to convert.

                Returns:
                    int: Screen y-coordinate.
                """
                # v in [y_min, y_max] -> y screen
                if y_max == y_min:
                    return py + ph // 2
                t = (v - y_min) / (y_max - y_min)
                t = max(0.0, min(1.0, t))
                return int(py + ph - t * ph)

            points = []
            # X step so curve occupies full panel width
            dx = float(pw - 1) / float(max(n - 1, 1))
            for i, v in enumerate(series):
                x = int(px + i * dx)
                y = to_y(float(v))
                points.append((x, y))
            self._pg.draw.lines(self._screen, color, False, points, 2)

        # Ranges
        # theta and theta_ref: ±20 deg
        draw_series(
            self._hist_theta_target_deg,
            (255, 200, 40),
            -1,
            1,
            plot1_x,
            plot1_y,
            plot1_w,
            plot1_h,
        )
        draw_series(
            self._hist_theta_deg,
            (80, 180, 255),
            -1,
            1,
            plot1_x,
            plot1_y,
            plot1_w,
            plot1_h,
        )

        # Second panel (elevator)
        plot2_x, plot2_y, plot2_w, plot2_h = draw_frame(
            base_x, bottom_plot_y, base_w, plot_h
        )
        for i in range(3):
            y = plot2_y + int(i * plot2_h / 2)
            self._pg.draw.line(
                self._screen,
                (50, 58, 66),
                (plot2_x, y),
                (plot2_x + plot2_w, y),
                1,
            )
        # Y-axis range labels for second panel (elevator)
        e_lim = float(self.max_stabilizer_angle_deg)
        p2_min, p2_max = -e_lim, e_lim
        label2_x = plot2_x - 60
        self._pg.draw.line(
            self._screen,
            (140, 140, 150),
            (plot2_x - 6, plot2_y),
            (plot2_x, plot2_y),
            1,
        )
        txt = self._small_font.render(f"{p2_max:+.0f} deg", True, (240, 240, 245))
        self._screen.blit(txt, (label2_x, plot2_y - 10))
        self._pg.draw.line(
            self._screen,
            (140, 140, 150),
            (plot2_x - 6, plot2_y + plot2_h),
            (plot2_x, plot2_y + plot2_h),
            1,
        )
        txt = self._small_font.render(f"{p2_min:+.0f} deg", True, (240, 240, 245))
        self._screen.blit(txt, (label2_x, plot2_y + plot2_h - 10))
        draw_series(
            self._hist_elev_deg,
            (255, 120, 80),
            -25.0,
            25.0,
            plot2_x,
            plot2_y,
            plot2_w,
            plot2_h,
        )

        # Legends
        legend1 = self._small_font.render(
            "theta_ref — yellow; theta — cyan (deg)", True, (220, 225, 235)
        )
        self._screen.blit(legend1, (plot1_x, plot1_y - 18))
        legend2 = self._small_font.render(
            "elevator — orange (deg)", True, (220, 225, 235)
        )
        self._screen.blit(legend2, (plot2_x, plot2_y - 18))

    def _init_pygame(self) -> None:
        """Lazy initialization of Pygame for rendering."""
        if self._pygame_initialized:
            return
        try:
            import importlib

            pygame = importlib.import_module("pygame")
        except ImportError as exc:
            raise ImportError(
                "Package 'pygame' is required for visualization:\n"
                "  pip install pygame"
            ) from exc
        self._pg = pygame
        self._pg.init()
        width, height = 900, 600
        self._screen = self._pg.display.set_mode((width, height))
        self._pg.display.set_caption("ImprovedB747Env — Pitch Control")
        self._clock = self._pg.time.Clock()
        self._font = self._pg.font.SysFont(None, 18)
        self._small_font = self._pg.font.SysFont(None, 14)
        # Load aircraft image (side view)
        try:
            img_path = os.path.join(
                os.path.dirname(__file__), "assets", "b747_design.png"
            )
            if os.path.isfile(img_path):
                loaded_img = self._pg.image.load(img_path).convert_alpha()
                self._plane_img = loaded_img
                # Scale to comfortable width, preserving aspect ratio
                target_w = 520
                w, h = loaded_img.get_width(), loaded_img.get_height()
                scale = target_w / max(1, w)
                target_h = int(h * scale)
                self._plane_img_scaled = self._pg.transform.smoothscale(
                    self._plane_img, (int(target_w), int(target_h))
                )
        except Exception:
            # If image loading fails, fall back to primitive drawing
            self._plane_img = None
            self._plane_img_scaled = None
        self._pygame_initialized = True
        self._pygame_closed = False

    def _draw_aircraft(
        self, theta_rad: float, _elevator_deg: float, center: tuple[int, int]
    ) -> None:
        """Draw aircraft: preferably b747.png sprite, otherwise primitives.

        Args:
            theta_rad (float): Pitch angle in radians.
            elevator_deg (float): Elevator deflection in degrees.
            center (tuple[int, int]): Center position (x, y) for drawing.
        """
        assert self._screen is not None
        assert self._pg is not None

        # If sprite loaded - draw it with rotation
        if self._plane_img_scaled is not None:
            theta_deg = float(np.rad2deg(theta_rad))
            rotated = self._pg.transform.rotate(self._plane_img_scaled, -theta_deg)
            rect = rotated.get_rect(center=center)
            self._screen.blit(rotated, rect)
            return

    def _draw_elevator_gauge(self, elevator_deg: float) -> None:
        """Elevator deflection indicator in degrees.

        Args:
            elevator_deg (float): Elevator deflection in degrees.
        """
        assert self._screen is not None
        assert self._font is not None
        assert self._pg is not None

        x0, y0, w, h = 200, 520, 500, 18
        self._pg.draw.rect(self._screen, (60, 60, 60), (x0, y0, w, h), border_radius=4)
        self._pg.draw.rect(
            self._screen, (120, 120, 120), (x0 + 2, y0 + 2, w - 4, h - 4), 1
        )

        min_deg = -self.max_stabilizer_angle_deg
        max_deg = self.max_stabilizer_angle_deg
        ratio = float(np.clip((elevator_deg - min_deg) / (max_deg - min_deg), 0.0, 1.0))
        marker_x = int(x0 + 3 + ratio * (w - 6))
        self._pg.draw.line(
            self._screen,
            (255, 180, 0),
            (marker_x, y0 - 4),
            (marker_x, y0 + h + 4),
            3,
        )

        txt = self._font.render(
            f"Elevator: {elevator_deg:+.1f} deg", True, (240, 240, 240)
        )
        self._screen.blit(txt, (x0, y0 - 24))

    def _draw_hud(self, theta_deg: float, target_deg: float) -> None:
        """Simple HUD panel for displaying current and target pitch.

        Args:
            theta_deg (float): Current pitch angle in degrees.
            target_deg (float): Target pitch angle in degrees.
        """
        assert self._screen is not None
        assert self._font is not None

        info = (
            f"Step: {self.current_step}  "
            f"Pitch: {theta_deg:+.2f} deg  "
            f"Target: {target_deg:+.2f} deg  "
            f"Reward: {self._last_reward:+.3f}"
        )
        txt = self._font.render(info, True, (240, 240, 240))
        self._screen.blit(txt, (16, 16))

    def render(self, mode: str = "human"):
        """2D flight visualization using Pygame.

        Features:
            - Aircraft position: screen center, rotated by current pitch
            - Elevator indicator: horizontal scale [-25, 25] deg
            - HUD: step, current/target pitch, reward

        Args:
            mode (str): Render mode. Only "human" is supported.
                Defaults to "human".
        """
        if mode != "human" or self._pygame_closed or self.state is None:
            return

        self._init_pygame()
        assert self._screen is not None
        assert self._clock is not None
        assert self._pg is not None

        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.close()
                return

        theta = float(self.state[self._idx_theta])
        theta_deg = float(np.rad2deg(theta))
        idx = int(np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1))
        target_theta = float(self.reference_signal[0, idx])
        target_deg = float(np.rad2deg(target_theta))
        elevator_deg = float(self.previous_action * self.max_stabilizer_angle_deg)

        # Background: neutral dark, no sky/ground distinction
        self._screen.fill((18, 22, 28))

        # Reference axes X (right) and Y (up) through aircraft center
        cx, cy = 460, 240
        self._pg.draw.line(
            self._screen,
            (80, 130, 220),
            (0, cy),
            (self._screen.get_width(), cy),
            1,
        )
        self._pg.draw.line(self._screen, (120, 200, 120), (cx, 0), (cx, 360), 1)
        # Arrows at axis ends
        self._pg.draw.polygon(
            self._screen,
            (80, 130, 220),
            [
                (self._screen.get_width() - 10, cy - 4),
                (self._screen.get_width() - 2, cy),
                (self._screen.get_width() - 10, cy + 4),
            ],
        )
        self._pg.draw.polygon(
            self._screen, (120, 200, 120), [(cx - 4, 8), (cx, 0), (cx + 4, 8)]
        )
        # Axis labels
        lblx = self._small_font.render("X", True, (180, 200, 255))
        lbly = self._small_font.render("Y", True, (160, 230, 160))
        self._screen.blit(lblx, (self._screen.get_width() - 24, cy + 6))
        self._screen.blit(lbly, (cx + 6, 2))

        self._draw_aircraft(theta, elevator_deg, (cx, cy))
        self._draw_elevator_gauge(elevator_deg)
        self._draw_hud(theta_deg, target_deg)
        # Update history and draw plots
        self._push_history(theta_deg, target_deg, elevator_deg)
        self._draw_timeseries()

        self._pg.display.flip()
        self._clock.tick(60)

    def close(self):
        """Close pygame window and clean up resources."""
        if self._pygame_initialized and not self._pygame_closed:
            try:
                if self._pg is not None:
                    self._pg.display.quit()
                    self._pg.quit()
            except Exception:  # noqa: S110
                pass
            finally:
                self._pygame_closed = True
                self._pygame_initialized = False
