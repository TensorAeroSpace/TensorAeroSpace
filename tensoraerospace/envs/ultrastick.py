"""
Module for modeling Ultrastick-25e aircraft.

Gymnasium environments for training agents to control longitudinal motion
of Ultrastick-25e. Includes pitch angle and angular velocity control.
"""

from typing import Any, Callable, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import Ultrastick


class LinearLongitudinalUltrastick(gym.Env):
    """Simulation of Ultrastick-25e in Gym environment for training AI agents.

    Args:
        initial_state: Initial state.
        reference_signal: Reference signal.
        number_time_steps: Number of simulation steps.
        tracking_states: Tracked states.
        state_space: State space.
        control_space: Control space.
        output_space: Full output space (including noise).
        reward_func: Reward function (WIP status).
    """

    def __init__(
        self,
        initial_state: Union[np.ndarray, List[float]],
        reference_signal: Union[np.ndarray, Callable],
        number_time_steps: int,
        tracking_states: Optional[List[str]] = None,
        state_space: Optional[List[str]] = None,
        control_space: Optional[List[str]] = None,
        output_space: Optional[List[str]] = None,
        reward_func: Optional[Callable] = None,
    ) -> None:
        """Initialize legacy Ultrastick environment."""
        self.max_action_value = 25.0
        self.initial_state = initial_state
        self.number_time_steps = number_time_steps
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["theta", "q"]
        )
        self.state_space = state_space if state_space is not None else ["theta", "q"]
        self.control_space = control_space if control_space is not None else ["stab"]
        self.output_space = output_space if output_space is not None else ["theta", "q"]
        self.selected_state_output = self.output_space
        self.reference_signal = reference_signal
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = Ultrastick(
            initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=None,
            t0=0,
        )
        # Map state_space to model's full state indices
        # Model's selected_states: ["u", "w", "q", "theta", "h"]
        model_state_names = self.model.selected_states
        self.state_space_indices = [
            model_state_names.index(state_name) for state_name in self.state_space
        ]
        self.indices_tracking_states = [
            self.state_space.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.action_space = spaces.Box(
            low=-25,
            high=25,
            shape=(len(self.control_space), 1),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_space), 1),
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
    def reward(state: np.ndarray, ref_signal: np.ndarray, ts: int) -> float:
        """Control evaluation.

        Args:
            state: Current state.
            ref_signal: Reference state.
            ts: Time step.

        Returns:
            float: Control evaluation.
        """
        ref_val = float(np.asarray(ref_signal[:, ts]).reshape(-1)[0])
        return float(abs(float(state[0]) - ref_val))

    def _get_info(self):
        """Return auxiliary info for Gym API (currently empty)."""
        return {}

    def step(self, action: np.ndarray):
        """Execute simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators.

        Returns:
            next_state (np.ndarray): Next state of control object.
            reward (np.ndarray): Evaluation of control algorithm actions.
            done (bool): Simulation status, completed or not.
            logging (any): Additional information (not used).
        """
        # Ensure action is 1D and get elevator value
        action_flat = np.asarray(action).flatten()
        ele_deg = float(action_flat[0])

        # Clip elevator to limits
        if ele_deg > self.max_action_value:
            ele_deg = self.max_action_value
        if ele_deg < self.max_action_value * -1:
            ele_deg = self.max_action_value * -1

        # Convert to radians and create model input [ele (rad), delta_t]
        # delta_t = 0 means no throttle change
        ele_rad = np.deg2rad(ele_deg)
        model_action = np.array([ele_rad, 0.0], dtype=np.float32)

        self.current_step += 1
        next_state_full = self.model.run_step(model_action)
        # Map full model state to state_space
        next_state = next_state_full[self.state_space_indices]
        reward = self.reward_func(
            next_state[self.indices_tracking_states],
            self.ref_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 2
        info = self._get_info()
        return (
            next_state.reshape([-1, 1]).astype(np.float32),
            reward,
            self.done,
            False,
            info,
        )

    def reset(self, seed=None, options=None):
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional options for initialization.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.done = False
        self.model = None  # type: ignore[assignment]
        self.model = Ultrastick(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
        )
        # Recompute state_space_indices after model recreation
        model_state_names = self.model.selected_states
        self.state_space_indices = [
            model_state_names.index(state_name) for state_name in self.state_space
        ]
        self.ref_signal = self.reference_signal
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )
        info = self._get_info()
        # Map initial_state to state_space
        initial_state_array = np.array(self.initial_state, dtype=np.float32).reshape(-1)
        observation = (
            initial_state_array[self.state_space_indices]
            .reshape([-1, 1])
            .astype(np.float32)
        )
        return observation, info

    def render(self):
        """Visual display of actions in environment (not implemented)."""
        raise NotImplementedError()


class ImprovedUltrastickEnv(gym.Env):
    """Improved Ultrastick longitudinal channel control environment.

    Features normalized action/observation spaces and a shaped reward.

    - Two controls: elevator (rad) and throttle (dimensionless 0..1)
    - Track a single state: pitch angle theta
    - Observation (normalized to [-1, 1]):
        [pitch_error, q, theta, prev_elev, prev_throttle]
    - Actions are normalized in [-1, 1] and scaled internally
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_elevator_deg: float = 0.0,
        initial_throttle: float = 0.0,
        use_initial_action_on_first_step: bool = True,
    ) -> None:
        """Initialize improved Ultrastick environment."""
        super().__init__()

        # Physical constraints and normalization parameters
        self.max_pitch_rad = float(np.deg2rad(30.0))
        self.max_pitch_rate_rad_s = float(np.deg2rad(30.0))
        self.max_elevator_deg = 15.0  # |ele| <= 15 deg (training)
        self.max_throttle = 1.0  # |delta_t| <= 1.0

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Simulation parameters
        self.dt = float(dt)
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        self.reference_signal = np.array(reference_signal, dtype=float)
        self.number_time_steps = int(number_time_steps)
        self.current_step = 0

        # Previous actions (normalized)
        self.initial_elevator_deg = float(initial_elevator_deg)
        self.initial_throttle = float(initial_throttle)
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.prev_elev_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_elevator_deg,
                -1.0,
                1.0,
            )
        )
        # store throttle history in [-1, 1] while actual throttle is [0, 1]
        self.prev_thr_norm = float(
            np.clip(2.0 * float(self.initial_throttle) - 1.0, -1.0, 1.0)
        )
        self.pre_prev_elev_norm = float(self.prev_elev_norm)
        self.pre_prev_thr_norm = float(self.prev_thr_norm)

        # Reward weights
        self.reward_scale = 0.08
        self.w_theta = 8.0
        self.w_q = 0.5
        self.w_action_elev = 0.003
        self.w_action_thr = 0.001
        self.w_smooth_elev = 0.003
        self.w_smooth_thr = 0.002
        self.w_jerk_elev = 0.0005
        self.w_jerk_thr = 0.0003

        # Action smoothing and progress shaping
        self.elev_alpha = 0.5
        self.k_progress = 0.05
        self._prev_e_theta = 0.0
        self._prev_e_q_rel = 0.0

        # Model (returns outputs y = C x + D u)
        self.model = Ultrastick(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
        )
        self.model.initialise_system(
            x0=self.initial_state, number_time_steps=self.number_time_steps
        )
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

    # Helper indices
    @property
    def _idx_q(self) -> int:
        """Index of pitch rate state."""
        # Ultrastick C matrix output order: [Va, alpha, theta, q, h]
        return 3

    @property
    def _idx_theta(self) -> int:
        """Index of pitch angle state."""
        # Ultrastick C matrix output order: [Va, alpha, theta, q, h]
        return 2

    def _flatten_action(self, a: Any) -> list[float]:
        """Flatten arbitrary nested action into list of floats."""
        if isinstance(a, (list, tuple)):
            out: list[float] = []
            for el in a:
                out.extend(self._flatten_action(el))
            return out
        if isinstance(a, np.ndarray):
            try:
                return a.astype(float).ravel().tolist()
            except Exception:
                return []
        try:
            if np.isscalar(a):
                return [float(a)]
        except Exception:
            pass
        try:
            return np.asarray(a, dtype=float).ravel().tolist()
        except Exception:
            return []

    def _to_norm_action(self, action: Any) -> np.ndarray:
        """Coerce incoming action to normalized shape (2,) in [-1, 1]."""
        try:
            arr = np.asarray(action, dtype=float)
        except Exception:
            arr_list = self._flatten_action(action)
            arr = np.asarray(arr_list, dtype=float)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        arr = np.ravel(arr)
        if arr.size == 0:
            arr = np.array([0.0, self.prev_thr_norm], dtype=float)
        elif arr.size == 1:
            arr = np.array([float(arr[0]), self.prev_thr_norm], dtype=float)
        else:
            arr = np.array([float(arr[0]), float(arr[1])], dtype=float)
        arr = np.clip(arr, -1.0, 1.0)
        return arr.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return normalized observation vector."""
        y = self.state.reshape(-1)
        theta = float(y[self._idx_theta])
        q = float(y[self._idx_q])

        # Target theta for current step (in rad)
        idx = int(
            np.clip(
                self.current_step,
                0,
                self.reference_signal.shape[1] - 1,
            )
        )
        target_theta = float(self.reference_signal[0, idx])

        # Normalized features
        pitch_error = target_theta - theta
        norm_pitch_error = float(np.clip(pitch_error / self.max_pitch_rad, -1.0, 1.0))
        norm_q = float(np.clip(q / self.max_pitch_rate_rad_s, -1.0, 1.0))
        norm_theta = float(np.clip(theta / self.max_pitch_rad, -1.0, 1.0))

        return np.array(
            [
                norm_pitch_error,
                norm_q,
                norm_theta,
                float(self.prev_elev_norm),
                float(self.prev_thr_norm),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment state and action history."""
        super().reset(seed=seed)
        self.model.initialise_system(
            self.initial_state,
            self.number_time_steps,
        )
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        # Reset action history
        self.prev_elev_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_elevator_deg,
                -1.0,
                1.0,
            )
        )
        self.prev_thr_norm = float(
            np.clip(2.0 * float(self.initial_throttle) - 1.0, -1.0, 1.0)
        )
        self.pre_prev_elev_norm = float(self.prev_elev_norm)
        self.pre_prev_thr_norm = float(self.prev_thr_norm)
        self._prev_e_theta = 0.0
        self._prev_e_q_rel = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Apply normalized elevator/throttle action and advance simulation."""
        # Normalize input (robust to nested/ragged actions)
        action = self._to_norm_action(action)

        # Scale actions
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            elev_deg = float(self.initial_elevator_deg)
            thr = float(self.initial_throttle)
        else:
            # commands (normalized)
            cmd_elev_norm = float(action[0])
            cmd_thr_norm = float(action[1])

            # Elevator: low-pass + rate limiting (deg)
            prev_elev_deg = float(self.prev_elev_norm) * self.max_elevator_deg
            cmd_elev_deg = cmd_elev_norm * self.max_elevator_deg
            elev_deg_pre = (
                self.elev_alpha * cmd_elev_deg + (1.0 - self.elev_alpha) * prev_elev_deg
            )
            deg_step_limit = 300.0 * self.dt  # deg per step
            elev_deg = float(
                np.clip(
                    elev_deg_pre,
                    prev_elev_deg - deg_step_limit,
                    prev_elev_deg + deg_step_limit,
                )
            )

            # Throttle mapped to [0, 1]
            thr = float(0.5 * (cmd_thr_norm + 1.0) * self.max_throttle)

        # Run model step (Ultrastick expects [ele (rad), delta_t])
        elev_rad = float(np.deg2rad(elev_deg))
        y = self.model.run_step(np.array([elev_rad, thr], dtype=float)).reshape(-1)
        self.state = y.copy()
        self.current_step += 1

        # Reward
        theta = float(y[self._idx_theta])
        q = float(y[self._idx_q])
        idx = int(
            np.clip(
                self.current_step,
                0,
                self.reference_signal.shape[1] - 1,
            )
        )
        target_theta = float(self.reference_signal[0, idx])

        # Reference theta derivative for damping term
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

        e_theta = float((theta - target_theta) / self.max_pitch_rad)
        e_q_rel = float((q - ref_theta_dot) / self.max_pitch_rate_rad_s)

        # Normalize actually applied actions
        elev_norm = float(np.clip(elev_deg / self.max_elevator_deg, -1.0, 1.0))
        thr_norm = float(np.clip((2.0 * thr / self.max_throttle) - 1.0, -1.0, 1.0))
        du_elev = elev_norm - float(self.prev_elev_norm)
        du_thr = thr_norm - float(self.prev_thr_norm)
        ddu_elev = (
            elev_norm
            - 2.0 * float(self.prev_elev_norm)
            + float(self.pre_prev_elev_norm)
        )
        ddu_thr = (
            thr_norm - 2.0 * float(self.prev_thr_norm) + float(self.pre_prev_thr_norm)
        )

        cost = (
            self.w_theta * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action_elev * (elev_norm**2)
            + self.w_action_thr * (thr_norm**2)
            + self.w_smooth_elev * (du_elev**2)
            + self.w_smooth_thr * (du_thr**2)
            + self.w_jerk_elev * (ddu_elev**2)
            + self.w_jerk_thr * (ddu_thr**2)
        )
        progress = float(
            self.k_progress
            * (
                (self._prev_e_theta**2 + self._prev_e_q_rel**2)
                - (e_theta**2 + e_q_rel**2)
            )
        )
        reward = float(-cost * self.reward_scale + progress)

        # Update action history
        self.pre_prev_elev_norm = float(self.prev_elev_norm)
        self.pre_prev_thr_norm = float(self.prev_thr_norm)
        self.prev_elev_norm = float(elev_norm)
        self.prev_thr_norm = float(thr_norm)
        # Update error history for progress shaping
        self._prev_e_theta = float(e_theta)
        self._prev_e_q_rel = float(e_q_rel)

        # Termination
        terminated = bool(abs(theta) > self.max_pitch_rad)
        truncated = bool(self.current_step >= self.number_time_steps - 2)

        info: dict[str, Any] = {
            "elevator_deg": float(elev_deg),
            "throttle": float(thr),
            "reward": float(reward),
            "cost": float(cost),
            "progress": float(progress),
            "e_theta2": float(e_theta**2),
            "e_q_rel2": float(e_q_rel**2),
            "elev_norm2": float(elev_norm**2),
            "thr_norm2": float(thr_norm**2),
            "du_elev2": float(du_elev**2),
            "du_thr2": float(du_thr**2),
            "ddu_elev2": float(ddu_elev**2),
            "ddu_thr2": float(ddu_thr**2),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Render is not implemented for ImprovedUltrastickEnv."""
        return None
