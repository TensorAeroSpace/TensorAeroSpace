"""LAPAN LSU-05 NG Gymnasium environments.

This module implements Gymnasium environments for longitudinal control of the
LAPAN Surveillance Aircraft (LSU)-05 NG model, including a legacy environment
(``LinearLongitudinalLAPAN``) and a normalized variant (``ImprovedLAPANEnv``).
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tensoraerospace.aerospacemodel import LAPAN


class LinearLongitudinalLAPAN(gym.Env):
    """Simulation of LAPAN control object in OpenAI Gym environment for training AI agents.

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
        initial_state: np.ndarray | list[float],
        reference_signal: np.ndarray | Callable,
        number_time_steps: int,
        tracking_states: list[str] | None = None,
        state_space: list[str] | None = None,
        control_space: list[str] | None = None,
        output_space: list[str] | None = None,
        reward_func: Callable | None = None,
        dt: float = 0.01,
    ) -> None:
        """Initialize LAPAN longitudinal environment."""
        super().__init__()
        self.initial_state = np.array(initial_state, dtype=float, copy=True).reshape(-1)
        self.dt = float(dt)
        if int(number_time_steps) != number_time_steps or number_time_steps < 2:
            raise ValueError("number_time_steps must be an integer >= 2")
        self.number_time_steps = number_time_steps = int(number_time_steps)
        self.tracking_states = (
            tracking_states if tracking_states is not None else ["theta", "q"]
        )
        self.state_space = state_space if state_space is not None else ["theta", "q"]
        self.control_space = control_space if control_space is not None else ["stab"]
        self.output_space = (
            output_space if output_space is not None else list(self.state_space)
        )
        self.selected_state_output = self.output_space
        if not self.output_space or not self.tracking_states:
            raise ValueError("output_space and tracking_states must be nonempty")
        if len(self.control_space) != 1:
            raise ValueError("LAPAN supports one elevator input")
        if callable(reference_signal):
            reference_signal = np.array(
                [
                    np.atleast_1d(reference_signal(i * self.dt))
                    for i in range(number_time_steps)
                ]
            ).T
        self.reference_signal = np.array(reference_signal, dtype=float, copy=True)
        if (
            self.reference_signal.ndim != 2
            or self.reference_signal.shape[1] < 1
            or not np.all(np.isfinite(self.reference_signal))
        ):
            raise ValueError(
                "reference_signal must be finite with shape (channels, T), T >= 1"
            )
        if self.reference_signal.shape[0] not in (1, len(self.tracking_states)):
            raise ValueError("reference channels must be one or match tracking_states")
        if reward_func:
            self.reward_func = reward_func
        else:
            self.reward_func = self.reward

        self.model = LAPAN(
            self.initial_state,
            number_time_steps=number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.indices_tracking_states = [
            self.model.list_state.index(self.tracking_states[i])
            for i in range(len(self.tracking_states))
        ]

        self.ref_signal = self.reference_signal
        # Preserve the public action contract in degrees; the plant uses radians.
        self.max_action_value = 25.0
        self.max_elevator_angle_deg = self.max_action_value
        self.action_space = spaces.Box(
            low=-self.max_elevator_angle_deg,
            high=self.max_elevator_angle_deg,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.output_space),), dtype=np.float32
        )

        self.current_step = 0
        self.done = False

    def _get_info(self) -> dict[str, float | np.ndarray]:
        """Return auxiliary info for Gym API (currently empty)."""
        return {}

    @staticmethod
    def reward(state: np.ndarray, ref_signal: np.ndarray, ts: int) -> float:
        """Evaluate control performance.

        Args:
            state (np.ndarray): Current state.
            ref_signal (np.ndarray): Reference signal.
            ts (int): Time step.

        Returns:
            float: Control evaluation reward.
        """
        ts_safe = int(np.clip(ts, 0, ref_signal.shape[1] - 1))
        reference = ref_signal[:, ts_safe]
        tracked = np.asarray(state).reshape(-1)
        # A single reference retains the first-tracked-state objective.
        error = tracked[:1] - reference if reference.size == 1 else tracked - reference
        return -float(np.mean(np.abs(error)))

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | np.ndarray]]:
        """Execute one simulation step.

        Args:
            action (np.ndarray): Control signal array for selected actuators.

        Returns:
            tuple: Tuple containing:
                - next_state (np.ndarray): Next state of the control object.
                - reward (np.ndarray): Evaluation of control algorithm actions.
                - done (bool): Simulation status, whether completed or not.
                - truncated (bool): Whether episode was truncated.
                - info (dict): Additional information.
        """
        action = np.asarray(action).reshape(-1)
        if action.size != 1 or not np.all(np.isfinite(action)):
            raise ValueError("action must contain one finite elevator command")
        action_rad = np.deg2rad(
            np.clip(action, -self.max_action_value, self.max_action_value)
        )
        next_state = self.model.run_step(action_rad)
        self.current_step += 1
        reward = self.reward_func(
            np.asarray(self.model.xt).reshape(-1, 1)[self.indices_tracking_states],
            self.reference_signal,
            self.current_step,
        )
        self.done = self.current_step >= self.number_time_steps - 1
        info = self._get_info()
        info["applied_action"] = np.rad2deg(
            self.model.store_input[:, self.model.time_step - 1]
        ).astype(np.float32)

        return (
            np.asarray(next_state).reshape(-1).astype(np.float32),
            float(reward),
            False,
            self.done,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
        """Reset simulation environment to initial conditions.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Additional initialization options. Defaults to None.

        Returns:
            tuple: Tuple containing:
                - observation (np.ndarray): Initial observation.
                - info (dict): Additional information.
        """
        super().reset(seed=seed)

        self.model = LAPAN(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=self.output_space,
            t0=0,
            dt=self.dt,
        )
        self.ref_signal = self.reference_signal
        self.current_step = 0
        self.done = False
        info = self._get_info()
        observation = np.array(self.initial_state, dtype=np.float32)[
            self.model.selected_state_index
        ].reshape(-1)
        return observation, info

    def render(self) -> None:
        """Visual rendering of actions in the environment. Work in progress.

        Raises:
            NotImplementedError: Rendering is not yet implemented.
        """
        raise NotImplementedError("Rendering is not implemented for LAPANEnv.")


class ImprovedLAPANEnv(gym.Env):
    """LAPAN env with normalized spaces; internal units are radians."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_state: np.ndarray,
        reference_signal: np.ndarray,
        number_time_steps: int,
        dt: float = 0.01,
        initial_elevator_deg: float = 0.0,
        use_initial_action_on_first_step: bool = True,
    ) -> None:
        """Initialize normalized LAPAN environment."""
        super().__init__()

        # Physical/normalization limits
        self.max_pitch_rad = np.deg2rad(20.0)
        self.max_pitch_rate_rad_s = np.deg2rad(5.0)
        self.max_elevator_angle_rad = np.deg2rad(25.0)
        self.max_elevator_angle_deg = 25.0

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Simulation parameters
        self.dt = float(dt)
        self.initial_state = np.array(initial_state, dtype=float).reshape(-1)
        self.reference_signal = np.array(reference_signal, dtype=float)
        if int(number_time_steps) != number_time_steps or number_time_steps < 2:
            raise ValueError("number_time_steps must be an integer >= 2")
        self.number_time_steps = int(number_time_steps)
        if (
            self.reference_signal.ndim != 2
            or self.reference_signal.shape[0] != 1
            or self.reference_signal.shape[1] < 1
            or not np.all(np.isfinite(self.reference_signal))
        ):
            raise ValueError(
                "reference_signal must be finite with shape (1, T), T >= 1"
            )
        self.current_step = 0
        # LAPAN state order: [u, w, q, theta]
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)

        # Initial elevator and action history (normalized)
        if not np.isfinite(initial_elevator_deg):
            raise ValueError("initial_elevator_deg must be finite")
        self.initial_elevator_deg = float(
            np.clip(
                initial_elevator_deg,
                -self.max_elevator_angle_deg,
                self.max_elevator_angle_deg,
            )
        )
        self.initial_action_norm = float(
            np.clip(
                self.initial_elevator_deg / self.max_elevator_angle_deg,
                -1.0,
                1.0,
            )
        )
        self.use_initial_action_on_first_step = bool(use_initial_action_on_first_step)
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0

        # Reward shaping
        self.reward_scale = 0.1
        self.w_pitch = 5.0
        self.w_q = 0.2
        self.w_action = 0.003
        self.w_smooth = 0.01
        self.w_jerk = 0.001

        # Store init args for helpers (only public __init__ parameters —
        # avoid capturing locals() which includes derived variables)
        self.init_args = {
            "initial_state": initial_state,
            "reference_signal": reference_signal,
            "number_time_steps": number_time_steps,
            "dt": dt,
            "initial_elevator_deg": initial_elevator_deg,
            "use_initial_action_on_first_step": use_initial_action_on_first_step,
        }

        # Underlying LAPAN model (keep full state output order)
        # Constructor already invokes initialise_system internally.
        self.model = LAPAN(
            self.initial_state,
            number_time_steps=self.number_time_steps,
            selected_state_output=None,
            t0=0,
            dt=self.dt,
            initial_control=float(np.deg2rad(self.initial_elevator_deg)),
        )

    # Helper indices based on LAPAN state order [u, w, q, theta]
    @property
    def _idx_q(self) -> int:
        """Index of pitch rate state."""
        return 2

    @property
    def _idx_theta(self) -> int:
        """Index of pitch angle state."""
        return 3

    def _get_obs(self) -> np.ndarray:
        """Return normalized observation vector."""
        # Model provides theta, q in radians now
        theta_rad = float(self.state[self._idx_theta])
        q_rad_s = float(self.state[self._idx_q])
        idx = int(np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1))
        target_theta = float(self.reference_signal[0, idx])  # radians

        pitch_error = target_theta - theta_rad
        norm_pitch_error = float(np.clip(pitch_error / self.max_pitch_rad, -1.0, 1.0))
        norm_q = float(np.clip(q_rad_s / self.max_pitch_rate_rad_s, -1.0, 1.0))
        norm_theta = float(np.clip(theta_rad / self.max_pitch_rad, -1.0, 1.0))
        norm_prev_action = float(self.previous_action)

        return np.array(
            [norm_pitch_error, norm_q, norm_theta, norm_prev_action],
            dtype=np.float32,
        )

    def get_init_args(self):
        """Return initialization arguments for reproducibility."""
        init_args = self.init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        return init_args

    def reset(self, seed=None, options=None):
        """Reset environment state and counters."""
        super().reset(seed=seed)
        self.model.initialise_system(
            self.initial_state,
            self.number_time_steps,
        )
        self.state = np.array(self.initial_state, dtype=float).reshape(-1)
        self.current_step = 0
        self.previous_action = float(self.initial_action_norm)
        self.pre_previous_action = float(self.initial_action_norm)
        self._last_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Apply normalized action and advance simulation by one step."""
        # action in [-1, 1]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1 or not np.all(np.isfinite(action)):
            raise ValueError("action must contain one finite elevator command")
        action = np.clip(action, -1.0, 1.0)

        # Scale to radians (LAPAN input now expects radians), optionally use
        # initial elevator at first step (converted to rad)
        if self.current_step == 0 and self.use_initial_action_on_first_step:
            scaled_action_rad = np.array(
                [np.deg2rad(self.initial_elevator_deg)],
                dtype=np.float32,
            )
        else:
            scaled_action_rad = action * self.max_elevator_angle_rad

        # LAPAN model expects radians for elevator input
        self.state = self.model.run_step(scaled_action_rad).reshape(-1)
        self.current_step += 1

        # Reward (model outputs are radians)
        theta_rad = float(self.state[self._idx_theta])
        q_rad_s = float(self.state[self._idx_q])
        idx_safe = int(
            np.clip(self.current_step, 0, self.reference_signal.shape[1] - 1)
        )
        target_theta = float(self.reference_signal[0, idx_safe])  # radians

        # Reference derivative for relative q penalty
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

        e_theta = float((theta_rad - target_theta) / self.max_pitch_rad)
        e_q_rel = float((q_rad_s - ref_theta_dot) / self.max_pitch_rate_rad_s)

        u_applied_norm = float(
            self.model.store_input[0, self.model.time_step - 1]
            / self.max_elevator_angle_rad
        )
        u = u_applied_norm
        du = u_applied_norm - float(self.previous_action)
        ddu = (
            u_applied_norm
            - 2.0 * float(self.previous_action)
            + float(self.pre_previous_action)
        )

        cost = (
            self.w_pitch * (e_theta**2)
            + self.w_q * (e_q_rel**2)
            + self.w_action * (u**2)
            + self.w_smooth * (du**2)
            + self.w_jerk * (ddu**2)
        )
        reward = float(-cost) * float(self.reward_scale)

        self.pre_previous_action = float(self.previous_action)
        self.previous_action = float(u_applied_norm)
        terminated = False
        if abs(theta_rad) > self.max_pitch_rad:
            reward = -100.0
            terminated = True

        self._last_reward = float(reward)
        truncated = self.current_step >= self.number_time_steps - 1

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {"elevator_deg": float(u_applied_norm * self.max_elevator_angle_deg)},
        )

    def render(self, mode: str = "human"):
        """Rendering not implemented for LAPAN environment."""
        return

    def close(self):
        """Close environment resources."""
        return
