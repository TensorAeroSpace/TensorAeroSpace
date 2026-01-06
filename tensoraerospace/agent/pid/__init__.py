from __future__ import annotations

"""PID-based control baselines.

This module provides utilities for running classic PID controllers and logging
their performance in TensorAeroSpace environments.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)


class StateSpaceNotAvailable(Exception):
    """Exception raised when state-space matrices are not available.

    This exception is raised when trying to use MATLAB-style tuning methods
    on an environment that does not provide state-space matrices (A, B, C, D).
    """

    message = (
        "State-space matrices (A, B, C, D) are not available in the environment model. "
        "MATLAB-style tuning requires an environment with a linear state-space model."
    )


@dataclass
class MATLABTuneResult:
    """Result of MATLAB-style PID tuning.

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        settling_time (float): Achieved settling time in seconds.
        overshoot (float): Achieved overshoot in percent.
        ise (float): Integral Squared Error.
        method (str): Tuning method name.
    """

    kp: float
    ki: float
    kd: float
    settling_time: float
    overshoot: float
    ise: float
    method: str = "MATLAB-Style"

    def __repr__(self):
        return (
            f"MATLABTuneResult(Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}, "
            f"settling_time={self.settling_time:.2f}s, overshoot={self.overshoot:.2f}%)"
        )


class PID(BaseRLModel):
    """PID controller implementation for control systems.

    This class implements a PID (Proportional-Integral-Derivative) controller
    for automatic control systems. The PID controller uses proportional (P),
    integral (I), and derivative (D) components to compute the control signal.

    Args:
        env: Gymnasium environment. Defaults to None.
        kp (float): Proportional gain. Defaults to 1.
        ki (float): Integral gain. Defaults to 1.
        kd (float): Derivative gain. Defaults to 0.5.
        dt (float): Time step (time difference between consecutive updates). Defaults to 0.01.

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step.
        integral (float): Accumulated integral value.
        prev_error (float): Previous error value for derivative computation.
        env: Gymnasium environment.

    Example:
        >>> pid = PID(env=env, kp=0.1, ki=0.01, kd=0.05, dt=1)
        >>> control_signal = pid.select_action(10, 7)
    """

    def __init__(
        self,
        env: Env | None = None,
        kp: float = 1.0,
        ki: float = 1.0,
        kd: float = 0.5,
        dt: float = 0.01,
    ) -> None:
        """Initialize PID controller parameters."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        # Simulink-style: derivative on measurement (avoids derivative kick on setpoint steps)
        self.prev_measurement = 0
        self.env = env

    def select_action(self, setpoint: float, measurement: float) -> float:
        """Compute and return control signal based on setpoint and measurement.

        This method uses the current measurement and setpoint to compute the error,
        then applies the PID algorithm to compute the control signal.

        Args:
            setpoint (float): Desired value that the system should reach.
            measurement (float): Current measured value.

        Returns:
            float: Control signal computed by the PID controller.

        Example:
            >>> pid = PID(env=env, kp=0.1, ki=0.01, kd=0.05, dt=1)
            >>> control_signal = pid.select_action(10, 7)
            >>> print(control_signal)
        """
        # NOTE: This implementation follows MATLAB/Simulink default behavior:
        # - derivative on measurement (prevents derivative kick on setpoint steps)
        # - simple anti-windup via conditional integration when output saturates

        error = float(setpoint) - float(measurement)
        dt = float(self.dt) if self.dt is not None else 0.0

        # Derivative term (on measurement)
        if dt > 0:
            derivative = -(float(measurement) - float(self.prev_measurement)) / dt
            integral_candidate = float(self.integral) + error * dt
        else:
            derivative = 0.0
            integral_candidate = float(self.integral)

        output_unsat = (
            float(self.kp) * error
            + float(self.ki) * integral_candidate
            + float(self.kd) * derivative
        )
        output = output_unsat

        # Optional saturation + anti-windup (if action limits are available)
        if self.env is not None and hasattr(self.env, "action_space"):
            try:
                low = float(np.asarray(self.env.action_space.low).reshape(-1)[0])
                high = float(np.asarray(self.env.action_space.high).reshape(-1)[0])
                output = float(np.clip(output, low, high))
                if output != output_unsat:
                    # Saturated: do not integrate further (anti-windup)
                    integral_candidate = float(self.integral)
                    output_unsat = (
                        float(self.kp) * error
                        + float(self.ki) * integral_candidate
                        + float(self.kd) * derivative
                    )
                    output = float(np.clip(output_unsat, low, high))
            except Exception:
                # If action_space is not a Box-like object, ignore saturation
                pass

        self.integral = float(integral_candidate)
        self.prev_error = float(error)
        self.prev_measurement = float(measurement)
        return float(output)

    def reset(self) -> None:
        """Reset PID controller internal state.

        Resets integral accumulator and previous error to zero.
        Should be called before starting a new control episode.
        """
        self.integral = 0
        self.prev_error = 0
        self.prev_measurement = 0

    @staticmethod
    def _align_reference_to_observation_units(
        env: Env, reference_signal: np.ndarray, track_state_idx: int
    ) -> np.ndarray:
        """Align reference signal units to environment observation units.

        TensorAeroSpace environments are not fully consistent in angle units:
        - Some environments expose angles in radians
        - `LinearLongitudinalB747` exposes q/theta in degrees (observations),
          while its reference is commonly provided in radians

        This helper converts reference values to match observation units when it is safe
        and unambiguous.
        """
        ref = np.asarray(reference_signal, dtype=float)
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Only B747 linear env converts q/theta observations to degrees (see envs/b747.py)
        if (
            getattr(unwrapped.__class__, "__module__", "")
            == "tensoraerospace.envs.b747"
            and getattr(unwrapped.__class__, "__name__", "") == "LinearLongitudinalB747"
        ):
            names = getattr(unwrapped, "output_space", None) or getattr(
                unwrapped, "state_space", None
            )
            if isinstance(names, list) and len(names) > track_state_idx:
                if names[track_state_idx] in ("theta", "q"):
                    return np.rad2deg(ref)
        return ref

    @staticmethod
    def _make_step_reference(
        reference_signal: np.ndarray,
        dt: float,
        n_steps: int,
        step_at_ratio: float = 0.2,
    ) -> np.ndarray:
        """Create a step reference matching scale of a given reference signal."""
        ref = np.asarray(reference_signal, dtype=float)
        if ref.ndim > 1:
            base = ref[0, :]
        else:
            base = ref

        base = np.asarray(base).reshape(-1)
        if base.size == 0:
            base = np.zeros(n_steps, dtype=float)

        min_v = float(np.min(base))
        max_v = float(np.max(base))
        amp = max_v - min_v
        if abs(amp) < 1e-9:
            amp = 1.0

        idx = int(np.clip(int(step_at_ratio * n_steps), 1, max(1, n_steps - 1)))
        step = np.full((n_steps,), min_v, dtype=float)
        step[idx:] = min_v + amp
        return step.reshape(1, -1)

    @staticmethod
    def _make_sine_reference(
        reference_signal: np.ndarray,
        dt: float,
        n_steps: int,
        cycles: float = 3.0,
        around_final: bool = True,
    ) -> np.ndarray:
        """Create a sinusoidal reference matching scale of a given reference signal."""
        ref = np.asarray(reference_signal, dtype=float)
        if ref.ndim > 1:
            base = ref[0, :]
        else:
            base = ref

        base = np.asarray(base).reshape(-1)
        if base.size == 0:
            base = np.zeros(n_steps, dtype=float)

        min_v = float(np.min(base))
        max_v = float(np.max(base))
        amp = (max_v - min_v) * 0.5
        if abs(amp) < 1e-9:
            amp = 1.0

        offset = float(base[-1]) if around_final else float(np.mean(base))

        t = np.arange(n_steps, dtype=float) * float(dt)
        total_t = max(float(dt) * float(n_steps), float(dt))
        freq_hz = float(cycles) / total_t
        sine = offset + amp * np.sin(2.0 * np.pi * freq_hz * t)
        return sine.reshape(1, -1)

    @staticmethod
    def _check_state_space_available(
        env: Env,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check if environment has state-space matrices and return them.

        Args:
            env: Gymnasium environment with model attribute.

        Returns:
            Tuple of (A, B, C, D) matrices.

        Raises:
            StateSpaceNotAvailable: If matrices are not available.
        """
        # Get unwrapped environment
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

        # Check for model attribute
        if not hasattr(unwrapped, "model"):
            raise StateSpaceNotAvailable(
                "Environment does not have 'model' attribute. "
                "MATLAB-style tuning requires a linear state-space model."
            )

        model = unwrapped.model

        # Check for state-space matrices
        required_matrices = ["A", "B", "C", "D"]
        for matrix_name in required_matrices:
            if not hasattr(model, matrix_name):
                raise StateSpaceNotAvailable(
                    f"Model does not have '{matrix_name}' matrix. "
                    f"MATLAB-style tuning requires matrices: {required_matrices}"
                )

        A = np.array(model.A)
        B = np.array(model.B)
        C = np.array(model.C)
        D = np.array(model.D)

        # Validate matrix dimensions
        n_states = A.shape[0]
        if A.shape[1] != n_states:
            raise StateSpaceNotAvailable(
                f"Matrix A must be square. Got shape {A.shape}"
            )
        if B.shape[0] != n_states:
            raise StateSpaceNotAvailable(
                f"Matrix B must have {n_states} rows. Got shape {B.shape}"
            )

        return A, B, C, D

    def tune_matlab_style(
        self,
        track_state_idx: int = 0,
        target_settling_time: Optional[float] = None,
        target_overshoot: float = 10.0,
        n_iterations: int = 100,
        verbose: bool = True,
        mode: str = "step_response",
    ) -> MATLABTuneResult:
        """MATLAB-style PID tuning using state-space model optimization.

        This method implements PID tuning similar to MATLAB Simulink PID Tuner.
        It requires the environment to have a model with state-space matrices (A, B, C, D).

        Two optimization modes are available:
        - "step_response": Primary objective is step response (settling time, overshoot),
          with an additional secondary check on tracking (sinusoid) to avoid oscillatory
          controllers.
        - "tracking": Primary objective is tracking (RMSE), with an additional secondary
          check on step response to avoid controllers that behave poorly on setpoint steps.

        Args:
            track_state_idx (int): Index of the state to track (in output vector).
                Defaults to 0.
            target_settling_time (float, optional): Target settling time in seconds.
                If None, uses 50% of simulation time. Only used in "step_response" mode.
            target_overshoot (float): Target maximum overshoot in percent.
                Defaults to 10.0. Only used in "step_response" mode.
            n_iterations (int): Number of optimization iterations.
                Defaults to 100.
            verbose (bool): Whether to print progress. Defaults to True.
            mode (str): Optimization mode. Options:
                - "step_response": Minimize settling time, overshoot, static error
                - "tracking": Minimize RMSE and phase lag for signal tracking
                Defaults to "step_response".

        Returns:
            MATLABTuneResult: Optimized PID parameters and performance metrics.

        Raises:
            StateSpaceNotAvailable: If environment does not have state-space matrices.
            ValueError: If environment is not set or invalid mode.

        Example:
            >>> # Step response optimization
            >>> result = pid.tune_matlab_style(track_state_idx=0, mode="step_response")
            >>> # Tracking optimization (for sinusoids, etc.)
            >>> result = pid.tune_matlab_style(track_state_idx=0, mode="tracking")
        """
        if self.env is None:
            raise ValueError(
                "Environment not set. Create PID with env parameter or set self.env"
            )

        # Validate mode
        valid_modes = ["step_response", "tracking"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Check state-space availability
        A, B, C, D = self._check_state_space_available(self.env)

        mode_emoji = "📊" if mode == "step_response" else "🌊"
        mode_desc = "Step Response" if mode == "step_response" else "Signal Tracking"

        if verbose:
            print(f"\n{mode_emoji} MATLAB-Style PID Optimization ({mode_desc})")
            print("-" * 60)
            print(f"   System dimension: {A.shape[0]} states")
            print(f"   Matrices: A={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")

        # Get simulation parameters from environment
        unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        # Get number of time steps
        if hasattr(unwrapped, "number_time_steps"):
            n_steps = unwrapped.number_time_steps
        else:
            n_steps = 1000  # Default

        # Get dt
        if hasattr(unwrapped, "dt"):
            dt = unwrapped.dt
        else:
            dt = self.dt

        # Get reference signal
        if hasattr(unwrapped, "reference_signal"):
            reference_signal = np.array(unwrapped.reference_signal)
        elif hasattr(unwrapped, "ref_signal"):
            reference_signal = np.array(unwrapped.ref_signal)
        else:
            # Default step reference
            reference_signal = np.ones((1, n_steps)) * np.deg2rad(5.0)

        # Align reference units to observation units when needed (B747 q/theta degrees)
        reference_signal = self._align_reference_to_observation_units(
            self.env, reference_signal, track_state_idx
        )

        if target_settling_time is None:
            target_settling_time = n_steps * dt * 0.5

        if verbose:
            print(f"   Simulation steps: {n_steps}, dt: {dt}s")
            print(f"   Mode: {mode_desc}")
            if mode == "step_response":
                print(f"   Target settling time: {target_settling_time:.1f}s")
                print(f"   Target overshoot: {target_overshoot}%")
            else:
                print(f"   Objective: Minimize RMSE and phase lag")

        # Compute DC gain for sign determination (Simulink-like automatic sign)
        try:
            # DC gain = -C @ inv(A) @ B (for stable systems)
            dc_gain = float(
                -C[track_state_idx : track_state_idx + 1, :]
                @ np.linalg.solve(A, B[:, 0:1])
            )
        except np.linalg.LinAlgError:
            dc_gain = -1.0  # Default for unstable systems

        sign = -1 if dc_gain < 0 else 1

        if verbose:
            print(f"   DC Gain: {dc_gain:.4f}")

        # Prepare secondary references for robust tuning
        # (step + sine) to obtain one set of gains that behaves reasonably in both cases.
        step_ref = self._make_step_reference(reference_signal, dt=dt, n_steps=n_steps)
        sine_ref = self._make_sine_reference(
            reference_signal, dt=dt, n_steps=n_steps, cycles=3.0, around_final=True
        )

        # Cost helpers
        def _rmse_from_metrics(m: Dict[str, float]) -> float:
            # ISE is ∫e^2 dt ≈ sum(e^2)*dt
            n_pts = max(1, int(m.get("n_points", 1)))
            return float(np.sqrt(float(m["ise"]) / max(1e-12, float(dt) * n_pts)))

        def _control_penalty(m: Dict[str, float]) -> float:
            # Penalize aggressive / saturated control (helps stability across signals)
            ctrl_rms = float(m.get("control_rms", 0.0))
            dctrl_rms = float(m.get("dcontrol_rms", 0.0))
            sat_frac = float(m.get("sat_frac", 0.0))
            return 0.02 * ctrl_rms + 0.02 * dctrl_rms + 300.0 * sat_frac

        def _step_cost(m: Dict[str, float]) -> float:
            settling_time = float(m["settling_time"])
            overshoot_val = float(m["overshoot"])
            static_error = float(m["static_error"])
            ise = float(m["ise"])
            cost = 0.0
            cost += settling_time * 2.0
            cost += max(0.0, overshoot_val - float(target_overshoot)) * 10.0
            cost += abs(static_error) * 100.0
            cost += ise * 0.001
            cost += _control_penalty(m)
            return cost

        def _tracking_cost(m: Dict[str, float]) -> float:
            rmse = _rmse_from_metrics(m)
            iae = float(m["iae"])
            cost = 0.0
            cost += rmse * 80.0
            cost += iae * 0.05
            cost += _control_penalty(m)
            return cost

        # Cost function for optimization (robust across step + tracking)
        def compute_cost(params: np.ndarray) -> float:
            """Compute cost for given PID parameters."""
            kp_p, ki_p, kd_p = params
            # Enforce correct loop sign (Simulink-like); optimize magnitudes only
            kp = float(sign) * float(kp_p)
            ki = float(sign) * float(ki_p)
            kd = float(sign) * float(kd_p)

            # Simulate closed-loop system
            try:
                if mode == "step_response":
                    # Primary: step response on provided reference (assumed step-like)
                    m_primary = self._simulate_closed_loop(
                        kp, ki, kd, track_state_idx, reference_signal
                    )
                    # Secondary: tracking on sinusoid around final value
                    m_secondary = self._simulate_closed_loop(
                        kp, ki, kd, track_state_idx, sine_ref
                    )
                    return _step_cost(m_primary) + 0.25 * _tracking_cost(m_secondary)

                # mode == "tracking"
                m_primary = self._simulate_closed_loop(
                    kp, ki, kd, track_state_idx, reference_signal
                )
                m_secondary = self._simulate_closed_loop(
                    kp, ki, kd, track_state_idx, step_ref
                )
                return _tracking_cost(m_primary) + 0.25 * _step_cost(m_secondary)
            except Exception:
                return 1e6

        # Optimization using scipy differential evolution
        try:
            from scipy.optimize import differential_evolution
        except ImportError:
            raise ImportError(
                "scipy is required for MATLAB-style tuning. "
                "Install it with: pip install scipy"
            )

        # Search bounds on magnitudes (sign is applied separately)
        if mode == "tracking":
            bounds = [(0.0, 20.0), (0.0, 10.0), (0.0, 15.0)]
        else:
            bounds = [(0.0, 50.0), (0.0, 50.0), (0.0, 50.0)]

        # Initial guess (magnitudes)
        x0 = np.array([1.0, 0.5, 0.5], dtype=float)

        if verbose:
            print(f"\n   🔄 Running optimization ({n_iterations} iterations)...")

        # Progress tracking with tqdm
        pbar = None
        best_cost = [float("inf")]

        if verbose:
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(
                    total=n_iterations,
                    desc="   Optimization",
                    unit="iter",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
            except ImportError:
                pass  # tqdm not available, continue without progress bar

        def callback(xk, convergence=None):
            """Callback for progress updates."""
            if pbar is not None:
                pbar.update(1)
                cost = compute_cost(xk)
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    pbar.set_postfix({"cost": f"{cost:.2f}"})
            return False

        # Run optimization
        result = differential_evolution(
            compute_cost,
            bounds,
            maxiter=n_iterations,
            seed=42,
            disp=False,
            polish=True,
            x0=x0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            callback=callback if verbose else None,
            updating="deferred",
        )

        if pbar is not None:
            pbar.close()

        kp_opt_p, ki_opt_p, kd_opt_p = result.x
        kp_opt = float(sign) * float(kp_opt_p)
        ki_opt = float(sign) * float(ki_opt_p)
        kd_opt = float(sign) * float(kd_opt_p)

        # Get final metrics
        # Primary + secondary reports
        if mode == "step_response":
            final_primary = self._simulate_closed_loop(
                kp_opt, ki_opt, kd_opt, track_state_idx, reference_signal
            )
            final_secondary = self._simulate_closed_loop(
                kp_opt, ki_opt, kd_opt, track_state_idx, sine_ref
            )
        else:
            final_primary = self._simulate_closed_loop(
                kp_opt, ki_opt, kd_opt, track_state_idx, reference_signal
            )
            final_secondary = self._simulate_closed_loop(
                kp_opt, ki_opt, kd_opt, track_state_idx, step_ref
            )

        # Update PID parameters
        self.kp = kp_opt
        self.ki = ki_opt
        self.kd = kd_opt

        # Compute RMSE for tracking mode
        rmse = float(
            np.sqrt(
                final_primary["ise"]
                / max(1e-12, float(dt) * max(1, int(final_primary.get("n_points", 1))))
            )
        )

        method_name = f"MATLAB-Style ({mode_desc})"
        tune_result = MATLABTuneResult(
            kp=kp_opt,
            ki=ki_opt,
            kd=kd_opt,
            settling_time=final_primary["settling_time"],
            overshoot=final_primary["overshoot"],
            ise=final_primary["ise"],
            method=method_name,
        )

        if verbose:
            print(f"\n   ✅ Optimization completed!")
            print(f"   Kp={kp_opt:.4f}, Ki={ki_opt:.4f}, Kd={kd_opt:.4f}")
            if mode == "step_response":
                print(
                    f"   [Primary step] Settling time: {final_primary['settling_time']:.2f}s"
                )
                print(f"   [Primary step] Overshoot: {final_primary['overshoot']:.2f}%")
                print(
                    f"   [Primary step] Static error: {final_primary['static_error']:.4f}"
                )
                print(
                    f"   [Secondary sine] RMSE: {_rmse_from_metrics(final_secondary):.4f}"
                )
            else:
                print(f"   [Primary tracking] RMSE: {rmse:.4f}")
                print(f"   [Primary tracking] IAE: {final_primary['iae']:.4f}")
                print(
                    f"   [Secondary step] Overshoot: {final_secondary['overshoot']:.2f}%"
                )
                print(
                    f"   [Secondary step] Settling time: {final_secondary['settling_time']:.2f}s"
                )

        return tune_result

    def _simulate_closed_loop(
        self,
        kp: float,
        ki: float,
        kd: float,
        track_state_idx: int,
        reference_signal: np.ndarray,
    ) -> Dict[str, float]:
        """Simulate closed-loop system with given PID parameters.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            track_state_idx: Index of tracked state.
            reference_signal: Reference signal array.

        Returns:
            Dictionary with performance metrics.
        """
        # Create a fresh environment reference for simulation
        unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        # Temporarily disable reward (speed + avoid reward/unit mismatches)
        prev_use_reward = None
        if hasattr(unwrapped, "use_reward"):
            try:
                prev_use_reward = bool(getattr(unwrapped, "use_reward"))
                setattr(unwrapped, "use_reward", False)
            except Exception:
                prev_use_reward = None

        # Get dt
        dt = getattr(unwrapped, "dt", self.dt)

        # Get number of steps
        n_steps = (
            reference_signal.shape[1]
            if reference_signal.ndim > 1
            else len(reference_signal)
        )
        n_steps = min(n_steps, getattr(unwrapped, "number_time_steps", n_steps) - 2)

        # Action limits (if available)
        low = high = None
        if hasattr(self.env, "action_space"):
            try:
                low = float(np.asarray(self.env.action_space.low).reshape(-1)[0])
                high = float(np.asarray(self.env.action_space.high).reshape(-1)[0])
            except Exception:
                low = high = None

        # Initialize PID state
        integral = 0.0
        prev_measurement = 0.0

        # Storage for response
        response = []
        reference = []

        # Initial observation
        obs, _ = self.env.reset()
        if obs.ndim > 1:
            current_value = float(obs[track_state_idx, 0])
        else:
            current_value = float(obs[track_state_idx])
        prev_measurement = float(current_value)

        control_history: list[float] = []
        sat_count = 0

        for step in range(n_steps):
            # Get reference value
            if reference_signal.ndim > 1:
                ref_val = float(
                    reference_signal[0, min(step, reference_signal.shape[1] - 1)]
                )
            else:
                ref_val = float(reference_signal[min(step, len(reference_signal) - 1)])

            # PID computation (Simulink-style):
            # - derivative on measurement (avoids setpoint derivative kick)
            # - anti-windup via conditional integration when saturated
            error = ref_val - current_value
            if dt > 0:
                derivative = -(current_value - prev_measurement) / dt
                integral_candidate = integral + error * dt
            else:
                derivative = 0.0
                integral_candidate = integral

            control_unsat = kp * error + ki * integral_candidate + kd * derivative
            control = float(control_unsat)

            if low is not None and high is not None:
                control = float(np.clip(control, low, high))
                if control != float(control_unsat):
                    sat_count += 1
                    # anti-windup: do not integrate if saturated
                    integral_candidate = integral
                    control_unsat = (
                        kp * error + ki * integral_candidate + kd * derivative
                    )
                    control = float(np.clip(control_unsat, low, high))

            integral = float(integral_candidate)
            prev_measurement = float(current_value)
            control_history.append(float(control))

            # Step environment
            obs, _, terminated, truncated, _ = self.env.step(np.array([control]))

            if obs.ndim > 1:
                current_value = float(obs[track_state_idx, 0])
            else:
                current_value = float(obs[track_state_idx])

            response.append(current_value)
            reference.append(ref_val)

            if terminated or truncated:
                break

        response = np.array(response)
        reference = np.array(reference)

        # Compute metrics
        metrics = self._compute_metrics(reference, response, dt)

        # Control effort / smoothness metrics
        n_pts = int(len(response))
        ctrl = np.asarray(control_history, dtype=float).reshape(-1)
        if ctrl.size > 0:
            control_rms = float(np.sqrt(np.mean(ctrl**2)))
            control_max = float(np.max(np.abs(ctrl)))
            if dt > 0 and ctrl.size > 1:
                dctrl = np.diff(ctrl) / float(dt)
                dcontrol_rms = float(np.sqrt(np.mean(dctrl**2)))
            else:
                dcontrol_rms = 0.0
            sat_frac = (
                float(sat_count) / float(max(1, ctrl.size))
                if (low is not None and high is not None)
                else 0.0
            )
        else:
            control_rms = 0.0
            control_max = 0.0
            dcontrol_rms = 0.0
            sat_frac = 0.0

        metrics.update(
            {
                "n_points": n_pts,
                "control_rms": control_rms,
                "control_max": control_max,
                "dcontrol_rms": dcontrol_rms,
                "sat_frac": sat_frac,
            }
        )

        # Restore reward flag
        if prev_use_reward is not None:
            try:
                setattr(unwrapped, "use_reward", prev_use_reward)
            except Exception:
                pass

        return metrics

    @staticmethod
    def _compute_metrics(
        reference: np.ndarray, response: np.ndarray, dt: float
    ) -> Dict[str, float]:
        """Compute control performance metrics.

        Args:
            reference: Reference signal array.
            response: System response array.
            dt: Time step.

        Returns:
            Dictionary with metrics: settling_time, overshoot, static_error, ise, iae.
        """
        n = min(len(reference), len(response))
        reference = reference[:n]
        response = response[:n]

        # Final reference value (steady state target)
        ref_final = reference[-1] if len(reference) > 0 else 0.0

        # Static error (at end of simulation)
        static_error = float(response[-1] - ref_final) if len(response) > 0 else 0.0

        # ISE - Integral Squared Error
        error = response - reference
        ise = float(np.sum(error**2) * dt)

        # IAE - Integral Absolute Error
        iae = float(np.sum(np.abs(error)) * dt)

        # Overshoot calculation
        if abs(ref_final) > 1e-10:
            # Find maximum deviation from final value in direction of overshoot
            if ref_final > 0:
                max_val = np.max(response)
                overshoot_val = max(0, (max_val - ref_final) / ref_final * 100)
            else:
                min_val = np.min(response)
                overshoot_val = max(0, (ref_final - min_val) / abs(ref_final) * 100)
        else:
            overshoot_val = 0.0

        # Settling time (5% criterion)
        tolerance = 0.05 * abs(ref_final) if abs(ref_final) > 1e-10 else 0.05
        settled = np.abs(response - ref_final) <= tolerance

        # Find last time we exited the settling band
        settling_idx = n  # Default to end
        for i in range(n - 1, -1, -1):
            if not settled[i]:
                settling_idx = i + 1
                break
        else:
            settling_idx = 0  # Always settled

        settling_time = float(settling_idx * dt)

        return {
            "settling_time": settling_time,
            "overshoot": float(overshoot_val),
            "static_error": float(static_error),
            "ise": ise,
            "iae": iae,
        }

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Get environment and agent parameters for saving.

        Returns:
            dict: Dictionary with environment and agent policy parameters.
        """
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        print(env_name)
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"
        env_params = {}

        # Добавление информации о пространстве действий и пространстве состояний
        try:
            action_space = str(self.env.action_space)
            env_params["action_space"] = action_space
        except AttributeError:
            pass

        try:
            observation_space = str(self.env.observation_space)
            env_params["observation_space"] = observation_space
        except AttributeError:
            pass

        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)

        policy_params = {
            "ki": self.ki,
            "kp": self.kp,
            "kd": self.kd,
            "dt": self.dt,
        }
        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save PID model to the specified directory.

        If path is not specified, creates a directory with current date and time.

        Args:
            path (str, optional): Path where the model will be saved. If None,
                creates a directory with current date and time.

        Returns:
            Path: Path to the directory with saved model.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Текущая дата и время в формате 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Создание пути в текущем каталоге с датой и временем

        save_dir = path / date_str
        config_path = save_dir / "config.json"

        # Создание директории, если она не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Сохранение модели
        config = self.get_param_env()
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)

        return save_dir

    @classmethod
    def __load(cls, path: str | Path) -> "PID":
        """Load PID model from the specified directory.

        Args:
            path (str or Path): Path to directory with saved model.

        Returns:
            PID: Loaded PID model instance.

        Raises:
            TheEnvironmentDoesNotMatch: If agent type does not match expected.
        """
        path = Path(path)
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            # Десериализуем параметры среды, преобразуя списки в numpy массивы
            env_params = deserialize_env_params(config["env"]["params"])
            env = get_class_from_string(config["env"]["name"])(**env_params)
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])

        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: str | None = None,
        version: str | None = None,
    ) -> "PID":
        """Load pretrained model from local path or Hugging Face Hub.

        Args:
            repo_name (str): Repository name or local path to model.
            access_token (str, optional): Access token for Hugging Face Hub.
            version (str, optional): Model version to load.

        Returns:
            PID: Loaded PID model instance.
        """
        path = Path(repo_name)
        # Проверяем существование пути (включая относительные пути)
        if path.exists() and path.is_dir():
            new_agent = cls.__load(path)
            return new_agent
        # Проверяем, является ли это локальным путем (начинается с ./ или ../)
        elif (
            repo_name.startswith(("./", "../")) or "/" in repo_name or "\\" in repo_name
        ):
            # Это локальный путь, но директория не существует
            raise FileNotFoundError(f"Локальная директория не найдена: {repo_name}")
        else:
            # Это имя репозитория для Hugging Face Hub
            folder_path = super().from_pretrained(repo_name, access_token, version)
            new_agent = cls.__load(folder_path)
            return new_agent
