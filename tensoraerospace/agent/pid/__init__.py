"""PID-based control baselines.

This module provides utilities for running classic PID controllers and logging
their performance in TensorAeroSpace environments.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

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

    def __init__(self, env=None, kp=1, ki=1, kd=0.5, dt=0.01):
        """Initialize PID controller parameters."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.env = env

    def select_action(self, setpoint, measurement):
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
        error = setpoint - measurement
        self.integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

    def reset(self):
        """Reset PID controller internal state.

        Resets integral accumulator and previous error to zero.
        Should be called before starting a new control episode.
        """
        self.integral = 0
        self.prev_error = 0

    @staticmethod
    def _check_state_space_available(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        - "step_response": Optimizes for step response (settling time, overshoot)
        - "tracking": Optimizes for signal tracking (RMSE, phase lag)

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

        # Compute DC gain for sign determination
        try:
            # DC gain = -C @ inv(A) @ B (for stable systems)
            dc_gain = float(-C[track_state_idx:track_state_idx+1, :] @ np.linalg.solve(A, B[:, 0:1]))
        except np.linalg.LinAlgError:
            dc_gain = -1.0  # Default for unstable systems

        sign = -1 if dc_gain < 0 else 1

        if verbose:
            print(f"   DC Gain: {dc_gain:.4f}")

        # Cost function for optimization
        def compute_cost(params: np.ndarray) -> float:
            """Compute cost for given PID parameters."""
            kp, ki, kd = params

            # Simulate closed-loop system
            try:
                metrics = self._simulate_closed_loop(
                    kp, ki, kd, track_state_idx, reference_signal
                )

                if mode == "step_response":
                    # Step response mode: minimize settling time, overshoot
                    settling_time = metrics["settling_time"]
                    overshoot_val = metrics["overshoot"]
                    static_error = metrics["static_error"]
                    ise = metrics["ise"]

                    cost = 0.0
                    cost += settling_time * 2.0  # Settling time weight
                    cost += max(0, overshoot_val - target_overshoot) * 10.0  # Overshoot penalty
                    cost += abs(static_error) * 100.0  # Static error penalty
                    cost += ise * 0.001  # ISE contribution
                else:
                    # Tracking mode: minimize RMSE while maintaining stability
                    ise = metrics["ise"]
                    iae = metrics["iae"]
                    overshoot_val = metrics["overshoot"]
                    settling_time_val = metrics["settling_time"]

                    # RMSE from ISE
                    n_points = max(1, int(n_steps * 0.9))
                    rmse = np.sqrt(ise / n_points) if n_points > 0 else ise

                    cost = 0.0
                    # Primary: minimize tracking error
                    cost += rmse * 50.0
                    cost += iae * 0.05

                    # Stability constraints (prevent oscillations)
                    cost += max(0, overshoot_val - 30.0) * 5.0  # Limit overshoot
                    cost += settling_time_val * 0.5  # Still care about settling

                    # Penalize extreme gains (cause instability)
                    cost += max(0, abs(kd) - 10.0) * 2.0  # Limit Kd to prevent oscillations
                    cost += max(0, abs(ki) - 5.0) * 1.0  # Don't completely remove Ki

                return cost
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

        # Search bounds (tighter for tracking to ensure stability)
        if mode == "tracking":
            bounds = [(-20, 20), (-10, 10), (-15, 15)]  # Tighter bounds for stability
        else:
            bounds = [(-50, 50), (-50, 50), (-50, 50)]

        # Initial guess based on DC gain
        x0 = np.array([sign * 1.0, sign * 0.5, sign * 0.5])

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

        kp_opt, ki_opt, kd_opt = result.x

        # Get final metrics
        final_metrics = self._simulate_closed_loop(
            kp_opt, ki_opt, kd_opt, track_state_idx, reference_signal
        )

        # Update PID parameters
        self.kp = kp_opt
        self.ki = ki_opt
        self.kd = kd_opt

        # Compute RMSE for tracking mode
        n_points = max(1, int(n_steps * 0.9))
        rmse = np.sqrt(final_metrics["ise"] / n_points) if n_points > 0 else 0.0

        method_name = f"MATLAB-Style ({mode_desc})"
        tune_result = MATLABTuneResult(
            kp=kp_opt,
            ki=ki_opt,
            kd=kd_opt,
            settling_time=final_metrics["settling_time"],
            overshoot=final_metrics["overshoot"],
            ise=final_metrics["ise"],
            method=method_name,
        )

        if verbose:
            print(f"\n   ✅ Optimization completed!")
            print(f"   Kp={kp_opt:.4f}, Ki={ki_opt:.4f}, Kd={kd_opt:.4f}")
            if mode == "step_response":
                print(f"   Settling time: {final_metrics['settling_time']:.2f}s")
                print(f"   Overshoot: {final_metrics['overshoot']:.2f}%")
                print(f"   Static error: {final_metrics['static_error']:.4f}")
            else:
                print(f"   RMSE: {rmse:.4f}")
                print(f"   IAE: {final_metrics['iae']:.4f}")
                print(f"   Overshoot: {final_metrics['overshoot']:.2f}% (stability check)")
                print(f"   Settling time: {final_metrics['settling_time']:.2f}s")

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
        # Create a fresh environment copy for simulation
        unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        # Reset environment
        self.env.reset()

        # Get dt
        dt = getattr(unwrapped, "dt", self.dt)

        # Get number of steps
        n_steps = reference_signal.shape[1] if reference_signal.ndim > 1 else len(reference_signal)
        n_steps = min(n_steps, getattr(unwrapped, "number_time_steps", n_steps) - 2)

        # Initialize PID state
        integral = 0.0
        prev_error = 0.0

        # Storage for response
        response = []
        reference = []

        # Initial observation
        obs, _ = self.env.reset()
        if obs.ndim > 1:
            current_value = float(obs[track_state_idx, 0])
        else:
            current_value = float(obs[track_state_idx])

        for step in range(n_steps):
            # Get reference value
            if reference_signal.ndim > 1:
                ref_val = float(reference_signal[0, min(step, reference_signal.shape[1] - 1)])
            else:
                ref_val = float(reference_signal[min(step, len(reference_signal) - 1)])

            # PID computation
            error = ref_val - current_value
            integral = integral + error * dt
            derivative = (error - prev_error) / dt if dt > 0 else 0.0
            control = kp * error + ki * integral + kd * derivative
            prev_error = error

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
        ise = float(np.sum(error ** 2) * dt)

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

    def get_param_env(self):
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

    def save(self, path=None):
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
    def __load(cls, path):
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
    def from_pretrained(cls, repo_name, access_token=None, version=None):
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
