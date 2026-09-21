"""Incremental Approximate Dynamic Programming (iADP) agent.

Based on the TU Delft / DLR flight-test paper

    Konatala, Milz, Weiser, Looye, van Kampen,
    *"Flight Testing Reinforcement Learning based Online Adaptive Flight
    Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024,
    DOI: 10.2514/6.2024-2402.

The controller implements the three main blocks in Fig. 2 of the paper:

1. **Incremental model identification.** An online fixed-forgetting RLS
   tracks the parameter matrix ``Θ̃ = [F̃; G̃]^T`` of the locally
   linearised plant ``ΔX_{t+1} ≈ F̃ ΔX_t + G̃ Δδ_t`` where the
   augmented state ``X_t = [x_t; x_t^r]`` stacks system states and the
   reference being tracked. See :class:`IncrementalRLS`.

2. **Policy evaluation.** A quadratic value-function approximation
   ``V_π(X_t) = X_t^T P̃ X_t`` is fitted by batch least-squares to the
   Bellman residuals on a sliding window of the most recent transitions.
   Each sample contributes one scalar equation
   ``(X_t ⊗ X_t)^T · vec(P̃^{j+1}) = c_t + γ · X_{t+1}^T P̃^{j} X_{t+1}``
   (paper eq. (10)), and ``P̃`` is symmetrised after each solve.

3. **Policy improvement.** Minimising the cost-to-go w.r.t. ``Δδ_t``
   gives the closed-form control increment of paper eq. (11)::

       Δδ_t = −(R + γ G̃^T P̃ G̃)^{-1}
              · [R δ_{t-1} + γ G̃^T P̃ X̂_t + γ G̃^T P̃ F̃ ΔX̂_t].

The one-step cost is the standard LQT quadratic form,
``c_t = (y_t − y_t^r)^T Q (y_t − y_t^r) + δ_t^T R δ_t`` (paper eq. (5)),
and the augmented state ``X_t = [x_t; x_t^r]`` is constructed internally
from the user-supplied observation and reference so the public
``predict`` / ``learn`` API matches the other online adaptive-critic
agents of ``tensoraerospace``.

Use ``IADPConfig.paper(...)`` for unregularized batch evaluation, independent
plant/reference state dimensions and the published CLA/SLA phase schedules.
The default paper profile is continuous learning, including its initial
open-loop identification phase. ``IADPConfig()`` uses the same equations with configurable experiment timing.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np

from tensoraerospace.optimization.agent import OptimizableAgent

from .rls import IncrementalRLS


@dataclass
class IADPConfig:
    """Configuration of the single published iADP update law.

    ``paper(...)`` supplies the reported CLA/SLA experiment schedule. Direct
    construction configures another experiment with the same equations.
    ``Q`` weights output error, ``R`` weights actual control, ``gamma`` is the
    Bellman discount, and ``gamma_rls`` is fixed RLS forgetting per sample.
    Output maps have shapes (n_output, n_state) and (n_output, n_reference).
    ``P_init`` is the initial quadratic value kernel. Its default is a
    tracking-shaped positive seed, not the unpublished flight-test tuning.

    Critic evaluation is unregularized batch least squares: no eigenvalue
    projection, update blending or alternative policy inverse is available.
    Window length, minimum samples and update cadence remain configurable.
    Sufficient sample count does not imply informative excitation.
    """

    dt: float = 0.01
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    gamma: float = 0.8
    gamma_rls: float = 0.995
    phi_init: float = 1e2
    policy_eval_window: int = 200
    policy_eval_every: int = 50
    policy_eval_iterations: int = 1
    policy_eval_warmup_updates: int = 20
    model_learning_only_steps: int = 0
    excitation_signal: Optional[np.ndarray] = None
    F_init: Optional[np.ndarray] = None
    G_init: Optional[np.ndarray] = None
    P_init: Optional[np.ndarray] = None
    u_magnitude_limit: float = 25.0
    u_rate_limit: float = 60.0
    seed: Optional[int] = None
    history: dict = field(default_factory=dict)
    policy_eval_min_samples: Optional[int] = None
    # Paper architecture: independent plant/reference states and output maps.
    n_reference: Optional[int] = None
    output_matrix: Optional[np.ndarray] = None
    reference_output_matrix: Optional[np.ndarray] = None
    learning_mode: str = "continuous"
    controller_training_steps: Optional[int] = None
    policy_training_start_step: int = 0
    continuous_excitation_signal: Optional[np.ndarray] = None

    @classmethod
    def paper(
        cls,
        *,
        excitation_signal: np.ndarray,
        dt: float = 0.001,
        learning_mode: str = "continuous",
        model_learning_seconds: float = 20.0,
        controller_training_seconds: float = 40.0,
        critic_window_seconds: float = 20.0,
        critic_update_hz: float = 20.0,
        **kwargs: Any,
    ) -> "IADPConfig":
        """Konatala (2024), Fig. 2 and Section III.B experimental protocol.

        Continuous learning retains online identification after excitation.
        Sequential learning freezes the model after identification and freezes
        the critic after controller training. Durations are configurable;
        they are experiment settings, not universal aircraft tuning.
        No ridge, PSD projection or blending is added to the paper's LS solve.
        """

        def ticks(seconds: float) -> int:
            """Convert a positive duration to an exact integer number of sampling
            intervals.
            """
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(seconds):
                raise ValueError("dt and durations must be finite and positive")
            value = seconds / dt
            if value <= 0 or not np.isclose(value, round(value)):
                raise ValueError("duration must be a positive integer multiple of dt")
            return int(round(value))

        if not np.isfinite(critic_update_hz) or critic_update_hz <= 0:
            raise ValueError("critic_update_hz must be positive")
        model_steps = ticks(model_learning_seconds)
        excitation = np.asarray(excitation_signal, dtype=float)
        if (
            excitation.ndim != 2
            or excitation.shape[0] < model_steps
            or not np.isfinite(excitation).all()
            or not np.any(np.ptp(excitation, axis=0) > 0)
        ):
            raise ValueError(
                "paper schedule requires varying excitation covering model learning"
            )
        training_steps = ticks(controller_training_seconds)
        window = ticks(critic_window_seconds)
        settings: dict[str, Any] = dict(
            dt=dt,
            learning_mode=learning_mode,
            model_learning_only_steps=model_steps,
            controller_training_steps=training_steps,
            excitation_signal=excitation_signal,
            policy_eval_window=window,
            policy_eval_min_samples=window,
            policy_eval_every=ticks(1 / critic_update_hz),
            policy_eval_warmup_updates=0,
            # The reported SLA trial fits during the final five seconds of
            # controller training (55--60 s with the default phase lengths).
            policy_training_start_step=(
                model_steps + max(0, training_steps - round(5.0 / dt))
                if learning_mode == "sequential"
                else 0
            ),
        )
        settings.update(kwargs)
        return cls(**settings)


def _as_array(value: Any) -> Optional[np.ndarray]:
    """Return a numpy-float64 copy of ``value`` or ``None``."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    return arr.copy()


def _validate_cost_weight(weight: np.ndarray, name: str) -> np.ndarray:
    """Equation (5) requires finite, symmetric positive-semidefinite weights."""
    if not np.all(np.isfinite(weight)) or not np.allclose(
        weight, weight.T, rtol=1e-12, atol=0.0
    ):
        raise ValueError(f"{name} must be finite and symmetric")
    symmetric = 0.5 * weight + 0.5 * weight.T
    tolerance = 8 * np.finfo(float).eps * len(weight) * np.max(np.abs(weight))
    if np.linalg.eigvalsh(symmetric).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


class IADPAgent(OptimizableAgent):
    """Incremental Approximate Dynamic Programming control agent.

    The agent tracks a user-supplied reference on the observed state and
    re-identifies the plant online in continuous learning mode. See the
    module-level docstring for the full algorithm.

    Args:
        n_state: Number of observed plant states ``x_t``. The augmented
            state dimension is ``n_state + n_reference``; ``n_reference``
            defaults to ``n_state`` for compatibility.
        n_control: Number of control channels ``δ_t``.
        config: :class:`IADPConfig` instance. Use ``IADPConfig.paper`` for
            the published algorithm profile. Tuning is plant-specific.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        config: Optional[IADPConfig] = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else IADPConfig()
        self._initialize_tracking_config()

        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        self._initialize_cost_weights()

        # --- incremental model identifier ---
        self.rls = IncrementalRLS(
            n_output=self.n_aug,
            n_regressor=self.n_aug + self.n_control,
            gamma_rls=self.cfg.gamma_rls,
            phi_init=self.cfg.phi_init,
        )
        F_init = _as_array(self.cfg.F_init)
        if F_init is not None:
            if F_init.shape != (self.n_aug, self.n_aug):
                raise ValueError(
                    f"F_init must have shape ({self.n_aug}, {self.n_aug}),"
                    f" got {F_init.shape}"
                )
            self.rls.theta[: self.n_aug, :] = F_init.T
        G_init = _as_array(self.cfg.G_init)
        if G_init is not None:
            if G_init.shape != (self.n_aug, self.n_control):
                raise ValueError(
                    f"G_init must have shape ({self.n_aug}, {self.n_control}),"
                    f" got {G_init.shape}"
                )
            self.rls.theta[self.n_aug :, :] = G_init.T

        # --- kernel matrix (value function parameters) ---
        P_init = _as_array(self.cfg.P_init)
        if P_init is None:
            # The article does not publish its initial P. Couple the plant
            # and reference outputs with a positive tracking-shaped seed.
            error_map = np.hstack([self.C, -self.Cr])
            P_init = error_map.T @ self.Q @ error_map + 1e-6 * np.eye(self.n_aug)
        if P_init.shape != (self.n_aug, self.n_aug):
            raise ValueError(
                f"P_init must have shape ({self.n_aug}, {self.n_aug}),"
                f" got {P_init.shape}"
            )
        self.P = 0.5 * (P_init + P_init.T)
        self.P = _validate_cost_weight(self.P, "P_init")
        if not np.isfinite(self.rls.theta).all():
            raise ValueError("F_init and G_init must be finite")

        self._validate_policy_sample_threshold()

        # --- rolling state ---
        self._X_prev: Optional[np.ndarray] = None
        self._delta_prev = np.zeros(self.n_control, dtype=np.float64)
        self._last_X: Optional[np.ndarray] = None
        self._last_dX: np.ndarray = np.zeros(self.n_aug, dtype=np.float64)
        self._last_delta: np.ndarray = np.zeros(self.n_control, dtype=np.float64)
        self._last_d_delta: np.ndarray = np.zeros(self.n_control, dtype=np.float64)
        self._step: int = 0

        # Transition window: list of (X_t, X_{t+1}, c_t).
        self._window: collections.deque = collections.deque(
            maxlen=int(self.cfg.policy_eval_window)
        )

    def _initialize_tracking_config(self) -> None:
        """Validate output maps and the configured learning schedule."""
        self.n_reference = (
            self.n_state if self.cfg.n_reference is None else int(self.cfg.n_reference)
        )
        if min(self.n_state, self.n_control, self.n_reference) <= 0:
            raise ValueError("state, control and reference dimensions must be positive")
        self.n_aug = self.n_state + self.n_reference
        self.C = (
            np.eye(self.n_state)
            if self.cfg.output_matrix is None
            else np.asarray(self.cfg.output_matrix, dtype=float)
        )
        if self.C.ndim != 2 or self.C.shape[1] != self.n_state:
            raise ValueError("output_matrix must have n_state columns")
        self.n_output = self.C.shape[0]
        self.Cr = (
            np.eye(self.n_reference)
            if self.cfg.reference_output_matrix is None
            else np.asarray(self.cfg.reference_output_matrix, dtype=float)
        )
        if self.Cr.shape != (self.n_output, self.n_reference):
            raise ValueError(
                "reference_output_matrix must map reference states to outputs"
            )
        if not np.isfinite(self.C).all() or not np.isfinite(self.Cr).all():
            raise ValueError("output maps must be finite")
        if self.cfg.learning_mode not in ("continuous", "sequential"):
            raise ValueError("learning_mode must be continuous or sequential")
        if self.cfg.learning_mode == "sequential" and (
            self.cfg.controller_training_steps is None
            or self.cfg.controller_training_steps <= 0
        ):
            raise ValueError(
                "sequential learning requires controller_training_steps > 0"
            )
        if not 0.0 < self.cfg.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")

    def _initialize_cost_weights(self) -> None:
        """Build and validate the quadratic output and control costs."""
        # --- quadratic weights ---
        Q = _as_array(self.cfg.Q)
        if Q is None:
            Q = np.eye(self.n_output, dtype=np.float64)
        if Q.shape != (self.n_output, self.n_output):
            raise ValueError(
                f"Q must have shape ({self.n_output}, {self.n_output}), got {Q.shape}"
            )
        R = _as_array(self.cfg.R)
        if R is None:
            R = np.eye(self.n_control, dtype=np.float64)
        if R.shape != (self.n_control, self.n_control):
            raise ValueError(
                f"R must have shape ({self.n_control}, {self.n_control}),"
                f" got {R.shape}"
            )
        self.Q = _validate_cost_weight(Q, "Q")
        self.R = _validate_cost_weight(R, "R")

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @property
    def F(self) -> np.ndarray:
        """Current ``F̃`` estimate, shape ``(n_aug, n_aug)``."""
        return self.rls.theta[: self.n_aug, :].T.copy()

    @property
    def G(self) -> np.ndarray:
        """Current ``G̃`` estimate, shape ``(n_aug, n_control)``."""
        return self.rls.theta[self.n_aug :, :].T.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _slice_reference(self, reference: np.ndarray, time_step: int) -> np.ndarray:
        """Select and copy the reference sample, broadcasting scalar commands as needed."""
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 0:
            return np.full(self.n_reference, float(ref), dtype=np.float64)
        if ref.ndim == 1:
            if ref.size == self.n_reference:
                return ref.astype(np.float64, copy=True)
            idx = int(np.clip(time_step, 0, ref.size - 1))
            return np.full(self.n_reference, float(ref[idx]), dtype=np.float64)
        if ref.ndim == 2:
            idx = int(np.clip(time_step, 0, ref.shape[1] - 1))
            col = ref[:, idx]
            if col.size == self.n_reference:
                return col.astype(np.float64, copy=True)
            if col.size == 1:
                return np.full(self.n_reference, float(col[0]), dtype=np.float64)
            raise ValueError(
                f"reference column has {col.size} entries, expected {self.n_reference}"
            )
        raise ValueError("reference must be scalar, 1-D, or 2-D")

    def _augment(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Concatenate measured plant states and reference states for the value
        function.
        """
        return np.concatenate([x, ref])

    def _compute_policy_increment(
        self, X_t: np.ndarray, dX_t: np.ndarray
    ) -> np.ndarray:
        """Paper eq. (11): closed-form cost-to-go minimiser."""
        F = self.F
        G = self.G
        P = self.P
        R = self.R
        gamma = float(self.cfg.gamma)

        GTP = G.T @ P  # (n_control, n_aug)
        H = R + gamma * (GTP @ G)  # (n_control, n_control)
        rhs = R @ self._delta_prev + gamma * (GTP @ X_t) + gamma * (GTP @ (F @ dX_t))
        return np.asarray(-np.linalg.solve(H, rhs))

    def _excitation(self, step: int) -> np.ndarray:
        """Return the open-loop identification control at ``step``."""
        exc = self.cfg.excitation_signal
        if exc is None:
            return np.zeros(self.n_control, dtype=np.float64)
        exc_arr = np.asarray(exc, dtype=np.float64)
        if exc_arr.ndim == 1:
            if exc_arr.size == self.n_control:
                return exc_arr.astype(np.float64, copy=True)
            idx = int(np.clip(step, 0, exc_arr.size - 1))
            return np.full(self.n_control, float(exc_arr[idx]), dtype=np.float64)
        if exc_arr.ndim == 2:
            if exc_arr.shape[1] != self.n_control:
                raise ValueError(
                    f"excitation_signal second dim must be {self.n_control},"
                    f" got {exc_arr.shape[1]}"
                )
            idx = int(np.clip(step, 0, exc_arr.shape[0] - 1))
            return np.asarray(exc_arr[idx].astype(np.float64, copy=True))
        raise ValueError("excitation_signal must be 1-D or 2-D")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        """Current experimental phase, independent of reference time indexing."""
        if self._step < self.cfg.model_learning_only_steps:
            return "model_learning"
        if self.cfg.learning_mode == "sequential" and self._step >= (
            self.cfg.model_learning_only_steps
            + cast(int, self.cfg.controller_training_steps)
        ):
            return "assessment"
        return "controller_training"

    def reset(self, *, initial_action: Optional[np.ndarray] = None) -> None:
        """Clear per-episode rolling state (keeps learned ``F̃``, ``G̃``,
        ``P̃``)."""
        self._X_prev = None
        self._delta_prev = np.zeros(self.n_control, dtype=np.float64)
        self._last_X = None
        self._last_dX = np.zeros(self.n_aug, dtype=np.float64)
        self._last_delta = np.zeros(self.n_control, dtype=np.float64)
        self._last_d_delta = np.zeros(self.n_control, dtype=np.float64)
        self._step = 0
        self._window.clear()
        if initial_action is not None:
            initial = np.asarray(initial_action, dtype=float).reshape(-1)
            if initial.size != self.n_control or not np.isfinite(initial).all():
                raise ValueError("initial_action must contain n_control finite values")
            self._delta_prev = initial.copy()

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Compute the commanded control for the current step.

        Args:
            x_obs: System state ``x_t`` of shape ``(n_state,)``.
            reference: Reference signal. Accepted shapes: scalar,
                ``(n_state,)`` (per-channel constant), ``(T,)`` (shared
                schedule across channels) or ``(n_state, T)``
                (per-channel schedule).
            time_step: Index into the reference schedule.
            deterministic: Unused — kept for API parity with stochastic
                agents.

        Returns:
            Control command ``δ`` of shape ``(n_control,)``, rate- and
            magnitude-limited per the configuration.
        """
        del deterministic
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        if x.size != self.n_state:
            raise ValueError(f"x_obs must have length {self.n_state}, got {x.size}")
        ref = self._slice_reference(reference, time_step)
        X_t = self._augment(x, ref)
        if not np.isfinite(X_t).all():
            raise ValueError("state and reference must be finite")

        if self._X_prev is not None:
            dX_t = X_t - self._X_prev
        else:
            dX_t = np.zeros(self.n_aug, dtype=np.float64)

        if self._step < int(self.cfg.model_learning_only_steps):
            # Initial open-loop identification phase.
            delta_cmd = self._excitation(self._step)
            d_delta = delta_cmd - self._delta_prev
        else:
            d_delta = self._compute_policy_increment(X_t, dX_t)
            if (
                self.cfg.continuous_excitation_signal is not None
                and self.phase != "assessment"
            ):
                excitation = np.asarray(
                    self.cfg.continuous_excitation_signal, dtype=float
                )
                if (
                    excitation.ndim != 2
                    or excitation.shape[1] != self.n_control
                    or not len(excitation)
                    or not np.isfinite(excitation).all()
                ):
                    raise ValueError(
                        "continuous_excitation_signal must have shape (T, n_control) and be finite"
                    )
                index = (self._step - self.cfg.model_learning_only_steps) % len(
                    excitation
                )
                d_delta = d_delta + excitation[index]
        # Excitation and policy commands share the same actuator envelope.
        du_max = float(self.cfg.u_rate_limit) * float(self.cfg.dt)
        d_delta = np.clip(d_delta, -du_max, du_max)
        delta_cmd = self._delta_prev + d_delta
        delta_cmd = np.clip(
            delta_cmd,
            -float(self.cfg.u_magnitude_limit),
            float(self.cfg.u_magnitude_limit),
        )
        # Re-derive the actual applied increment after saturation.
        d_delta = delta_cmd - self._delta_prev

        # Cache for learn().
        self._last_X = X_t.copy()
        self._last_dX = dX_t.copy()
        self._last_delta = delta_cmd.copy()
        self._last_d_delta = d_delta.copy()
        return delta_cmd

    def learn(
        self,
        next_x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
        *,
        applied_action: Optional[np.ndarray] = None,
    ) -> dict:
        """Update the online estimators from the newly observed state.

        Must be called once per environment step, **after**
        :meth:`predict` and the corresponding ``env.step(δ)``.

        Args:
            next_x_obs: System state measured at ``t + 1``.
            reference: Reference signal (same shape conventions as in
                :meth:`predict`).
            time_step: Same index passed to :meth:`predict`.
            applied_action: Actual actuator input for this transition in the
                same units as predict(). Pass feedback if the plant clips or
                lags the command. None assumes the command was applied exactly.

        Returns:
            Scalar diagnostics: RLS residual norm, ``F̃``/``G̃``/``P̃``
            norms, and the one-step cost.
        """
        x_next = np.asarray(next_x_obs, dtype=np.float64).reshape(-1)
        if x_next.size != self.n_state:
            raise ValueError(
                f"next_x_obs must have length {self.n_state}, got {x_next.size}"
            )
        ref_next = self._slice_reference(reference, time_step + 1)
        X_next = self._augment(x_next, ref_next)
        applied = np.asarray(
            self._last_delta if applied_action is None else applied_action,
            dtype=np.float64,
        ).reshape(-1)
        if applied.size != self.n_control or not np.all(np.isfinite(applied)):
            raise ValueError("applied_action must contain n_control finite values")
        if not np.all(np.isfinite(X_next)):
            raise ValueError("next state and reference must be finite")
        self._last_delta = applied.copy()
        self._last_d_delta = applied - self._delta_prev

        eps_norm = 0.0
        cost_t = 0.0
        if self._last_X is not None:
            # Incremental identification needs two consecutive transitions.
            # After reset there is no measured X_{t-1}; assuming dX_t=0
            # attributes the initial free response to control effectiveness.
            W = np.concatenate([self._last_dX, self._last_d_delta])
            if self._X_prev is not None:
                dX_target = X_next - self._last_X
                if (
                    self.cfg.learning_mode == "continuous"
                    or self.phase == "model_learning"
                ):
                    eps = self.rls.update(W, dX_target)
                else:
                    eps = dX_target - self.rls.predict(W)
                eps_norm = float(np.linalg.norm(eps))

            # Cost evaluated at time ``t`` using the action δ_t that was
            # applied to get from X̂_t to X̂_{t+1}.
            x_t = self._last_X[: self.n_state]
            r_t = self._last_X[self.n_state :]
            err = self.C @ x_t - self.Cr @ r_t
            cost_t = float(
                err @ self.Q @ err + self._last_delta @ self.R @ self._last_delta
            )

            # Fig. 2 / Eq. 10 bootstrap from the incremental model, not the
            # noisy next measurement used to identify that model. The first
            # transition has no previous state increment, so retain its
            # measured target while the incremental history warms up.
            value_next = X_next
            if self._X_prev is not None:
                value_next = self._last_X + self.rls.predict(W)
            self._learn_value_transition(self._last_X, value_next, cost_t)

        # Keep X_t as the previous state for predict(X_{t+1}). Storing
        # X_{t+1} here would make every subsequent state increment zero.
        self._X_prev = self._last_X.copy() if self._last_X is not None else None
        if self._last_delta is not None:
            self._delta_prev = self._last_delta.copy()
        self._step += 1

        return {
            "rls_pred_error_norm": eps_norm,
            "cost": cost_t,
            "F_norm": float(np.linalg.norm(self.F)),
            "G_norm": float(np.linalg.norm(self.G)),
            "P_norm": float(np.linalg.norm(self.P)),
        }

    # ------------------------------------------------------------------
    # Policy evaluation — batch LS fit of the kernel matrix P̃.
    # ------------------------------------------------------------------
    def _validate_policy_sample_threshold(self) -> None:
        """Reject an explicit data threshold that cannot fit in the window."""
        count = self.cfg.policy_eval_min_samples
        if count is not None and (
            not np.isfinite(count)
            or int(count) != count
            or count < max(self.n_aug**2, 4)
            or count > self.cfg.policy_eval_window
        ):
            raise ValueError(
                "policy_eval_min_samples must be an integer between "
                "max(n_aug**2, 4) and policy_eval_window"
            )

    def _learn_value_transition(
        self, state: np.ndarray, next_state: np.ndarray, cost: float
    ) -> None:
        """Train the critic only after the model-only identification phase."""
        if self.phase in ("model_learning", "assessment"):
            return
        self._window.append(
            {"X": state.copy(), "Xnext": next_state.copy(), "cost": cost}
        )
        ready = self.rls.num_updates >= int(
            self.cfg.policy_eval_warmup_updates
        ) and len(self._window) >= max(
            self.n_aug**2, 4, self.cfg.policy_eval_min_samples or 0
        )
        every = max(1, int(self.cfg.policy_eval_every))
        if (
            ready
            and self._step >= self.cfg.policy_training_start_step
            and (self._step + 1) % every == 0
        ):
            self._policy_evaluation()

    def _policy_evaluation(self) -> None:
        """Fit ``P̃`` to the Bellman residuals over the current window."""
        n_aug = self.n_aug
        count = len(self._window)
        if count < 2:
            return
        states = np.stack([sample["X"] for sample in self._window])
        following = np.stack([sample["Xnext"] for sample in self._window])
        costs = np.asarray([sample["cost"] for sample in self._window])
        design = np.einsum("ni,nj->nij", states, states).reshape(count, -1)
        kernel = self.P.copy()
        for _ in range(max(1, int(self.cfg.policy_eval_iterations))):
            target = costs + self.cfg.gamma * np.einsum(
                "ni,ij,nj->n", following, kernel, following
            )
            # SVD least squares is the Moore-Penrose solution in Fig. 2.
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            kernel = coefficients.reshape(n_aug, n_aug)
            kernel = 0.5 * (kernel + kernel.T)
        self.P = kernel

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config dict for :meth:`save`."""
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict.pop("history", None)
        for key in (
            "Q",
            "R",
            "F_init",
            "G_init",
            "P_init",
            "excitation_signal",
            "output_matrix",
            "reference_output_matrix",
            "continuous_excitation_signal",
        ):
            value = cfg_dict.get(key)
            if isinstance(value, np.ndarray):
                cfg_dict[key] = value.tolist()
        return {
            "policy": {
                "name": agent_name,
                "params": {
                    "n_state": self.n_state,
                    "n_control": self.n_control,
                },
                "config": cfg_dict,
            },
        }

    def save(self, path: Union[str, Path, None] = None) -> str:
        """Write the agent to a directory.

        Files produced:
            * ``config.json`` — agent / config metadata with all arrays
              serialised as lists.
            * ``rls.npz`` — ``theta``, ``Phi`` and update counter.
            * ``value.npz`` — current kernel matrix ``P̃``.
            * ``weights.npz`` — ``Q`` and ``R`` (as applied, including
              any defaults filled in by the constructor).
            * ``loop_state.npz`` — rolling state (``X_prev``,
              ``delta_prev``, last-tick caches, step counter). With this
              present a saved checkpoint resumes bit-identically mid
              episode.
            * ``window.npz`` — the transition buffer used by the policy
              evaluator, stored as stacked ``X`` / ``Xnext`` / ``cost``
              arrays.

        Args:
            path: Base directory (``None`` → CWD).

        Returns:
            Absolute path to the created run directory.
        """
        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        np.savez(
            run_dir / "rls.npz",
            theta=self.rls.theta,
            Phi=self.rls.Phi,
            num_updates=np.asarray(self.rls.num_updates),
            last_residual=self.rls.last_residual,
        )
        np.savez(run_dir / "value.npz", P=self.P)
        np.savez(run_dir / "weights.npz", Q=self.Q, R=self.R)

        np.savez(
            run_dir / "loop_state.npz",
            X_prev=(
                self._X_prev
                if self._X_prev is not None
                else np.zeros(self.n_aug, dtype=np.float64)
            ),
            has_X_prev=np.asarray(self._X_prev is not None),
            delta_prev=self._delta_prev,
            last_X=(
                self._last_X
                if self._last_X is not None
                else np.zeros(self.n_aug, dtype=np.float64)
            ),
            has_last_X=np.asarray(self._last_X is not None),
            last_dX=self._last_dX,
            last_delta=self._last_delta,
            last_d_delta=self._last_d_delta,
            step=np.asarray(self._step),
        )

        window = list(self._window)
        if window:
            np.savez(
                run_dir / "window.npz",
                X=np.stack([s["X"] for s in window]),
                Xnext=np.stack([s["Xnext"] for s in window]),
                cost=np.asarray([s["cost"] for s in window], dtype=np.float64),
            )
        else:
            np.savez(
                run_dir / "window.npz",
                X=np.zeros((0, self.n_aug), dtype=np.float64),
                Xnext=np.zeros((0, self.n_aug), dtype=np.float64),
                cost=np.zeros((0,), dtype=np.float64),
            )
        return str(run_dir)

    @classmethod
    def _load_from_dir(cls, folder: Union[str, Path]) -> "IADPAgent":
        """Reconstruct an agent from a :meth:`save` directory."""
        folder_p = Path(folder)
        config_path = folder_p / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {str(folder_p)!r}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy = cfg.get("policy", {})
        params = policy.get("params", {})
        cfg_dict = dict(policy.get("config", {}))

        for key in (
            "Q",
            "R",
            "F_init",
            "G_init",
            "P_init",
            "excitation_signal",
            "output_matrix",
            "reference_output_matrix",
            "continuous_excitation_signal",
        ):
            value = cfg_dict.get(key)
            if value is not None:
                cfg_dict[key] = np.asarray(value, dtype=np.float64)

        agent_cfg = IADPConfig(**cfg_dict)
        agent = cls(
            n_state=params["n_state"],
            n_control=params["n_control"],
            config=agent_cfg,
        )

        rls_path = folder_p / "rls.npz"
        if rls_path.exists():
            with np.load(rls_path) as npz:
                agent.rls.theta = npz["theta"]
                agent.rls.Phi = npz["Phi"]
                agent.rls.num_updates = int(npz["num_updates"])
                agent.rls.last_residual = npz["last_residual"]

        value_path = folder_p / "value.npz"
        if value_path.exists():
            with np.load(value_path) as npz:
                agent.P = npz["P"]

        weights_path = folder_p / "weights.npz"
        if weights_path.exists():
            with np.load(weights_path) as npz:
                agent.Q = npz["Q"]
                agent.R = npz["R"]

        loop_path = folder_p / "loop_state.npz"
        if loop_path.exists():
            with np.load(loop_path) as npz:
                agent._X_prev = npz["X_prev"] if bool(npz["has_X_prev"]) else None
                agent._delta_prev = npz["delta_prev"]
                agent._last_X = npz["last_X"] if bool(npz["has_last_X"]) else None
                agent._last_dX = npz["last_dX"]
                agent._last_delta = npz["last_delta"]
                agent._last_d_delta = npz["last_d_delta"]
                agent._step = int(npz["step"])

        window_path = folder_p / "window.npz"
        if window_path.exists():
            with np.load(window_path) as npz:
                X = npz["X"]
                Xn = npz["Xnext"]
                cost = npz["cost"]
            agent._window.clear()
            for i in range(X.shape[0]):
                agent._window.append(
                    {
                        "X": X[i].copy(),
                        "Xnext": Xn[i].copy(),
                        "cost": float(cost[i]),
                    }
                )
        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "IADPAgent":
        """Load an agent from a local directory or Hugging Face Hub.

        Args:
            repo_name: Local folder path, or ``namespace/repo_name`` on
                the Hugging Face Hub.
            access_token: Hub access token for private repos.
            version: Hub revision / branch / tag.

        Returns:
            IADPAgent: Reconstructed agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(
                f"Local directory not found: '{repo_name}'. Please check the path."
            )

        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls._load_from_dir(folder_path)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a :meth:`save` directory to the Hugging Face Hub.

        Args:
            repo_name: Target repository id, e.g. ``"me/my-iadp"``.
            folder_path: Local folder produced by :meth:`save`.
            access_token: Hub access token.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )
