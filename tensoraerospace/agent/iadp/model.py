"""Incremental Approximate Dynamic Programming (iADP) agent.

Based on the TU Delft / DLR flight-test paper

    Konatala, Milz, Weiser, Looye, van Kampen,
    *"Flight Testing Reinforcement Learning based Online Adaptive Flight
    Control Laws on CS-25 Class Aircraft"*, AIAA SCITECH 2024,
    DOI: 10.2514/6.2024-2402.

The controller combines three blocks, exactly mirroring Fig. 2 of the
paper:

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

The agent operates in the paper's **Continuous Learning Approach** by
default — model learning, controller training and controller assessment
run concurrently from the first step. Provide
``model_learning_only_steps > 0`` with ``excitation_signal`` to recover
the **Sequential Learning Approach** in which the first ``N`` ticks are
an open-loop identification window.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .rls import IncrementalRLS


@dataclass
class IADPConfig:
    """Hyper-parameters for :class:`IADPAgent`.

    Args:
        dt: Control step [s].
        Q: Tracking-error weight matrix of shape ``(n_state, n_state)``.
            Defaults to the identity.
        R: Control weight matrix of shape ``(n_control, n_control)``.
            Defaults to the identity.
        gamma: Discount factor ``γ ∈ (0, 1)`` used in the Bellman
            equation of eq. (3). Smaller values keep the cost-to-go
            short-sighted; the paper uses values around ``0.6–0.9``
            tuned offline via MOPS.
        gamma_rls: Constant RLS forgetting factor for the incremental
            model. See :class:`IncrementalRLS`.
        phi_init: Initial RLS covariance scale.
        policy_eval_window: Number of recent transitions used to fit
            ``P̃`` at each policy-evaluation tick. Must be larger than
            ``n_aug^2`` for the LS to be well-posed.
        policy_eval_every: Stride in ``learn()`` ticks between
            policy-evaluation updates. The paper reports a 20 Hz
            controller-training loop against a 1 kHz model-learning
            loop — ``policy_eval_every ≈ 50`` at a ``dt = 0.01`` sim
            replicates that.
        policy_eval_iterations: Inner fixed-point sweeps per
            policy-evaluation tick. One sweep is usually enough.
        policy_eval_regularization: Ridge term added to the LS normal
            equations for numerical stability.
        policy_eval_warmup_updates: Skip policy evaluation until the RLS
            identifier has seen at least this many updates. Gives the
            incremental model a chance to settle before the LS sees
            noisy ``F̃``/``G̃`` estimates.
        enforce_psd: When True, after each batch-LS solve for ``P̃`` the
            eigenvalues are clipped to ``[psd_floor, +∞)`` and the
            matrix is reconstructed. The paper defines ``P̃`` to be
            positive-definite so that the cost-to-go stays convex; in
            finite-window LS this isn't automatic, and a non-PSD
            ``P̃`` can destabilise the closed-form policy eq. (11).
        psd_floor: Lower bound for the eigenvalue clip used when
            ``enforce_psd`` is True.
        policy_eval_blend: Exponential-moving-average coefficient for
            ``P̃`` updates. After each batch-LS solve the new kernel
            matrix ``P_solved`` is mixed with the previous one via
            ``P̃ ← blend·P_solved + (1-blend)·P̃``. The default ``1.0``
            replaces ``P̃`` outright (the paper's behaviour) but causes
            a small step change in the policy every
            ``policy_eval_every`` ticks — visible as a sawtooth on the
            control trace. Values around ``0.2–0.4`` smooth the control
            output without measurably degrading tracking, and are
            analogous to soft-target updates used elsewhere in RL.
        model_learning_only_steps: Number of initial steps during which
            the agent ignores the policy and outputs
            ``excitation_signal`` (or zero if it is ``None``). Replicates
            the paper's 20–25 s open-loop Sequential Learning phase.
        excitation_signal: Optional ``(T, n_control)`` schedule of
            absolute control values used during
            ``model_learning_only_steps``. When ``None`` the agent
            outputs zero during the open-loop window.
        F_init: Optional warm-start for ``F̃``, shape
            ``(n_aug, n_aug)``.
        G_init: Optional warm-start for ``G̃``, shape
            ``(n_aug, n_control)``.
        P_init: Optional warm-start for the kernel matrix ``P̃``, shape
            ``(n_aug, n_aug)``. Defaults to the identity, which gives a
            non-trivial first policy output even before any
            policy-evaluation tick has fired.
        u_magnitude_limit: Hard magnitude clamp on the per-channel
            absolute control value (matches actuator envelope).
        u_rate_limit: Maximum ``|Δδ|`` per second per channel.
        pinv_rcond: Cut-off passed to ``numpy.linalg.pinv`` on the
            policy-improvement matrix inversion.
        seed: Optional RNG seed — unused by the deterministic update
            rules but forwarded to NumPy for parity with stochastic
            agents.
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
    policy_eval_regularization: float = 1e-4
    policy_eval_warmup_updates: int = 20
    enforce_psd: bool = True
    psd_floor: float = 1e-6
    policy_eval_blend: float = 1.0
    model_learning_only_steps: int = 0
    excitation_signal: Optional[np.ndarray] = None
    F_init: Optional[np.ndarray] = None
    G_init: Optional[np.ndarray] = None
    P_init: Optional[np.ndarray] = None
    u_magnitude_limit: float = 25.0
    u_rate_limit: float = 60.0
    pinv_rcond: float = 1e-8
    seed: Optional[int] = None
    history: dict = field(default_factory=dict)


def _as_array(value: Any) -> Optional[np.ndarray]:
    """Return a numpy-float64 copy of ``value`` or ``None``."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    return arr.copy()


class IADPAgent:
    """Incremental Approximate Dynamic Programming control agent.

    The agent tracks a user-supplied reference on the observed state and
    continuously re-identifies the plant model online. See the
    module-level docstring for the full algorithm.

    Args:
        n_state: Number of controlled system states ``x_t``. The
            augmented state ``X_t = [x_t; x_t^r]`` has dimension
            ``2 · n_state``.
        n_control: Number of control channels ``δ_t``.
        config: :class:`IADPConfig` instance. Defaults fit a moderately
            fast fixed-wing inner loop at ``dt = 0.01`` s.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        config: Optional[IADPConfig] = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.n_aug = 2 * self.n_state
        self.cfg = config if config is not None else IADPConfig()

        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        # --- quadratic weights ---
        Q = _as_array(self.cfg.Q)
        if Q is None:
            Q = np.eye(self.n_state, dtype=np.float64)
        if Q.shape != (self.n_state, self.n_state):
            raise ValueError(
                f"Q must have shape ({self.n_state}, {self.n_state}), got {Q.shape}"
            )
        R = _as_array(self.cfg.R)
        if R is None:
            R = np.eye(self.n_control, dtype=np.float64)
        if R.shape != (self.n_control, self.n_control):
            raise ValueError(
                f"R must have shape ({self.n_control}, {self.n_control}),"
                f" got {R.shape}"
            )
        self.Q = Q
        self.R = R

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
            P_init = np.eye(self.n_aug, dtype=np.float64)
        if P_init.shape != (self.n_aug, self.n_aug):
            raise ValueError(
                f"P_init must have shape ({self.n_aug}, {self.n_aug}),"
                f" got {P_init.shape}"
            )
        self.P = 0.5 * (P_init + P_init.T)

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
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 0:
            return np.full(self.n_state, float(ref), dtype=np.float64)
        if ref.ndim == 1:
            if ref.size == self.n_state:
                return ref.astype(np.float64, copy=True)
            idx = int(np.clip(time_step, 0, ref.size - 1))
            return np.full(self.n_state, float(ref[idx]), dtype=np.float64)
        if ref.ndim == 2:
            idx = int(np.clip(time_step, 0, ref.shape[1] - 1))
            col = ref[:, idx]
            if col.size == self.n_state:
                return col.astype(np.float64, copy=True)
            if col.size == 1:
                return np.full(self.n_state, float(col[0]), dtype=np.float64)
            raise ValueError(
                f"reference column has {col.size} entries, expected {self.n_state}"
            )
        raise ValueError("reference must be scalar, 1-D, or 2-D")

    def _augment(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
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
        H_inv = np.linalg.pinv(H, rcond=float(self.cfg.pinv_rcond))
        return np.asarray(-(H_inv @ rhs))

    def _excitation(self, step: int) -> np.ndarray:
        """Return the open-loop control at ``step`` (SLA phase)."""
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
    def reset(self) -> None:
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

        if self._X_prev is not None:
            dX_t = X_t - self._X_prev
        else:
            dX_t = np.zeros(self.n_aug, dtype=np.float64)

        if self._step < int(self.cfg.model_learning_only_steps):
            # Sequential-learning open-loop phase.
            delta_cmd = self._excitation(self._step)
            d_delta = delta_cmd - self._delta_prev
        else:
            d_delta = self._compute_policy_increment(X_t, dX_t)
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
    ) -> dict:
        """Update the online estimators from the newly observed state.

        Must be called once per environment step, **after**
        :meth:`predict` and the corresponding ``env.step(δ)``.

        Args:
            next_x_obs: System state measured at ``t + 1``.
            reference: Reference signal (same shape conventions as in
                :meth:`predict`).
            time_step: Same index passed to :meth:`predict`.

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

        eps_norm = 0.0
        cost_t = 0.0
        if self._last_X is not None:
            # Target: ΔX̂_t = X̂_t − X̂_{t-1}. On the first call after
            # predict() this is the same as ``X_next − last_X``.
            dX_target = X_next - self._last_X
            W = np.concatenate([self._last_dX, self._last_d_delta])
            eps = self.rls.update(W, dX_target)
            eps_norm = float(np.linalg.norm(eps))

            # Cost evaluated at time ``t`` using the action δ_t that was
            # applied to get from X̂_t to X̂_{t+1}.
            x_t = self._last_X[: self.n_state]
            r_t = self._last_X[self.n_state :]
            err = x_t - r_t
            cost_t = float(
                err @ self.Q @ err + self._last_delta @ self.R @ self._last_delta
            )

            self._window.append(
                {
                    "X": self._last_X.copy(),
                    "Xnext": X_next.copy(),
                    "cost": cost_t,
                }
            )

            # Policy evaluation on the controller-training cadence.
            ready = self.rls.num_updates >= int(
                self.cfg.policy_eval_warmup_updates
            ) and len(self._window) >= max(self.n_aug**2, 4)
            every = max(1, int(self.cfg.policy_eval_every))
            if ready and (self._step + 1) % every == 0:
                self._policy_evaluation()

        # Roll state.
        self._X_prev = X_next.copy()
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
    def _policy_evaluation(self) -> None:
        """Fit ``P̃`` to the Bellman residuals over the current window."""
        n_aug = self.n_aug
        N = len(self._window)
        if N < 2:
            return

        A = np.zeros((N, n_aug * n_aug), dtype=np.float64)
        # Precompute regressor rows; they don't depend on the inner sweep.
        for i, s in enumerate(self._window):
            X = s["X"]
            A[i] = np.outer(X, X).ravel()

        P_j = self.P.copy()
        lam = float(self.cfg.policy_eval_regularization)
        AtA = A.T @ A + lam * np.eye(n_aug * n_aug, dtype=np.float64)
        # Right-hand side depends on P through the discounted next-step
        # value; recompute per inner sweep.
        for _ in range(max(1, int(self.cfg.policy_eval_iterations))):
            b = np.empty(N, dtype=np.float64)
            for i, s in enumerate(self._window):
                Xn = s["Xnext"]
                b[i] = s["cost"] + float(self.cfg.gamma) * float(Xn @ P_j @ Xn)
            try:
                p = np.linalg.solve(AtA, A.T @ b)
            except np.linalg.LinAlgError:
                p = np.linalg.pinv(AtA) @ (A.T @ b)
            P_new = p.reshape(n_aug, n_aug)
            P_new = 0.5 * (P_new + P_new.T)
            if bool(self.cfg.enforce_psd):
                eigvals, eigvecs = np.linalg.eigh(P_new)
                eigvals = np.clip(eigvals, float(self.cfg.psd_floor), None)
                P_new = (eigvecs * eigvals) @ eigvecs.T
            P_j = P_new

        # Exponential-moving-average blend with the incumbent ``P̃`` to
        # smooth the step-change in the policy at every evaluation tick.
        blend = float(np.clip(self.cfg.policy_eval_blend, 0.0, 1.0))
        if blend < 1.0:
            self.P = blend * P_j + (1.0 - blend) * self.P
        else:
            self.P = P_j

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config dict for :meth:`save`."""
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict.pop("history", None)
        for key in ("Q", "R", "F_init", "G_init", "P_init", "excitation_signal"):
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

        for key in ("Q", "R", "F_init", "G_init", "P_init", "excitation_signal"):
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
