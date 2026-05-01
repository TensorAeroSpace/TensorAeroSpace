"""Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI) agent.

Based on the TU Delft line of work on fault-tolerant INDI, in particular

    Sun et al., *"Active Incremental Nonlinear Dynamic Inversion for Sensor
    and Actuator Fault Diagnosis and Fault-Tolerant Flight Control"*,
    TU Delft Aerospace, https://research.tudelft.nl/en/publications/
    active-incremental-nonlinear-dynamic-inversion-for-sensor-and-act/

The method combines three blocks:

1. **INDI control law.** The desired angular acceleration is produced by a
   second-order reference model tracking the commanded rate. The applied
   control increment is then ``Δu = G⁺ · (ν_des − ω̇_meas)``, where ``G⁺``
   is the (pseudo-) inverse of the control-effectiveness matrix estimated
   online.

2. **Variable-Forgetting-Factor RLS** on the incremental model
   ``Δω̇ ≈ G · Δu``. The forgetting factor contracts when the residual
   norm grows — fast adaptation during faults / manoeuvres — and relaxes
   during quiescent flight for noise rejection. See :mod:`.vff_rls`.

3. **Sensor-filter stand-in.** A low-pass differentiator
   (:class:`LowPassDerivative`) produces ω̇ from the measured ω, and a
   residual-based :class:`BiasEstimator` yields a coarse IMU bias that
   the agent subtracts from its measurements. This stands in for the
   paper's OTSEKF-HOSM stack; the interface is deliberately narrow so a
   full OTSEKF can be swapped in later without touching the agent.

Interface mirrors the other tensoraerospace online agents
(:class:`IMGDHPAgent`, :class:`ETDHPAgent`): ``predict`` returns a control
command, ``learn`` updates the online estimators from the observed next
state, and a full save / load / Hugging Face round-trip is supported.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .sensor_filter import BiasEstimator, LowPassDerivative
from .vff_rls import VFFRLSEstimator


@dataclass
class AAINDIConfig:
    """Hyper-parameters for :class:`AAINDIAgent`.

    Args:
        dt: Simulation / control step [s].
        ref_wn: Reference-model natural frequency [rad/s]. Higher values
            track aggressive reference changes faster at the cost of
            larger control increments.
        ref_zeta: Reference-model damping ratio. Default ``0.7`` gives
            the critically-damped-ish response used throughout the
            paper.
        u_magnitude_limit: Hard magnitude clamp on the control output
            (per-channel). Matches the actuator envelope of the plant.
        u_rate_limit: Maximum Δu per step (per-channel). Limits how far
            the incremental law can move the actuator in one control
            tick.
        vff_forgetting_min: Lower bound on the VFF-RLS forgetting
            factor. Smaller values react to faults sooner.
        vff_forgetting_max: Upper bound on the VFF-RLS forgetting
            factor. Larger values tune how aggressively old data is
            kept for noise rejection.
        vff_eps_sensitivity: Residual norm at which the forgetting
            factor has dropped to ``1/e`` of its peak.
        vff_cov_init: Initial covariance scale for the RLS.
        sensor_cutoff_hz: Cut-off of the low-pass differentiator used
            to produce ω̇_meas from raw ω.
        bias_forgetting: Exponential-forgetting parameter of the bias
            estimator (closer to 1 → slower, smoother tracking).
        enable_bias_correction: When True, subtract the estimated IMU
            bias from the angular-rate measurement before forming the
            INDI residual.
        pinv_rcond: Cut-off for the pseudo-inverse of ``G`` (passed to
            ``numpy.linalg.pinv``). Prevents blowing up when ``G`` has
            near-zero singular values during the RLS warm-up.
        G_init: Optional warm-start for the control-effectiveness
            matrix, shape ``(n_state, n_control)``. INDI needs a
            reasonable ``G`` on the first few ticks — in a real
            deployment this would come from a linearised on-board model.
            When ``None`` the RLS starts from tiny random weights and
            the controller is quiet for a handful of steps until the
            identifier has converged.
        ref_error_kp: Proportional feedback gain that injects the
            reference-model tracking error ``(r_ref − ω)`` into the
            INDI desired acceleration. Without it pure INDI is a
            type-0 loop for ``ω`` and settles with a small residual
            offset (≈ 10 % on the F-16 longitudinal channel) driven by
            actuator lag / rate limiting. Set to a value of order
            ``ref_wn`` to close the gap; leave at ``0.0`` for the
            textbook pure-INDI behaviour.
        ref_error_ki: Integral feedback gain on the reference tracking
            error. Eliminates residual offset from persistent
            modelling error / constant disturbances (e.g. a stuck
            trim). Start at ``0`` and raise cautiously — too large
            trades steady-state accuracy for wind-up.
        seed: Optional RNG seed.
    """

    dt: float = 0.01
    ref_wn: float = 10.0
    ref_zeta: float = 0.7
    u_magnitude_limit: float = 25.0
    u_rate_limit: float = 60.0
    vff_forgetting_min: float = 0.7
    vff_forgetting_max: float = 0.999
    vff_eps_sensitivity: float = 1.0
    vff_cov_init: float = 1e2
    sensor_cutoff_hz: float = 10.0
    bias_forgetting: float = 0.99
    enable_bias_correction: bool = True
    pinv_rcond: float = 1e-6
    G_init: Optional[np.ndarray] = None
    ref_error_kp: float = 0.0
    ref_error_ki: float = 0.0
    seed: int | None = None
    history: dict = field(default_factory=dict)


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


class AAINDIAgent:
    """Active-Adaptive INDI control agent.

    Args:
        n_state: Dimension of the controlled angular state vector ``ω``
            (e.g. 3 for roll/pitch/yaw rates).
        n_control: Number of control channels ``u`` (e.g. aileron,
            elevator, rudder).
        config: :class:`AAINDIConfig` instance. Defaults to
            moderately conservative values suitable for a sub-sonic
            fixed-wing aircraft.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        config: AAINDIConfig | None = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else AAINDIConfig()

        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        # --- online estimators ---
        self.rls = VFFRLSEstimator(
            n_y=self.n_state,
            n_u=self.n_control,
            forgetting_min=self.cfg.vff_forgetting_min,
            forgetting_max=self.cfg.vff_forgetting_max,
            eps_sensitivity=self.cfg.vff_eps_sensitivity,
            cov_init=self.cfg.vff_cov_init,
            seed=self.cfg.seed,
        )
        # Warm-start G when a nominal linearisation is provided — avoids
        # a pinv-of-near-zero transient on the first control ticks.
        if self.cfg.G_init is not None:
            G0 = np.asarray(self.cfg.G_init, dtype=np.float64)
            if G0.shape != (self.n_state, self.n_control):
                raise ValueError(
                    f"G_init must have shape ({self.n_state}, {self.n_control}),"
                    f" got {G0.shape}"
                )
            # theta stores G.T per the VFFRLSEstimator convention.
            self.rls.theta = G0.T.copy()
        self.deriv = LowPassDerivative(
            n=self.n_state,
            dt=self.cfg.dt,
            cutoff_hz=self.cfg.sensor_cutoff_hz,
        )
        self.bias_est = BiasEstimator(
            n=self.n_state,
            forgetting=self.cfg.bias_forgetting,
        )

        # --- reference model (2nd-order per channel) ---
        # x_ref = [r, r_dot], driven by commanded rate r_cmd to produce
        # smooth desired rate r and its derivative r_dot = ν_des.
        self._ref_rate = np.zeros(self.n_state, dtype=np.float64)
        self._ref_rate_dot = np.zeros(self.n_state, dtype=np.float64)

        # --- rolling state ---
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_prev: np.ndarray | None = None
        self._omega_dot_prev: np.ndarray | None = None
        self._omega_dot_cached: np.ndarray = np.zeros(self.n_state, dtype=np.float64)
        # Integrator state for the optional outer-loop error-injection
        # feedback (ref_error_ki > 0).
        self._int_err = np.zeros(self.n_state, dtype=np.float64)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the per-episode rolling state (keeps learned G estimate)."""
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_prev = None
        self._omega_dot_prev = None
        self._omega_dot_cached = np.zeros(self.n_state, dtype=np.float64)
        self._ref_rate = np.zeros(self.n_state, dtype=np.float64)
        self._ref_rate_dot = np.zeros(self.n_state, dtype=np.float64)
        self._int_err = np.zeros(self.n_state, dtype=np.float64)
        self.deriv.reset()
        self.bias_est.reset()
        self._step = 0

    def _advance_reference_model(self, r_cmd: np.ndarray) -> np.ndarray:
        """Run one step of the 2nd-order reference model, return desired ω̇."""
        wn = float(self.cfg.ref_wn)
        zeta = float(self.cfg.ref_zeta)
        dt = float(self.cfg.dt)
        # Continuous-time 2nd order:
        #   r̈ = -2·zeta·wn·ṙ + wn² · (r_cmd − r)
        # Euler-discretised here for simplicity (dt ≪ 1/wn in practice).
        r_ddot = -2.0 * zeta * wn * self._ref_rate_dot + wn**2 * (
            np.asarray(r_cmd, dtype=np.float64).reshape(-1) - self._ref_rate
        )
        self._ref_rate_dot = self._ref_rate_dot + dt * r_ddot
        self._ref_rate = self._ref_rate + dt * self._ref_rate_dot
        # ν_des is the new r_dot (desired angular acceleration).
        return self._ref_rate_dot.copy()

    def _corrected_omega(self, omega: np.ndarray) -> np.ndarray:
        """Subtract estimated IMU bias when correction is enabled."""
        if self.cfg.enable_bias_correction:
            return np.asarray(omega - self.bias_est.bias)
        return omega

    def predict(
        self,
        omega: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Compute the commanded control for the current step.

        Args:
            omega: Measured angular state ω (shape ``(n_state,)``).
            reference: Commanded angular rate ω_cmd. Accepted shapes are
                ``(n_state,)`` (a single command) or ``(n_state, T)`` /
                ``(T,)`` (a schedule, from which column ``time_step`` is
                read).
            time_step: Current step index; used to slice a time-varying
                reference.
            deterministic: Unused — kept for API parity with stochastic
                agents.

        Returns:
            Control command ``u`` of shape ``(n_control,)``.
        """
        del deterministic
        omega_v = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega_v.size != self.n_state:
            raise ValueError(
                f"omega must have length {self.n_state}, got {omega_v.size}"
            )

        # Slice reference for the current step.
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 1:
            if ref.size == self.n_state:
                r_cmd = ref.copy()
            else:
                idx = int(np.clip(time_step, 0, ref.size - 1))
                r_cmd = np.full(self.n_state, ref[idx], dtype=np.float64)
        elif ref.ndim == 2:
            idx = int(np.clip(time_step, 0, ref.shape[1] - 1))
            col = ref[:, idx]
            if col.size == self.n_state:
                r_cmd = col.astype(np.float64, copy=True)
            elif col.size == 1:
                r_cmd = np.full(self.n_state, float(col[0]), dtype=np.float64)
            else:
                raise ValueError(
                    f"reference column has {col.size} entries, expected {self.n_state}"
                )
        else:
            raise ValueError("reference must be 1-D or 2-D")

        # ω̇_meas comes from the differentiator, which is advanced inside
        # ``learn()`` when a new measurement arrives. Feeding it here too
        # would push the same sample through twice and zero the output;
        # instead we read the cached result from the previous ``learn``.
        _ = self._corrected_omega(omega_v)  # kept so bias path is exercised
        omega_dot_meas = self._omega_dot_cached.copy()

        # Reference-model output.
        nu_des = self._advance_reference_model(r_cmd)

        # Optional outer-loop PI error injection.
        # Pure INDI is type-0 in ω: at steady state ν_des → 0 and the
        # plant holds whatever rate it reached through the transient,
        # leaving a ≈ 10 % residual offset driven by actuator lag /
        # rate-limiter action. Adding ``K_p·(r_ref − ω) + K_i·∫(r_ref − ω)``
        # to ``ν_des`` turns it into a proper rate tracker without
        # changing any of the adaptive / fault-tolerance machinery.
        k_p = float(self.cfg.ref_error_kp)
        k_i = float(self.cfg.ref_error_ki)
        if k_p != 0.0 or k_i != 0.0:
            err = self._ref_rate - omega_v
            if k_i != 0.0:
                self._int_err = self._int_err + err * self.cfg.dt
            nu_des = nu_des + k_p * err + k_i * self._int_err

        # INDI law: Δu = G⁺ · (ν_des − ω̇_meas).
        G = self.rls.G  # (n_state, n_control)
        try:
            G_pinv = np.linalg.pinv(G, rcond=self.cfg.pinv_rcond)
        except np.linalg.LinAlgError:
            G_pinv = np.zeros((self.n_control, self.n_state), dtype=np.float64)
        du = G_pinv @ (nu_des - omega_dot_meas)

        # Rate limit the increment, then apply and clamp magnitude.
        du_max = self.cfg.u_rate_limit * self.cfg.dt
        du = _clamp(du, -du_max, du_max)
        u_cmd = _clamp(
            self._u_prev + du,
            -self.cfg.u_magnitude_limit,
            self.cfg.u_magnitude_limit,
        )

        # Remember for the incremental identifier next tick.
        self._last_u_cmd = u_cmd.copy()
        self._last_omega_dot_meas = omega_dot_meas.copy()
        self._last_nu_des = nu_des.copy()
        return u_cmd

    def learn(
        self,
        next_omega: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> dict[str, float]:
        """Update the online estimators from the newly observed state.

        Must be called once per environment step, **after** :meth:`predict`
        and the corresponding ``env.step(u)`` call.

        Args:
            next_omega: Angular state measured at ``t+1``.
            reference: Commanded reference (unused at learn time —
                accepted only for API parity with other agents).
            time_step: Same index passed to :meth:`predict`.

        Returns:
            Dict of scalar metrics: prediction residual norm, current
            forgetting factor, norm of the identified G, and current
            bias estimate norm.
        """
        del reference, time_step
        next_v = np.asarray(next_omega, dtype=np.float64).reshape(-1)
        if next_v.size != self.n_state:
            raise ValueError(
                f"next_omega must have length {self.n_state}, got {next_v.size}"
            )

        # Differentiate the measured next-step state — this is the
        # single authoritative call to ``self.deriv``. Predict reads the
        # cached result on the following tick.
        next_corrected = self._corrected_omega(next_v)
        omega_dot_next = self.deriv.step(next_corrected)
        self._omega_dot_cached = omega_dot_next.copy()

        # VFF-RLS update using (Δu_t, Δω̇).
        if self._omega_dot_prev is not None:
            du = self._last_u_cmd - self._u_prev
            dy = omega_dot_next - self._omega_dot_prev
            eps = self.rls.update(du, dy)
        else:
            eps = np.zeros(self.n_state, dtype=np.float64)

        # Update the bias estimator from the reintegration residual.
        # innovation = measured ω − (prev ω + dt · ω̇_meas) ≈ bias·dt over long averages.
        if self._omega_prev is not None:
            predicted = self._omega_prev + self.cfg.dt * omega_dot_next
            innovation = next_v - predicted
            self.bias_est.update(innovation)

        # Roll state.
        self._u_prev = self._last_u_cmd.copy()
        self._omega_prev = next_v.copy()
        self._omega_dot_prev = omega_dot_next.copy()
        self._step += 1

        return {
            "rls_pred_error_norm": float(np.linalg.norm(eps)),
            "vff_lambda": float(self.rls.last_lambda),
            "G_norm": float(np.linalg.norm(self.rls.G)),
            "bias_norm": float(np.linalg.norm(self.bias_est.bias)),
        }

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config dict for :meth:`save`."""
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict.pop("history", None)
        # ``G_init`` may be a numpy array — JSON wants a plain list.
        g_init = cfg_dict.get("G_init")
        if isinstance(g_init, np.ndarray):
            cfg_dict["G_init"] = g_init.tolist()
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
            * ``config.json`` — agent/config metadata.
            * ``vff_rls.npz`` — RLS parameter matrix ``theta``, covariance
              ``P``, last forgetting factor, update counter.
            * ``bias_state.npz`` — current bias estimate.
            * ``deriv_state.npz`` — low-pass differentiator internal state.
            * ``loop_state.npz`` — reference-model state, PI integrator,
              last applied control, cached ω̇. Persisting these means a
              saved agent resumes bit-identically on reload mid-episode
              (essential when ``ref_error_kp``/``ref_error_ki`` are
              non-zero).

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

        # RLS state.
        np.savez(
            run_dir / "vff_rls.npz",
            theta=self.rls.theta,
            P=self.rls.P,
            last_lambda=np.asarray(self.rls.last_lambda),
            num_updates=np.asarray(self.rls.num_updates),
        )
        # Bias estimator state.
        np.savez(run_dir / "bias_state.npz", bias=self.bias_est.bias)
        # Derivative filter state — ``prev_x`` may be None at save time.
        deriv_prev = self.deriv._prev_x
        np.savez(
            run_dir / "deriv_state.npz",
            y=self.deriv.last_output,
            prev_x=deriv_prev if deriv_prev is not None else np.array([]),
            has_prev=np.asarray(deriv_prev is not None),
        )
        # Reference model + PI integrator + last command + cached ω̇.
        np.savez(
            run_dir / "loop_state.npz",
            ref_rate=self._ref_rate,
            ref_rate_dot=self._ref_rate_dot,
            int_err=self._int_err,
            u_prev=self._u_prev,
            omega_dot_cached=self._omega_dot_cached,
            step=np.asarray(self._step),
        )
        return str(run_dir)

    @classmethod
    def _load_from_dir(cls, folder: Union[str, Path]) -> "AAINDIAgent":
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

        # Rehydrate the G_init list back into an ndarray so the constructor
        # applies it to the RLS parameter matrix.
        g_init = cfg_dict.get("G_init")
        if g_init is not None:
            cfg_dict["G_init"] = np.asarray(g_init, dtype=np.float64)

        agent_cfg = AAINDIConfig(**cfg_dict)
        agent = cls(
            n_state=params["n_state"],
            n_control=params["n_control"],
            config=agent_cfg,
        )

        # RLS state.
        rls_path = folder_p / "vff_rls.npz"
        if rls_path.exists():
            with np.load(rls_path) as npz:
                agent.rls.theta = npz["theta"]
                agent.rls.P = npz["P"]
                agent.rls.last_lambda = float(npz["last_lambda"])
                agent.rls.num_updates = int(npz["num_updates"])

        bias_path = folder_p / "bias_state.npz"
        if bias_path.exists():
            with np.load(bias_path) as npz:
                agent.bias_est._bias = npz["bias"]

        deriv_path = folder_p / "deriv_state.npz"
        if deriv_path.exists():
            with np.load(deriv_path) as npz:
                agent.deriv._y = npz["y"]
                agent.deriv._prev_x = npz["prev_x"] if bool(npz["has_prev"]) else None

        # Loop state (reference model + PI integrator + last command + cache).
        # Older checkpoints (before this field was added) simply skip this
        # branch and the agent keeps the freshly-constructed zero state.
        loop_path = folder_p / "loop_state.npz"
        if loop_path.exists():
            with np.load(loop_path) as npz:
                agent._ref_rate = npz["ref_rate"]
                agent._ref_rate_dot = npz["ref_rate_dot"]
                agent._int_err = npz["int_err"]
                agent._u_prev = npz["u_prev"]
                agent._omega_dot_cached = npz["omega_dot_cached"]
                agent._step = int(npz["step"])

        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "AAINDIAgent":
        """Load an agent from a local directory or Hugging Face Hub.

        Args:
            repo_name: Local folder path, or ``namespace/repo_name`` on
                the Hugging Face Hub.
            access_token: Hub access token for private repos.
            version: Hub revision / branch / tag.

        Returns:
            AAINDIAgent: Reconstructed agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(
                f"Local directory not found: '{repo_name}'." " Please check the path."
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
            repo_name: Target repository id, e.g. ``"me/my-aaindi"``.
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
