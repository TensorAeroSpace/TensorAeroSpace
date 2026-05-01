"""AIDIAgent — orchestrator for the Adaptive Incremental Dynamic Inversion
controller (Ul Haq, Atmaca, van Kampen, AIAA 2026-1744).

The agent composes:
    * an outer-loop block (CStarController, RollReferenceModel,
      SideslipCompensator, SpeedController, LinearController),
    * the AIDI inner law: ``Δu = (Θ ⊙ G_nominal)⁺ · (ν_des − ω̇_meas)``,
    * a per-row VFF-RLS that adapts Θ online,
    * Pseudo-Control Hedging linking the inner-loop deficit back into
      the reference models.

The agent is model-agnostic: an ``OnboardCEModel`` instance is supplied
at construction and queried each tick for the linearisation of
``∂ω̇/∂u`` around the current operating point.

Units: ``u_magnitude_limit`` / ``u_rate_limit`` are interpreted in the
same units as the control vector that ``onboard_ce(x, u)`` expects on
its second argument. For :class:`F16NonlinearOnboardCE` that is
**radians** (the default config values are pre-converted from the
F-16 mechanical envelope of ±25° / 60°·s⁻¹).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Reuse the well-tested low-pass differentiator from aa_indi for ω̇_meas.
from tensoraerospace.agent.aa_indi.sensor_filter import LowPassDerivative

from .allocator import MoorePenroseAllocator
from .onboard_ce import OnboardCEModel
from .pch import PseudoControlHedge
from .ref_models import (
    CStarController,
    LinearController,
    RollReferenceModel,
    SideslipCompensator,
    SpeedController,
)
from .scaling_rls import ScalingRLS
from .utils import reconstruct_n_z

logger = logging.getLogger(__name__)


REQUIRED_OBS_KEYS = ("omega", "alpha", "beta", "theta", "phi", "V")
REQUIRED_REF_KEYS = ("C_star", "phi_cmd", "beta_cmd", "V_cmd")


@dataclass
class AIDIConfig:
    """Hyper-parameters for :class:`AIDIAgent`."""

    dt: float = 0.01

    # Inner-loop (allocator + actuator clamps). Defaults are in RADIANS to
    # match the F-16 nonlinear ODE convention; a 25° envelope ≈ 0.4363 rad.
    u_magnitude_limit: float = math.radians(25.0)
    u_rate_limit: float = math.radians(60.0)
    pinv_rcond: float = 1e-6
    # Only fall back to Δu = 0 when ``G`` is essentially singular —
    # numpy.linalg.pinv with ``rcond`` already handles graceful degradation
    # on weak singular directions. The default threshold is generous so we
    # only trip on truly broken matrices (zero row / column).
    cond_threshold: float = 1e12
    sensor_cutoff_hz: float = 15.0

    # Scaling-RLS.
    rls_lambda_min: float = 0.7
    rls_lambda_max: float = 0.999
    rls_sigma0: float = 1e-3
    rls_memory_length: int = 100
    rls_cov_init: float = 1.0
    # Cross-axis consistency check (paper §III.C, page 10) only helps when
    # control surfaces are *redundantly* mapped to the same moment axes
    # (Flying V's 5-surface layout). On a 3×3 plant like the F-16 nonlinear
    # angular env, surfaces are mostly axis-aligned so per-row updates
    # legitimately differ — averaging them erases the signal. Default to a
    # very loose threshold (≈ off); set ≤ 1e-6 only on truly redundant
    # plants.
    rls_consistency_threshold: float = 10.0

    # PCH.
    pch_freeze_after: int = 30
    pch_gap_tol: float = 1e-3

    # C* longitudinal.
    cstar_kp: float = 1.5
    cstar_ki: float = 0.5
    cstar_V_co: float = 122.6
    cstar_i_clip: float = 5.0

    # Roll reference model.
    roll_omega_n: float = 2.5
    roll_zeta: float = 0.7

    # Sideslip compensator.
    sideslip_kp: float = 1.5
    sideslip_ki: float = 0.1
    sideslip_i_clip: float = 5.0

    # Speed controller (off by default for constant-airspeed envs).
    speed_kp: float = 0.0
    speed_ki: float = 0.0
    speed_kd: float = 0.0
    speed_enabled: bool = False

    # Linear controller — additional rate-error feedback.
    rate_kp: tuple = (0.0, 0.0, 0.0)

    seed: int | None = None
    history: dict = field(default_factory=dict)


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


class AIDIAgent:
    """Adaptive Incremental Dynamic Inversion control agent."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        onboard_ce: OnboardCEModel,
        config: AIDIConfig | None = None,
    ) -> None:
        if onboard_ce.n_state != n_state or onboard_ce.n_control != n_control:
            raise ValueError(
                f"onboard_ce shape mismatch: agent expects "
                f"({n_state}, {n_control}), onboard_ce reports "
                f"({onboard_ce.n_state}, {onboard_ce.n_control})"
            )
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else AIDIConfig()
        self.onboard_ce = onboard_ce

        # --- Inner loop -----------------------------------------------
        self.rls = ScalingRLS(
            n_y=self.n_state,
            n_u=self.n_control,
            lambda_min=self.cfg.rls_lambda_min,
            lambda_max=self.cfg.rls_lambda_max,
            sigma0=self.cfg.rls_sigma0,
            memory_length=self.cfg.rls_memory_length,
            cov_init=self.cfg.rls_cov_init,
            consistency_threshold=self.cfg.rls_consistency_threshold,
            seed=self.cfg.seed,
        )
        self.allocator = MoorePenroseAllocator(
            rcond=self.cfg.pinv_rcond,
            cond_threshold=self.cfg.cond_threshold,
        )
        self.deriv = LowPassDerivative(
            n=self.n_state,
            dt=self.cfg.dt,
            cutoff_hz=self.cfg.sensor_cutoff_hz,
        )

        # --- PCH and outer loop ---------------------------------------
        self.pch = PseudoControlHedge(
            n_y=self.n_state,
            freeze_after=self.cfg.pch_freeze_after,
            gap_tol=self.cfg.pch_gap_tol,
        )
        self.cstar = CStarController(
            kp=self.cfg.cstar_kp,
            ki=self.cfg.cstar_ki,
            V_co=self.cfg.cstar_V_co,
            dt=self.cfg.dt,
            i_clip=self.cfg.cstar_i_clip,
        )
        self.roll_ref = RollReferenceModel(
            omega_n=self.cfg.roll_omega_n,
            zeta=self.cfg.roll_zeta,
            dt=self.cfg.dt,
        )
        self.sideslip = SideslipCompensator(
            kp=self.cfg.sideslip_kp,
            ki=self.cfg.sideslip_ki,
            dt=self.cfg.dt,
            i_clip=self.cfg.sideslip_i_clip,
        )
        self.speed = SpeedController(
            kp=self.cfg.speed_kp,
            ki=self.cfg.speed_ki,
            kd=self.cfg.speed_kd,
            dt=self.cfg.dt,
            enabled=self.cfg.speed_enabled,
        )
        self.linear = LinearController(
            rate_kp=np.asarray(self.cfg.rate_kp, dtype=np.float64),
            n_y=self.n_state,
        )

        # --- Rolling state --------------------------------------------
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_dot_cached = np.zeros(self.n_state, dtype=np.float64)
        self._omega_prev: np.ndarray | None = None
        self._omega_dot_prev: np.ndarray | None = None
        self._last_u_cmd = np.zeros(self.n_control, dtype=np.float64)
        self._last_nu_des = np.zeros(self.n_state, dtype=np.float64)
        self._alpha_prev: float | None = None
        self._last_G_nominal: np.ndarray | None = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_obs(self, obs: dict) -> None:
        missing = [k for k in REQUIRED_OBS_KEYS if k not in obs]
        if missing:
            raise KeyError(
                f"observation is missing required keys {missing}. "
                f"AIDI needs: {REQUIRED_OBS_KEYS}"
            )

    def _check_refs(self, refs: dict) -> None:
        missing = [k for k in REQUIRED_REF_KEYS if k not in refs]
        if missing:
            raise KeyError(
                f"references is missing required keys {missing}. "
                f"AIDI needs: {REQUIRED_REF_KEYS}"
            )

    def _resolve_n_z(self, obs: dict, q: float) -> float:
        if "n_z" in obs:
            return float(obs["n_z"])
        alpha = float(obs["alpha"])
        alpha_dot = (
            (alpha - self._alpha_prev) / self.cfg.dt
            if self._alpha_prev is not None
            else 0.0
        )
        return reconstruct_n_z(
            alpha=alpha,
            alpha_dot=alpha_dot,
            q=q,
            V=float(obs["V"]),
            theta=float(obs["theta"]),
            phi=float(obs["phi"]),
        )

    def _build_state_vector(self, obs: dict) -> np.ndarray:
        """Either pass through ``obs['state']`` or reconstruct a 14-vector."""
        if "state" in obs:
            return np.asarray(obs["state"], dtype=np.float64).reshape(-1)
        x = np.zeros(14, dtype=np.float64)
        omega = np.asarray(obs["omega"], dtype=np.float64).reshape(-1)
        x[0] = float(obs["alpha"])
        x[1] = float(obs["beta"])
        x[2] = float(omega[0])
        x[3] = float(omega[1])
        x[4] = float(omega[2])
        x[7] = float(obs["theta"])
        return x

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode rolling state — keeps Θ and P (lifelong adaptation)."""
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_dot_cached = np.zeros(self.n_state, dtype=np.float64)
        self._omega_prev = None
        self._omega_dot_prev = None
        self._last_u_cmd = np.zeros(self.n_control, dtype=np.float64)
        self._last_nu_des = np.zeros(self.n_state, dtype=np.float64)
        self._alpha_prev = None
        self._last_G_nominal = None
        self.deriv.reset()
        self.pch.reset()
        self.cstar.reset()
        self.roll_ref.reset()
        self.sideslip.reset()
        self.speed.reset()
        self._step = 0

    def predict(
        self,
        observation: dict,
        references: dict,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del deterministic, time_step
        self._check_obs(observation)
        self._check_refs(references)

        omega = np.asarray(observation["omega"], dtype=np.float64).reshape(-1)
        if omega.size != self.n_state:
            raise ValueError(f"omega must have length {self.n_state}, got {omega.size}")

        # Use the cached ω̇_meas — `learn` advances the differentiator.
        omega_dot_meas = self._omega_dot_cached.copy()

        # PCH from previous-tick demand vs current measurement.
        hedge = self.pch.update(
            nu_des_prev=self._last_nu_des,
            omega_dot_meas=omega_dot_meas,
        )

        # Outer loop — desired rates (p_des, q_des, r_des).
        q = float(omega[1])
        n_z = self._resolve_n_z(observation, q)
        q_des = self.cstar.step(
            c_star_cmd=float(references["C_star"]),
            n_z=n_z,
            q=q,
            V=float(observation["V"]),
            hedge=float(hedge[1]),
        )
        p_des = self.roll_ref.step(
            phi_cmd=float(references["phi_cmd"]),
            phi=float(observation["phi"]),
            hedge=float(hedge[0]),
        )
        r_des = self.sideslip.step(
            beta_cmd=float(references["beta_cmd"]),
            beta=float(observation["beta"]),
            hedge=float(hedge[2]),
        )
        # Speed PID is exposed but discarded here — auto-throttle slot.
        _ = self.speed.step(
            V_cmd=float(references["V_cmd"]),
            V=float(observation["V"]),
        )
        omega_des = np.array([p_des, q_des, r_des], dtype=np.float64)
        nu_des = self.linear.combine(omega_des=omega_des, omega=omega)

        # Inner loop — AIDI law.
        x_for_ce = self._build_state_vector(observation)
        G_nominal = self.onboard_ce(x_for_ce, self._u_prev)
        G_eff = self.rls.theta * G_nominal
        du = self.allocator.allocate(G_eff, nu_des, omega_dot_meas)

        # Rate / magnitude clamps.
        du_max = self.cfg.u_rate_limit * self.cfg.dt
        du = _clamp(du, -du_max, du_max)
        u_cmd = _clamp(
            self._u_prev + du,
            -self.cfg.u_magnitude_limit,
            self.cfg.u_magnitude_limit,
        )

        # Bookkeeping for `learn` and next tick.
        self._last_u_cmd = u_cmd.copy()
        self._last_nu_des = nu_des.copy()
        self._alpha_prev = float(observation["alpha"])
        self._last_G_nominal = G_nominal.copy()
        return u_cmd

    def learn(
        self,
        next_observation: dict,
        references: dict,
        time_step: int = 0,
    ) -> Dict[str, float]:
        del references, time_step
        self._check_obs(next_observation)
        omega = np.asarray(
            next_observation["omega"],
            dtype=np.float64,
        ).reshape(-1)
        if omega.size != self.n_state:
            raise ValueError(f"omega must have length {self.n_state}, got {omega.size}")

        omega_dot_next = self.deriv.step(omega)
        self._omega_dot_cached = omega_dot_next.copy()

        residuals = np.zeros(self.n_state, dtype=np.float64)
        if self._omega_dot_prev is not None and self._last_G_nominal is not None:
            du = self._last_u_cmd - self._u_prev
            domega = omega_dot_next - self._omega_dot_prev
            residuals = self.rls.update(du, domega, self._last_G_nominal)

        self._u_prev = self._last_u_cmd.copy()
        self._omega_prev = omega.copy()
        self._omega_dot_prev = omega_dot_next.copy()
        self._step += 1

        G_norm = (
            float(np.linalg.norm(self.rls.theta * self._last_G_nominal))
            if self._last_G_nominal is not None
            else 0.0
        )
        return {
            "residual_norm": float(np.linalg.norm(residuals)),
            "lambda_min": float(np.min(self.rls.last_lambda)),
            "G_norm": G_norm,
            "frozen_axes": int(self.pch.is_frozen.sum()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict.pop("history", None)
        cfg_dict["rate_kp"] = list(cfg_dict.get("rate_kp", []))
        return {
            "policy": {
                "name": agent_name,
                "params": {"n_state": self.n_state, "n_control": self.n_control},
                "config": cfg_dict,
            },
        }

    def save(self, path: Union[str, Path, None] = None) -> str:
        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        np.savez(
            run_dir / "scaling_rls.npz",
            theta=self.rls.theta,
            P=self.rls.P,
            last_lambda=self.rls.last_lambda,
            last_residual=self.rls.last_residual,
            num_updates=np.asarray(self.rls.num_updates),
        )
        np.savez(
            run_dir / "outer_state.npz",
            cstar_int=np.asarray(self.cstar._int_err),
            sideslip_int=np.asarray(self.sideslip._int_err),
            speed_int=np.asarray(self.speed._int_err),
            speed_prev_err=np.asarray(self.speed._prev_err),
            roll_phi=np.asarray(self.roll_ref._phi),
            roll_phi_dot=np.asarray(self.roll_ref._phi_dot),
        )
        np.savez(
            run_dir / "pch_state.npz",
            last_hedge=self.pch.last_hedge,
            sat_counter=self.pch.saturation_counter,
            is_frozen=self.pch.is_frozen,
        )
        deriv_prev = self.deriv._prev_x
        np.savez(
            run_dir / "deriv_state.npz",
            y=self.deriv.last_output,
            prev_x=deriv_prev if deriv_prev is not None else np.array([]),
            has_prev=np.asarray(deriv_prev is not None),
        )
        np.savez(
            run_dir / "loop_state.npz",
            u_prev=self._u_prev,
            omega_dot_cached=self._omega_dot_cached,
            omega_prev=(
                self._omega_prev if self._omega_prev is not None else np.array([])
            ),
            has_omega_prev=np.asarray(self._omega_prev is not None),
            omega_dot_prev=(
                self._omega_dot_prev
                if self._omega_dot_prev is not None
                else np.array([])
            ),
            has_omega_dot_prev=np.asarray(self._omega_dot_prev is not None),
            last_u_cmd=self._last_u_cmd,
            last_nu_des=self._last_nu_des,
            alpha_prev=np.asarray(
                self._alpha_prev if self._alpha_prev is not None else 0.0
            ),
            has_alpha_prev=np.asarray(self._alpha_prev is not None),
            last_G_nominal=(
                self._last_G_nominal
                if self._last_G_nominal is not None
                else np.array([])
            ),
            has_last_G_nominal=np.asarray(self._last_G_nominal is not None),
            step=np.asarray(self._step),
        )
        return str(run_dir)

    @classmethod
    def _load_from_dir(
        cls,
        folder: Union[str, Path],
        onboard_ce: OnboardCEModel,
    ) -> "AIDIAgent":
        folder_p = Path(folder)
        with open(folder_p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy = cfg.get("policy", {})
        params = policy.get("params", {})
        cfg_dict = dict(policy.get("config", {}))
        cfg_dict["rate_kp"] = tuple(cfg_dict.get("rate_kp", (0.0, 0.0, 0.0)))
        agent_cfg = AIDIConfig(**cfg_dict)
        agent = cls(
            n_state=params["n_state"],
            n_control=params["n_control"],
            onboard_ce=onboard_ce,
            config=agent_cfg,
        )

        with np.load(folder_p / "scaling_rls.npz") as npz:
            agent.rls.theta = npz["theta"]
            agent.rls.P = npz["P"]
            agent.rls.last_lambda = npz["last_lambda"]
            agent.rls.last_residual = npz["last_residual"]
            agent.rls.num_updates = int(npz["num_updates"])

        with np.load(folder_p / "outer_state.npz") as npz:
            agent.cstar._int_err = float(npz["cstar_int"])
            agent.sideslip._int_err = float(npz["sideslip_int"])
            agent.speed._int_err = float(npz["speed_int"])
            agent.speed._prev_err = float(npz["speed_prev_err"])
            agent.roll_ref._phi = float(npz["roll_phi"])
            agent.roll_ref._phi_dot = float(npz["roll_phi_dot"])

        with np.load(folder_p / "pch_state.npz") as npz:
            agent.pch.last_hedge = npz["last_hedge"]
            agent.pch.saturation_counter = npz["sat_counter"].astype(np.int32)
            agent.pch.is_frozen = npz["is_frozen"].astype(bool)

        with np.load(folder_p / "deriv_state.npz") as npz:
            agent.deriv._y = npz["y"]
            agent.deriv._prev_x = npz["prev_x"] if bool(npz["has_prev"]) else None

        with np.load(folder_p / "loop_state.npz") as npz:
            agent._u_prev = npz["u_prev"]
            agent._omega_dot_cached = npz["omega_dot_cached"]
            agent._omega_prev = (
                npz["omega_prev"] if bool(npz["has_omega_prev"]) else None
            )
            agent._omega_dot_prev = (
                npz["omega_dot_prev"] if bool(npz["has_omega_dot_prev"]) else None
            )
            agent._last_u_cmd = npz["last_u_cmd"]
            agent._last_nu_des = npz["last_nu_des"]
            agent._alpha_prev = (
                float(npz["alpha_prev"]) if bool(npz["has_alpha_prev"]) else None
            )
            agent._last_G_nominal = (
                npz["last_G_nominal"] if bool(npz["has_last_G_nominal"]) else None
            )
            agent._step = int(npz["step"])

        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        *,
        onboard_ce: OnboardCEModel,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "AIDIAgent":
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p, onboard_ce=onboard_ce)
        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name,
            token=access_token,
            revision=version,
        )
        return cls._load_from_dir(folder_path, onboard_ce=onboard_ce)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )
