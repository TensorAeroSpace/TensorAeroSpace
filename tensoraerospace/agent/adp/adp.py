# flake8: noqa
"""Adaptive Critic Design / ADP agent.

This implementation follows the Adaptive Critic Design (ACD) / Approximate
Dynamic Programming idea from:

Prokhorov D.V., Wunsch D.C. “Adaptive critic designs: A case study for
neurocontrol.” Neural Networks, 8(9), 1995, pp. 1367–1372.

We implement paper-inspired ACD designs (HDP/DHP/GDHP and AD variants) and a
canonical ADHDP-style actor-critic:
  - actor:  a = π(R)
  - critic: J = J(R, a)  (action-dependent cost-to-go)
  - TD learning: J(R_t, a_t) ≈ U_t + γ J(R_{t+1}, π(R_{t+1}))
  - actor improvement: minimize J(R, π(R)) w.r.t actor params

For convenience we also keep a more "practical" DDPG-like variant under
`design="ddpg"` (optional replay/target networks).

Notes:
  - The environment is assumed to follow Gymnasium API.
  - We treat the environment reward as "utility" and convert to cost via:
        cost = -reward
    so minimizing cost-to-go is equivalent to maximizing return.
"""

from __future__ import annotations

import datetime
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from ..metrics import create_metric_writer
from .networks import (
    DeterministicActor,
    JCritic,
    JLambdaActionCritic,
    JLambdaCritic,
    LambdaCritic,
    QCritic,
    polyak_update,
)
from .replay import ReplayBuffer


def _as_flat_np(x: Any) -> np.ndarray:
    """Convert env observation to a flat float32 numpy array."""
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


class ADP(BaseRLModel):
    """Adaptive Critic Design (ADP) agent for continuous control."""

    def __init__(
        self,
        env: Any,
        *,
        design: str = "adhdp",
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42,
        # Exploration (training-time only)
        exploration_std: float = 0.1,
        # Learning mode
        use_replay: bool = False,
        memory_capacity: int = 200_000,
        batch_size: int = 64,
        updates_per_step: int = 1,
        # Optional target networks (Polyak averaging). Disabled by default to
        # stay closer to classic online ACD; can be enabled for stability.
        use_target_networks: bool = False,
        tau: float = 0.01,
        # Logging
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 100,
        # ADHDP stabilization: optional PD baseline computed from the observation.
        # NOTE: for canonical ADHDP (paper-style) we do NOT compose actions as
        # baseline+residual. Instead, use warm-start below to initialize the actor as
        # a stabilizing controller, then train with pure ACD updates.
        adhdp_use_baseline: bool = False,
        adhdp_warmstart_actor_episodes: int = 0,
        adhdp_warmstart_actor_epochs: int = 2,
        # Paper-style training schedule: keep actor fixed for first N episodes so the
        # critic can adapt on stable trajectories.
        adhdp_critic_warmup_episodes: int = 0,
        # Paper Section III-style alternating cycles (also applicable to AD forms):
        # train critic for Nc episodes with fixed actor, then train actor for Na episodes
        # with fixed critic, and repeat.
        adhdp_critic_cycle_episodes: int = 0,
        adhdp_action_cycle_episodes: int = 0,
        # Use environment cost as utility U for ADHDP updates (paper-style).
        # If enabled and env returns info["cost_total"], we train on reward=-cost_total
        # instead of the shaped environment reward (which can include bonuses/progress).
        adhdp_use_env_cost: bool = True,
        # Optional ADHDP stabilization (practical): keep actor close to PD stabilizer early
        # in training, then decay this regularizer.
        #
        # This helps prevent the actor from drifting into saturated actions while the
        # critic is still inaccurate.
        adhdp_actor_bc_l2: float = 0.0,
        adhdp_actor_bc_decay: float = 1.0,
        # DHP utility weights (used only when design="dhp")
        dhp_w_theta: float = 5.0,
        dhp_w_q: float = 0.2,
        dhp_w_u: float = 0.01,
        dhp_w_du: float = 0.02,
        dhp_use_env_cost: bool = True,
        # Optional stabilization for classic ACD (HDP/DHP/GDHP and AD variants):
        # the paper recommends *starting* with a stabilizing controller to keep the
        # closed loop stable early in training, but the ACD update rules themselves
        # do NOT require any "baseline" or expert policy.
        #
        # If enabled, the learned actor is treated as a residual controller:
        #   u = u_baseline + dhp_residual_scale * delta_u
        dhp_use_baseline: bool = False,
        dhp_baseline_type: str = "pd",
        dhp_baseline_kp: float = 0.6,
        dhp_baseline_ki: float = 0.0,
        dhp_baseline_kd: float = 0.2,
        dhp_pid_use_normalized_theta: bool = True,
        dhp_pid_mode: str = "norm",
        # Scale of the learned policy output (or residual if baseline is enabled).
        # Use 1.0 for "pure" learned control without a baseline.
        dhp_residual_scale: float = 1.0,
        # Optional paper-style warm start:
        # The paper recommends starting critic training with an actor that already acts
        # as a stabilizing controller, to keep the closed-loop stable while the critic adapts.
        #
        # This is *not* a "baseline controller" inside ACD equations; it is just actor
        # initialization. We implement it by collecting (R(t) -> u_baseline) samples from
        # a PD/PID stabilizer and doing a short supervised fit of the actor.
        dhp_warmstart_actor_episodes: int = 0,
        dhp_warmstart_actor_epochs: int = 2,
        dhp_warmstart_actor_disable_baseline_after: bool = True,
        # Extra stabilization: keep residual close to 0 when using a strong baseline.
        # Adds actor_loss += dhp_actor_delta_l2 * mean(delta_u^2)
        dhp_actor_delta_l2: float = 0.0,
        # Paper Section III: alternating critic/action training cycles (keep the other fixed)
        dhp_critic_cycle_episodes: int = 0,
        dhp_action_cycle_episodes: int = 0,
    ) -> None:
        super().__init__()
        self.env = env
        self.design = str(design).lower().strip()
        self.gamma = float(gamma)
        self.exploration_std = float(exploration_std)
        self.use_replay = bool(use_replay)
        self.batch_size = int(batch_size)
        self.updates_per_step = int(updates_per_step)
        self.use_target_networks = bool(use_target_networks)
        self.tau = float(tau)
        self.hidden_size = int(hidden_size)

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.updates_per_step < 1:
            raise ValueError("updates_per_step must be >= 1")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.exploration_std < 0.0:
            raise ValueError("exploration_std must be >= 0")
        if self.use_target_networks and not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1] when use_target_networks=True")

        self.device = torch.device(device)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Spaces (for action scaling)
        obs_dim = int(getattr(self.env.observation_space, "shape", (0,))[0])
        act_dim = int(getattr(self.env.action_space, "shape", (0,))[0])
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(
                f"ADP expects Box-like spaces. Got obs_dim={obs_dim}, act_dim={act_dim}"
            )

        action_low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(
            -1
        )

        hidden_sizes = (int(self.hidden_size), int(self.hidden_size))

        # DHP needs access to physical state dimension + model Jacobians A,B
        self._dhp_state_dim: Optional[int] = None
        self._dhp_A: Optional[torch.Tensor] = None
        self._dhp_B: Optional[torch.Tensor] = None
        self._dhp_dt: Optional[float] = None
        self._dhp_ref: Optional[np.ndarray] = None
        self._dhp_w_theta = float(dhp_w_theta)
        self._dhp_w_q = float(dhp_w_q)
        self._dhp_w_u = float(dhp_w_u)
        self._dhp_w_du = float(dhp_w_du)
        self._dhp_use_env_cost = bool(dhp_use_env_cost)
        self._dhp_use_baseline = bool(dhp_use_baseline)
        self._dhp_baseline_type = str(dhp_baseline_type).lower().strip()
        self._dhp_baseline_kp = float(dhp_baseline_kp)
        self._dhp_baseline_ki = float(dhp_baseline_ki)
        self._dhp_baseline_kd = float(dhp_baseline_kd)
        self._dhp_pid_use_normalized_theta = bool(dhp_pid_use_normalized_theta)
        self._dhp_pid_mode = str(dhp_pid_mode).lower().strip()
        self._dhp_residual_scale = float(dhp_residual_scale)
        self._dhp_warmstart_actor_episodes = int(dhp_warmstart_actor_episodes)
        self._dhp_warmstart_actor_epochs = int(dhp_warmstart_actor_epochs)
        self._dhp_warmstart_actor_disable_baseline_after = bool(
            dhp_warmstart_actor_disable_baseline_after
        )
        self._dhp_actor_delta_l2 = float(dhp_actor_delta_l2)
        self._dhp_critic_cycle_episodes = int(dhp_critic_cycle_episodes)
        self._dhp_action_cycle_episodes = int(dhp_action_cycle_episodes)
        self._prev_u_norm = np.zeros((act_dim,), dtype=np.float32)
        self._prev2_u_norm = np.zeros((act_dim,), dtype=np.float32)
        self._dhp_pid = None
        self._adhdp_use_baseline = bool(adhdp_use_baseline)
        self._adhdp_warmstart_actor_episodes = int(adhdp_warmstart_actor_episodes)
        self._adhdp_warmstart_actor_epochs = int(adhdp_warmstart_actor_epochs)
        self._adhdp_critic_warmup_episodes = int(adhdp_critic_warmup_episodes)
        self._adhdp_critic_cycle_episodes = int(adhdp_critic_cycle_episodes)
        self._adhdp_action_cycle_episodes = int(adhdp_action_cycle_episodes)
        self._adhdp_use_env_cost = bool(adhdp_use_env_cost)
        self._adhdp_actor_bc_l2 = float(adhdp_actor_bc_l2)
        self._adhdp_actor_bc_decay = float(adhdp_actor_bc_decay)
        self._adhdp_actor_bc_coef = float(adhdp_actor_bc_l2)

        supported_designs = (
            "adhdp",  # canonical action-dependent HDP (online, no replay/targets)
            "ddpg",  # practical DDPG-like variant (optional replay/targets)
            "dhp",
            "gdhp",
            "hdp",
            "addhp",
            "adgdhp",
        )
        if self.design not in supported_designs:
            raise ValueError(
                f"design must be one of {supported_designs}, got {self.design!r}"
            )

        self._is_model_based_design = self.design in (
            "dhp",
            "gdhp",
            "hdp",
            "addhp",
            "adgdhp",
        )

        # Replay/target networks are *not* part of canonical ACD/ADHDP (paper),
        # but we allow them for the practical `design="ddpg"` variant only.
        if self.design == "adhdp" and (self.use_replay or self.use_target_networks):
            raise ValueError(
                "Canonical design='adhdp' does not support replay/target networks. "
                "Use design='ddpg' for a practical DDPG-like variant."
            )
        if self._is_model_based_design and (
            self.use_replay or self.use_target_networks
        ):
            raise ValueError(
                "use_replay/use_target_networks are only supported for design='ddpg'"
            )
        if self.design == "adhdp" and self._adhdp_use_baseline:
            raise ValueError(
                "Canonical design='adhdp' does not support baseline+residual action composition. "
                "Use adhdp_warmstart_actor_episodes>0 to initialize the actor as a stabilizing PD controller "
                "(paper-style), or use design='ddpg' if you need residual stabilization."
            )

        if self._is_model_based_design:
            init_state = getattr(self.env, "initial_state", None)
            if init_state is None and hasattr(self.env, "unwrapped"):
                init_state = getattr(self.env.unwrapped, "initial_state", None)
            state_dim = (
                int(np.asarray(init_state).reshape(-1).shape[0])
                if init_state is not None
                else 0
            )
            if state_dim < 1:
                # Fallback to observation dimension, but DHP will likely be invalid without model Jacobians.
                state_dim = obs_dim
            self._dhp_state_dim = state_dim

            # Actor takes an observable vector R(t).
            # For tracking tasks (ImprovedB747Env) we extend it with reference signals:
            #   R(t) = [x(t), theta_ref(t), q_ref(t)]
            ref_sig = getattr(self.env, "reference_signal", None)
            if ref_sig is None and hasattr(self.env, "unwrapped"):
                ref_sig = getattr(self.env.unwrapped, "reference_signal", None)
            if ref_sig is None:
                raise ValueError(
                    f"{self.design!r} design requires env.reference_signal (pitch reference trajectory)."
                )
            self._dhp_ref = np.asarray(ref_sig, dtype=float)
            self._dhp_ref_dim = 2
            self._dhp_input_dim = int(state_dim + int(self._dhp_ref_dim))
            self.actor = DeterministicActor(
                self._dhp_input_dim,
                act_dim,
                hidden_sizes=hidden_sizes,
                action_low=action_low,
                action_high=action_high,
            ).to(self.device)

            # Critics per ACD family (Prokhorov & Wunsch 1997):
            #  - HDP:       J(R)
            #  - DHP:       lambda = dJ/dR
            #  - GDHP:      J(R) and lambda = dJ/dR (shared trunk, Fig. 5)
            #  - ADGDHP:    J(R,A) and (dJ/dR, dJ/dA) (Fig. 7)
            if self.design == "hdp":
                self.critic = JCritic(
                    self._dhp_input_dim, hidden_sizes=hidden_sizes
                ).to(self.device)
            elif self.design == "dhp":
                self.critic = LambdaCritic(
                    self._dhp_input_dim,
                    input_dim=self._dhp_input_dim,
                    hidden_sizes=hidden_sizes,
                ).to(self.device)
            elif self.design == "gdhp":
                self.critic = JLambdaCritic(
                    input_dim=self._dhp_input_dim,
                    r_dim=self._dhp_input_dim,
                    hidden_sizes=hidden_sizes,
                ).to(self.device)
            elif self.design in ("addhp", "adgdhp"):
                self.critic = JLambdaActionCritic(
                    r_dim=self._dhp_input_dim,
                    a_dim=act_dim,
                    hidden_sizes=hidden_sizes,
                ).to(self.device)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported model-based design: {self.design!r}")

            # Pull linear model matrices from env.model if available (ImprovedB747Env)
            mdl = getattr(self.env, "model", None)
            if mdl is None and hasattr(self.env, "unwrapped"):
                mdl = getattr(self.env.unwrapped, "model", None)
            A = getattr(mdl, "filt_A", None)
            B = getattr(mdl, "filt_B", None)
            if A is None or B is None:
                raise ValueError(
                    "DHP design requires env.model.filt_A and env.model.filt_B (linear model Jacobians)."
                )
            A_x = torch.as_tensor(
                np.asarray(A), dtype=torch.float32, device=self.device
            )

            # Effective B for u_norm: u_rad = deg2rad(max_deg)*u_norm (for ImprovedB747Env)
            max_deg = getattr(self.env, "max_stabilizer_angle_deg", None)
            if max_deg is None and hasattr(self.env, "unwrapped"):
                max_deg = getattr(self.env.unwrapped, "max_stabilizer_angle_deg", None)
            max_deg_f = float(max_deg) if max_deg is not None else 1.0
            self._env_max_stabilizer_angle_deg = float(max_deg_f)
            u_norm_to_rad = float(np.deg2rad(max_deg_f)) if max_deg is not None else 1.0
            B_x = torch.as_tensor(
                np.asarray(B) * u_norm_to_rad, dtype=torch.float32, device=self.device
            )

            # Extended Jacobians for R(t) = [x(t), ref(t)] with ref treated as exogenous:
            #   R_{t+1} = [A_x x_t + B_x u_t, ref_{t+1}]
            r_dim = int(self._dhp_input_dim)
            self._dhp_A = torch.zeros(
                (r_dim, r_dim), dtype=torch.float32, device=self.device
            )
            self._dhp_A[: int(state_dim), : int(state_dim)] = A_x
            self._dhp_B = torch.zeros(
                (r_dim, act_dim), dtype=torch.float32, device=self.device
            )
            self._dhp_B[: int(state_dim), :] = B_x

            self._dhp_dt = float(getattr(self.env, "dt", 0.01))

            # Prefer matching environment tracking cost (ImprovedB747Env) for stability
            self._env_max_pitch_rad = float(
                getattr(self.env, "max_pitch_rad", np.deg2rad(20.0))
            )
            self._env_max_pitch_rate_rad_s = float(
                getattr(self.env, "max_pitch_rate_rad_s", np.deg2rad(5.0))
            )
            self._env_ref_theta_dot_clip_rad_s = float(
                getattr(
                    self.env, "ref_theta_dot_clip_rad_s", self._env_max_pitch_rate_rad_s
                )
            )
            self._env_w_pitch = float(getattr(self.env, "w_pitch", self._dhp_w_theta))
            self._env_w_q = float(getattr(self.env, "w_q", self._dhp_w_q))
            self._env_w_action = float(getattr(self.env, "w_action", self._dhp_w_u))
            self._env_w_smooth = float(getattr(self.env, "w_smooth", self._dhp_w_du))
            self._env_w_jerk = float(getattr(self.env, "w_jerk", 0.0))

            # Optional PID baseline (from tensoraerospace.agent.pid)
            need_pid = self._dhp_baseline_type == "pid" and (
                bool(getattr(self, "_dhp_use_baseline", False))
                or int(getattr(self, "_dhp_warmstart_actor_episodes", 0) or 0) > 0
            )
            if need_pid:
                try:
                    from ..pid import (  # local import to avoid unnecessary dependency chain
                        PID,
                    )
                except Exception as exc:  # pragma: no cover
                    raise ImportError(
                        "Failed to import PID baseline from tensoraerospace.agent.pid"
                    ) from exc
                pid_mode = str(getattr(self, "_dhp_pid_mode", "norm")).lower().strip()
                if pid_mode not in ("norm", "deg"):
                    raise ValueError("dhp_pid_mode must be 'norm' or 'deg'")
                pid_env = None if pid_mode == "deg" else self.env
                self._dhp_pid = PID(
                    env=pid_env,
                    kp=float(self._dhp_baseline_kp),
                    ki=float(self._dhp_baseline_ki),
                    kd=float(self._dhp_baseline_kd),
                    dt=float(self._dhp_dt or 0.01),
                )
        else:
            # Default ADHDP-like (value / Q) on observation space
            self.actor = DeterministicActor(
                obs_dim,
                act_dim,
                hidden_sizes=hidden_sizes,
                action_low=action_low,
                action_high=action_high,
            ).to(self.device)
            self.critic = QCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(
                self.device
            )

        self.actor_optim = Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(critic_lr))

        # Optional target networks (practical variant only)
        self.actor_target: Optional[DeterministicActor]
        self.critic_target: Optional[QCritic]
        if self.design == "ddpg" and self.use_target_networks:
            self.actor_target = DeterministicActor(
                obs_dim,
                act_dim,
                hidden_sizes=hidden_sizes,
                action_low=action_low,
                action_high=action_high,
            ).to(self.device)
            self.critic_target = QCritic(
                obs_dim, act_dim, hidden_sizes=hidden_sizes
            ).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.actor_target = None
            self.critic_target = None

        # Optional replay buffer (practical variant only)
        self.memory: Optional[ReplayBuffer]
        if self.design == "ddpg" and self.use_replay:
            self.memory = ReplayBuffer(int(memory_capacity), seed=self.seed)
        else:
            self.memory = None

        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = create_metric_writer(self.log_dir)
        self.log_every_updates = int(log_every_updates)
        if self.log_every_updates < 1:
            raise ValueError("log_every_updates must be >= 1")

        self._updates = 0

    # ---- common API ----
    def get_env(self):
        return self.env

    def select_action(self, state: np.ndarray, *, evaluate: bool = False) -> np.ndarray:
        """Select action for a single observation."""
        obs = _as_flat_np(state)

        # For model-based ACD designs we often train the actor as a residual on top
        # of a stabilizing baseline controller. During evaluation we must apply the
        # same composition.
        if bool(getattr(self, "_is_model_based_design", False)):
            state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
            r_dim = int(getattr(self, "_dhp_input_dim", 0) or 0)
            if state_dim < 1 or r_dim < 1:
                raise ValueError(
                    f"{self.design!r} select_action requires a model-based initialization."
                )
            if obs.shape[0] < r_dim:
                raise ValueError(
                    f"{self.design!r} select_action expects R(t) with dim={r_dim}. "
                    f"Got shape={obs.shape}."
                )

            x = obs[:state_dim]
            theta_ref = float(obs[state_dim]) if r_dim >= state_dim + 2 else 0.0
            q_ref = float(obs[state_dim + 1]) if r_dim >= state_dim + 2 else 0.0

            # baseline u_base in normalized action units (u_norm)
            u_base = 0.0
            if bool(getattr(self, "_dhp_use_baseline", False)):
                if (
                    str(getattr(self, "_dhp_baseline_type", "pd")) == "pid"
                    and getattr(self, "_dhp_pid", None) is not None
                ):
                    pid_mode = (
                        str(getattr(self, "_dhp_pid_mode", "norm")).lower().strip()
                    )
                    if pid_mode == "deg":
                        sp = float(np.rad2deg(theta_ref))
                        meas = float(np.rad2deg(float(x[3])))
                        control_deg = float(self._dhp_pid.select_action(sp, meas))
                        u_base = float(control_deg) / float(
                            getattr(self, "_env_max_stabilizer_angle_deg", 1.0)
                        )
                    else:
                        if bool(getattr(self, "_dhp_pid_use_normalized_theta", True)):
                            sp = float(theta_ref) / float(
                                getattr(self, "_env_max_pitch_rad", 1.0)
                            )
                            meas = float(x[3]) / float(
                                getattr(self, "_env_max_pitch_rad", 1.0)
                            )
                        else:
                            sp = float(theta_ref)
                            meas = float(x[3])
                        u_base = float(self._dhp_pid.select_action(sp, meas))
                else:
                    # PD baseline using theta/q errors (normalized like env)
                    inv_theta = 1.0 / float(getattr(self, "_env_max_pitch_rad", 1.0))
                    inv_q = 1.0 / float(getattr(self, "_env_max_pitch_rate_rad_s", 1.0))
                    e_theta_n = (float(theta_ref) - float(x[3])) * float(inv_theta)
                    e_q_n = (float(q_ref) - float(x[2])) * float(inv_q)
                    u_base = float(getattr(self, "_dhp_baseline_kp", 0.0)) * float(
                        e_theta_n
                    ) + float(getattr(self, "_dhp_baseline_kd", 0.0)) * float(e_q_n)

            obs_t = torch.as_tensor(
                obs[:r_dim], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                delta = self.actor(obs_t).squeeze(0)
                base = torch.full_like(delta, float(u_base))
                u = base + float(getattr(self, "_dhp_residual_scale", 1.0)) * delta
                u = u.cpu().numpy()

            act = u.astype(np.float32)
        else:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                act = self.actor(obs_t).squeeze(0).cpu().numpy()

        if not evaluate and self.exploration_std > 0.0:
            act = act + np.random.normal(
                0.0, self.exploration_std, size=act.shape
            ).astype(np.float32)

        # Always clip to env bounds
        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        return np.clip(act, low, high).astype(np.float32)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Alias for compatibility with some agent APIs."""
        return self.select_action(state, evaluate=bool(deterministic))

    def reset(self) -> None:
        """Reset agent's internal episodic state.

        For DHP with PID baseline this is important for evaluation rollouts:
        the PID has an integrator state that must not leak between episodes,
        otherwise periodic evaluations can look noisy/erratic.
        """
        if bool(getattr(self, "_is_model_based_design", False)):
            try:
                self._prev_u_norm = np.zeros_like(self._prev_u_norm)
                self._prev2_u_norm = np.zeros_like(self._prev2_u_norm)
            except Exception:
                pass
            if getattr(self, "_dhp_pid", None) is not None:
                try:
                    self._dhp_pid.reset()
                except Exception:
                    pass
        # Reset ADHDP PID integrator state (if used)
        try:
            self._adhdp_pid_i = 0.0
        except Exception:
            pass

    def _dhp_baseline_u_norm(
        self, *, x: np.ndarray, theta_ref: float, q_ref: float
    ) -> float:
        """Compute baseline stabilizing action in env action units.

        This helper is used for:
          - residual composition when dhp_use_baseline=True
          - actor warm-start (paper-style initialization)
        """
        baseline_type = str(getattr(self, "_dhp_baseline_type", "pd")).lower().strip()
        if baseline_type == "pid":
            pid = getattr(self, "_dhp_pid", None)
            if pid is None:
                raise ValueError(
                    "PID baseline requested but PID instance is not initialized. "
                    "Set dhp_baseline_type='pid' and enable dhp_use_baseline or dhp_warmstart_actor_episodes>0."
                )
            pid_mode = str(getattr(self, "_dhp_pid_mode", "norm")).lower().strip()
            if pid_mode == "deg":
                sp = float(np.rad2deg(theta_ref))
                meas = float(np.rad2deg(float(x[3])))
                control_deg = float(pid.select_action(sp, meas))
                u_base = float(control_deg) / float(
                    getattr(self, "_env_max_stabilizer_angle_deg", 1.0)
                )
            else:
                if bool(getattr(self, "_dhp_pid_use_normalized_theta", True)):
                    sp = float(theta_ref) / float(
                        getattr(self, "_env_max_pitch_rad", 1.0)
                    )
                    meas = float(x[3]) / float(getattr(self, "_env_max_pitch_rad", 1.0))
                else:
                    sp = float(theta_ref)
                    meas = float(x[3])
                u_base = float(pid.select_action(sp, meas))
            return float(np.clip(u_base, -1.0, 1.0))

        # Default: PD baseline using theta/q errors (normalized like env)
        inv_theta = 1.0 / float(getattr(self, "_env_max_pitch_rad", 1.0))
        inv_q = 1.0 / float(getattr(self, "_env_max_pitch_rate_rad_s", 1.0))
        e_theta_n = (float(theta_ref) - float(x[3])) * float(inv_theta)
        e_q_n = (float(q_ref) - float(x[2])) * float(inv_q)
        u_base = float(getattr(self, "_dhp_baseline_kp", 0.0)) * float(
            e_theta_n
        ) + float(getattr(self, "_dhp_baseline_kd", 0.0)) * float(e_q_n)
        return float(np.clip(u_base, -1.0, 1.0))

    def _warmstart_actor_from_baseline(
        self, *, episodes: int, max_steps: int | None
    ) -> None:
        """Warm-start actor by imitating a stabilizing baseline controller.

        This matches the paper's practical recommendation: start critic training with a
        stabilizing actor, without introducing any baseline into ACD update equations.
        """
        episodes = int(episodes)
        if episodes <= 0:
            return
        if not bool(getattr(self, "_is_model_based_design", False)):
            return

        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
        state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
        if state_dim < 1:
            raise ValueError("Warm-start requires a valid _dhp_state_dim")
        if getattr(self, "_dhp_ref", None) is None:
            raise ValueError(
                "Warm-start requires env.reference_signal (DHP-style tracking)"
            )

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for _ep in range(episodes):
            _obs, _info = self.env.reset()
            # Reset PID integrator per episode if used
            if getattr(self, "_dhp_pid", None) is not None:
                try:
                    self._dhp_pid.reset()
                except Exception:
                    pass

            done = False
            steps = 0
            while not done:
                idx_t = int(getattr(self.env, "current_step", steps))
                x = _as_flat_np(getattr(self.env, "state"))

                ref = np.asarray(getattr(self, "_dhp_ref"), dtype=float)
                idx = int(np.clip(idx_t, 0, int(ref.shape[1]) - 1))
                idx_prev = int(np.clip(idx_t - 1, 0, int(ref.shape[1]) - 1))
                theta_ref = float(ref[0, idx])
                theta_ref_prev = float(ref[0, idx_prev])
                q_ref = float(
                    (theta_ref - theta_ref_prev)
                    / float(getattr(self, "_dhp_dt", 0.01) or 0.01)
                )

                r_t = np.concatenate([x[:state_dim], [theta_ref, q_ref]]).astype(
                    np.float32
                )
                u_base = float(
                    self._dhp_baseline_u_norm(x=x, theta_ref=theta_ref, q_ref=q_ref)
                )

                # Supervised pair (R -> u)
                xs.append(r_t)
                ys.append(np.full((act_dim,), u_base, dtype=np.float32))

                action = np.full((act_dim,), u_base, dtype=np.float32)
                low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                    -1
                )
                high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(
                    -1
                )
                action = np.clip(action, low, high).astype(np.float32)

                _obs, _reward, terminated, truncated, _info = self.env.step(action)
                done = bool(terminated or truncated)
                steps += 1
                if max_steps is not None and steps >= int(max_steps):
                    break

        if len(xs) < 1:
            return

        x_t = torch.as_tensor(
            np.stack(xs, axis=0), dtype=torch.float32, device=self.device
        )
        y_t = torch.as_tensor(
            np.stack(ys, axis=0), dtype=torch.float32, device=self.device
        )

        # Use the same LR as the actor optimizer
        lr = float(self.actor_optim.param_groups[0].get("lr", 3e-4))
        opt = Adam(self.actor.parameters(), lr=lr)
        epochs = int(getattr(self, "_dhp_warmstart_actor_epochs", 1) or 1)
        for _ in range(max(1, epochs)):
            pred = self.actor(x_t)
            loss = F.mse_loss(pred, y_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            opt.step()

    # ---- learning ----
    def _adhdp_baseline_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute a PD/PID baseline action from normalized observation.

        This is primarily intended for ImprovedB747Env where the observation is:
          obs = [norm_pitch_error, norm_q, norm_theta, norm_prev_action]   (dim=4)
        and optionally (include_reference_in_obs=True):
          obs = [..., norm_theta_ref, norm_ref_theta_dot]                 (dim=6)

        Returns: tensor with shape (B, act_dim) in normalized action units.
        """
        if obs.ndim != 2:
            raise ValueError(f"Expected obs shape (B, obs_dim), got {tuple(obs.shape)}")
        if int(obs.shape[1]) < 2:
            raise ValueError(
                "PD baseline requires observation with at least 2 dims (pitch_error, q)"
            )

        e_theta_n = obs[:, 0]  # target - theta (normalized)
        q_n = obs[:, 1]  # q (normalized)
        # If available, use reference theta-dot (normalized) as q_ref
        q_ref_n = obs[:, 5] if int(obs.shape[1]) >= 6 else torch.zeros_like(q_n)
        e_q_n = q_ref_n - q_n

        kp = float(getattr(self, "_dhp_baseline_kp", 0.0))
        ki = float(getattr(self, "_dhp_baseline_ki", 0.0))
        kd = float(getattr(self, "_dhp_baseline_kd", 0.0))
        baseline_type = str(getattr(self, "_dhp_baseline_type", "pd")).lower().strip()
        if baseline_type not in ("pd", "pid"):
            baseline_type = "pd"

        if baseline_type == "pid":
            # Stateful integrator is maintained per episode (canonical ADHDP is online, no replay).
            dt = float(getattr(self.env, "dt", 0.1))
            i_prev = float(getattr(self, "_adhdp_pid_i", 0.0) or 0.0)
            i_new = i_prev + float(torch.mean(e_theta_n).item()) * dt
            # Simple anti-windup (keep integral bounded)
            i_clip = float(getattr(self, "_adhdp_pid_i_clip", 1.0) or 1.0)
            if i_clip > 0.0:
                i_new = float(np.clip(i_new, -i_clip, i_clip))
            self._adhdp_pid_i = float(i_new)
            u = kp * e_theta_n + ki * float(i_new) + kd * e_q_n
        else:
            u = kp * e_theta_n + kd * e_q_n

        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
        u = u.reshape(-1, 1).repeat(1, act_dim)
        return torch.clamp(u, -1.0, 1.0)

    def _adhdp_policy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Policy action for canonical ADHDP."""
        return self.actor(obs)

    def _warmstart_adhdp_actor_from_pd(
        self, *, episodes: int, max_steps: int | None
    ) -> None:
        """Warm-start ADHDP actor by imitating PD stabilizer computed from observation.

        This matches the paper's practical recommendation (stabilizing initial actor),
        without introducing baseline+residual composition into the ADHDP equations.
        """
        episodes = int(episodes)
        if episodes <= 0:
            return

        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for _ep in range(episodes):
            obs, _info = self.env.reset()
            obs = _as_flat_np(obs)
            # reset PID integrator at episode start
            try:
                self._adhdp_pid_i = 0.0
            except Exception:
                pass
            done = False
            steps = 0
            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    u_base = self._adhdp_baseline_action(obs_t).squeeze(0).cpu().numpy()
                u_base = np.asarray(u_base, dtype=np.float32).reshape(-1)
                if u_base.shape[0] == 1 and act_dim > 1:
                    u_base = np.repeat(u_base, act_dim).astype(np.float32)

                xs.append(obs.astype(np.float32))
                ys.append(u_base.astype(np.float32))

                low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                    -1
                )
                high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(
                    -1
                )
                action = np.clip(u_base, low, high).astype(np.float32)

                next_obs, _reward, terminated, truncated, _info = self.env.step(action)
                obs = _as_flat_np(next_obs)
                done = bool(terminated or truncated)
                steps += 1
                if max_steps is not None and steps >= int(max_steps):
                    break

        if len(xs) < 1:
            return

        x_t = torch.as_tensor(
            np.stack(xs, axis=0), dtype=torch.float32, device=self.device
        )
        y_t = torch.as_tensor(
            np.stack(ys, axis=0), dtype=torch.float32, device=self.device
        )

        lr = float(self.actor_optim.param_groups[0].get("lr", 3e-4))
        opt = Adam(self.actor.parameters(), lr=lr)
        epochs = int(getattr(self, "_adhdp_warmstart_actor_epochs", 1) or 1)
        for _ in range(max(1, epochs)):
            pred = self.actor(x_t)
            loss = F.mse_loss(pred, y_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            opt.step()

    def _dhp_targets_and_grads(
        self,
        *,
        r_t: torch.Tensor,
        r_tp1: torch.Tensor,
        u_t: torch.Tensor,
        u_prev: torch.Tensor,
        u_prev2: torch.Tensor,
        t_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (lambda_target, g_u) for DHP at time t.

        r_t, r_tp1: shape (r_dim,) where r_dim = state_dim + ref_dim
        u_t, u_prev: shape (act_dim,)
        """
        assert self._dhp_A is not None and self._dhp_B is not None
        state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
        if state_dim < 1:
            raise ValueError("DHP requires a valid _dhp_state_dim")

        theta = r_t[3]
        q = r_t[2]
        # Reference components are expected in r_t for tracking tasks: [x, theta_ref, q_ref]
        theta_ref_t = (
            r_t[state_dim] if r_t.shape[0] >= state_dim + 2 else torch.zeros_like(theta)
        )
        q_ref_t = (
            r_t[state_dim + 1] if r_t.shape[0] >= state_dim + 2 else torch.zeros_like(q)
        )
        if self._dhp_use_env_cost:
            # normalized errors like ImprovedB747Env cost_base
            e_theta = (theta - theta_ref_t) / float(self._env_max_pitch_rad)
            e_q = (q - q_ref_t) / float(self._env_max_pitch_rate_rad_s)
        else:
            e_theta = theta - theta_ref_t
            e_q = q - q_ref_t
        du = u_t - u_prev
        ddu = u_t - 2.0 * u_prev + u_prev2

        # Utility U is COST (positive); we minimize it
        if self._dhp_use_env_cost:
            w_theta = float(self._env_w_pitch)
            w_q = float(self._env_w_q)
            w_u = float(self._env_w_action)
            w_du = float(self._env_w_smooth)
            w_jerk = float(self._env_w_jerk)
            inv_theta = 1.0 / float(self._env_max_pitch_rad)
            inv_q = 1.0 / float(self._env_max_pitch_rate_rad_s)
        else:
            w_theta = float(self._dhp_w_theta)
            w_q = float(self._dhp_w_q)
            w_u = float(self._dhp_w_u)
            w_du = float(self._dhp_w_du)
            w_jerk = 0.0
            inv_theta = 1.0
            inv_q = 1.0

        # Derivatives of U wrt R (q, theta, and also wrt reference signals if present)
        dU_dR = torch.zeros_like(r_t)
        dU_dR[2] = 2.0 * w_q * e_q * float(inv_q)
        dU_dR[3] = 2.0 * w_theta * e_theta * float(inv_theta)
        if r_t.shape[0] >= state_dim + 2:
            # d/d theta_ref and d/d q_ref (sign is negative)
            dU_dR[state_dim] = -2.0 * w_theta * e_theta * float(inv_theta)
            dU_dR[state_dim + 1] = -2.0 * w_q * e_q * float(inv_q)

        # Derivative of U wrt u
        dU_du = 2.0 * w_u * u_t + 2.0 * w_du * du + 2.0 * w_jerk * ddu

        # lambda at t+1 from critic (conditioned on R(t+1))
        with torch.no_grad():
            out_tp1 = self.critic(r_tp1.unsqueeze(0))
            # DHP: critic returns lambda tensor; GDHP: returns (J, lambda)
            lam_tp1 = out_tp1[1] if isinstance(out_tp1, tuple) else out_tp1
            lam_tp1 = lam_tp1.squeeze(0)  # (r_dim,)

        # g_u = dU/du + gamma * B^T * lambda_{t+1}
        g_u = dU_du + self.gamma * (self._dhp_B.T @ lam_tp1).reshape_as(dU_du)

        # du/dR from actor (for lambda target)
        # Note: compute jacobian row-wise via autograd.grad for each action dim.
        du_dR_rows = []
        for i in range(int(u_t.shape[0])):
            grad_i = torch.autograd.grad(
                outputs=u_t[i],
                inputs=r_t,
                grad_outputs=torch.ones_like(u_t[i]),
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            du_dR_rows.append(grad_i)
        du_dR = torch.stack(du_dR_rows, dim=0)  # (act_dim, r_dim)

        # lambda target: dU/dR + gamma*A^T*lam_{t+1} + (du/dR)^T * g_u
        lam_target = dU_dR + self.gamma * (self._dhp_A.T @ lam_tp1) + (du_dR.T @ g_u)
        return lam_target.detach(), g_u.detach()

    def _td_update_batch(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        rew_b: np.ndarray,
        next_obs_b: np.ndarray,
        done_bootstrap_b: np.ndarray,
        *,
        do_critic_update: bool = True,
        do_actor_update: bool = True,
    ) -> Tuple[float, float]:
        """One update step on a given batch. Returns (critic_loss, actor_loss).

        do_actor_update=False is used for paper-style schedules where the critic is
        trained first with a fixed stabilizing actor.
        """
        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)
        # reward comes in as (B,1) from replay; accept (B,) too.
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=self.device).reshape(
            -1, 1
        )
        next_obs_t = torch.as_tensor(
            next_obs_b, dtype=torch.float32, device=self.device
        )
        done_t = torch.as_tensor(
            done_bootstrap_b, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        # Convert reward to cost
        cost_t = -rew_t

        # Practical variant uses target networks; canonical ADHDP uses online networks.
        actor_next = self.actor_target if self.actor_target is not None else self.actor
        critic_next = (
            self.critic_target if self.critic_target is not None else self.critic
        )

        with torch.no_grad():
            next_act_t = actor_next(next_obs_t)
            q_next = critic_next(next_obs_t, next_act_t)
            target_q = cost_t + (1.0 - done_t) * self.gamma * q_next

        # Critic update
        critic_loss_t: torch.Tensor
        if bool(do_critic_update):
            q = self.critic(obs_t, act_t)
            critic_loss_t = F.mse_loss(q, target_q)
            self.critic_optim.zero_grad()
            critic_loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()
        else:
            critic_loss_t = torch.as_tensor(
                0.0, dtype=torch.float32, device=self.device
            )

        # Actor update: minimize critic (cost-to-go)
        actor_loss_t: torch.Tensor
        if bool(do_actor_update):
            actor_act = self.actor(obs_t)
            actor_loss_t = self.critic(obs_t, actor_act).mean()
            # Optional behavioral-cloning regularizer towards PD stabilizer
            if self.design == "adhdp":
                coef = float(getattr(self, "_adhdp_actor_bc_coef", 0.0) or 0.0)
                if coef > 0.0:
                    with torch.no_grad():
                        u_base = self._adhdp_baseline_action(obs_t)
                    actor_loss_t = actor_loss_t + coef * F.mse_loss(actor_act, u_base)
            self.actor_optim.zero_grad()
            actor_loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()
        else:
            actor_loss_t = torch.as_tensor(0.0, dtype=torch.float32, device=self.device)

        # Polyak updates (practical variant only)
        if (
            self.design == "ddpg"
            and self.use_target_networks
            and self.actor_target is not None
            and self.critic_target is not None
        ):
            polyak_update(self.actor_target, self.actor, tau=self.tau)
            polyak_update(self.critic_target, self.critic, tau=self.tau)

        return float(critic_loss_t.item()), float(actor_loss_t.item())

    def train(self, *args, **kwargs) -> None:
        """Train for a number of episodes.

        Args:
            num_episodes (int): Number of episodes.
            max_steps (int | None): Optional per-episode cap.
        """
        num_episodes = (
            int(args[0]) if len(args) > 0 else int(kwargs.get("num_episodes", 1))
        )
        max_steps = kwargs.get("max_steps", None)
        max_steps_i = int(max_steps) if max_steps is not None else None

        total_steps = 0
        # Paper-style initialization: warm-start ADHDP actor as PD stabilizer
        if self.design == "adhdp":
            warm_ep = int(getattr(self, "_adhdp_warmstart_actor_episodes", 0) or 0)
            if warm_ep > 0:
                self._warmstart_adhdp_actor_from_pd(
                    episodes=warm_ep, max_steps=max_steps_i
                )
        # Paper-style initialization: warm-start actor as a stabilizing controller
        # before beginning ACD updates. This keeps the closed loop stable while the critic adapts.
        warm_ep = int(getattr(self, "_dhp_warmstart_actor_episodes", 0) or 0)
        if warm_ep > 0 and bool(getattr(self, "_is_model_based_design", False)):
            self._warmstart_actor_from_baseline(episodes=warm_ep, max_steps=max_steps_i)
            if bool(getattr(self, "_dhp_warmstart_actor_disable_baseline_after", True)):
                # After warm-start we typically want "pure" ACD updates (no residual baseline).
                self._dhp_use_baseline = False

        for ep in range(num_episodes):
            obs, _info = self.env.reset()
            obs = _as_flat_np(obs)
            ep_reward = 0.0
            steps = 0
            done = False
            # Reset ADHDP PID integrator at episode start (important for stable warm-start/BC targets)
            if self.design == "adhdp":
                try:
                    self._adhdp_pid_i = 0.0
                except Exception:
                    pass
            # Decay ADHDP actor BC regularizer per episode (if enabled)
            if self.design == "adhdp":
                try:
                    decay = float(getattr(self, "_adhdp_actor_bc_decay", 1.0) or 1.0)
                    if decay < 1.0:
                        self._adhdp_actor_bc_coef = float(
                            getattr(self, "_adhdp_actor_bc_coef", 0.0) or 0.0
                        ) * float(decay)
                except Exception:
                    pass
            # Reset action history per episode to avoid cross-episode coupling in du/jerk penalties.
            # (Important for tracking tasks where the episode restarts the plant state.)
            if bool(getattr(self, "_is_model_based_design", False)):
                self._prev_u_norm = np.zeros_like(self._prev_u_norm)
                self._prev2_u_norm = np.zeros_like(self._prev2_u_norm)
            if (
                bool(getattr(self, "_is_model_based_design", False))
                and getattr(self, "_dhp_pid", None) is not None
            ):
                # Reset PID baseline internal state per episode
                try:
                    self._dhp_pid.reset()
                except Exception:
                    pass

            # Section III (Prokhorov & Wunsch): alternate critic's and action's training cycles.
            # If cycle lengths are 0, fall back to updating both each step (legacy behavior).
            dhp_cycle_c = int(getattr(self, "_dhp_critic_cycle_episodes", 0) or 0)
            dhp_cycle_a = int(getattr(self, "_dhp_action_cycle_episodes", 0) or 0)
            dhp_phase = "both"
            if (
                bool(getattr(self, "_is_model_based_design", False))
                and dhp_cycle_c > 0
                and dhp_cycle_a > 0
            ):
                cycle_len = dhp_cycle_c + dhp_cycle_a
                pos = int(ep % cycle_len)
                dhp_phase = "critic" if pos < dhp_cycle_c else "actor"
            do_critic = dhp_phase in ("both", "critic")
            do_actor = dhp_phase in ("both", "actor")

            while not done:
                if self.design == "dhp":
                    # Use physical state x from env; ignore normalized observation for learning.
                    idx_t = int(getattr(self.env, "current_step", steps))
                    x_np = _as_flat_np(getattr(self.env, "state"))
                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                    x_t.requires_grad_(True)
                    # Build R(t) = [x(t), theta_ref(t), q_ref(t)]
                    assert self._dhp_ref is not None
                    idx = int(np.clip(idx_t, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx_prev = int(
                        np.clip(idx_t - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref = float(self._dhp_ref[0, idx])
                    theta_ref_prev = float(self._dhp_ref[0, idx_prev])
                    q_ref = float(
                        (theta_ref - theta_ref_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_t = torch.cat(
                        [
                            x_t,
                            torch.tensor(
                                [theta_ref, q_ref],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    # Baseline stabilizing controller (PD on normalized errors), then residual from actor.
                    if self._dhp_use_baseline:
                        # Baseline is treated as non-differentiable stabilizer (stop-grad through x)
                        x_det = x_t.detach()
                        if (
                            getattr(self, "_dhp_baseline_type", "pd") == "pid"
                            and getattr(self, "_dhp_pid", None) is not None
                        ):
                            # PID baseline on pitch tracking: setpoint vs measurement (optionally normalized)
                            pid_mode = (
                                str(getattr(self, "_dhp_pid_mode", "norm"))
                                .lower()
                                .strip()
                            )
                            if pid_mode == "deg":
                                # Compatibility with example/comparison PID tuning notebooks:
                                # PID inputs/outputs are in degrees; convert output to u_norm.
                                sp = float(np.rad2deg(theta_ref))
                                meas = float(np.rad2deg(float(x_det[3])))
                                control_deg = float(
                                    self._dhp_pid.select_action(sp, meas)
                                )
                                u_base = float(control_deg) / float(
                                    self._env_max_stabilizer_angle_deg
                                )
                            else:
                                if bool(
                                    getattr(self, "_dhp_pid_use_normalized_theta", True)
                                ):
                                    sp = float(theta_ref) / float(
                                        self._env_max_pitch_rad
                                    )
                                    meas = float(x_det[3]) / float(
                                        self._env_max_pitch_rad
                                    )
                                else:
                                    sp = float(theta_ref)
                                    meas = float(x_det[3])
                                u_base = float(self._dhp_pid.select_action(sp, meas))
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                        else:
                            # PD baseline using theta/q errors (normalized)
                            e_theta_n = (float(theta_ref) - float(x_det[3])) / float(
                                self._env_max_pitch_rad
                            )
                            e_q_n = (float(q_ref) - float(x_det[2])) / float(
                                self._env_max_pitch_rate_rad_s
                            )
                            u_base = float(self._dhp_baseline_kp) * float(
                                e_theta_n
                            ) + float(self._dhp_baseline_kd) * float(e_q_n)
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                    else:
                        u_base_t = torch.zeros(
                            (1,), dtype=torch.float32, device=self.device
                        )

                    # Actor outputs an action (optionally used as a residual on top of baseline).
                    # Keep constraints via clamp (piecewise differentiable) rather than extra tanh.
                    delta = self.actor(r_t.unsqueeze(0)).squeeze(0)  # requires grad
                    low_t = torch.as_tensor(
                        np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                            -1
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    high_t = torch.as_tensor(
                        np.asarray(
                            self.env.action_space.high, dtype=np.float32
                        ).reshape(-1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    u_pol = u_base_t + float(self._dhp_residual_scale) * delta
                    u_det = torch.clamp(u_pol, low_t, high_t)
                    u_np = u_det.detach().cpu().numpy()
                    if self.exploration_std > 0.0:
                        u_np = u_np + np.random.normal(
                            0.0, self.exploration_std, size=u_np.shape
                        ).astype(np.float32)
                    # clip to env bounds
                    low = np.asarray(
                        self.env.action_space.low, dtype=np.float32
                    ).reshape(-1)
                    high = np.asarray(
                        self.env.action_space.high, dtype=np.float32
                    ).reshape(-1)
                    u_np = np.clip(u_np, low, high).astype(np.float32)

                    next_obs, reward, terminated, truncated, _info = self.env.step(u_np)
                    next_obs = _as_flat_np(next_obs)
                    x1_np = _as_flat_np(getattr(self.env, "state"))
                    x_tp1 = torch.as_tensor(
                        x1_np, dtype=torch.float32, device=self.device
                    )
                    # R(t+1) for critic: use reference at idx_t+1
                    idx1 = int(np.clip(idx_t + 1, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx1_prev = int(
                        np.clip(idx1 - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref1 = float(self._dhp_ref[0, idx1])
                    theta_ref1_prev = float(self._dhp_ref[0, idx1_prev])
                    q_ref1 = float(
                        (theta_ref1 - theta_ref1_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_tp1 = torch.cat(
                        [
                            x_tp1,
                            torch.tensor(
                                [theta_ref1, q_ref1],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    # Critic update (lambda)
                    u_prev_t = torch.as_tensor(
                        self._prev_u_norm, dtype=torch.float32, device=self.device
                    )
                    u_prev2_t = torch.as_tensor(
                        self._prev2_u_norm, dtype=torch.float32, device=self.device
                    )
                    # Use deterministic actor output for gradients (DHP needs du/dx)
                    u_t = u_det

                    lam_target, g_u = self._dhp_targets_and_grads(
                        r_t=r_t,
                        r_tp1=r_tp1,
                        u_t=u_t,
                        u_prev=u_prev_t,
                        u_prev2=u_prev2_t,
                        t_idx=idx_t,
                    )
                    critic_loss_t = None
                    if do_critic:
                        lam_pred = self.critic(r_t.unsqueeze(0)).squeeze(0)
                        critic_loss_t = F.mse_loss(lam_pred, lam_target)
                        self.critic_optim.zero_grad()
                        critic_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=1.0
                        )
                        self.critic_optim.step()

                    # Actor update: gradient descent using g_u at output (paper Eq. (9)/(10) style)
                    # actor_loss = sum(u * g_u) so d/dtheta = du/dtheta * g_u
                    actor_loss_t = None
                    if do_actor:
                        r_t_det = torch.cat(
                            [
                                x_t.detach(),
                                torch.tensor(
                                    [theta_ref, q_ref],
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            ],
                            dim=0,
                        )
                        delta_det = self.actor(r_t_det.unsqueeze(0)).squeeze(0)
                        u_pol_det = (
                            u_base_t + float(self._dhp_residual_scale) * delta_det
                        )
                        u_for_grad = torch.clamp(u_pol_det, low_t, high_t)
                        actor_loss_t = torch.sum(u_for_grad * g_u)
                        reg = float(getattr(self, "_dhp_actor_delta_l2", 0.0) or 0.0)
                        if reg > 0.0:
                            actor_loss_t = actor_loss_t + reg * torch.mean(
                                delta_det * delta_det
                            )
                        self.actor_optim.zero_grad()
                        actor_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=1.0
                        )
                        self.actor_optim.step()

                    # Store previously *policy* action (deterministic, bounded) for smoothness/jerk penalties
                    # and DHP gradients. Using the noisy applied action here creates a mismatch between
                    # (u_t used for gradients) and (u_prev used for du/jerk), which can cause drift.
                    self._prev2_u_norm = self._prev_u_norm.copy()
                    self._prev_u_norm = (
                        u_det.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        if critic_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/critic_lambda",
                                float(critic_loss_t.item()),
                                self._updates,
                            )
                        if actor_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/actor", float(actor_loss_t.item()), self._updates
                            )
                        self.writer.add_scalar(
                            "train/dhp_phase",
                            (
                                0.0
                                if dhp_phase == "critic"
                                else (1.0 if dhp_phase == "actor" else 2.0)
                            ),
                            self._updates,
                        )

                elif self.design == "gdhp":
                    # GDHP (Fig. 5): critic learns both J and lambda=dJ/dR.
                    idx_t = int(getattr(self.env, "current_step", steps))
                    x_np = _as_flat_np(getattr(self.env, "state"))
                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                    x_t.requires_grad_(True)
                    assert self._dhp_ref is not None

                    state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
                    idx = int(np.clip(idx_t, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx_prev = int(
                        np.clip(idx_t - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref = float(self._dhp_ref[0, idx])
                    theta_ref_prev = float(self._dhp_ref[0, idx_prev])
                    q_ref = float(
                        (theta_ref - theta_ref_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_t = torch.cat(
                        [
                            x_t,
                            torch.tensor(
                                [theta_ref, q_ref],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    # Baseline stabilizer (optional)
                    if self._dhp_use_baseline:
                        x_det = x_t.detach()
                        if (
                            getattr(self, "_dhp_baseline_type", "pd") == "pid"
                            and getattr(self, "_dhp_pid", None) is not None
                        ):
                            pid_mode = (
                                str(getattr(self, "_dhp_pid_mode", "norm"))
                                .lower()
                                .strip()
                            )
                            if pid_mode == "deg":
                                sp = float(np.rad2deg(theta_ref))
                                meas = float(np.rad2deg(float(x_det[3])))
                                control_deg = float(
                                    self._dhp_pid.select_action(sp, meas)
                                )
                                u_base = float(control_deg) / float(
                                    self._env_max_stabilizer_angle_deg
                                )
                            else:
                                if bool(
                                    getattr(self, "_dhp_pid_use_normalized_theta", True)
                                ):
                                    sp = float(theta_ref) / float(
                                        self._env_max_pitch_rad
                                    )
                                    meas = float(x_det[3]) / float(
                                        self._env_max_pitch_rad
                                    )
                                else:
                                    sp = float(theta_ref)
                                    meas = float(x_det[3])
                                u_base = float(self._dhp_pid.select_action(sp, meas))
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                        else:
                            e_theta_n = (float(theta_ref) - float(x_det[3])) / float(
                                self._env_max_pitch_rad
                            )
                            e_q_n = (float(q_ref) - float(x_det[2])) / float(
                                self._env_max_pitch_rate_rad_s
                            )
                            u_base = float(self._dhp_baseline_kp) * float(
                                e_theta_n
                            ) + float(self._dhp_baseline_kd) * float(e_q_n)
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                    else:
                        u_base_t = torch.zeros(
                            (1,), dtype=torch.float32, device=self.device
                        )

                    delta = self.actor(r_t.unsqueeze(0)).squeeze(0)  # requires grad
                    low_t = torch.as_tensor(
                        np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                            -1
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    high_t = torch.as_tensor(
                        np.asarray(
                            self.env.action_space.high, dtype=np.float32
                        ).reshape(-1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    u_pol = u_base_t + float(self._dhp_residual_scale) * delta
                    u_det = torch.clamp(u_pol, low_t, high_t)

                    u_np = u_det.detach().cpu().numpy()
                    if self.exploration_std > 0.0:
                        u_np = u_np + np.random.normal(
                            0.0, self.exploration_std, size=u_np.shape
                        ).astype(np.float32)
                    low = np.asarray(
                        self.env.action_space.low, dtype=np.float32
                    ).reshape(-1)
                    high = np.asarray(
                        self.env.action_space.high, dtype=np.float32
                    ).reshape(-1)
                    u_np = np.clip(u_np, low, high).astype(np.float32)

                    next_obs, reward, terminated, truncated, _info = self.env.step(u_np)
                    next_obs = _as_flat_np(next_obs)

                    x1_np = _as_flat_np(getattr(self.env, "state"))
                    x_tp1 = torch.as_tensor(
                        x1_np, dtype=torch.float32, device=self.device
                    )
                    idx1 = int(np.clip(idx_t + 1, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx1_prev = int(
                        np.clip(idx1 - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref1 = float(self._dhp_ref[0, idx1])
                    theta_ref1_prev = float(self._dhp_ref[0, idx1_prev])
                    q_ref1 = float(
                        (theta_ref1 - theta_ref1_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_tp1 = torch.cat(
                        [
                            x_tp1,
                            torch.tensor(
                                [theta_ref1, q_ref1],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    u_prev_t = torch.as_tensor(
                        self._prev_u_norm, dtype=torch.float32, device=self.device
                    )
                    u_prev2_t = torch.as_tensor(
                        self._prev2_u_norm, dtype=torch.float32, device=self.device
                    )
                    u_t = u_det

                    # DHP portion (lambda target and g_u for actor)
                    lam_target, g_u = self._dhp_targets_and_grads(
                        r_t=r_t,
                        r_tp1=r_tp1,
                        u_t=u_t,
                        u_prev=u_prev_t,
                        u_prev2=u_prev2_t,
                        t_idx=idx_t,
                    )

                    # HDP portion (J target)
                    if self._dhp_use_env_cost:
                        w_theta = float(self._env_w_pitch)
                        w_q = float(self._env_w_q)
                        w_u = float(self._env_w_action)
                        w_du = float(self._env_w_smooth)
                        w_jerk = float(self._env_w_jerk)
                        e_theta = (r_t[3] - r_t[state_dim]) / float(
                            self._env_max_pitch_rad
                        )
                        e_q = (r_t[2] - r_t[state_dim + 1]) / float(
                            self._env_max_pitch_rate_rad_s
                        )
                    else:
                        w_theta = float(self._dhp_w_theta)
                        w_q = float(self._dhp_w_q)
                        w_u = float(self._dhp_w_u)
                        w_du = float(self._dhp_w_du)
                        w_jerk = 0.0
                        e_theta = r_t[3] - r_t[state_dim]
                        e_q = r_t[2] - r_t[state_dim + 1]

                    du = u_t - u_prev_t
                    ddu = u_t - 2.0 * u_prev_t + u_prev2_t
                    cost_t = (
                        w_theta * (e_theta * e_theta)
                        + w_q * (e_q * e_q)
                        + w_u * torch.sum(u_t * u_t)
                        + w_du * torch.sum(du * du)
                        + w_jerk * torch.sum(ddu * ddu)
                    )
                    bootstrap = 0.0 if bool(terminated) else 1.0
                    with torch.no_grad():
                        j_tp1, _lam_tp1 = self.critic(r_tp1.unsqueeze(0))
                        j_tp1_s = j_tp1.squeeze(0).squeeze(-1)
                    j_target = (
                        cost_t.detach() + float(bootstrap) * float(self.gamma) * j_tp1_s
                    )

                    critic_loss_t = None
                    if do_critic:
                        j_pred, lam_pred = self.critic(r_t.unsqueeze(0))
                        j_pred_s = j_pred.squeeze(0).squeeze(-1)
                        lam_pred_s = lam_pred.squeeze(0)
                        critic_loss_t = F.mse_loss(j_pred_s, j_target) + F.mse_loss(
                            lam_pred_s, lam_target
                        )
                        self.critic_optim.zero_grad()
                        critic_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=1.0
                        )
                        self.critic_optim.step()

                    actor_loss_t = None
                    if do_actor:
                        r_t_det = torch.cat(
                            [
                                x_t.detach(),
                                torch.tensor(
                                    [theta_ref, q_ref],
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            ],
                            dim=0,
                        )
                        delta_det = self.actor(r_t_det.unsqueeze(0)).squeeze(0)
                        u_pol_det = (
                            u_base_t + float(self._dhp_residual_scale) * delta_det
                        )
                        u_for_grad = torch.clamp(u_pol_det, low_t, high_t)
                        actor_loss_t = torch.sum(u_for_grad * g_u)
                        reg = float(getattr(self, "_dhp_actor_delta_l2", 0.0) or 0.0)
                        if reg > 0.0:
                            actor_loss_t = actor_loss_t + reg * torch.mean(
                                delta_det * delta_det
                            )
                        self.actor_optim.zero_grad()
                        actor_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=1.0
                        )
                        self.actor_optim.step()

                    self._prev2_u_norm = self._prev_u_norm.copy()
                    self._prev_u_norm = (
                        u_det.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        if critic_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/critic_gdhp",
                                float(critic_loss_t.item()),
                                self._updates,
                            )
                        if actor_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/actor", float(actor_loss_t.item()), self._updates
                            )
                        self.writer.add_scalar(
                            "train/dhp_phase",
                            (
                                0.0
                                if dhp_phase == "critic"
                                else (1.0 if dhp_phase == "actor" else 2.0)
                            ),
                            self._updates,
                        )

                elif self.design == "hdp":
                    # HDP: critic learns J(R). Actor is improved via model-based one-step lookahead.
                    idx_t = int(getattr(self.env, "current_step", steps))
                    x_np = _as_flat_np(getattr(self.env, "state"))
                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                    assert self._dhp_ref is not None

                    state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
                    idx = int(np.clip(idx_t, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx_prev = int(
                        np.clip(idx_t - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref = float(self._dhp_ref[0, idx])
                    theta_ref_prev = float(self._dhp_ref[0, idx_prev])
                    q_ref = float(
                        (theta_ref - theta_ref_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_t = torch.cat(
                        [
                            x_t,
                            torch.tensor(
                                [theta_ref, q_ref],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    # Baseline stabilizer (optional)
                    if self._dhp_use_baseline:
                        x_det = x_t.detach()
                        if (
                            getattr(self, "_dhp_baseline_type", "pd") == "pid"
                            and getattr(self, "_dhp_pid", None) is not None
                        ):
                            pid_mode = (
                                str(getattr(self, "_dhp_pid_mode", "norm"))
                                .lower()
                                .strip()
                            )
                            if pid_mode == "deg":
                                sp = float(np.rad2deg(theta_ref))
                                meas = float(np.rad2deg(float(x_det[3])))
                                control_deg = float(
                                    self._dhp_pid.select_action(sp, meas)
                                )
                                u_base = float(control_deg) / float(
                                    self._env_max_stabilizer_angle_deg
                                )
                            else:
                                if bool(
                                    getattr(self, "_dhp_pid_use_normalized_theta", True)
                                ):
                                    sp = float(theta_ref) / float(
                                        self._env_max_pitch_rad
                                    )
                                    meas = float(x_det[3]) / float(
                                        self._env_max_pitch_rad
                                    )
                                else:
                                    sp = float(theta_ref)
                                    meas = float(x_det[3])
                                u_base = float(self._dhp_pid.select_action(sp, meas))
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                        else:
                            e_theta_n = (float(theta_ref) - float(x_det[3])) / float(
                                self._env_max_pitch_rad
                            )
                            e_q_n = (float(q_ref) - float(x_det[2])) / float(
                                self._env_max_pitch_rate_rad_s
                            )
                            u_base = float(self._dhp_baseline_kp) * float(
                                e_theta_n
                            ) + float(self._dhp_baseline_kd) * float(e_q_n)
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                    else:
                        u_base_t = torch.zeros(
                            (1,), dtype=torch.float32, device=self.device
                        )

                    low_t = torch.as_tensor(
                        np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                            -1
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    high_t = torch.as_tensor(
                        np.asarray(
                            self.env.action_space.high, dtype=np.float32
                        ).reshape(-1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # Deterministic policy action for env interaction (no grad required here)
                    with torch.no_grad():
                        delta = self.actor(r_t.unsqueeze(0)).squeeze(0)
                        u_det = torch.clamp(
                            u_base_t + float(self._dhp_residual_scale) * delta,
                            low_t,
                            high_t,
                        )
                    u_np = u_det.detach().cpu().numpy()
                    if self.exploration_std > 0.0:
                        u_np = u_np + np.random.normal(
                            0.0, self.exploration_std, size=u_np.shape
                        ).astype(np.float32)
                    low = np.asarray(
                        self.env.action_space.low, dtype=np.float32
                    ).reshape(-1)
                    high = np.asarray(
                        self.env.action_space.high, dtype=np.float32
                    ).reshape(-1)
                    u_np = np.clip(u_np, low, high).astype(np.float32)

                    next_obs, reward, terminated, truncated, _info = self.env.step(u_np)
                    next_obs = _as_flat_np(next_obs)

                    x1_np = _as_flat_np(getattr(self.env, "state"))
                    x_tp1 = torch.as_tensor(
                        x1_np, dtype=torch.float32, device=self.device
                    )
                    idx1 = int(np.clip(idx_t + 1, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx1_prev = int(
                        np.clip(idx1 - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref1 = float(self._dhp_ref[0, idx1])
                    theta_ref1_prev = float(self._dhp_ref[0, idx1_prev])
                    q_ref1 = float(
                        (theta_ref1 - theta_ref1_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_tp1 = torch.cat(
                        [
                            x_tp1,
                            torch.tensor(
                                [theta_ref1, q_ref1],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    u_prev_t = torch.as_tensor(
                        self._prev_u_norm, dtype=torch.float32, device=self.device
                    )
                    u_prev2_t = torch.as_tensor(
                        self._prev2_u_norm, dtype=torch.float32, device=self.device
                    )

                    if self._dhp_use_env_cost:
                        w_theta = float(self._env_w_pitch)
                        w_q = float(self._env_w_q)
                        w_u = float(self._env_w_action)
                        w_du = float(self._env_w_smooth)
                        w_jerk = float(self._env_w_jerk)
                        e_theta = (r_t[3] - r_t[state_dim]) / float(
                            self._env_max_pitch_rad
                        )
                        e_q = (r_t[2] - r_t[state_dim + 1]) / float(
                            self._env_max_pitch_rate_rad_s
                        )
                    else:
                        w_theta = float(self._dhp_w_theta)
                        w_q = float(self._dhp_w_q)
                        w_u = float(self._dhp_w_u)
                        w_du = float(self._dhp_w_du)
                        w_jerk = 0.0
                        e_theta = r_t[3] - r_t[state_dim]
                        e_q = r_t[2] - r_t[state_dim + 1]
                    du = u_det - u_prev_t
                    ddu = u_det - 2.0 * u_prev_t + u_prev2_t
                    cost_t = (
                        w_theta * (e_theta * e_theta)
                        + w_q * (e_q * e_q)
                        + w_u * torch.sum(u_det * u_det)
                        + w_du * torch.sum(du * du)
                        + w_jerk * torch.sum(ddu * ddu)
                    )
                    bootstrap = 0.0 if bool(terminated) else 1.0
                    with torch.no_grad():
                        j_tp1 = self.critic(r_tp1.unsqueeze(0)).squeeze(0).squeeze(-1)
                    j_target = (
                        cost_t.detach() + float(bootstrap) * float(self.gamma) * j_tp1
                    )

                    critic_loss_t = None
                    if do_critic:
                        j_pred = self.critic(r_t.unsqueeze(0)).squeeze(0).squeeze(-1)
                        critic_loss_t = F.mse_loss(j_pred, j_target)
                        self.critic_optim.zero_grad()
                        critic_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=1.0
                        )
                        self.critic_optim.step()

                    actor_loss_t = None
                    if do_actor:
                        # Model-based one-step lookahead: minimize U + gamma*J(R_{t+1}(R_t, A_t))
                        r_t_det = torch.cat(
                            [
                                x_t.detach(),
                                torch.tensor(
                                    [theta_ref, q_ref],
                                    dtype=torch.float32,
                                    device=self.device,
                                ),
                            ],
                            dim=0,
                        )
                        delta_det = self.actor(r_t_det.unsqueeze(0)).squeeze(0)
                        u_pol_det = (
                            u_base_t + float(self._dhp_residual_scale) * delta_det
                        )
                        u_for_grad = torch.clamp(u_pol_det, low_t, high_t)
                        du_g = u_for_grad - u_prev_t
                        ddu_g = u_for_grad - 2.0 * u_prev_t + u_prev2_t
                        cost_g = (
                            w_theta * (e_theta.detach() * e_theta.detach())
                            + w_q * (e_q.detach() * e_q.detach())
                            + w_u * torch.sum(u_for_grad * u_for_grad)
                            + w_du * torch.sum(du_g * du_g)
                            + w_jerk * torch.sum(ddu_g * ddu_g)
                        )
                        assert self._dhp_A is not None and self._dhp_B is not None
                        r_pred = (self._dhp_A @ r_t_det) + (self._dhp_B @ u_for_grad)
                        if r_pred.shape[0] >= state_dim + 2:
                            r_pred = r_pred.clone()
                            r_pred[state_dim] = torch.tensor(
                                theta_ref1, dtype=torch.float32, device=self.device
                            )
                            r_pred[state_dim + 1] = torch.tensor(
                                q_ref1, dtype=torch.float32, device=self.device
                            )
                        j_pred_next = (
                            self.critic(r_pred.unsqueeze(0)).squeeze(0).squeeze(-1)
                        )
                        actor_loss_t = (
                            cost_g + float(bootstrap) * float(self.gamma) * j_pred_next
                        )
                        self.actor_optim.zero_grad()
                        actor_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=1.0
                        )
                        self.actor_optim.step()

                    self._prev2_u_norm = self._prev_u_norm.copy()
                    self._prev_u_norm = (
                        u_det.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        if critic_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/critic_hdp",
                                float(critic_loss_t.item()),
                                self._updates,
                            )
                        if actor_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/actor_hdp",
                                float(actor_loss_t.item()),
                                self._updates,
                            )
                        self.writer.add_scalar(
                            "train/dhp_phase",
                            (
                                0.0
                                if dhp_phase == "critic"
                                else (1.0 if dhp_phase == "actor" else 2.0)
                            ),
                            self._updates,
                        )

                elif self.design in ("addhp", "adgdhp"):
                    # ADGDHP: critic learns J(R,A) and its gradients wrt (R,A).
                    # ADDHP is treated as ADGDHP without the scalar-J loss.
                    idx_t = int(getattr(self.env, "current_step", steps))
                    x_np = _as_flat_np(getattr(self.env, "state"))
                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                    assert self._dhp_ref is not None

                    state_dim = int(getattr(self, "_dhp_state_dim", 0) or 0)
                    idx = int(np.clip(idx_t, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx_prev = int(
                        np.clip(idx_t - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref = float(self._dhp_ref[0, idx])
                    theta_ref_prev = float(self._dhp_ref[0, idx_prev])
                    q_ref = float(
                        (theta_ref - theta_ref_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_t = torch.cat(
                        [
                            x_t,
                            torch.tensor(
                                [theta_ref, q_ref],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    # Baseline stabilizer (optional)
                    if self._dhp_use_baseline:
                        x_det = x_t.detach()
                        if (
                            getattr(self, "_dhp_baseline_type", "pd") == "pid"
                            and getattr(self, "_dhp_pid", None) is not None
                        ):
                            pid_mode = (
                                str(getattr(self, "_dhp_pid_mode", "norm"))
                                .lower()
                                .strip()
                            )
                            if pid_mode == "deg":
                                sp = float(np.rad2deg(theta_ref))
                                meas = float(np.rad2deg(float(x_det[3])))
                                control_deg = float(
                                    self._dhp_pid.select_action(sp, meas)
                                )
                                u_base = float(control_deg) / float(
                                    self._env_max_stabilizer_angle_deg
                                )
                            else:
                                if bool(
                                    getattr(self, "_dhp_pid_use_normalized_theta", True)
                                ):
                                    sp = float(theta_ref) / float(
                                        self._env_max_pitch_rad
                                    )
                                    meas = float(x_det[3]) / float(
                                        self._env_max_pitch_rad
                                    )
                                else:
                                    sp = float(theta_ref)
                                    meas = float(x_det[3])
                                u_base = float(self._dhp_pid.select_action(sp, meas))
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                        else:
                            e_theta_n = (float(theta_ref) - float(x_det[3])) / float(
                                self._env_max_pitch_rad
                            )
                            e_q_n = (float(q_ref) - float(x_det[2])) / float(
                                self._env_max_pitch_rate_rad_s
                            )
                            u_base = float(self._dhp_baseline_kp) * float(
                                e_theta_n
                            ) + float(self._dhp_baseline_kd) * float(e_q_n)
                            u_base_t = torch.tensor(
                                [u_base], dtype=torch.float32, device=self.device
                            )
                            u_base_t = torch.clamp(u_base_t, -1.0, 1.0).detach()
                    else:
                        u_base_t = torch.zeros(
                            (1,), dtype=torch.float32, device=self.device
                        )

                    low_t = torch.as_tensor(
                        np.asarray(self.env.action_space.low, dtype=np.float32).reshape(
                            -1
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    high_t = torch.as_tensor(
                        np.asarray(
                            self.env.action_space.high, dtype=np.float32
                        ).reshape(-1),
                        dtype=torch.float32,
                        device=self.device,
                    )

                    delta = self.actor(r_t.unsqueeze(0)).squeeze(0)
                    u_pol = u_base_t + float(self._dhp_residual_scale) * delta
                    u_det = torch.clamp(u_pol, low_t, high_t)

                    u_np = u_det.detach().cpu().numpy()
                    if self.exploration_std > 0.0:
                        u_np = u_np + np.random.normal(
                            0.0, self.exploration_std, size=u_np.shape
                        ).astype(np.float32)
                    low = np.asarray(
                        self.env.action_space.low, dtype=np.float32
                    ).reshape(-1)
                    high = np.asarray(
                        self.env.action_space.high, dtype=np.float32
                    ).reshape(-1)
                    u_np = np.clip(u_np, low, high).astype(np.float32)

                    next_obs, reward, terminated, truncated, _info = self.env.step(u_np)
                    next_obs = _as_flat_np(next_obs)

                    x1_np = _as_flat_np(getattr(self.env, "state"))
                    x_tp1 = torch.as_tensor(
                        x1_np, dtype=torch.float32, device=self.device
                    )
                    idx1 = int(np.clip(idx_t + 1, 0, int(self._dhp_ref.shape[1]) - 1))
                    idx1_prev = int(
                        np.clip(idx1 - 1, 0, int(self._dhp_ref.shape[1]) - 1)
                    )
                    theta_ref1 = float(self._dhp_ref[0, idx1])
                    theta_ref1_prev = float(self._dhp_ref[0, idx1_prev])
                    q_ref1 = float(
                        (theta_ref1 - theta_ref1_prev) / float(self._dhp_dt or 0.01)
                    )
                    r_tp1 = torch.cat(
                        [
                            x_tp1,
                            torch.tensor(
                                [theta_ref1, q_ref1],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )

                    u_prev_t = torch.as_tensor(
                        self._prev_u_norm, dtype=torch.float32, device=self.device
                    )
                    u_prev2_t = torch.as_tensor(
                        self._prev2_u_norm, dtype=torch.float32, device=self.device
                    )

                    # Cost derivatives dU/dR and dU/du (same as in DHP helper)
                    if self._dhp_use_env_cost:
                        w_theta = float(self._env_w_pitch)
                        w_q = float(self._env_w_q)
                        w_u = float(self._env_w_action)
                        w_du = float(self._env_w_smooth)
                        w_jerk = float(self._env_w_jerk)
                        inv_theta = 1.0 / float(self._env_max_pitch_rad)
                        inv_q = 1.0 / float(self._env_max_pitch_rate_rad_s)
                        e_theta = (r_t[3] - r_t[state_dim]) * float(inv_theta)
                        e_q = (r_t[2] - r_t[state_dim + 1]) * float(inv_q)
                    else:
                        w_theta = float(self._dhp_w_theta)
                        w_q = float(self._dhp_w_q)
                        w_u = float(self._dhp_w_u)
                        w_du = float(self._dhp_w_du)
                        w_jerk = 0.0
                        inv_theta = 1.0
                        inv_q = 1.0
                        e_theta = r_t[3] - r_t[state_dim]
                        e_q = r_t[2] - r_t[state_dim + 1]

                    dU_dR = torch.zeros_like(r_t)
                    dU_dR[2] = 2.0 * w_q * e_q * float(inv_q)
                    dU_dR[3] = 2.0 * w_theta * e_theta * float(inv_theta)
                    dU_dR[state_dim] = -2.0 * w_theta * e_theta * float(inv_theta)
                    dU_dR[state_dim + 1] = -2.0 * w_q * e_q * float(inv_q)

                    du0 = u_det - u_prev_t
                    ddu0 = u_det - 2.0 * u_prev_t + u_prev2_t
                    dU_du = 2.0 * w_u * u_det + 2.0 * w_du * du0 + 2.0 * w_jerk * ddu0

                    # Compute (R,A) gradients at t+1 from critic and actor Jacobian du_{t+1}/dR_{t+1}
                    r_tp1_req = r_tp1.detach().clone()
                    r_tp1_req.requires_grad_(True)

                    # Baseline at t+1 (stop-grad)
                    if self._dhp_use_baseline:
                        x1_det = x_tp1.detach()
                        if (
                            getattr(self, "_dhp_baseline_type", "pd") == "pid"
                            and getattr(self, "_dhp_pid", None) is not None
                        ):
                            pid_mode = (
                                str(getattr(self, "_dhp_pid_mode", "norm"))
                                .lower()
                                .strip()
                            )
                            if pid_mode == "deg":
                                sp = float(np.rad2deg(theta_ref1))
                                meas = float(np.rad2deg(float(x1_det[3])))
                                control_deg = float(
                                    self._dhp_pid.select_action(sp, meas)
                                )
                                u_base1 = float(control_deg) / float(
                                    self._env_max_stabilizer_angle_deg
                                )
                            else:
                                if bool(
                                    getattr(self, "_dhp_pid_use_normalized_theta", True)
                                ):
                                    sp = float(theta_ref1) / float(
                                        self._env_max_pitch_rad
                                    )
                                    meas = float(x1_det[3]) / float(
                                        self._env_max_pitch_rad
                                    )
                                else:
                                    sp = float(theta_ref1)
                                    meas = float(x1_det[3])
                                u_base1 = float(self._dhp_pid.select_action(sp, meas))
                            u_base1_t = torch.tensor(
                                [u_base1], dtype=torch.float32, device=self.device
                            )
                            u_base1_t = torch.clamp(u_base1_t, -1.0, 1.0).detach()
                        else:
                            e_theta_n1 = (float(theta_ref1) - float(x1_det[3])) / float(
                                self._env_max_pitch_rad
                            )
                            e_q_n1 = (float(q_ref1) - float(x1_det[2])) / float(
                                self._env_max_pitch_rate_rad_s
                            )
                            u_base1 = float(self._dhp_baseline_kp) * float(
                                e_theta_n1
                            ) + float(self._dhp_baseline_kd) * float(e_q_n1)
                            u_base1_t = torch.tensor(
                                [u_base1], dtype=torch.float32, device=self.device
                            )
                            u_base1_t = torch.clamp(u_base1_t, -1.0, 1.0).detach()
                    else:
                        u_base1_t = torch.zeros(
                            (1,), dtype=torch.float32, device=self.device
                        )

                    delta1 = self.actor(r_tp1_req.unsqueeze(0)).squeeze(0)
                    u_tp1_det = torch.clamp(
                        u_base1_t + float(self._dhp_residual_scale) * delta1,
                        low_t,
                        high_t,
                    )

                    du_dR_rows = []
                    for i in range(int(u_tp1_det.shape[0])):
                        grad_i = torch.autograd.grad(
                            outputs=u_tp1_det[i],
                            inputs=r_tp1_req,
                            grad_outputs=torch.ones_like(u_tp1_det[i]),
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=False,
                        )[0]
                        du_dR_rows.append(grad_i)
                    du_dR_tp1 = torch.stack(du_dR_rows, dim=0)  # (act_dim, r_dim)

                    with torch.no_grad():
                        j1, jr1, ja1 = self.critic(
                            r_tp1.unsqueeze(0), u_tp1_det.detach().unsqueeze(0)
                        )
                        j1_s = j1.squeeze(0).squeeze(-1)
                        jr1_s = jr1.squeeze(0)
                        ja1_s = ja1.squeeze(0)

                    lam_tilde_tp1 = jr1_s + (du_dR_tp1.T @ ja1_s)
                    assert self._dhp_A is not None and self._dhp_B is not None
                    jr_target = dU_dR + float(self.gamma) * (
                        self._dhp_A.T @ lam_tilde_tp1
                    )
                    ja_target = dU_du + float(self.gamma) * (
                        self._dhp_B.T @ lam_tilde_tp1
                    ).reshape_as(dU_du)

                    # Scalar J target (only for ADGDHP)
                    bootstrap = 0.0 if bool(terminated) else 1.0
                    cost_t = (
                        w_theta * (e_theta * e_theta)
                        + w_q * (e_q * e_q)
                        + w_u * torch.sum(u_det * u_det)
                        + w_du * torch.sum(du0 * du0)
                        + w_jerk * torch.sum(ddu0 * ddu0)
                    )
                    j_target = (
                        cost_t.detach() + float(bootstrap) * float(self.gamma) * j1_s
                    )

                    critic_loss_t = None
                    if do_critic:
                        j_pred, jr_pred, ja_pred = self.critic(
                            r_t.unsqueeze(0), u_det.unsqueeze(0)
                        )
                        jr_pred_s = jr_pred.squeeze(0)
                        ja_pred_s = ja_pred.squeeze(0)
                        critic_loss_t = F.mse_loss(jr_pred_s, jr_target) + F.mse_loss(
                            ja_pred_s, ja_target
                        )
                        if self.design == "adgdhp":
                            j_pred_s = j_pred.squeeze(0).squeeze(-1)
                            critic_loss_t = critic_loss_t + F.mse_loss(
                                j_pred_s, j_target
                            )
                        self.critic_optim.zero_grad()
                        critic_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.critic.parameters(), max_norm=1.0
                        )
                        self.critic_optim.step()

                    actor_loss_t = None
                    if do_actor:
                        r_t_det = r_t.detach()
                        delta_det = self.actor(r_t_det.unsqueeze(0)).squeeze(0)
                        u_pol_det = (
                            u_base_t + float(self._dhp_residual_scale) * delta_det
                        )
                        u_for_grad = torch.clamp(u_pol_det, low_t, high_t)
                        with torch.no_grad():
                            _j_c, _jr_c, ja_c = self.critic(
                                r_t_det.unsqueeze(0), u_for_grad.detach().unsqueeze(0)
                            )
                            ja_c_s = ja_c.squeeze(0)
                        actor_loss_t = torch.sum(u_for_grad * ja_c_s)
                        reg = float(getattr(self, "_dhp_actor_delta_l2", 0.0) or 0.0)
                        if reg > 0.0:
                            actor_loss_t = actor_loss_t + reg * torch.mean(
                                delta_det * delta_det
                            )
                        self.actor_optim.zero_grad()
                        actor_loss_t.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=1.0
                        )
                        self.actor_optim.step()

                    self._prev2_u_norm = self._prev_u_norm.copy()
                    self._prev_u_norm = (
                        u_det.detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        if critic_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/critic_adgdhp",
                                float(critic_loss_t.item()),
                                self._updates,
                            )
                        if actor_loss_t is not None:
                            self.writer.add_scalar(
                                "loss/actor_adgdhp",
                                float(actor_loss_t.item()),
                                self._updates,
                            )
                        self.writer.add_scalar(
                            "train/dhp_phase",
                            (
                                0.0
                                if dhp_phase == "critic"
                                else (1.0 if dhp_phase == "actor" else 2.0)
                            ),
                            self._updates,
                        )

                else:
                    # Canonical ADHDP-style update on observation space:
                    # online TD on J(R,A) with actor improvement via critic gradients.
                    adhdp_warm_c = int(
                        getattr(self, "_adhdp_critic_warmup_episodes", 0) or 0
                    )
                    # Alternating cycles (if configured)
                    c_ep = int(getattr(self, "_adhdp_critic_cycle_episodes", 0) or 0)
                    a_ep = int(getattr(self, "_adhdp_action_cycle_episodes", 0) or 0)
                    phase = "both"
                    if c_ep > 0 and a_ep > 0:
                        cyc = c_ep + a_ep
                        pos = int(ep % cyc)
                        phase = "critic" if pos < c_ep else "actor"

                    do_critic = phase in ("both", "critic")
                    do_actor = phase in ("both", "actor")
                    # Warmup overrides: first N episodes train critic only
                    if ep < adhdp_warm_c:
                        do_critic = True
                        do_actor = False

                    # Avoid exploration noise while actor is frozen to keep trajectories stable.
                    act = self.select_action(obs, evaluate=not do_actor)
                    next_obs, reward, terminated, truncated, info = self.env.step(act)
                    next_obs = _as_flat_np(next_obs)

                done_env = bool(terminated or truncated)
                # Bootstrap stops only on true termination (not time limit)
                done_bootstrap = float(bool(terminated))

                # Logging uses the environment reward.
                ep_reward += float(reward)
                steps += 1
                total_steps += 1

                if (
                    self.design == "ddpg"
                    and self.use_replay
                    and self.memory is not None
                ):
                    self.memory.push(obs, act, float(reward), next_obs, done_bootstrap)
                    if len(self.memory) >= self.batch_size:
                        for _ in range(self.updates_per_step):
                            b = self.memory.sample(self.batch_size)
                            critic_loss, actor_loss = self._td_update_batch(*b)
                            self._updates += 1
                            if (self._updates % self.log_every_updates) == 0:
                                self.writer.add_scalar(
                                    "loss/critic", critic_loss, self._updates
                                )
                                self.writer.add_scalar(
                                    "loss/actor", actor_loss, self._updates
                                )
                elif self.design in ("adhdp", "ddpg"):
                    # For canonical ADHDP it is more stable to train on the utility U (cost),
                    # not on shaped reward. If env provides cost_total, use reward=-cost_total.
                    reward_for_update = float(reward)
                    if self.design == "adhdp" and bool(
                        getattr(self, "_adhdp_use_env_cost", True)
                    ):
                        try:
                            cost_total = float(info.get("cost_total"))  # type: ignore[union-attr]
                            reward_scale = float(getattr(self.env, "reward_scale", 1.0))
                            reward_for_update = -cost_total * reward_scale
                        except Exception:
                            reward_for_update = float(reward)
                    # Online update on the single transition
                    critic_loss, actor_loss = self._td_update_batch(
                        obs_b=obs.reshape(1, -1),
                        act_b=act.reshape(1, -1),
                        rew_b=np.asarray([[reward_for_update]], dtype=np.float32),
                        next_obs_b=next_obs.reshape(1, -1),
                        done_bootstrap_b=np.asarray(
                            [[done_bootstrap]], dtype=np.float32
                        ),
                        do_critic_update=(
                            bool(do_critic) if self.design == "adhdp" else True
                        ),
                        do_actor_update=(
                            bool(do_actor) if self.design == "adhdp" else True
                        ),
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        self.writer.add_scalar(
                            "loss/critic", critic_loss, self._updates
                        )
                        self.writer.add_scalar("loss/actor", actor_loss, self._updates)

                obs = next_obs
                done = done_env
                if max_steps_i is not None and steps >= max_steps_i:
                    break

            self.writer.add_scalar("performance/episode_reward", float(ep_reward), ep)
            self.writer.add_scalar("performance/episode_length", int(steps), ep)
            self.writer.add_scalar("train/total_steps", int(total_steps), ep)
            if bool(getattr(self, "_is_model_based_design", False)):
                self.writer.add_scalar(
                    "train/dhp_phase_episode",
                    (
                        0.0
                        if dhp_phase == "critic"
                        else (1.0 if dhp_phase == "actor" else 2.0)
                    ),
                    ep,
                )

        self.writer.flush()

    # ---- persistence (HF-style similar to SAC/DDPG) ----
    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        try:
            if "tensoraerospace" in env_name:
                env_params = serialize_env(self.env)
        except Exception:
            env_params = {}

        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        policy_params = {
            "design": str(self.design),
            "gamma": self.gamma,
            "exploration_std": self.exploration_std,
            "use_replay": self.use_replay,
            "memory_capacity": int(getattr(self.memory, "capacity", 0) or 0),
            "batch_size": self.batch_size,
            "updates_per_step": self.updates_per_step,
            "use_target_networks": self.use_target_networks,
            "tau": self.tau,
            "device": self.device.type,
            "seed": self.seed,
            "actor_lr": float(self.actor_optim.defaults.get("lr", 3e-4)),
            "critic_lr": float(self.critic_optim.defaults.get("lr", 3e-4)),
            "hidden_size": int(self.hidden_size),
            "adhdp_use_baseline": bool(getattr(self, "_adhdp_use_baseline", False)),
            "adhdp_warmstart_actor_episodes": int(
                getattr(self, "_adhdp_warmstart_actor_episodes", 0) or 0
            ),
            "adhdp_warmstart_actor_epochs": int(
                getattr(self, "_adhdp_warmstart_actor_epochs", 0) or 0
            ),
            "adhdp_critic_warmup_episodes": int(
                getattr(self, "_adhdp_critic_warmup_episodes", 0) or 0
            ),
            "adhdp_critic_cycle_episodes": int(
                getattr(self, "_adhdp_critic_cycle_episodes", 0) or 0
            ),
            "adhdp_action_cycle_episodes": int(
                getattr(self, "_adhdp_action_cycle_episodes", 0) or 0
            ),
            "adhdp_use_env_cost": bool(getattr(self, "_adhdp_use_env_cost", True)),
            "adhdp_actor_bc_l2": float(getattr(self, "_adhdp_actor_bc_l2", 0.0) or 0.0),
            "adhdp_actor_bc_decay": float(
                getattr(self, "_adhdp_actor_bc_decay", 1.0) or 1.0
            ),
        }
        if bool(getattr(self, "_is_model_based_design", False)):
            policy_params.update(
                {
                    "dhp_w_theta": float(self._dhp_w_theta),
                    "dhp_w_q": float(self._dhp_w_q),
                    "dhp_w_u": float(self._dhp_w_u),
                    "dhp_w_du": float(self._dhp_w_du),
                    "dhp_use_env_cost": bool(self._dhp_use_env_cost),
                    "dhp_use_baseline": bool(self._dhp_use_baseline),
                    "dhp_baseline_type": str(getattr(self, "_dhp_baseline_type", "pd")),
                    "dhp_baseline_kp": float(self._dhp_baseline_kp),
                    "dhp_baseline_ki": float(getattr(self, "_dhp_baseline_ki", 0.0)),
                    "dhp_baseline_kd": float(self._dhp_baseline_kd),
                    "dhp_pid_use_normalized_theta": bool(
                        getattr(self, "_dhp_pid_use_normalized_theta", True)
                    ),
                    "dhp_pid_mode": str(getattr(self, "_dhp_pid_mode", "norm")),
                    "dhp_actor_delta_l2": float(
                        getattr(self, "_dhp_actor_delta_l2", 0.0) or 0.0
                    ),
                    "dhp_residual_scale": float(self._dhp_residual_scale),
                    "dhp_warmstart_actor_episodes": int(
                        getattr(self, "_dhp_warmstart_actor_episodes", 0) or 0
                    ),
                    "dhp_warmstart_actor_epochs": int(
                        getattr(self, "_dhp_warmstart_actor_epochs", 0) or 0
                    ),
                    "dhp_warmstart_actor_disable_baseline_after": bool(
                        getattr(
                            self, "_dhp_warmstart_actor_disable_baseline_after", True
                        )
                    ),
                    "dhp_critic_cycle_episodes": int(
                        getattr(self, "_dhp_critic_cycle_episodes", 0) or 0
                    ),
                    "dhp_action_cycle_episodes": int(
                        getattr(self, "_dhp_action_cycle_episodes", 0) or 0
                    ),
                }
            )

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(
        self, path: Union[str, Path, None] = None, *, save_gradients: bool = False
    ) -> str:
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = path / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        actor_path = run_dir / "actor.pth"
        critic_path = run_dir / "critic.pth"
        actor_target_path = run_dir / "actor_target.pth"
        critic_target_path = run_dir / "critic_target.pth"
        actor_optim_path = run_dir / "actor_optim.pth"
        critic_optim_path = run_dir / "critic_optim.pth"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)
        if (
            self.use_target_networks
            and self.actor_target is not None
            and self.critic_target is not None
        ):
            torch.save(self.actor_target, actor_target_path)
            torch.save(self.critic_target, critic_target_path)

        if save_gradients:
            torch.save(self.actor_optim.state_dict(), actor_optim_path)
            torch.save(self.critic_optim.state_dict(), critic_optim_path)

        return str(run_dir)

    @staticmethod
    def _filter_kwargs_for_init(
        env_cls: type, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(env_cls.__init__)
        except (TypeError, ValueError):
            return kwargs

        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return kwargs

        allowed: set[str] = set()
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                allowed.add(name)
        return {k: v for k, v in kwargs.items() if k in allowed}

    @classmethod
    def __load(
        cls,
        path: Union[str, Path],
        *,
        load_gradients: bool = False,
    ) -> "ADP":
        path = Path(path)
        config_path = path / "config.json"
        actor_path = path / "actor.pth"
        critic_path = path / "critic.pth"
        actor_target_path = path / "actor_target.pth"
        critic_target_path = path / "critic_target.pth"
        actor_optim_path = path / "actor_optim.pth"
        critic_optim_path = path / "critic_optim.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch

        # Recreate env
        env_cfg = config.get("env", {})
        env_cls_path = env_cfg.get("name")
        env_params = dict(env_cfg.get("params", {}) or {})

        if env_cls_path and "tensoraerospace" in str(env_cls_path):
            env_cls = get_class_from_string(env_cls_path)
            env_params = cls._filter_kwargs_for_init(env_cls, env_params)
            env = env_cls(**env_params)
        else:
            env = get_class_from_string(env_cls_path)() if env_cls_path else None

        p = dict(config.get("policy", {}).get("params", {}) or {})
        # Device fallback
        dev = str(p.get("device", "cpu"))
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        if dev == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            dev = "cpu"
        p["device"] = dev

        new_agent = cls(env=env, **p)

        new_agent.actor = torch.load(
            actor_path, map_location=new_agent.device, weights_only=False
        ).to(new_agent.device)
        new_agent.critic = torch.load(
            critic_path, map_location=new_agent.device, weights_only=False
        ).to(new_agent.device)

        if (
            new_agent.use_target_networks
            and actor_target_path.exists()
            and critic_target_path.exists()
        ):
            new_agent.actor_target = torch.load(
                actor_target_path, map_location=new_agent.device, weights_only=False
            ).to(new_agent.device)
            new_agent.critic_target = torch.load(
                critic_target_path, map_location=new_agent.device, weights_only=False
            ).to(new_agent.device)

        # Reinitialize optimizers
        actor_lr = float(p.get("actor_lr", 3e-4))
        critic_lr = float(p.get("critic_lr", 3e-4))
        new_agent.actor_optim = Adam(new_agent.actor.parameters(), lr=actor_lr)
        new_agent.critic_optim = Adam(new_agent.critic.parameters(), lr=critic_lr)

        if load_gradients:
            if actor_optim_path.exists():
                new_agent.actor_optim.load_state_dict(
                    torch.load(
                        actor_optim_path,
                        map_location=new_agent.device,
                        weights_only=False,
                    )
                )
            if critic_optim_path.exists():
                new_agent.critic_optim.load_state_dict(
                    torch.load(
                        critic_optim_path,
                        map_location=new_agent.device,
                        weights_only=False,
                    )
                )
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        *,
        load_gradients: bool = False,
    ) -> "ADP":
        # 1) local folder
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        # 2) explicit path-like but missing
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Local directory not found: '{repo_name}'.")
            return cls.__load(p, load_gradients=load_gradients)

        # 3) Hugging Face repo id
        folder_path = super().from_pretrained(repo_name, access_token, version)
        return cls.__load(folder_path, load_gradients=load_gradients)
