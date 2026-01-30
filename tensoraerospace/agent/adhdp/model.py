# flake8: noqa
"""Canonical ADHDP (Action-Dependent Heuristic Dynamic Programming) agent.

This is a standalone, paper-inspired implementation of ADHDP:
  - Critic learns action-dependent cost-to-go J(R, a) via online TD
  - Actor improves by minimizing J(R, pi(R))

The implementation is intentionally lightweight and compatible with Gymnasium-like
environments used in TensorAeroSpace (e.g., ImprovedB747Env).

Notes on stabilization options:
  - Warm-start: initialize the actor by imitating a stabilizing baseline (PD/PID)
  - Critic warmup / alternating cycles: train critic first on a fixed actor policy,
    then train the actor with a fixed critic (paper Section III idea)
  - Optional BC regularizer: keep the actor close to the baseline early, then decay
    the regularizer to zero.

This class is separate from `tensoraerospace.agent.ADP` to keep the canonical ADHDP
logic isolated from other ACD/ADP variants.
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

from ..adp.networks import DeterministicActor, QCritic
from ..base import (
    BaseRLModel,
    get_class_from_string,
    serialize_env,
)
from ..metrics import create_metric_writer


def _as_flat_np(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


class ADHDP(BaseRLModel):
    """Action-Dependent Heuristic Dynamic Programming (ADHDP) agent.

    ADHDP is a model-free reinforcement learning algorithm from the Adaptive Critic
    Designs (ACD) family. Unlike HDP which requires a plant model, ADHDP learns an
    action-dependent cost-to-go function J(R, a) that directly takes both state and
    action as inputs. The actor is improved by minimizing this critic output via
    backpropagation through the critic network.

    The algorithm follows the framework from Prokhorov & Wunsch (1997):
      - Critic learns: J(R_t, a_t) ≈ U_t + γ J(R_{t+1}, π(R_{t+1}))
      - Actor minimizes: J(R_t, π(R_t)) via gradient through critic

    This is the canonical, paper-style actor-critic without modern stabilization
    tricks (replay buffer, target networks). It uses online TD(0) learning.

    Example:
        >>> import numpy as np
        >>> from tensoraerospace.agent import ADHDP
        >>> from tensoraerospace.envs.b747 import ImprovedB747Env
        >>>
        >>> def sine_ref(steps, amp_deg=2.0, freq_hz=0.05, dt=0.1):
        ...     t = np.arange(steps) * dt
        ...     return (np.deg2rad(amp_deg) * np.sin(2*np.pi*freq_hz*t)).reshape(1,-1)
        >>>
        >>> env = ImprovedB747Env(
        ...     initial_state=np.zeros(4),
        ...     reference_signal=sine_ref(300),
        ...     number_time_steps=300,
        ...     dt=0.1,
        ...     include_reference_in_obs=True,
        ... )
        >>> agent = ADHDP(env, paper_strict=True, gamma=0.99)
        >>> agent.train(num_episodes=100)

    References:
        - Prokhorov D.V., Wunsch D.C. "Adaptive Critic Designs."
          IEEE Trans. Neural Networks, vol. 8, no. 5, pp. 997-1007, 1997.
        - Werbos P.J. "A menu of designs for reinforcement learning over time."
          Neural Networks for Control, MIT Press, 1990.

    Attributes:
        actor: Neural network that outputs control action π(R).
        critic: Neural network that estimates action-dependent cost-to-go J(R, a).
        env: The Gymnasium-compatible environment.
        gamma: Discount factor for future costs.
        paper_strict: If True, uses canonical ADHDP without baseline mixing.
    """

    def __init__(
        self,
        env: Any,
        *,
        paper_strict: bool = False,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42,
        policy_mode: str = "direct",
        residual_scale: float = 0.2,
        use_baseline_in_critic_phases: bool = False,
        action_momentum: float = 0.0,
        action_max_abs: float = 1.0,
        action_selection: str = "actor",
        action_grad_steps: int = 0,
        action_grad_lr: float = 0.0,
        action_grad_step_clip: float = 0.0,
        action_grad_u_l2: float = 0.0,
        action_grad_du_l2: float = 0.0,
        actor_update_mode: str = "minimize_critic",
        actor_distill_coef: float = 1.0,
        actor_distill_steps: int = 10,
        actor_distill_lr: float = 0.03,
        # If True and actor_update_mode="distill_critic_gradient", execute the teacher policy
        # (critic-gradient argmin) during training rollouts after baseline warmup.
        # This is baseline-free and prevents the (still-learning) actor from driving the
        # system into out-of-distribution states before it has distilled the teacher.
        distill_execute_teacher: bool = False,
        # Optional noise for teacher rollouts (default 0.0 = deterministic teacher).
        teacher_rollout_noise_std: float = 0.0,
        exploration_std: float = 0.02,
        critic_updates_per_step: int = 1,
        actor_updates_per_step: int = 1,
        initial_state_noise_std: float = 0.0,
        reference_roll_steps: int = 0,
        reference_noise_std: float = 0.0,
        baseline_warmup_episodes: int = 0,
        critic_warmup_episodes: int = 0,
        critic_cycle_episodes: int = 0,
        action_cycle_episodes: int = 0,
        use_env_cost: bool = True,
        warmstart_actor_episodes: int = 0,
        warmstart_actor_epochs: int = 2,
        baseline_type: str = "pid",
        baseline_kp: float = -24.6295,
        baseline_ki: float = -0.2486,
        baseline_kd: float = -7.8179,
        pid_i_clip: float = 1.0,
        actor_bc_l2: float = 0.0,
        actor_bc_decay: float = 1.0,
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 500,
    ) -> None:
        """Initialize the ADHDP agent.

        Args:
            env: Gymnasium-compatible environment with Box observation and action
                spaces. For best results, use ImprovedB747Env with
                `include_reference_in_obs=True`.

            paper_strict: If True, enforce canonical ADHDP configuration:
                - No residual baseline in executed policy
                - No baseline substitution during critic-only phases
                - No BC regularizer
                This matches the original Prokhorov & Wunsch formulation. Default: False.

            gamma: Discount factor for future costs. Range: [0, 1]. Default: 0.99.

            actor_lr: Learning rate for the actor network optimizer (Adam).
                Default: 1e-4.

            critic_lr: Learning rate for the critic network optimizer (Adam).
                Default: 1e-4.

            hidden_size: Number of neurons in each hidden layer of both actor and
                critic networks. Both use two hidden layers with Tanh activation.
                Default: 256.

            device: Torch device for computation ('cpu', 'cuda', 'mps', or
                torch.device instance). Default: 'cpu'.

            seed: Random seed for reproducibility. Default: 42.

            policy_mode: Policy composition mode:
                - "direct": Pure actor output (default for paper_strict)
                - "residual": u = u_baseline + residual_scale * pi(R)
                Default: "direct".

            residual_scale: Scale of learned residual when policy_mode="residual".
                Default: 0.2.

            use_baseline_in_critic_phases: If True, execute baseline controller
                during critic-only training phases. Default: False.

            action_momentum: Momentum for action smoothing: u = m*u_prev + (1-m)*u_new.
                Range: [0, 1). Default: 0.0 (disabled).

            action_max_abs: Maximum action magnitude (safety envelope) in normalized
                units. Range: (0, 1]. Default: 1.0.

            action_selection: How to select actions:
                - "actor": Use actor network pi(R) (default)
                - "critic_gradient": Optimize action by minimizing J(R,a) w.r.t. a
                Default: "actor".

            action_grad_steps: Number of gradient steps for critic-based action
                optimization (only used if action_selection="critic_gradient").
                Default: 0.

            action_grad_lr: Learning rate for action optimization. Default: 0.0.

            action_grad_step_clip: Maximum step size for action gradient updates.
                Default: 0.0 (no clipping).

            action_grad_u_l2: L2 regularization on action magnitude during action
                optimization. Default: 0.0.

            action_grad_du_l2: L2 regularization on action change (trust region)
                during action optimization. Default: 0.0.

            actor_update_mode: Actor training mode:
                - "minimize_critic": Minimize J(R, pi(R)) via backprop through critic
                - "distill_critic_gradient": Supervised distillation of critic-gradient policy
                Default: "minimize_critic".

            actor_distill_coef: Loss coefficient for distillation mode. Default: 1.0.

            actor_distill_steps: Gradient steps per distillation target. Default: 10.

            actor_distill_lr: Learning rate for distillation optimization. Default: 0.03.

            exploration_std: Standard deviation of Gaussian noise added to actions
                during training. Default: 0.02.

            critic_updates_per_step: Number of critic gradient updates per environment
                step (MATLAB-style "epochs per step"). Default: 1.

            actor_updates_per_step: Number of actor gradient updates per environment
                step. Default: 1.

            initial_state_noise_std: Standard deviation of noise added to initial
                state for trajectory randomization (persistent excitation).
                Default: 0.0.

            reference_roll_steps: Maximum random roll (shift) of reference signal
                at episode start. Default: 0.

            reference_noise_std: Standard deviation of noise added to reference
                signal. Default: 0.0.

            baseline_warmup_episodes: Episodes running only baseline controller
                for critic warmup (paper Section III). Default: 0.

            critic_warmup_episodes: Episodes with frozen actor (critic-only
                training). Default: 0.

            critic_cycle_episodes: Episodes per critic-only cycle in alternating
                training schedule. Default: 0 (no alternating).

            action_cycle_episodes: Episodes per actor-only cycle in alternating
                training schedule. Default: 0.

            use_env_cost: If True, use environment's cost_total instead of shaped
                reward for TD target. Default: True.

            warmstart_actor_episodes: Episodes to pre-train actor by imitating
                baseline via supervised learning. Default: 0.

            warmstart_actor_epochs: Supervised epochs per warmstart episode.
                Default: 2.

            baseline_type: Baseline controller type: "pd" or "pid". Default: "pid".

            baseline_kp: Proportional gain for baseline (tuned for B747).
                Default: -24.6295.

            baseline_ki: Integral gain for PID baseline. Default: -0.2486.

            baseline_kd: Derivative gain for baseline. Default: -7.8179.

            pid_i_clip: Anti-windup integral clipping for PID. Default: 1.0.

            actor_bc_l2: L2 regularization coefficient keeping actor close to
                baseline (behavioral cloning). Default: 0.0.

            actor_bc_decay: Decay rate for actor_bc_l2 per episode. Default: 1.0.

            log_dir: Directory path for TensorBoard logs. If None, logging is
                disabled. Default: None.

            log_every_updates: Frequency of logging (every N gradient updates).
                Default: 500.

        Raises:
            ValueError: If observation or action space is not Box-like.
        """
        super().__init__()
        self.env = env
        self.paper_strict = bool(paper_strict)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.policy_mode = str(policy_mode).lower().strip()
        if self.policy_mode not in ("direct", "residual"):
            self.policy_mode = "direct"
        self.residual_scale = float(residual_scale)
        self.use_baseline_in_critic_phases = bool(use_baseline_in_critic_phases)
        self.action_momentum = float(action_momentum)
        if not (0.0 <= float(self.action_momentum) < 1.0):
            self.action_momentum = 0.0
        self.action_max_abs = float(action_max_abs)
        if not (0.0 < float(self.action_max_abs) <= 1.0):
            self.action_max_abs = 1.0

        self.action_selection = str(action_selection).lower().strip()
        if self.action_selection not in ("actor", "critic_gradient"):
            self.action_selection = "actor"
        self.action_grad_steps = int(max(0, int(action_grad_steps)))
        self.action_grad_lr = float(action_grad_lr)
        if self.action_grad_lr < 0.0:
            self.action_grad_lr = 0.0
        self.action_grad_step_clip = float(action_grad_step_clip)
        if self.action_grad_step_clip < 0.0:
            self.action_grad_step_clip = 0.0
        self.action_grad_u_l2 = float(action_grad_u_l2)
        if self.action_grad_u_l2 < 0.0:
            self.action_grad_u_l2 = 0.0
        self.action_grad_du_l2 = float(action_grad_du_l2)
        if self.action_grad_du_l2 < 0.0:
            self.action_grad_du_l2 = 0.0

        self.actor_update_mode = str(actor_update_mode).lower().strip()
        if self.actor_update_mode not in ("minimize_critic", "distill_critic_gradient"):
            self.actor_update_mode = "minimize_critic"
        self.actor_distill_coef = float(actor_distill_coef)
        if self.actor_distill_coef < 0.0:
            self.actor_distill_coef = 0.0
        self.actor_distill_steps = int(max(0, int(actor_distill_steps)))
        self.actor_distill_lr = float(actor_distill_lr)
        if self.actor_distill_lr < 0.0:
            self.actor_distill_lr = 0.0
        self.distill_execute_teacher = bool(distill_execute_teacher)
        self.teacher_rollout_noise_std = float(teacher_rollout_noise_std)
        if self.teacher_rollout_noise_std < 0.0:
            self.teacher_rollout_noise_std = 0.0
        self.exploration_std = float(exploration_std)
        self.hidden_size = int(hidden_size)
        self.critic_updates_per_step = int(max(1, critic_updates_per_step))
        self.actor_updates_per_step = int(max(1, actor_updates_per_step))

        # Trajectory variety (recommended by paper; replaces action-noise heavy exploration)
        self.initial_state_noise_std = float(initial_state_noise_std)
        self.reference_roll_steps = int(reference_roll_steps)
        self.reference_noise_std = float(reference_noise_std)

        self.baseline_warmup_episodes = int(max(0, int(baseline_warmup_episodes)))
        self.critic_warmup_episodes = int(critic_warmup_episodes)
        self.critic_cycle_episodes = int(critic_cycle_episodes)
        self.action_cycle_episodes = int(action_cycle_episodes)

        self.use_env_cost = bool(use_env_cost)

        self.warmstart_actor_episodes = int(warmstart_actor_episodes)
        self.warmstart_actor_epochs = int(warmstart_actor_epochs)

        self.baseline_type = str(baseline_type).lower().strip()
        self.baseline_kp = float(baseline_kp)
        self.baseline_ki = float(baseline_ki)
        self.baseline_kd = float(baseline_kd)
        self.pid_i_clip = float(pid_i_clip)
        # PID baseline internal state (used only when baseline_type="pid" and called sequentially).
        self._pid_i = 0.0
        self._pid_prev_meas = 0.0

        self.actor_bc_l2 = float(actor_bc_l2)
        self.actor_bc_decay = float(actor_bc_decay)
        self._actor_bc_coef = float(actor_bc_l2)

        # Enforce paper-strict mode after reading user settings.
        # Baseline may still be used for warm-start dataset generation (actor imitation),
        # but it must not be mixed into the executed policy afterward.
        if bool(self.paper_strict):
            self.policy_mode = "direct"
            self.residual_scale = 0.0
            self.use_baseline_in_critic_phases = False
            # Disable BC regularizer (not in the canonical ADHDP description).
            self.actor_bc_l2 = 0.0
            self._actor_bc_coef = 0.0
            self.actor_bc_decay = 1.0
            # Paper-strict + no-baseline often works best with critic-gradient action refinement.
            # If user didn't explicitly configure it, pick conservative defaults.
            if self.action_selection == "actor":
                # Keep actor selection by default, but ensure the user can switch easily in notebooks.
                pass

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        obs_dim = int(getattr(self.env.observation_space, "shape", (0,))[0])
        act_dim = int(getattr(self.env.action_space, "shape", (0,))[0])
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(
                f"ADHDP expects Box-like spaces. Got obs_dim={obs_dim}, act_dim={act_dim}"
            )

        action_low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(
            -1
        )
        hidden_sizes = (int(hidden_size), int(hidden_size))
        self._action_low_t = torch.as_tensor(
            action_low, dtype=torch.float32, device=self.device
        )
        self._action_high_t = torch.as_tensor(
            action_high, dtype=torch.float32, device=self.device
        )

        # Save baseline initial state / reference for per-episode randomization (if available).
        self._initial_state_base: Optional[np.ndarray] = None
        try:
            init_s = getattr(self.env, "initial_state", None)
            if init_s is not None:
                self._initial_state_base = (
                    np.asarray(init_s, dtype=np.float32).reshape(-1).copy()
                )
        except (TypeError, ValueError, AttributeError):
            self._initial_state_base = None

        self._reference_signal_base: Optional[np.ndarray] = None
        try:
            ref = getattr(self.env, "reference_signal", None)
            if ref is not None:
                self._reference_signal_base = np.asarray(ref, dtype=np.float32).copy()
        except (TypeError, ValueError, AttributeError):
            self._reference_signal_base = None

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

        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = create_metric_writer(self.log_dir)
        self.log_every_updates = int(log_every_updates)
        self._updates = 0

    # ---- baseline helpers ----
    def reset(self) -> None:
        # reset PID integrator
        self._pid_i = 0.0
        self._pid_prev_meas = 0.0

    def _baseline_action(self, obs_t: torch.Tensor) -> torch.Tensor:
        """PD/PID baseline from normalized observation.

        Expected obs format (ImprovedB747Env):
          [norm_pitch_error, norm_q, norm_theta, norm_prev_action] (+ optional refs)
        """
        if obs_t.ndim != 2:
            raise ValueError(
                f"Expected obs shape (B, obs_dim), got {tuple(obs_t.shape)}"
            )
        if int(obs_t.shape[1]) < 2:
            raise ValueError("Baseline requires observation with at least 2 dims.")

        # NOTE:
        # Baseline must depend ONLY on observation (not on self.env state), otherwise
        # evaluation on a different env instance (common in notebooks) will silently break.
        #
        # For ImprovedB747Env, obs is normalized but we can reconstruct radians using env.max_pitch_rad.
        # That allows us to match `tensoraerospace.agent.pid.PID` baseline behavior (derivative on measurement).
        e_theta_n = obs_t[:, 0]  # (theta_ref - theta) / max_pitch_rad
        q_n = obs_t[:, 1]
        q_ref_n = obs_t[:, 5] if int(obs_t.shape[1]) >= 6 else torch.zeros_like(q_n)
        e_q_n = q_ref_n - q_n  # normalized pitch-rate error (used by PD fallback)

        kp = float(self.baseline_kp)
        ki = float(self.baseline_ki)
        kd = float(self.baseline_kd)

        baseline_type = str(self.baseline_type).lower().strip()
        if baseline_type not in ("pd", "pid"):
            baseline_type = "pd"

        if baseline_type == "pid":
            dt = float(getattr(self.env, "dt", 0.1))
            # Reconstruct radians if possible (ImprovedB747Env); otherwise fall back to normalized units.
            max_pitch_rad = getattr(self.env, "max_pitch_rad", None)
            if max_pitch_rad is not None:
                max_pitch_rad_f = float(max_pitch_rad)
                theta_n = (
                    obs_t[:, 2]
                    if int(obs_t.shape[1]) >= 3
                    else torch.zeros_like(e_theta_n)
                )
                if int(obs_t.shape[1]) >= 5:
                    theta_ref_n = obs_t[:, 4]
                else:
                    theta_ref_n = theta_n + e_theta_n
                meas = theta_n * max_pitch_rad_f
                setp = theta_ref_n * max_pitch_rad_f
                err = setp - meas
            else:
                # Fallback: operate directly on normalized error/measurement.
                meas = torch.zeros_like(e_theta_n)
                setp = e_theta_n
                err = e_theta_n

            # Keep PID state only for sequential use (batch==1).
            i_curr = float(self._pid_i)
            prev_meas = float(self._pid_prev_meas)
            if int(obs_t.shape[0]) == 1:
                meas0 = float(meas[0].item())
                err0 = float(err[0].item())
                # Derivative on measurement (Simulink-style: avoids setpoint derivative kick)
                derivative = 0.0
                if dt > 0.0:
                    derivative = -(meas0 - prev_meas) / dt
                i_candidate = i_curr + err0 * dt
                if self.pid_i_clip > 0.0:
                    i_candidate = float(
                        np.clip(i_candidate, -self.pid_i_clip, self.pid_i_clip)
                    )
                u_unsat = (
                    float(kp) * err0 + float(ki) * i_candidate + float(kd) * derivative
                )
                u_sat = float(np.clip(u_unsat, -1.0, 1.0))
                if u_sat != u_unsat:
                    # anti-windup: do not integrate further if saturated
                    i_candidate = i_curr
                    u_unsat = (
                        float(kp) * err0
                        + float(ki) * i_candidate
                        + float(kd) * derivative
                    )
                    u_sat = float(np.clip(u_unsat, -1.0, 1.0))
                self._pid_i = float(i_candidate)
                self._pid_prev_meas = float(meas0)
                u = torch.as_tensor([[u_sat]], dtype=torch.float32, device=obs_t.device)
            else:
                # Batch call (e.g. warm-start dataset collection): stateless, no derivative, no integration update.
                u = (float(kp) * err + float(ki) * float(i_curr)).reshape(-1, 1)
                u = torch.clamp(u, -1.0, 1.0)
                act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
                return u.repeat(1, act_dim)
        else:
            u = kp * e_theta_n + kd * e_q_n

        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
        u = u.reshape(-1, 1).repeat(1, act_dim)
        # Baseline is defined in normalized control coordinates for B747-like envs.
        return torch.clamp(u, -1.0, 1.0)

    def _policy_action_t(
        self,
        obs_t: torch.Tensor,
        *,
        evaluate: bool,
        force_baseline: bool = False,
    ) -> torch.Tensor:
        """Return policy action tensor for a batch of observations."""
        if bool(force_baseline):
            return self._baseline_action(obs_t)

        # --- choose raw action ---
        if (
            self.action_selection == "critic_gradient"
            and self.action_grad_steps > 0
            and self.action_grad_lr > 0.0
        ):
            # Initialize from previous action if present; else start from zero.
            if int(obs_t.shape[1]) >= 4:
                a_init = obs_t[:, 3].reshape(-1, 1)
                if int(self._action_low_t.numel()) > 1:
                    a_init = a_init.repeat(1, int(self._action_low_t.numel()))
            else:
                a_init = torch.zeros(
                    (int(obs_t.shape[0]), int(self._action_low_t.numel())),
                    device=obs_t.device,
                )
            a = self._optimize_action_by_critic(
                obs_t, a_init=a_init, n_steps=int(self.action_grad_steps)
            )
            # Training-time exploration for critic-gradient mode (persistent excitation).
            if (not bool(evaluate)) and float(self.exploration_std) > 0.0:
                a = a + torch.randn_like(a) * float(self.exploration_std)
        else:
            # Actor-based action
            if self.policy_mode == "residual":
                base = self._baseline_action(obs_t)
                res = self.actor(obs_t)
                a = base + float(self.residual_scale) * res
            else:
                # direct
                a = self.actor(obs_t)
                if (not bool(evaluate)) and float(self.exploration_std) > 0.0:
                    a = a + torch.randn_like(a) * float(self.exploration_std)

        # Optional action momentum using previous action from observation (Markov).
        # ImprovedB747Env includes norm_prev_action at obs[3].
        m = float(self.action_momentum)
        if m > 0.0 and int(obs_t.shape[1]) >= 4:
            u_prev = obs_t[:, 3].reshape(-1, 1)
            if int(a.shape[1]) > 1:
                u_prev = u_prev.repeat(1, int(a.shape[1]))
            a = m * u_prev + (1.0 - m) * a

        # Baseline-free safety envelope (normalized control units)
        u_max = float(self.action_max_abs)
        if u_max < 1.0:
            a = torch.clamp(a, -u_max, u_max)

        return torch.max(torch.min(a, self._action_high_t), self._action_low_t)

    def _optimize_action_by_critic(
        self, obs_t: torch.Tensor, *, a_init: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """Minimize critic J(obs, a) w.r.t. a via gradient steps (HDPy-style).

        This is baseline-free and does not require actor training. It can be more stable
        than actor updates when critic is only locally accurate.
        """
        # NOTE: callers (train/select_action) often wrap action selection in `torch.no_grad()`.
        # For critic-gradient action selection we must temporarily re-enable autograd.
        a = a_init.detach().clone()
        # respect bounds + safety envelope
        u_max = float(self.action_max_abs)
        if u_max < 1.0:
            a = torch.clamp(a, -u_max, u_max)
        a = torch.max(torch.min(a, self._action_high_t), self._action_low_t)

        lr = float(self.action_grad_lr)
        if lr <= 0.0 or int(n_steps) <= 0:
            return a

        # Optional regularizers (trust-region-like): keep action bounded and near previous action.
        u_prev = None
        if int(obs_t.shape[1]) >= 4:
            u_prev = obs_t[:, 3].reshape(-1, 1)
            if int(a.shape[1]) > 1:
                u_prev = u_prev.repeat(1, int(a.shape[1]))

        # Keep critic weights fixed (we only need dJ/da).
        prev_req = [p.requires_grad for p in self.critic.parameters()]
        try:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            with torch.enable_grad():
                for _ in range(int(n_steps)):
                    a = a.detach().requires_grad_(True)
                    # Main objective: minimize critic J(s,a)
                    j = self.critic(obs_t, a).mean()
                    # Regularize large actions (helps prevent boundary exploitation when critic is inaccurate)
                    if float(self.action_grad_u_l2) > 0.0:
                        j = j + float(self.action_grad_u_l2) * (a.pow(2).mean())
                    # Regularize large action changes (trust region around previous action)
                    if u_prev is not None and float(self.action_grad_du_l2) > 0.0:
                        j = j + float(self.action_grad_du_l2) * (
                            (a - u_prev).pow(2).mean()
                        )
                    (grad_a,) = torch.autograd.grad(
                        j, a, create_graph=False, retain_graph=False
                    )
                    # Gradient DESCENT on J
                    step = -lr * grad_a
                    if float(self.action_grad_step_clip) > 0.0:
                        step = torch.clamp(
                            step,
                            -float(self.action_grad_step_clip),
                            float(self.action_grad_step_clip),
                        )
                    a = (a + step).detach()
                    if u_max < 1.0:
                        a = torch.clamp(a, -u_max, u_max)
                    a = torch.max(torch.min(a, self._action_high_t), self._action_low_t)
        finally:
            for p, r in zip(self.critic.parameters(), prev_req):
                p.requires_grad_(r)
        return a

    def _teacher_action_t(self, obs_t: torch.Tensor, *, evaluate: bool) -> torch.Tensor:
        """Teacher policy: critic-gradient argmin_a J(obs,a), with optional training noise."""
        # Initialize from previous action if present; else start from zero.
        if int(obs_t.shape[1]) >= 4:
            a_init = obs_t[:, 3].reshape(-1, 1)
            if int(self._action_low_t.numel()) > 1:
                a_init = a_init.repeat(1, int(self._action_low_t.numel()))
        else:
            a_init = torch.zeros(
                (int(obs_t.shape[0]), int(self._action_low_t.numel())),
                device=obs_t.device,
            )
        a = self._optimize_action_by_critic(
            obs_t, a_init=a_init, n_steps=int(self.action_grad_steps)
        )
        # Teacher-rollout noise is independent from exploration_std (which is for actor-based rollouts).
        if (not bool(evaluate)) and float(self.teacher_rollout_noise_std) > 0.0:
            a = a + torch.randn_like(a) * float(self.teacher_rollout_noise_std)
        # Apply same post-processing as main policy.
        m = float(self.action_momentum)
        if m > 0.0 and int(obs_t.shape[1]) >= 4:
            u_prev = obs_t[:, 3].reshape(-1, 1)
            if int(a.shape[1]) > 1:
                u_prev = u_prev.repeat(1, int(a.shape[1]))
            a = m * u_prev + (1.0 - m) * a
        u_max = float(self.action_max_abs)
        if u_max < 1.0:
            a = torch.clamp(a, -u_max, u_max)
        return torch.max(torch.min(a, self._action_high_t), self._action_low_t)

    def _maybe_randomize_env_for_episode(self) -> None:
        """Apply lightweight per-episode randomization (paper Section III recommendation).

        - Randomize env.initial_state around its configured initial_state.
        - Randomize env.reference_signal by rolling + optional noise.
        """
        # 1) initial state randomization
        if (
            self._initial_state_base is not None
            and float(self.initial_state_noise_std) > 0.0
        ):
            try:
                noise = np.random.normal(
                    0.0,
                    float(self.initial_state_noise_std),
                    size=self._initial_state_base.shape,
                ).astype(np.float32)
                new_init = (self._initial_state_base + noise).astype(np.float32)
                setattr(self.env, "initial_state", new_init.reshape(-1))
            except (TypeError, ValueError, AttributeError):
                pass

        # 2) reference signal randomization
        if self._reference_signal_base is not None and (
            int(self.reference_roll_steps) != 0 or float(self.reference_noise_std) > 0.0
        ):
            try:
                ref = np.asarray(self._reference_signal_base, dtype=np.float32).copy()
                if int(self.reference_roll_steps) != 0:
                    k = int(
                        np.random.randint(
                            -abs(int(self.reference_roll_steps)),
                            abs(int(self.reference_roll_steps)) + 1,
                        )
                    )
                    ref = np.roll(ref, shift=k, axis=-1)
                if float(self.reference_noise_std) > 0.0:
                    ref = ref + np.random.normal(
                        0.0, float(self.reference_noise_std), size=ref.shape
                    ).astype(np.float32)
                    # Clip to env limits if present (e.g. max_pitch_rad)
                    max_pitch_rad = getattr(self.env, "max_pitch_rad", None)
                    if max_pitch_rad is not None:
                        ref = np.clip(
                            ref, -float(max_pitch_rad), float(max_pitch_rad)
                        ).astype(np.float32)
                setattr(self.env, "reference_signal", ref)
            except (TypeError, ValueError, AttributeError):
                pass

    def _warmstart_actor(self, *, episodes: int, max_steps: int | None) -> None:
        episodes = int(episodes)
        if episodes <= 0:
            return

        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []

        for _ep in range(episodes):
            obs, _info = self.env.reset()
            obs = _as_flat_np(obs)
            self.reset()
            done = False
            steps = 0
            while not done:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    u_base = self._baseline_action(obs_t).squeeze(0).cpu().numpy()
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
        if self.policy_mode == "residual":
            # Residual policy: baseline is added externally, so we want actor ≈ 0 initially.
            y_t = torch.zeros_like(y_t)

        lr = float(self.actor_optim.param_groups[0].get("lr", 3e-4))
        opt = Adam(self.actor.parameters(), lr=lr)
        for _ in range(max(1, int(self.warmstart_actor_epochs))):
            pred = self.actor(x_t)
            loss = F.mse_loss(pred, y_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            opt.step()

    # ---- common API ----
    def get_env(self):
        return self.env

    def select_action(self, state: np.ndarray, *, evaluate: bool = False) -> np.ndarray:
        obs = _as_flat_np(state)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            act_t = self._policy_action_t(
                obs_t, evaluate=bool(evaluate), force_baseline=False
            )
            act = np.asarray(act_t.squeeze(0).cpu().numpy(), dtype=np.float32)
        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        return np.clip(act, low, high).astype(np.float32)

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Compatibility wrapper around `select_action`.

        Accepts either:
          - predict(state)
          - predict(state, deterministic=True/False)
        """
        if len(args) < 1 and "state" not in kwargs:
            raise ValueError(
                "ADHDP.predict expects `state` as the first positional argument."
            )
        state = kwargs.get("state", args[0] if len(args) > 0 else None)
        deterministic = kwargs.get("deterministic", True)
        if len(args) > 1:
            deterministic = bool(args[1])
        return self.select_action(
            np.asarray(state, dtype=np.float32), evaluate=bool(deterministic)
        )

    # ---- learning ----
    def _td_update(
        self,
        *,
        obs: np.ndarray,
        act: np.ndarray,
        reward_for_update: float,
        next_obs: np.ndarray,
        done_bootstrap: float,
        do_critic_update: bool,
        do_actor_update: bool,
        force_baseline_policy: bool = False,
    ) -> Tuple[float, float]:
        obs_t = torch.as_tensor(
            obs.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        act_t = torch.as_tensor(
            act.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        next_obs_t = torch.as_tensor(
            next_obs.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        done_t = torch.as_tensor(
            [[done_bootstrap]], dtype=torch.float32, device=self.device
        )

        # Convert reward to cost
        cost_t = -torch.as_tensor(
            [[reward_for_update]], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_act_t = self._policy_action_t(
                next_obs_t, evaluate=True, force_baseline=bool(force_baseline_policy)
            )
            q_next = self.critic(next_obs_t, next_act_t)
            target_q = cost_t + (1.0 - done_t) * self.gamma * q_next

        critic_loss_t = torch.as_tensor(0.0, dtype=torch.float32, device=self.device)
        if bool(do_critic_update):
            q = self.critic(obs_t, act_t)
            critic_loss_t = F.mse_loss(q, target_q)
            self.critic_optim.zero_grad()
            critic_loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()

        actor_loss_t = torch.as_tensor(0.0, dtype=torch.float32, device=self.device)
        if bool(do_actor_update):
            actor_act = self._policy_action_t(
                obs_t, evaluate=True, force_baseline=False
            )
            actor_loss_t = self.critic(obs_t, actor_act).mean()
            if float(self._actor_bc_coef) > 0.0:
                with torch.no_grad():
                    u_base = self._baseline_action(obs_t)
                if self.policy_mode == "residual":
                    # Regularize residual towards zero (i.e., stay close to baseline)
                    res = self.actor(obs_t)
                    actor_loss_t = actor_loss_t + float(
                        self._actor_bc_coef
                    ) * F.mse_loss(res, torch.zeros_like(res))
                else:
                    actor_loss_t = actor_loss_t + float(
                        self._actor_bc_coef
                    ) * F.mse_loss(actor_act, u_base)
            self.actor_optim.zero_grad()
            actor_loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optim.step()

        return float(critic_loss_t.item()), float(actor_loss_t.item())

    def _critic_update_with_target(
        self,
        *,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        target_q: torch.Tensor,
        n_steps: int,
    ) -> float:
        """Run N critic gradient steps on a fixed TD target (semi-gradient)."""
        last = 0.0
        for _ in range(max(1, int(n_steps))):
            q = self.critic(obs_t, act_t)
            loss_t = F.mse_loss(q, target_q)
            self.critic_optim.zero_grad()
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optim.step()
            last = float(loss_t.item())
        return float(last)

    def _actor_update(self, *, obs_t: torch.Tensor, n_steps: int) -> float:
        """Run N actor gradient steps (critic weights held fixed)."""
        last = 0.0
        # Freeze critic weights to avoid wasting compute (still allows dJ/da).
        prev_req = [p.requires_grad for p in self.critic.parameters()]
        try:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            for _ in range(max(1, int(n_steps))):
                actor_act = self._policy_action_t(
                    obs_t, evaluate=True, force_baseline=False
                )
                # Default actor objective: minimize critic output J(s, π(s)).
                # In distillation mode we intentionally DO NOT include this term, because
                # actor can exploit critic imperfections and diverge even when the
                # critic-gradient policy itself behaves well.
                loss_t = torch.as_tensor(0.0, dtype=torch.float32, device=obs_t.device)
                if str(self.actor_update_mode) == "minimize_critic":
                    loss_t = self.critic(obs_t, actor_act).mean()

                # Optional: stabilize actor by distilling the critic-gradient argmin policy.
                # This approximates the "actor as minimizer of J" while avoiding direct
                # exploitation of critic imperfections.
                if (
                    str(self.actor_update_mode) == "distill_critic_gradient"
                    and float(self.actor_distill_coef) > 0.0
                    and int(self.actor_distill_steps) > 0
                    and float(self.actor_distill_lr) > 0.0
                ):
                    # Use the same trust-region settings as the action-selection optimizer.
                    prev_lr = float(self.action_grad_lr)
                    prev_steps = int(self.action_grad_steps)
                    try:
                        self.action_grad_lr = float(self.actor_distill_lr)
                        self.action_grad_steps = int(self.actor_distill_steps)
                        # teacher action (no exploration)
                        if int(obs_t.shape[1]) >= 4:
                            a_init = obs_t[:, 3].reshape(-1, 1)
                            if int(actor_act.shape[1]) > 1:
                                a_init = a_init.repeat(1, int(actor_act.shape[1]))
                        else:
                            a_init = torch.zeros_like(actor_act)
                        with torch.no_grad():
                            a_star = self._optimize_action_by_critic(
                                obs_t,
                                a_init=a_init,
                                n_steps=int(self.action_grad_steps),
                            )
                        # Pure distillation loss (teacher matching).
                        loss_t = loss_t + float(self.actor_distill_coef) * F.mse_loss(
                            actor_act, a_star
                        )
                    finally:
                        self.action_grad_lr = float(prev_lr)
                        self.action_grad_steps = int(prev_steps)

                if float(self._actor_bc_coef) > 0.0:
                    with torch.no_grad():
                        u_base = self._baseline_action(obs_t)
                    if self.policy_mode == "residual":
                        res = self.actor(obs_t)
                        loss_t = loss_t + float(self._actor_bc_coef) * F.mse_loss(
                            res, torch.zeros_like(res)
                        )
                    else:
                        loss_t = loss_t + float(self._actor_bc_coef) * F.mse_loss(
                            actor_act, u_base
                        )
                self.actor_optim.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optim.step()
                last = float(loss_t.item())
        finally:
            for p, r in zip(self.critic.parameters(), prev_req):
                p.requires_grad_(r)
        return float(last)

    def train(self, *args, **kwargs) -> None:
        num_episodes = (
            int(args[0]) if len(args) > 0 else int(kwargs.get("num_episodes", 1))
        )
        max_steps = kwargs.get("max_steps", None)
        max_steps_i = int(max_steps) if max_steps is not None else None
        show_progress = bool(kwargs.get("show_progress", True))
        progress_desc = str(kwargs.get("progress_desc", "ADHDP train"))

        if self.warmstart_actor_episodes > 0:
            self._warmstart_actor(
                episodes=self.warmstart_actor_episodes, max_steps=max_steps_i
            )

        total_steps = 0
        ep_iter = range(num_episodes)
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]

                pbar = tqdm(ep_iter, desc=progress_desc, unit="ep")
                ep_iter = pbar
            except (ImportError, ModuleNotFoundError):
                pbar = None

        for ep in ep_iter:
            self._maybe_randomize_env_for_episode()
            obs, _info = self.env.reset()
            obs = _as_flat_np(obs)
            self.reset()

            # BC decay
            if self.actor_bc_decay < 1.0:
                self._actor_bc_coef = float(self._actor_bc_coef) * float(
                    self.actor_bc_decay
                )

            # alternating cycles
            phase = "both"
            c_ep = int(self.critic_cycle_episodes)
            a_ep = int(self.action_cycle_episodes)
            if c_ep > 0 and a_ep > 0:
                cyc = c_ep + a_ep
                pos = int(ep % cyc)
                phase = "critic" if pos < c_ep else "actor"
            do_critic = phase in ("both", "critic")
            do_actor = phase in ("both", "actor")
            if ep < int(self.critic_warmup_episodes):
                do_critic = True
                do_actor = False

            # Baseline-only warmup overrides all other schedules.
            baseline_warmup = bool(ep < int(self.baseline_warmup_episodes))
            if baseline_warmup:
                phase = "baseline_warmup"
                do_critic = True
                do_actor = False

            # During distillation, keep rollouts on the teacher policy (critic-gradient) to avoid
            # the still-learning actor driving the system out-of-distribution.
            execute_teacher = bool(
                (not bool(baseline_warmup))
                and bool(self.distill_execute_teacher)
                and str(self.action_selection) == "actor"
                and str(self.actor_update_mode) == "distill_critic_gradient"
                and int(self.action_grad_steps) > 0
                and float(self.action_grad_lr) > 0.0
            )
            if execute_teacher:
                phase = "teacher_rollout"

            # If actions are chosen by critic-gradient optimization, actor training is not used.
            if str(self.action_selection) == "critic_gradient":
                do_actor = False

            ep_reward = 0.0
            steps = 0
            done = False
            last_critic_loss = 0.0
            last_actor_loss = 0.0
            while not done:
                # Baseline used only during explicit baseline warmup episodes,
                # or (legacy) optionally during critic-only phases.
                force_baseline = bool(baseline_warmup) or bool(
                    (not bool(do_actor))
                    and bool(self.use_baseline_in_critic_phases)
                    and str(self.policy_mode) == "direct"
                )
                obs_t = torch.as_tensor(
                    obs.reshape(1, -1), dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    if bool(execute_teacher):
                        # Use deterministic teacher by default; enable noise via teacher_rollout_noise_std.
                        act_t = self._teacher_action_t(obs_t, evaluate=True)
                    else:
                        act_t = self._policy_action_t(
                            # Always treat action selection as "training-time" here.
                            # If you want deterministic actions, set exploration_std=0.0.
                            obs_t,
                            evaluate=False,
                            force_baseline=force_baseline,
                        )
                act = np.asarray(act_t.squeeze(0).cpu().numpy(), dtype=np.float32)
                next_obs, reward, terminated, truncated, info = self.env.step(act)
                next_obs = _as_flat_np(next_obs)

                done_env = bool(terminated or truncated)
                done_bootstrap = float(bool(terminated))
                ep_reward += float(reward)

                reward_for_update = float(reward)
                if self.use_env_cost:
                    try:
                        cost_total = float(info.get("cost_total"))
                        reward_scale = float(getattr(self.env, "reward_scale", 1.0))
                        # Default: use pure environment utility U ≈ cost_total (scaled like env reward).
                        # NOTE: ImprovedB747Env applies a large termination penalty to `reward`, but
                        # does NOT include it into `cost_total`. If we train purely on cost_total,
                        # the agent may fail to learn "avoid termination" and drift into unsafe
                        # states. So we keep cost_total for normal steps, but on termination steps
                        # we fall back to env reward (which includes termination penalty).
                        reward_for_update = -cost_total * reward_scale
                        if bool(terminated):
                            reward_for_update = float(reward)
                    except (TypeError, ValueError, AttributeError):
                        reward_for_update = float(reward)

                # --- MATLAB-style per-step update multipliers (semi-gradient critic target) ---
                obs_t_u = torch.as_tensor(
                    obs.reshape(1, -1), dtype=torch.float32, device=self.device
                )
                act_t_u = torch.as_tensor(
                    act.reshape(1, -1), dtype=torch.float32, device=self.device
                )
                next_obs_t_u = torch.as_tensor(
                    next_obs.reshape(1, -1), dtype=torch.float32, device=self.device
                )
                done_t_u = torch.as_tensor(
                    [[done_bootstrap]], dtype=torch.float32, device=self.device
                )
                cost_t_u = -torch.as_tensor(
                    [[reward_for_update]], dtype=torch.float32, device=self.device
                )

                critic_loss = 0.0
                actor_loss = 0.0
                if bool(do_critic):
                    with torch.no_grad():
                        if bool(execute_teacher):
                            next_act_t_u = self._teacher_action_t(
                                next_obs_t_u, evaluate=True
                            )
                        else:
                            next_act_t_u = self._policy_action_t(
                                next_obs_t_u,
                                evaluate=True,
                                force_baseline=bool(force_baseline),
                            )
                        q_next_u = self.critic(next_obs_t_u, next_act_t_u)
                        target_q_u = cost_t_u + (1.0 - done_t_u) * self.gamma * q_next_u
                    critic_loss = self._critic_update_with_target(
                        obs_t=obs_t_u,
                        act_t=act_t_u,
                        target_q=target_q_u,
                        n_steps=int(self.critic_updates_per_step),
                    )
                if bool(do_actor):
                    actor_loss = self._actor_update(
                        obs_t=obs_t_u, n_steps=int(self.actor_updates_per_step)
                    )
                last_critic_loss = float(critic_loss)
                last_actor_loss = float(actor_loss)

                self._updates += 1
                total_steps += 1
                steps += 1

                if (self._updates % int(self.log_every_updates)) == 0:
                    # Log losses only when the corresponding network was actually updated.
                    if bool(do_critic):
                        self.writer.add_scalar(
                            "loss/critic", float(critic_loss), self._updates
                        )
                    if bool(do_actor):
                        self.writer.add_scalar(
                            "loss/actor", float(actor_loss), self._updates
                        )
                    # Phase + action diagnostics (helps debug saturation/drift)
                    self.writer.add_scalar(
                        "train/do_critic",
                        1.0 if bool(do_critic) else 0.0,
                        self._updates,
                    )
                    self.writer.add_scalar(
                        "train/do_actor", 1.0 if bool(do_actor) else 0.0, self._updates
                    )
                    # Action saturation stats
                    try:
                        a = np.asarray(act, dtype=np.float32).reshape(-1)
                        hi = np.asarray(
                            self.env.action_space.high, dtype=np.float32
                        ).reshape(-1)
                        hi = np.maximum(np.abs(hi), 1e-6)
                        sat = float(np.mean(np.abs(a) >= 0.98 * hi))
                        self.writer.add_scalar(
                            "action/mean_abs", float(np.mean(np.abs(a))), self._updates
                        )
                        self.writer.add_scalar(
                            "action/sat_frac", float(sat), self._updates
                        )
                    except (TypeError, ValueError, AttributeError):
                        pass

                obs = next_obs
                done = done_env
                if max_steps_i is not None and steps >= max_steps_i:
                    break

            self.writer.add_scalar("performance/episode_reward", float(ep_reward), ep)
            self.writer.add_scalar("performance/episode_length", int(steps), ep)
            self.writer.add_scalar("train/total_steps", int(total_steps), ep)
            if pbar is not None:
                try:
                    pbar.set_postfix(
                        phase=str(phase),
                        rew=float(ep_reward),
                        steps=int(steps),
                        c_loss=float(last_critic_loss),
                        a_loss=float(last_actor_loss),
                    )
                except (TypeError, ValueError, AttributeError):
                    pass

        self.writer.flush()

    # ---- persistence (optional, aligned with other agents) ----
    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        try:
            if "tensoraerospace" in env_name:
                env_params = serialize_env(self.env)
        except (TypeError, ValueError, AttributeError):
            env_params = {}

        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        policy_params: Dict[str, Any] = {
            "paper_strict": bool(self.paper_strict),
            "gamma": float(self.gamma),
            "policy_mode": str(self.policy_mode),
            "residual_scale": float(self.residual_scale),
            "use_baseline_in_critic_phases": bool(self.use_baseline_in_critic_phases),
            "action_momentum": float(self.action_momentum),
            "action_max_abs": float(self.action_max_abs),
            "action_selection": str(self.action_selection),
            "action_grad_steps": int(self.action_grad_steps),
            "action_grad_lr": float(self.action_grad_lr),
            "action_grad_step_clip": float(self.action_grad_step_clip),
            "action_grad_u_l2": float(self.action_grad_u_l2),
            "action_grad_du_l2": float(self.action_grad_du_l2),
            "actor_update_mode": str(self.actor_update_mode),
            "actor_distill_coef": float(self.actor_distill_coef),
            "actor_distill_steps": int(self.actor_distill_steps),
            "actor_distill_lr": float(self.actor_distill_lr),
            "distill_execute_teacher": bool(self.distill_execute_teacher),
            "teacher_rollout_noise_std": float(self.teacher_rollout_noise_std),
            "exploration_std": float(self.exploration_std),
            "critic_updates_per_step": int(self.critic_updates_per_step),
            "actor_updates_per_step": int(self.actor_updates_per_step),
            "device": self.device.type,
            "seed": int(self.seed),
            "actor_lr": float(self.actor_optim.defaults.get("lr", 1e-4)),
            "critic_lr": float(self.critic_optim.defaults.get("lr", 1e-4)),
            "hidden_size": int(self.hidden_size),
            "initial_state_noise_std": float(self.initial_state_noise_std),
            "reference_roll_steps": int(self.reference_roll_steps),
            "reference_noise_std": float(self.reference_noise_std),
            "baseline_warmup_episodes": int(self.baseline_warmup_episodes),
            "critic_warmup_episodes": int(self.critic_warmup_episodes),
            "critic_cycle_episodes": int(self.critic_cycle_episodes),
            "action_cycle_episodes": int(self.action_cycle_episodes),
            "use_env_cost": bool(self.use_env_cost),
            "warmstart_actor_episodes": int(self.warmstart_actor_episodes),
            "warmstart_actor_epochs": int(self.warmstart_actor_epochs),
            "baseline_type": str(self.baseline_type),
            "baseline_kp": float(self.baseline_kp),
            "baseline_ki": float(self.baseline_ki),
            "baseline_kd": float(self.baseline_kd),
            "pid_i_clip": float(self.pid_i_clip),
            "actor_bc_l2": float(self.actor_bc_l2),
            "actor_bc_decay": float(self.actor_bc_decay),
        }
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
        actor_optim_path = run_dir / "actor_optim.pth"
        critic_optim_path = run_dir / "critic_optim.pth"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)
        if save_gradients:
            torch.save(self.actor_optim.state_dict(), actor_optim_path)
            torch.save(self.critic_optim.state_dict(), critic_optim_path)
        return str(run_dir)

    @staticmethod
    def _filter_kwargs_for_init(
        env_cls: type, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(env_cls)
        except (TypeError, ValueError):
            return kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}

    def load(self, *args, **kwargs) -> None:
        """Load model weights from a directory created by `save()`."""
        folder = kwargs.get("path", None) or (args[0] if len(args) > 0 else None)
        if folder is None:
            raise ValueError(
                "ADHDP.load(path=...) expects a folder path created by ADHDP.save()."
            )
        folder_p = Path(folder)

        actor_path = folder_p / "actor.pth"
        critic_path = folder_p / "critic.pth"
        if not actor_path.exists() or not critic_path.exists():
            raise FileNotFoundError(
                f"Missing actor/critic files in {str(folder_p)!r} (expected actor.pth, critic.pth)"
            )

        self.actor = torch.load(
            actor_path, map_location=self.device, weights_only=False
        )
        self.critic = torch.load(
            critic_path, map_location=self.device, weights_only=False
        )

    @classmethod
    def from_dir(cls, folder: Union[str, Path]) -> "ADHDP":
        """Instantiate env+agent from `save()` directory (config.json + actor/critic)."""
        folder_p = Path(folder)
        config_path = folder_p / "config.json"
        actor_path = folder_p / "actor.pth"
        critic_path = folder_p / "critic.pth"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {str(folder_p)!r}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        env_info = cfg.get("env", {})
        policy_info = cfg.get("policy", {})
        env_name = env_info.get("name")
        env_params = env_info.get("params", {})
        policy_params = policy_info.get("params", {})
        if not env_name:
            raise ValueError("Invalid config: missing env.name")
        env_cls: type = get_class_from_string(env_name)
        env_kwargs = cls._filter_kwargs_for_init(env_cls, dict(env_params))
        env = env_cls(**env_kwargs)

        agent = cls(env=env, **policy_params)
        agent.actor = torch.load(
            actor_path, map_location=agent.device, weights_only=False
        )
        agent.critic = torch.load(
            critic_path, map_location=agent.device, weights_only=False
        )
        return agent
