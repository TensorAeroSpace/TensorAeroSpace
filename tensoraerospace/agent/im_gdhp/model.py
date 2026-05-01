"""Incremental Model-based Global Dual Heuristic Programming agent.

Implements the control scheme from Sun & van Kampen (2021),
*"Intelligent adaptive optimal control using incremental model-based
global dual heuristic programming subject to partial observability"*,
Applied Soft Computing 103, 107153, with the system identification
block formulated as block RLS on the incremental model (Zhou et al.
2020; Sun & van Kampen 2019). The implementation is self-contained in
pure PyTorch + NumPy and operates directly on the observation vector
returned by a Gymnasium environment — which is typically a strict
subset of the full internal state, hence "partial observability".

High-level algorithm::

    At each step t, given the newly observed y_t and reference r_t:

      1. Form the augmented observation o_t = [y_t; r_t; e_t] with
         e_t = y_t[tracking] − r_t.
      2. Actor produces u_t = π_θ(o_t).
      3. Execute u_t in the environment, observe y_{t+1}.
      4. Compute the one-step cost c_t = e_tᵀ Q e_t + ρ ‖u_t − u_{t-1}‖².
      5. If t ≥ 2, update the incremental model via one RLS step using
         (y_{t-2}, y_{t-1}, y_t, u_{t-2}, u_{t-1}) — this gives us
         current estimates of A_t and B_t.
      6. Compute λ(t+1) and J(t+1) by passing the predicted observation
         ŷ_{t+1} (from the incremental model) through the target /
         online critic.
      7. Critic loss:
            L_J  = (J(o_t) − (c_t + γ J(o_{t+1})))²
            L_λ  = ‖λ(o_t) − (∂c_t/∂y + γ A_tᵀ λ(o_{t+1}))‖²
            L    = L_J + β L_λ
      8. Actor loss: minimise ``c_t + γ J(ô_{t+1})`` where ``ô_{t+1}``
         is built from the predicted ``ŷ_{t+1}`` obtained via the
         incremental model. The autograd graph carries the gradient
         through ``B_t`` back to the actor weights.

The resulting agent has three differentiable blocks — actor, critic,
and the RLS-identified linear increment model — and does not require
any prior knowledge of the plant dynamics.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn, optim

from .incremental_model import IncrementalModelRLS
from .networks import GDHPActor, GDHPCritic


@dataclass
class IMGDHPConfig:
    """Hyper-parameters for :class:`IMGDHPAgent`.

    Args:
        gamma: Discount factor γ.
        actor_hidden: Hidden layer sizes of the actor MLP.
        critic_hidden: Hidden layer sizes of the critic backbone.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        beta_lambda: Weight of the λ-loss in the GDHP critic objective.
        track_Q: Diagonal weights of the quadratic tracking cost over
            the tracked output channels. Length must equal ``n_track``.
        action_rate_penalty: ρ coefficient penalising ‖Δu‖² in the
            one-step cost.
        forgetting: RLS forgetting factor for the incremental model.
        cov_init: Initial scale of the RLS covariance matrix.
        warmup_steps: Number of initial steps during which only the
            incremental model is updated (actor and critic are held
            fixed) so that ``A`` and ``B`` stabilise before being used
            for policy improvement.
        critic_only_steps: Number of additional steps beyond
            ``warmup_steps`` during which the critic is updated but
            the actor is held fixed. This gives the value function time
            to settle before the policy gradient starts acting on it,
            which empirically dampens DHP oscillations.
        critic_updates_per_step: Number of gradient steps to take on
            the critic for every environment transition. Multiple
            critic steps per env step accelerate Bellman convergence
            without waiting for more samples.
        target_update_tau: Polyak coefficient for the soft target
            critic update ``θ_target ← (1-τ)·θ_target + τ·θ``. When
            ``0.0`` the target network is disabled and the online
            critic is used for Bellman bootstrapping (legacy
            behaviour). Typical values ``1e-3`` to ``1e-2``.
        critic_weight_decay: L2 regularisation coefficient passed to
            the critic's ``Adam`` optimiser. Prevents the λ-head from
            drifting during long online runs.
        obs_scale: Optional per-component multiplier applied to the raw
            observation and the reference before feeding them to the
            actor and critic. Useful to compensate for very small
            (rad-scale) physical units. Length ``n_obs`` or ``None``.
        max_grad_norm: Gradient clipping applied to both actor and
            critic optimisers.
        exploration_noise_std: Std of zero-mean Gaussian noise added to
            the actor output during training. Set to ``0`` to disable.
        u_max: Per-channel absolute bound on the control output of the
            actor, in the units expected by the environment (e.g. deg
            for the F-16 envs in tensoraerospace).
        device: Torch device for the networks.
        seed: Optional seed for torch / numpy RNGs.
    """

    gamma: float = 0.95
    actor_hidden: Sequence[int] = (32, 32)
    critic_hidden: Sequence[int] = (64, 64)
    actor_lr: float = 1e-3
    critic_lr: float = 5e-3
    beta_lambda: float = 1.0
    track_Q: Sequence[float] = (1.0,)
    action_rate_penalty: float = 1e-3
    forgetting: float = 0.9995
    cov_init: float = 1e2
    warmup_steps: int = 5
    critic_only_steps: int = 0
    critic_updates_per_step: int = 1
    target_update_tau: float = 0.0
    critic_weight_decay: float = 0.0
    obs_scale: Sequence[float] | None = None
    max_grad_norm: float = 5.0
    exploration_noise_std: float = 0.0
    u_max: float = 25.0
    device: str = "cpu"
    seed: int | None = None
    history: dict = field(default_factory=dict)


class IMGDHPAgent:
    """Online IM-GDHP control agent with partial observability support.

    The agent's public training interface mirrors the online style of
    :class:`tensoraerospace.agent.ihdp.IHDPAgent`: at each environment
    step the caller invokes :meth:`predict` with the latest observation
    and reference, gets back the next control command, executes it in
    the environment, then calls :meth:`learn` with the new observation
    to perform one RLS + critic + actor update. A convenience
    :meth:`train` loop wraps this for episodic training against a
    Gymnasium environment.

    Args:
        n_obs: Length of the environment observation vector ``y``.
        n_action: Number of control channels.
        reference_size: Length of the reference vector at each time step
            (typically 1 per tracked channel).
        tracking_indices: Indices into ``y`` of the states that should
            track the reference. Used to build the scalar tracking error
            that drives the reward. Defaults to ``[0]``.
        config: Optional :class:`IMGDHPConfig` instance. Fields not set
            fall back to the class defaults.
    """

    def __init__(
        self,
        n_obs: int,
        n_action: int,
        reference_size: int = 1,
        tracking_indices: Sequence[int] | None = None,
        config: IMGDHPConfig | None = None,
    ) -> None:
        self.n_obs = int(n_obs)
        self.n_action = int(n_action)
        self.reference_size = int(reference_size)
        self.tracking_indices = (
            list(tracking_indices) if tracking_indices is not None else [0]
        )
        if len(self.tracking_indices) == 0:
            raise ValueError("tracking_indices must contain at least one index")
        self.cfg = config if config is not None else IMGDHPConfig()

        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
            np.random.seed(int(self.cfg.seed))

        if len(self.cfg.track_Q) != len(self.tracking_indices):
            raise ValueError(
                f"track_Q has length {len(self.cfg.track_Q)} but tracking_indices "
                f"has length {len(self.tracking_indices)}"
            )

        self.device = torch.device(self.cfg.device)

        # Augmented-observation layout: [y (n_obs); ref (reference_size);
        # tracking_error (len(tracking_indices))]. The tracking_error is
        # redundant given y and ref, but providing it explicitly helps the
        # actor/critic learn faster.
        self.augmented_size = (
            self.n_obs + self.reference_size + len(self.tracking_indices)
        )

        self.actor = GDHPActor(
            in_features=self.augmented_size,
            n_u=self.n_action,
            hidden_sizes=tuple(self.cfg.actor_hidden),
            u_max=self.cfg.u_max,
        ).to(self.device)

        self.critic = GDHPCritic(
            in_features=self.augmented_size,
            n_y=self.n_obs,
            hidden_sizes=tuple(self.cfg.critic_hidden),
        ).to(self.device)

        # Target critic for Bellman bootstrapping. Always constructed so
        # that hot-swapping ``target_update_tau`` at runtime stays
        # consistent — when ``tau == 0`` the target net tracks the
        # online net exactly (hard-copied below) and is never updated,
        # which reproduces the legacy behaviour.
        self.target_critic = GDHPCritic(
            in_features=self.augmented_size,
            n_y=self.n_obs,
            hidden_sizes=tuple(self.cfg.critic_hidden),
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.Adam(
            self.critic.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=float(self.cfg.critic_weight_decay),
        )

        self.incremental_model = IncrementalModelRLS(
            n_y=self.n_obs,
            n_u=self.n_action,
            forgetting=self.cfg.forgetting,
            cov_init=self.cfg.cov_init,
            seed=self.cfg.seed,
        )

        self._Q = torch.as_tensor(
            np.asarray(self.cfg.track_Q, dtype=np.float64),
            dtype=torch.float32,
            device=self.device,
        )

        if self.cfg.obs_scale is not None:
            scale = np.asarray(self.cfg.obs_scale, dtype=np.float64).reshape(-1)
            if scale.size != self.n_obs:
                raise ValueError(
                    f"obs_scale has length {scale.size}, expected {self.n_obs}"
                )
            self._obs_scale_np = scale
            self._obs_scale_t = torch.as_tensor(
                scale, dtype=torch.float32, device=self.device
            )
        else:
            self._obs_scale_np = np.ones(self.n_obs, dtype=np.float64)
            self._obs_scale_t = torch.ones(
                self.n_obs, dtype=torch.float32, device=self.device
            )

        # Rolling buffers: one step of history suffices because learn()
        # already has access to (y_{t-1}, y_t, y_{t+1}) and (u_{t-1}, u_t)
        # once called after env.step().
        self._y_tm1: np.ndarray | None = None
        self._u_tm1: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._last_augmented: np.ndarray | None = None
        self._total_steps = 0

        # Metric log populated by :meth:`train`.
        self.history: dict[str, list[float]] = {
            "episode_return": [],
            "critic_loss": [],
            "actor_loss": [],
            "rls_pred_error_norm": [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reference_at(self, reference_signal: np.ndarray, time_step: int) -> np.ndarray:
        """Fetch ``reference_signal[:, t]`` with safe clipping at the end."""
        ref_arr = np.asarray(reference_signal, dtype=np.float64)
        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(1, -1)
        t_safe = int(np.clip(time_step, 0, ref_arr.shape[1] - 1))
        return ref_arr[:, t_safe].reshape(-1)

    def _augment(self, y: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Build the augmented observation vector ``[y; ref; tracking_err]``.

        The raw observation and reference are multiplied by
        ``cfg.obs_scale`` (if supplied) to match the natural unit of the
        tracked channel — this lets a radian-scale observation feed an
        actor/critic that prefer O(1) inputs without hand-tuning the
        network initialisation.
        """
        y_v = np.asarray(y, dtype=np.float64).reshape(-1) * self._obs_scale_np
        ref_v = np.asarray(ref, dtype=np.float64).reshape(-1)
        if ref_v.size < self.reference_size:
            ref_v = np.concatenate([ref_v, np.zeros(self.reference_size - ref_v.size)])
        elif ref_v.size > self.reference_size:
            ref_v = ref_v[: self.reference_size]
        # Reference is measured in the same physical unit as the tracked
        # observation component; scale it accordingly.
        ref_scale = self._obs_scale_np[self.tracking_indices[0]]
        ref_v = ref_v * ref_scale
        track = y_v[self.tracking_indices]
        if ref_v.size == 1 and len(self.tracking_indices) > 1:
            err = track - ref_v[0]
        else:
            err = track - ref_v[: len(self.tracking_indices)]
        return np.concatenate([y_v, ref_v, err])

    def _augment_torch(self, y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Torch-compatible counterpart of :meth:`_augment`."""
        y_s = y * self._obs_scale_t
        ref_scale = float(self._obs_scale_t[self.tracking_indices[0]])
        ref_s = ref * ref_scale
        track = y_s[self.tracking_indices]
        if ref_s.numel() == 1 and len(self.tracking_indices) > 1:
            err = track - ref_s.expand(len(self.tracking_indices))
        else:
            err = track - ref_s[: len(self.tracking_indices)]
        return torch.cat([y_s, ref_s, err])

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the per-episode rolling history.

        Does *not* reset the learned weights, the incremental model
        covariance or the optimiser state — so learning progresses
        across episodes as expected.
        """
        self._y_tm1 = None
        self._u_tm1 = None
        self._last_action = None
        self._last_augmented = None
        self.incremental_model.reset()

    def predict(
        self,
        obs: np.ndarray,
        reference_signal: np.ndarray,
        time_step: int,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Compute the control action for a single time step.

        Args:
            obs: Current observation ``y_t`` of length ``n_obs``.
            reference_signal: Reference trajectory, shape
                ``(reference_size, T)`` or ``(T,)``.
            time_step: Current time index used to select
                ``reference_signal[:, time_step]``.
            deterministic: If True, no exploration noise is added.

        Returns:
            A NumPy array of length ``n_action`` with the commanded
            control.
        """
        y = np.asarray(obs, dtype=np.float64).reshape(-1)
        ref = self._reference_at(reference_signal, time_step)
        aug = self._augment(y, ref)

        self.actor.eval()
        with torch.no_grad():
            u_t = self.actor(
                torch.as_tensor(aug, dtype=torch.float32, device=self.device)
            )
        u = u_t.cpu().numpy().astype(np.float64)

        if (not deterministic) and self.cfg.exploration_noise_std > 0.0:
            u = u + self._rng_normal(self.n_action) * self.cfg.exploration_noise_std
            u = np.clip(u, -self.cfg.u_max, self.cfg.u_max)

        self._last_action = u.copy()
        self._last_augmented = aug.copy()
        return np.asarray(u)

    def _rng_normal(self, n: int) -> np.ndarray:
        return np.random.normal(size=(n,))

    def learn(
        self,
        next_obs: np.ndarray,
        reference_signal: np.ndarray,
        time_step: int,
    ) -> dict[str, float]:
        """Perform one online RLS + critic + actor update.

        Must be called *after* :meth:`predict` and the corresponding
        environment step. ``next_obs`` is ``y_{t+1}``, and ``time_step``
        is the index of the step that has just been executed (i.e. the
        same ``time_step`` that was passed to :meth:`predict`).

        Returns:
            Dict with latest scalar training metrics (``critic_loss``,
            ``actor_loss``, ``rls_pred_error_norm``).
        """
        if self._last_action is None or self._last_augmented is None:
            raise RuntimeError("learn() called before predict()")

        y_next = np.asarray(next_obs, dtype=np.float64).reshape(-1)
        u_t = self._last_action.copy()
        y_t_np = np.asarray(self._last_augmented[: self.n_obs]).copy()

        # --- 1. RLS update of the incremental model ---------------------
        rls_err_norm = float("nan")
        if self._y_tm1 is not None and self._u_tm1 is not None:
            # Freshest transition (y_{t-1}, y_t, y_{t+1}) with controls
            # (u_{t-1}, u_t): target = Δy_{t+1}, regressor = [Δy_t; Δu_t].
            eps = self.incremental_model.update(
                y_prev=self._y_tm1,
                y_curr=y_t_np,
                y_next=y_next,
                u_prev=self._u_tm1,
                u_curr=u_t,
            )
            rls_err_norm = float(np.linalg.norm(eps))

        # --- 2. Per-step tracking cost --------------------------------
        # Standard DHP convention: c_t depends on the state AT time t,
        # not the predicted next one. The tracking error at y_t vs the
        # reference at time t drives both the Bellman target for J and
        # the costate target for λ (through ∂c_t/∂y_t = 2 Q e_t).
        # The cost is evaluated in the *scaled* observation space so
        # that track_Q is interpreted consistently when obs_scale is
        # set — otherwise a scaling change would silently retune the
        # cost function.
        ref_now = self._reference_at(reference_signal, time_step)
        ref_next = self._reference_at(reference_signal, time_step + 1)

        ref_unit_scale = float(self._obs_scale_np[self.tracking_indices[0]])
        track_now_scaled = (
            y_t_np[self.tracking_indices] * self._obs_scale_np[self.tracking_indices]
        )
        if ref_now.size == 1 and len(self.tracking_indices) > 1:
            err_now = track_now_scaled - ref_now[0] * ref_unit_scale
        else:
            err_now = (
                track_now_scaled
                - ref_now[: len(self.tracking_indices)] * ref_unit_scale
            )

        Q_np = np.asarray(self.cfg.track_Q, dtype=np.float64)
        c_now_value = float(np.sum(Q_np * err_now**2))

        critic_loss_val = float("nan")
        actor_loss_val = float("nan")

        # --- 3. GDHP training (skipped during warm-up) -----------------
        if self._total_steps >= self.cfg.warmup_steps:
            # Multi-step critic update: iterate the critic optimiser
            # several times per env step to accelerate Bellman
            # convergence (useful when critic_only_steps is also > 0).
            for _ in range(max(1, int(self.cfg.critic_updates_per_step))):
                critic_loss_val = self._critic_update(
                    aug_t_np=self._last_augmented,
                    y_next_np=y_next,
                    ref_next_np=ref_next,
                    c_now_value=c_now_value,
                    err_now_np=err_now,
                )
            allow_actor = self._total_steps >= (
                self.cfg.warmup_steps + self.cfg.critic_only_steps
            )
            if allow_actor and self._y_tm1 is not None and self._u_tm1 is not None:
                actor_loss_val = self._actor_update(
                    y_t_np=y_t_np,
                    ref_now_np=ref_now,
                    ref_next_np=ref_next,
                    u_prev_np=self._u_tm1,
                    y_prev_np=self._y_tm1,
                )

        # --- 4. Shift the rolling history ------------------------------
        self._y_tm1 = y_t_np.copy()
        self._u_tm1 = u_t.copy()
        self._total_steps += 1

        metrics = {
            "critic_loss": critic_loss_val,
            "actor_loss": actor_loss_val,
            "rls_pred_error_norm": rls_err_norm,
        }
        return metrics

    # ------------------------------------------------------------------
    # Critic / actor updates
    # ------------------------------------------------------------------
    def _soft_update_target(self) -> None:
        """Polyak update of the target critic ``θ_tgt ← (1-τ)θ_tgt + τ θ``."""
        tau = float(self.cfg.target_update_tau)
        if tau <= 0.0:
            return
        with torch.no_grad():
            for tgt, src in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    def _critic_update(
        self,
        aug_t_np: np.ndarray,
        y_next_np: np.ndarray,
        ref_next_np: np.ndarray,
        c_now_value: float,
        err_now_np: np.ndarray,
    ) -> float:
        """Run one GDHP critic update targeting both ``J`` and ``λ``.

        ``J`` target: Bellman-style ``c_t + γ J(o_{t+1})``.
        ``λ`` target: ``∂c_t/∂y_t + γ Aᵀ λ(o_{t+1})`` where
        ``c_t = (y_t − r_t)ᵀ Q (y_t − r_t)`` depends on the state AT
        time ``t`` and therefore has a non-zero gradient only along the
        tracked components.

        When ``target_update_tau > 0`` the Bellman bootstrap uses the
        slow-moving target critic, which dampens the positive-feedback
        loop between actor and critic observed in long DHP runs.
        """
        aug_t = torch.as_tensor(aug_t_np, dtype=torch.float32, device=self.device)
        aug_next_np = self._augment(y_next_np, ref_next_np)
        aug_next = torch.as_tensor(aug_next_np, dtype=torch.float32, device=self.device)

        self.critic.train()
        bootstrap_net = (
            self.target_critic if self.cfg.target_update_tau > 0.0 else self.critic
        )
        with torch.no_grad():
            J_next, lam_next = bootstrap_net(aug_next)

        dc_dy = torch.zeros(self.n_obs, dtype=torch.float32, device=self.device)
        Q_np = np.asarray(self.cfg.track_Q, dtype=np.float64)
        for i, idx in enumerate(self.tracking_indices):
            dc_dy[idx] = float(2.0 * Q_np[i] * err_now_np[i])

        A_mat = torch.as_tensor(
            self.incremental_model.A, dtype=torch.float32, device=self.device
        )
        lambda_target = dc_dy + self.cfg.gamma * (A_mat.T @ lam_next)
        j_target = torch.tensor(
            c_now_value + self.cfg.gamma * float(J_next.detach().squeeze()),
            dtype=torch.float32,
            device=self.device,
        )

        J_pred, lam_pred = self.critic(aug_t)
        loss_j = (J_pred.squeeze() - j_target).pow(2)
        loss_lambda = ((lam_pred - lambda_target).pow(2)).sum()
        loss = loss_j + self.cfg.beta_lambda * loss_lambda

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_opt.step()
        self._soft_update_target()
        return float(loss.detach())

    def _actor_update(
        self,
        y_t_np: np.ndarray,
        ref_now_np: np.ndarray,
        ref_next_np: np.ndarray,
        u_prev_np: np.ndarray,
        y_prev_np: np.ndarray,
    ) -> float:
        """One actor update — Model-Predictive style loss.

        Minimises ``c(ŷ_{t+1}, r_{t+1}) + γ J(ô_{t+1}) + ρ ‖Δu‖²``
        where ``ŷ_{t+1}`` is built from the current incremental-model
        estimates: ``ŷ_{t+1} = y_t + A · (y_t − y_{t-1}) + B · (u −
        u_{t-1})``. The autograd graph flows through ``B · u`` back to
        the actor weights, which gives the correct policy gradient
        ``(2 Q e_{t+1} + λ(o_{t+1}))ᵀ · B · ∂u/∂W_a`` without manual
        Jacobian work. Keeping the direct one-step cost in the actor
        objective (ADP "action dependent" convention) gives the policy
        an immediate teaching signal even while the critic is still
        bootstrapping.
        """
        y_t = torch.as_tensor(y_t_np, dtype=torch.float32, device=self.device)
        y_prev = torch.as_tensor(y_prev_np, dtype=torch.float32, device=self.device)
        u_prev = torch.as_tensor(u_prev_np, dtype=torch.float32, device=self.device)
        ref_now = torch.as_tensor(ref_now_np, dtype=torch.float32, device=self.device)
        ref_next = torch.as_tensor(ref_next_np, dtype=torch.float32, device=self.device)

        A_mat = torch.as_tensor(
            self.incremental_model.A, dtype=torch.float32, device=self.device
        )
        B_mat = torch.as_tensor(
            self.incremental_model.B, dtype=torch.float32, device=self.device
        )

        aug_now = self._augment_torch(y_t, ref_now)
        u = self.actor(aug_now)
        du = u - u_prev

        y_next_pred = y_t + A_mat @ (y_t - y_prev) + B_mat @ du

        track = y_next_pred[self.tracking_indices]
        if ref_next.numel() == 1 and len(self.tracking_indices) > 1:
            err = track - ref_next.expand(len(self.tracking_indices))
        else:
            err = track - ref_next[: len(self.tracking_indices)]
        c_pred = torch.sum(self._Q * err.pow(2))

        aug_next_pred = self._augment_torch(y_next_pred, ref_next)
        J_next, _ = self.critic(aug_next_pred)

        loss = (
            c_pred
            + self.cfg.gamma * J_next.squeeze()
            + self.cfg.action_rate_penalty * du.pow(2).sum()
        )

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()
        return float(loss.detach())

    # ------------------------------------------------------------------
    # High-level training loop
    # ------------------------------------------------------------------
    def train(
        self,
        env: Any,
        num_episodes: int = 1,
        *,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Run episodic training against a Gymnasium environment.

        The env is expected to expose a ``reference_signal`` attribute
        (shape ``(reference_size, T)``) — matching the convention used
        by the existing ``tensoraerospace.envs.f16`` envs.

        Args:
            env: Gymnasium-like environment to train on.
            num_episodes: Number of episodes to run.
            max_steps: Optional cap on steps per episode.
            verbose: If True, print per-episode summaries.

        Returns:
            The accumulated training history (same as ``self.history``).
        """
        for ep in range(int(num_episodes)):
            obs, _info = env.reset()
            self.reset()
            done = False
            truncated = False
            ep_return = 0.0
            last_metrics = {
                "critic_loss": float("nan"),
                "actor_loss": float("nan"),
                "rls_pred_error_norm": float("nan"),
            }
            step = 0
            reference_signal = getattr(env, "reference_signal", None)
            if reference_signal is None:
                raise AttributeError(
                    "env must expose a 'reference_signal' attribute for IM-GDHP training"
                )

            while not (done or truncated):
                action = self.predict(obs, reference_signal, step)
                obs_next, reward, done, truncated, _info = env.step(action)
                last_metrics = self.learn(obs_next, reference_signal, step)
                ep_return += float(reward)
                obs = obs_next
                step += 1
                if max_steps is not None and step >= max_steps:
                    break

            self.history["episode_return"].append(ep_return)
            self.history["critic_loss"].append(last_metrics["critic_loss"])
            self.history["actor_loss"].append(last_metrics["actor_loss"])
            self.history["rls_pred_error_norm"].append(
                last_metrics["rls_pred_error_norm"]
            )
            if verbose:
                print(
                    f"[IM-GDHP] ep={ep + 1}/{num_episodes} "
                    f"return={ep_return:+.3f} "
                    f"critic={last_metrics['critic_loss']:.4f} "
                    f"actor={last_metrics['actor_loss']:.4f} "
                    f"rls|eps|={last_metrics['rls_pred_error_norm']:.4f}"
                )
        return self.history

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config for :meth:`save`.

        The agent has no bound environment (observations are fed in by
        the caller), so only the constructor signature and config
        dataclass are persisted.
        """
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        # ``cfg.history`` is a runtime cache (per-episode metrics) — never
        # persisted; reset to an empty dict on load.
        cfg_dict.pop("history", None)
        # Tuples → lists so the round-trip through JSON is type-stable.
        for key, value in list(cfg_dict.items()):
            if isinstance(value, tuple):
                cfg_dict[key] = list(value)
        return {
            "policy": {
                "name": agent_name,
                "params": {
                    "n_obs": self.n_obs,
                    "n_action": self.n_action,
                    "reference_size": self.reference_size,
                    "tracking_indices": list(self.tracking_indices),
                },
                "config": cfg_dict,
            },
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
        *,
        save_gradients: bool = False,
    ) -> str:
        """Write the agent to a directory.

        Files produced:
            * ``config.json`` — constructor kwargs + serialised
              :class:`IMGDHPConfig`.
            * ``actor.pth`` / ``critic.pth`` / ``target_critic.pth`` —
              PyTorch state dicts for the three networks.
            * ``incremental_model.npz`` — RLS ``theta`` and ``P``
              matrices plus scalar hyper-parameters.
            * ``actor_optim.pth`` / ``critic_optim.pth`` — optimiser
              state dicts (only when ``save_gradients=True``).

        Args:
            path: Base directory. If ``None``, uses CWD.
            save_gradients: Persist optimiser states so training can
                resume bitwise-identically from the checkpoint.

        Returns:
            Absolute path to the created run directory.
        """
        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Config
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        # Torch networks
        torch.save(self.actor.state_dict(), run_dir / "actor.pth")
        torch.save(self.critic.state_dict(), run_dir / "critic.pth")
        torch.save(self.target_critic.state_dict(), run_dir / "target_critic.pth")

        # RLS incremental model — theta + P + scalars.
        rls = self.incremental_model
        np.savez(
            run_dir / "incremental_model.npz",
            theta=rls.theta,
            P=rls.P,
            forgetting=np.asarray(rls.alpha),
            cov_init=np.asarray(rls.cov_init),
            num_updates=np.asarray(rls.num_updates),
        )

        if save_gradients:
            torch.save(self.actor_opt.state_dict(), run_dir / "actor_optim.pth")
            torch.save(self.critic_opt.state_dict(), run_dir / "critic_optim.pth")

        return str(run_dir)

    @classmethod
    def _load_from_dir(
        cls,
        folder: Union[str, Path],
        *,
        load_gradients: bool = False,
    ) -> "IMGDHPAgent":
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

        # Device fallback — a checkpoint saved on cuda/mps must still load
        # on a CPU-only host.
        dev = str(cfg_dict.get("device", "cpu"))
        if dev.startswith("cuda") and not torch.cuda.is_available():
            cfg_dict["device"] = "cpu"
        elif dev.startswith("mps") and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            cfg_dict["device"] = "cpu"

        agent_cfg = IMGDHPConfig(**cfg_dict)
        agent = cls(
            n_obs=params["n_obs"],
            n_action=params["n_action"],
            reference_size=params.get("reference_size", 1),
            tracking_indices=params.get("tracking_indices", [0]),
            config=agent_cfg,
        )

        # Torch networks
        agent.actor.load_state_dict(
            torch.load(
                folder_p / "actor.pth",
                map_location=agent.device,
                weights_only=False,
            )
        )
        agent.critic.load_state_dict(
            torch.load(
                folder_p / "critic.pth",
                map_location=agent.device,
                weights_only=False,
            )
        )
        target_path = folder_p / "target_critic.pth"
        if target_path.exists():
            agent.target_critic.load_state_dict(
                torch.load(target_path, map_location=agent.device, weights_only=False)
            )
        else:
            # Legacy checkpoint without a target critic — mirror online.
            agent.target_critic.load_state_dict(agent.critic.state_dict())

        # RLS incremental model
        rls_path = folder_p / "incremental_model.npz"
        if rls_path.exists():
            with np.load(rls_path) as npz:
                agent.incremental_model.theta = npz["theta"]
                agent.incremental_model.P = npz["P"]
                agent.incremental_model.alpha = float(npz["forgetting"])
                agent.incremental_model.cov_init = float(npz["cov_init"])
                agent.incremental_model.num_updates = int(npz["num_updates"])

        # Optimiser states
        if load_gradients:
            actor_opt = folder_p / "actor_optim.pth"
            critic_opt = folder_p / "critic_optim.pth"
            if actor_opt.exists():
                agent.actor_opt.load_state_dict(
                    torch.load(actor_opt, map_location=agent.device, weights_only=False)
                )
            if critic_opt.exists():
                agent.critic_opt.load_state_dict(
                    torch.load(
                        critic_opt, map_location=agent.device, weights_only=False
                    )
                )

        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "IMGDHPAgent":
        """Load an agent from a local directory or Hugging Face Hub.

        Args:
            repo_name: Local folder path, or ``namespace/repo_name`` on
                the Hugging Face Hub.
            access_token: Hub access token for private repos.
            version: Hub revision / branch / tag.
            load_gradients: Also restore optimiser state dicts.

        Returns:
            IMGDHPAgent: Reconstructed agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p, load_gradients=load_gradients)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(
                f"Local directory not found: '{repo_name}'." " Please check the path."
            )

        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls._load_from_dir(folder_path, load_gradients=load_gradients)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a :meth:`save` directory to the Hugging Face Hub.

        Args:
            repo_name: Target repository id, e.g. ``"me/my-imgdhp"``.
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
