"""Event-triggered Dual Heuristic Programming agent.

Reference:
    Bo Sun, Cheng Liu, Killian Dally, Erik-Jan van Kampen.
    "Intelligent Aircraft Stabilization Control with Event-Triggered
    Scheme", CEAS EuroGNC 2022.

High-level algorithm
--------------------

At each discrete step ``k`` with measurement ``x_k``:

1. **Event check.**  Compare ``‖x_k − x_et‖`` against the Lipschitz
   threshold ``ρ ‖x_et‖ (1 − (2ρ)^{k − k_trig})/(1 − 2ρ)``. If the
   bound is exceeded, re-arm the trigger and drop into the inner
   actor/critic update; otherwise hold the last control and skip
   training (the computational savings come from this branch).
2. **Plant-model gradients.**  Pass ``(x, u = actor(x))`` through the
   pre-trained ``PlantModelNN`` and extract ``F = ∂f/∂x`` and
   ``G = ∂f/∂u`` with row-wise autograd.
3. **DHP update.**  The critic predicts the costate ``λ(x) = ∂J/∂x``;
   the target is ``γ · Fᵀ λ(x_{k+1}) + ∂r/∂x`` with reward
   ``r = xᵀ Q x + Y(u)`` where ``Y`` is the bounded-control integral
   cost (Abu-Khalaf–Lewis). The actor target is the closed-form
   optimal control ``u* = u_b · tanh(−½γ/u_b · R⁻¹ Gᵀ λ(x_{k+1}))``.

All autograd happens inside this module; the caller only sees a
``predict``/``learn`` interface consistent with the rest of the
``tensoraerospace`` agents.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn, optim

from ..metrics import create_metric_writer, schema
from .event_trigger import EventTrigger
from .networks import ETDHPActor, ETDHPCritic, PlantModelNN


@dataclass
class ETDHPConfig:
    """Hyper-parameters for :class:`ETDHPAgent`.

    Args:
        actor_hidden: Tanh hidden-layer sizes for the actor MLP.
        critic_hidden: Tanh hidden-layer sizes for the critic MLP.
        model_hidden: Tanh hidden-layer sizes for the plant-model MLP.
        actor_lr: SGD learning rate for the actor.
        critic_lr: SGD learning rate for the critic.
        model_lr: Adam learning rate for the plant-model pre-training.
        model_epochs: Number of epochs for the offline plant-model fit
            (``fit_plant_model``). Online fine-tuning uses one epoch
            per triggered step when enabled via ``online_model_fit``.
        online_model_fit: Whether to keep adapting the plant model on
            every triggered step. Disabled by default — the reference
            paper freezes ``f_θ`` after the offline phase.
        Q: Diagonal weights of the quadratic state cost ``xᵀ Q x``.
            Length must equal ``n_state``.
        R: Diagonal weights of the quadratic control cost. Length must
            equal ``n_control``.
        gamma: Discount factor. Default ``1.0`` (finite-horizon /
            regulation-style, as in the reference).
        num_epochs_per_trigger: Inner-loop actor/critic iterations run
            each time the event trigger fires. The reference uses 10
            on the nonlinear F-16-like plant.
        u_bound: Per-channel actuator bound fed to the bounding layer
            of the actor. Interpreted in the same units as the action
            the env expects.
        rho: Lipschitz constant for the event trigger (see
            :class:`EventTrigger`). Must lie in ``(0, 0.5)``.
        trigger_floor: Minimum trigger threshold (state units). Small
            positive values suppress noise-induced retriggers when
            ``x ≈ 0``.
        exploration_fn: Optional callable ``(time_sec) -> np.ndarray``
            of length ``n_control``. If provided, the returned vector
            is added to the actor's target control during training,
            which injects persistent excitation into the NN update
            without perturbing the control actually applied to the
            plant. ``None`` disables exploration.
        weight_init_scale: Uniform-init bound for each NN weight.
        device: Torch device.
        seed: Optional seed for numpy and torch RNGs.
    """

    actor_hidden: Sequence[int] = (10, 10)
    critic_hidden: Sequence[int] = (10, 10)
    model_hidden: Sequence[int] = (10, 10)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    model_lr: float = 1e-3
    model_epochs: int = 200
    online_model_fit: bool = False
    Q: Sequence[float] = (1.0,)
    R: Sequence[float] = (1.0,)
    gamma: float = 1.0
    num_epochs_per_trigger: int = 10
    u_bound: float = 1.0
    rho: float = 0.1
    trigger_floor: float = 1e-3
    exploration_fn: Callable[[float], np.ndarray] | None = None
    weight_init_scale: float = 0.5
    device: str = "cpu"
    seed: int | None = None
    history: dict = field(default_factory=dict)


def _bounded_integral_cost(
    d_nn: torch.Tensor, u_bound: float, R: torch.Tensor
) -> torch.Tensor:
    """Compute the Abu-Khalaf–Lewis saturated-action integral cost ``Y(u)``.

    For a bounded actor ``u = u_b · tanh(D)``, the running cost that
    preserves optimality against the ``tanh`` squash is::

        Y(u) = 2 · u_b² · diag(R) · (tanh(D)·D + ½ log(1 − tanh(D)²))

    Summing over control channels (the result is a scalar).
    """
    t = torch.tanh(d_nn)
    # log1p(−t²) is numerically safer than log(1 − t²) near the
    # saturation band, but 1 − t² = 1/cosh(D)² so log(1 − tanh(D)²)
    # = −2 · log(cosh(D)) — the form used below avoids NaNs at |D| ≫ 1.
    log_term = -2.0 * torch.log(torch.cosh(d_nn))
    Y_per = t * d_nn + 0.5 * log_term
    return 2.0 * torch.dot(R * (u_bound**2), Y_per)


def _jacobians_per_output(
    y: torch.Tensor, x_inputs: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Compute ``dy[i]/dx`` for each output dimension ``i`` and each input.

    Returns a list of Jacobians with the same length as ``x_inputs``,
    each of shape ``(len(y), x_inputs[i].numel())``.
    """
    jacs = [torch.zeros((y.numel(), x.numel()), dtype=y.dtype) for x in x_inputs]
    n_out = y.numel()
    for i in range(n_out):
        grads = torch.autograd.grad(
            y[i],
            x_inputs,
            retain_graph=(i < n_out - 1),
            create_graph=False,
            allow_unused=True,
        )
        for k, g in enumerate(grads):
            if g is not None:
                jacs[k][i] = g.reshape(-1)
    return jacs


class ETDHPAgent:
    """Event-triggered DHP agent with bounded actor and costate critic.

    The agent operates on a **regulation state** ``x̃`` — typically the
    tracking error ``y − y_ref`` possibly augmented with unmeasured
    channels regulated to zero (elevator rate, etc.). The caller is
    responsible for turning the raw environment observation into that
    regulation state via :meth:`obs_to_state`, overridable by passing
    a custom ``state_transform`` on construction.

    Args:
        n_state: Length of the regulation state vector ``x̃``.
        n_control: Number of control channels.
        state_transform: Callable ``(obs, ref, time_step) -> x̃`` that
            builds the regulation state from the raw Gym observation
            and reference. If ``None`` (default), the state is assumed
            to already be in regulation form.
        config: Optional :class:`ETDHPConfig` instance.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        state_transform: (
            Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None
        ) = None,
        config: ETDHPConfig | None = None,
        log_dir: Union[str, Path, None] = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else ETDHPConfig()
        self._state_transform = state_transform

        if len(self.cfg.Q) != self.n_state:
            raise ValueError(
                f"Q has length {len(self.cfg.Q)} but n_state is {self.n_state}"
            )
        if len(self.cfg.R) != self.n_control:
            raise ValueError(
                f"R has length {len(self.cfg.R)} but n_control is {self.n_control}"
            )

        if self.cfg.seed is not None:
            torch.manual_seed(int(self.cfg.seed))
            np.random.seed(int(self.cfg.seed))

        self.device = torch.device(self.cfg.device)

        self.actor = ETDHPActor(
            n_state=self.n_state,
            n_control=self.n_control,
            hidden_sizes=tuple(self.cfg.actor_hidden),
            u_bound=self.cfg.u_bound,
            init_scale=self.cfg.weight_init_scale,
        ).to(self.device)
        self.critic = ETDHPCritic(
            n_state=self.n_state,
            hidden_sizes=tuple(self.cfg.critic_hidden),
            init_scale=self.cfg.weight_init_scale,
        ).to(self.device)
        self.plant_model = PlantModelNN(
            n_state=self.n_state,
            n_control=self.n_control,
            hidden_sizes=tuple(self.cfg.model_hidden),
            init_scale=self.cfg.weight_init_scale,
        ).to(self.device)

        # SGD to match the reference; the update per trigger is several
        # epochs over the same single-sample batch, so Adam's adaptive
        # statistics saturate quickly and behave worse than plain SGD.
        self.actor_opt = optim.SGD(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optim.SGD(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.model_opt = optim.Adam(self.plant_model.parameters(), lr=self.cfg.model_lr)

        self.Q = torch.as_tensor(
            np.asarray(self.cfg.Q, dtype=np.float64),
            dtype=torch.float32,
            device=self.device,
        )
        self.R = torch.as_tensor(
            np.asarray(self.cfg.R, dtype=np.float64),
            dtype=torch.float32,
            device=self.device,
        )

        self.event_trigger = EventTrigger(
            rho=self.cfg.rho,
            trigger_first_step=True,
            min_floor=self.cfg.trigger_floor,
        )

        self._last_action: np.ndarray = np.zeros(self.n_control, dtype=np.float64)
        self._last_state: np.ndarray | None = None
        self._last_time_sec: float = 0.0

        self.history: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "triggered": [],
            "condition": [],
            "norm_diff": [],
        }

        # ----- Unified TensorBoard metrics -----
        # Only create the writer when an explicit log_dir is provided so the
        # default no-logging path stays a true no-op (existing tests don't
        # pass log_dir and must continue working without TB side effects).
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = (
            create_metric_writer(self.log_dir, algo="etdhp")
            if self.log_dir is not None
            else None
        )
        self.global_env_step = 0
        self.update_count = 0

    # ------------------------------------------------------------------
    # State transforms and housekeeping
    # ------------------------------------------------------------------
    def obs_to_state(
        self, obs: np.ndarray, reference_signal: np.ndarray | None, time_step: int
    ) -> np.ndarray:
        """Turn a raw env observation into the regulation state ``x̃``.

        When no custom transform is supplied this is just the identity,
        which makes sense for pure stabilisation tasks. For tracking
        tasks the user can pass a ``state_transform`` that returns the
        tracking error (possibly padded with zero-regulated channels).
        """
        if self._state_transform is not None:
            return np.asarray(
                self._state_transform(obs, reference_signal, time_step),
                dtype=np.float64,
            ).reshape(-1)
        return np.asarray(obs, dtype=np.float64).reshape(-1)

    def reset(self) -> None:
        """Re-arm the event trigger for a new episode.

        Does not touch learned weights so training accumulates across
        episodes, consistent with the rest of the tensoraerospace
        agents.
        """
        self.event_trigger.reset()
        self._last_action = np.zeros(self.n_control, dtype=np.float64)
        self._last_state = None
        self._last_time_sec = 0.0

    # ------------------------------------------------------------------
    # Plant model pre-training
    # ------------------------------------------------------------------
    def fit_plant_model(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        *,
        batch_size: int = 256,
        epochs: int | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """Train :attr:`plant_model` on offline transitions ``(x, u, x')``.

        The agent expects the state tuples in **regulation form** (i.e.
        already transformed by :meth:`obs_to_state`). Typically a PE
        roll-out is collected on the real env, converted via
        ``obs_to_state`` and then handed here.

        Args:
            states: Array of shape ``(N, n_state)``.
            actions: Array of shape ``(N, n_control)``.
            next_states: Array of shape ``(N, n_state)``.
            batch_size: Mini-batch size for Adam.
            epochs: Overrides ``cfg.model_epochs`` when set.
            verbose: Print per-epoch MSE when True.

        Returns:
            Per-epoch MSE loss trace.
        """
        xs = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        us = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        xs_next = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        assert xs.shape[1] == self.n_state
        assert us.shape[1] == self.n_control
        assert xs_next.shape == xs.shape

        n = xs.shape[0]
        num_epochs = int(epochs) if epochs is not None else int(self.cfg.model_epochs)
        losses: list[float] = []
        loss_fn = nn.MSELoss()

        for ep in range(num_epochs):
            perm = torch.randperm(n, device=self.device)
            ep_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xu = torch.cat([xs[idx], us[idx]], dim=1)
                pred = self.plant_model(xu)
                loss = loss_fn(pred, xs_next[idx])
                self.model_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.model_opt.step()
                ep_loss += float(loss.detach())
                num_batches += 1
            ep_loss /= max(1, num_batches)
            losses.append(ep_loss)
            if verbose and (ep % max(1, num_epochs // 10) == 0 or ep == num_epochs - 1):
                print(f"  [plant-model] epoch {ep + 1}/{num_epochs}  MSE={ep_loss:.3e}")
        return losses

    # ------------------------------------------------------------------
    # Online predict / learn
    # ------------------------------------------------------------------
    def predict(
        self,
        obs: np.ndarray,
        reference_signal: np.ndarray | None = None,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Compute the control action for the current measurement.

        ``deterministic`` is accepted for API consistency with other
        tensoraerospace agents but is ignored — exploration is handled
        inside :meth:`learn` via ``cfg.exploration_fn``, so the control
        that reaches the plant is always the actor's current best.
        """
        del deterministic  # kept for API compatibility
        x_tilde = self.obs_to_state(obs, reference_signal, time_step)
        self._last_state = x_tilde.copy()
        self.actor.eval()
        with torch.no_grad():
            u_t, _ = self.actor(
                torch.as_tensor(x_tilde, dtype=torch.float32, device=self.device)
            )
        u = u_t.detach().cpu().numpy().astype(np.float64)
        self._last_action = u.copy()
        return u

    def learn(
        self,
        next_obs: np.ndarray,
        reference_signal: np.ndarray | None = None,
        time_step: int = 0,
        *,
        dt: float = 1.0,
    ) -> dict[str, float]:
        """Run the event-triggered update step.

        Must be called **after** :meth:`predict` and the environment
        step. When the event-trigger threshold is not crossed this
        method is essentially free — only the Lipschitz test runs, no
        autograd and no optimiser touch.

        Args:
            next_obs: Environment measurement ``y_{k+1}``.
            reference_signal: Optional reference trajectory, forwarded
                to the state transform.
            time_step: Discrete step index ``k`` of the measurement
                (same index that was passed to :meth:`predict`).
            dt: Simulation step (s). Used only to advance the internal
                wall-clock that ``cfg.exploration_fn`` receives.

        Returns:
            Dictionary of scalar metrics: ``triggered`` (1/0),
            ``norm_diff``, ``condition``, ``actor_loss``,
            ``critic_loss``.
        """
        x_measured = self.obs_to_state(next_obs, reference_signal, time_step + 1)

        triggered = self.event_trigger.should_trigger(x_measured, time_step + 1)
        metrics = {
            "triggered": float(triggered),
            "norm_diff": float(self.event_trigger.last_norm),
            "condition": float(self.event_trigger.last_condition),
            "actor_loss": float("nan"),
            "critic_loss": float("nan"),
        }

        self._last_time_sec += dt

        if not triggered:
            self.history["triggered"].append(0.0)
            self.history["condition"].append(metrics["condition"])
            self.history["norm_diff"].append(metrics["norm_diff"])
            return metrics

        # Build the NN-facing state: the update always happens at the
        # most recently captured (non-noisy) agent state, not the noisy
        # measurement, so the critic/actor see clean data.
        x_for_update = (
            self._last_state.copy()
            if self._last_state is not None
            else x_measured.copy()
        )
        a_loss, c_loss = self._run_inner_updates(x_for_update)
        metrics["actor_loss"] = a_loss
        metrics["critic_loss"] = c_loss

        # After updating, recompute the control so it picks up the new
        # weights *immediately* (the event trigger holds the control
        # between triggers, so a stale value would be applied otherwise).
        self.actor.eval()
        with torch.no_grad():
            u_t, _ = self.actor(
                torch.as_tensor(x_for_update, dtype=torch.float32, device=self.device)
            )
        self._last_action = u_t.detach().cpu().numpy().astype(np.float64)

        self.history["triggered"].append(1.0)
        self.history["condition"].append(metrics["condition"])
        self.history["norm_diff"].append(metrics["norm_diff"])
        self.history["actor_loss"].append(a_loss)
        self.history["critic_loss"].append(c_loss)

        if self.writer is not None:
            self.update_count += 1
            self.writer.add_scalar(
                schema.LOSS_ACTOR,
                float(a_loss),
                env_step=self.global_env_step,
            )
            self.writer.add_scalar(
                schema.LOSS_CRITIC,
                float(c_loss),
                env_step=self.global_env_step,
            )
            self.writer.add_scalar(
                schema.TRAIN_UPDATES,
                int(self.update_count),
                env_step=self.global_env_step,
            )
            self.writer.add_scalar(
                schema.TRAIN_LR,
                float(self.actor_opt.param_groups[0]["lr"]),
                env_step=self.global_env_step,
            )
        return metrics

    def last_action(self) -> np.ndarray:
        """Return the control that the env should actually apply.

        Between triggers this is the same action that was last produced
        by :meth:`predict`; on triggered steps it reflects the freshly
        updated actor.
        """
        return self._last_action.copy()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def _run_inner_updates(self, x_np: np.ndarray) -> tuple[float, float]:
        """Run ``num_epochs_per_trigger`` actor/critic SGD sweeps at ``x_np``.

        Each sweep mirrors equations (4)–(8) of the reference paper:

        * Compute ``(x_next, F, G)`` via the plant model.
        * Target control ``u* = u_b·tanh(−0.5 γ/u_b · R⁻¹ Gᵀ λ(x_next))``
          and actor MSE loss against ``u* + exploration``.
        * ``λ`` target ``γ Fᵀ λ(x_next) + ∂r/∂x`` with bounded-action
          reward ``r = xᵀ Q x + Y(u)``.
        """
        loss_actor_fn = nn.MSELoss()
        loss_critic_fn = nn.MSELoss()
        x_clean = torch.as_tensor(x_np, dtype=torch.float32, device=self.device).clone()

        last_actor_loss = float("nan")
        last_critic_loss = float("nan")

        for _ in range(int(self.cfg.num_epochs_per_trigger)):
            # ----- forward through model for Jacobians ---------------
            x_in = x_clean.detach().clone().requires_grad_(True)
            u_in, _ = self.actor(x_in)
            u_in = u_in.clone().requires_grad_(True)

            xu = torch.cat([x_in, u_in])
            x_next = self.plant_model(xu)

            F_jac, G_jac = _jacobians_per_output(x_next, [x_in, u_in])

            # ----- critic / optimal control --------------------------
            x_now_det = x_clean.detach()
            x_next_det = x_next.detach()
            lam_now = self.critic(x_now_det.requires_grad_(False))
            lam_next = self.critic(x_next_det)

            R_inv = torch.diag(1.0 / self.R)
            D = -0.5 * self.cfg.gamma / self.cfg.u_bound * (R_inv @ G_jac.T @ lam_next)
            u_opt = self.cfg.u_bound * torch.tanh(D)

            # ----- actor update --------------------------------------
            u_actor, d_nn = self.actor(x_clean)
            exploration = torch.zeros_like(u_actor)
            if self.cfg.exploration_fn is not None:
                e = np.asarray(
                    self.cfg.exploration_fn(float(self._last_time_sec)),
                    dtype=np.float64,
                ).reshape(-1)
                if e.size != self.n_control:
                    raise ValueError(
                        f"exploration_fn returned shape {e.shape}, "
                        f"expected ({self.n_control},)"
                    )
                exploration = torch.as_tensor(
                    e, dtype=torch.float32, device=self.device
                )
            actor_target = (u_opt.detach() + exploration).clamp(
                -self.cfg.u_bound, self.cfg.u_bound
            )
            loss_actor = loss_actor_fn(u_actor, actor_target)

            # ----- reward + costate target ---------------------------
            # Rebuild r with grad-path through x_clean so ∂r/∂x is
            # well-defined. Recompute actor on a grad-enabled copy to
            # get D_nn as a function of x.
            x_grad = x_clean.detach().clone().requires_grad_(True)
            _, d_nn_grad = self.actor(x_grad)
            Y = _bounded_integral_cost(d_nn_grad, self.cfg.u_bound, self.R)
            reward = torch.dot(x_grad * self.Q, x_grad) + Y
            (dr_dx,) = torch.autograd.grad(
                reward, x_grad, create_graph=False, retain_graph=False
            )

            lam_target = (self.cfg.gamma * (F_jac.T @ lam_next) + dr_dx).detach()
            loss_critic = loss_critic_fn(lam_now, lam_target)

            # ----- optimiser steps -----------------------------------
            self.actor_opt.zero_grad(set_to_none=True)
            loss_actor.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad(set_to_none=True)
            loss_critic.backward()
            self.critic_opt.step()

            last_actor_loss = float(loss_actor.detach())
            last_critic_loss = float(loss_critic.detach())

        return last_actor_loss, last_critic_loss

    # ------------------------------------------------------------------
    # Convenience training loop
    # ------------------------------------------------------------------
    def train_episode(
        self,
        env: Any,
        *,
        max_steps: int | None = None,
        reference_signal: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Run one episode of ET-DHP training against a Gym env.

        This is a thin wrapper intended for scripts/notebooks; the
        more flexible path is to call :meth:`predict` / :meth:`learn`
        directly and forward ``last_action`` to the environment.

        Args:
            env: Gymnasium env exposing ``reset``/``step``/``dt``.
            max_steps: Optional cap on episode length.
            reference_signal: Override the env's reference (defaults
                to ``env.reference_signal`` when present).

        Returns:
            Summary dict: ``steps``, ``triggers``, ``ep_return``.
        """
        obs, _info = env.reset()
        self.reset()
        ref = (
            reference_signal
            if reference_signal is not None
            else getattr(env, "reference_signal", None)
        )
        dt = float(getattr(env, "dt", 1.0))

        steps = 0
        ep_return = 0.0
        triggers = 0
        done = False
        truncated = False
        while not (done or truncated):
            self.predict(obs, ref, steps)
            action = self.last_action()
            obs_next, reward, done, truncated, _info = env.step(action)
            self.global_env_step += 1
            metrics = self.learn(obs_next, ref, steps, dt=dt)
            triggers += int(metrics["triggered"])
            ep_return += float(reward)
            obs = obs_next
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        if self.writer is not None:
            self.writer.log_episode(
                reward=float(ep_return),
                length=int(steps),
                env_step=int(self.global_env_step),
                terminated=bool(done),
                truncated=bool(truncated),
            )
            self.writer.flush()
            # Don't assert_contract_satisfied here — the user may call
            # train_episode() multiple times; contract is for the entire
            # training session and is checked in train().

        return {
            "steps": float(steps),
            "triggers": float(triggers),
            "ep_return": ep_return,
            "trigger_ratio": float(triggers) / max(1, steps),
        }

    def train(
        self,
        env: Any,
        *,
        num_episodes: int = 1,
        max_steps: int | None = None,
        reference_signal: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train ET-DHP for ``num_episodes`` episodes (unified interface).

        Thin wrapper around :meth:`train_episode` that loops the requested
        number of episodes, flushes the metrics writer, and asserts the
        unified-metrics contract once the session has ended.

        Args:
            env: Gymnasium env.
            num_episodes: Episode budget.
            max_steps: Optional per-episode step cap.
            reference_signal: Optional reference signal forwarded to
                :meth:`train_episode`.

        Returns:
            Aggregated session-level summary.
        """
        summaries: list[dict[str, float]] = []
        if self.writer is not None:
            # Guarantee the mandatory train/* tier even if no event-trigger
            # ever fires (high rho / aggressive trigger_floor configs).
            self.writer.add_scalar(
                schema.TRAIN_UPDATES,
                int(self.update_count),
                env_step=int(self.global_env_step),
            )
            self.writer.add_scalar(
                schema.TRAIN_LR,
                float(self.actor_opt.param_groups[0]["lr"]),
                env_step=int(self.global_env_step),
            )
        for _ in range(int(num_episodes)):
            summary = self.train_episode(
                env=env,
                max_steps=max_steps,
                reference_signal=reference_signal,
            )
            summaries.append(summary)
        if self.writer is not None:
            self.writer.flush()
            self.writer.assert_contract_satisfied()
        return {
            "num_episodes": float(len(summaries)),
            "total_steps": float(sum(s["steps"] for s in summaries)),
            "total_triggers": float(sum(s["triggers"] for s in summaries)),
        }

    # ------------------------------------------------------------------
    # Persistence — local save / load and Hugging Face Hub round-trip
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        """Build a JSON-serialisable config for :meth:`save`.

        The ``state_transform`` callable is never serialised — on load the
        caller must supply it again via :meth:`from_pretrained` /
        :meth:`_load_from_dir`. This matches how Gymnasium treats other
        non-picklable env-side components.
        """
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        # Drop runtime-only buffer.
        cfg_dict.pop("history", None)
        # Tuples → lists for JSON stability.
        for key, value in list(cfg_dict.items()):
            if isinstance(value, tuple):
                cfg_dict[key] = list(value)
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

    def save(
        self,
        path: Union[str, Path, None] = None,
        *,
        save_gradients: bool = False,
    ) -> str:
        """Write the agent to a directory.

        Files produced:
            * ``config.json`` — constructor kwargs + serialised
              :class:`ETDHPConfig`.
            * ``actor.pth`` / ``critic.pth`` / ``plant_model.pth`` —
              PyTorch state dicts.
            * ``event_trigger.json`` — supervisor counters and the
              state captured at the last trigger.
            * ``actor_optim.pth`` / ``critic_optim.pth`` /
              ``model_optim.pth`` — optimiser state dicts (only when
              ``save_gradients=True``).

        Args:
            path: Base directory (``None`` → CWD).
            save_gradients: Persist optimiser states for resume.

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
        torch.save(self.plant_model.state_dict(), run_dir / "plant_model.pth")

        # Event trigger internal state (array → list for JSON).
        et = self.event_trigger
        et_state = {
            "rho": float(et.rho),
            "trigger_first_step": bool(et.trigger_first_step),
            "min_floor": float(et.min_floor),
            "num_triggers": int(et.num_triggers),
            "step_at_trigger": int(et.step_at_trigger),
            "norm_state_at_trigger": float(et.norm_state_at_trigger),
            "last_condition": float(et.last_condition),
            "last_norm": float(et.last_norm),
            "first_call_done": bool(et._first_call_done),
            "state_at_trigger": (
                et.state_at_trigger.tolist()
                if et.state_at_trigger is not None
                else None
            ),
        }
        with open(run_dir / "event_trigger.json", "w", encoding="utf-8") as f:
            json.dump(et_state, f, indent=2)

        if save_gradients:
            torch.save(self.actor_opt.state_dict(), run_dir / "actor_optim.pth")
            torch.save(self.critic_opt.state_dict(), run_dir / "critic_optim.pth")
            torch.save(self.model_opt.state_dict(), run_dir / "model_optim.pth")

        return str(run_dir)

    @classmethod
    def _load_from_dir(
        cls,
        folder: Union[str, Path],
        *,
        state_transform: (
            Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None
        ) = None,
        load_gradients: bool = False,
    ) -> "ETDHPAgent":
        """Reconstruct an agent from a :meth:`save` directory.

        Args:
            folder: Directory previously created by :meth:`save`.
            state_transform: Optional callable re-attached to the loaded
                agent. ``save()`` cannot persist callables; supply the
                same transform here if you used one during training.
            load_gradients: Also restore optimiser state dicts.
        """
        folder_p = Path(folder)
        config_path = folder_p / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {str(folder_p)!r}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy = cfg.get("policy", {})
        params = policy.get("params", {})
        cfg_dict = dict(policy.get("config", {}))

        # Device fallback (cuda/mps → cpu when unavailable).
        dev = str(cfg_dict.get("device", "cpu"))
        if dev.startswith("cuda") and not torch.cuda.is_available():
            cfg_dict["device"] = "cpu"
        elif dev.startswith("mps") and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            cfg_dict["device"] = "cpu"

        agent_cfg = ETDHPConfig(**cfg_dict)
        agent = cls(
            n_state=params["n_state"],
            n_control=params["n_control"],
            state_transform=state_transform,
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
        agent.plant_model.load_state_dict(
            torch.load(
                folder_p / "plant_model.pth",
                map_location=agent.device,
                weights_only=False,
            )
        )

        # Event trigger
        et_path = folder_p / "event_trigger.json"
        if et_path.exists():
            with open(et_path, "r", encoding="utf-8") as f:
                et_state = json.load(f)
            et = agent.event_trigger
            et.num_triggers = int(et_state.get("num_triggers", 0))
            et.step_at_trigger = int(et_state.get("step_at_trigger", 0))
            et.norm_state_at_trigger = float(et_state.get("norm_state_at_trigger", 0.0))
            et.last_condition = float(et_state.get("last_condition", 0.0))
            et.last_norm = float(et_state.get("last_norm", 0.0))
            et._first_call_done = bool(et_state.get("first_call_done", False))
            sat = et_state.get("state_at_trigger")
            et.state_at_trigger = (
                np.asarray(sat, dtype=np.float64) if sat is not None else None
            )

        # Optimiser states
        if load_gradients:
            for attr, fname in (
                ("actor_opt", "actor_optim.pth"),
                ("critic_opt", "critic_optim.pth"),
                ("model_opt", "model_optim.pth"),
            ):
                pth = folder_p / fname
                if pth.exists():
                    getattr(agent, attr).load_state_dict(
                        torch.load(
                            pth,
                            map_location=agent.device,
                            weights_only=False,
                        )
                    )

        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        state_transform: (
            Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None
        ) = None,
        load_gradients: bool = False,
    ) -> "ETDHPAgent":
        """Load an agent from a local directory or the Hugging Face Hub.

        Args:
            repo_name: Local folder path, or ``namespace/repo_name`` on
                the Hugging Face Hub.
            access_token: Hub access token for private repos.
            version: Hub revision / branch / tag.
            state_transform: Re-attach the observation-to-regulation
                transform used during training (optional but required
                for tracking tasks).
            load_gradients: Also restore optimiser state dicts.

        Returns:
            ETDHPAgent: Reconstructed agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(
                p,
                state_transform=state_transform,
                load_gradients=load_gradients,
            )

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(
                f"Local directory not found: '{repo_name}'." " Please check the path."
            )

        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls._load_from_dir(
            folder_path,
            state_transform=state_transform,
            load_gradients=load_gradients,
        )

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a :meth:`save` directory to the Hugging Face Hub.

        Args:
            repo_name: Target repository id, e.g. ``"me/my-etdhp"``.
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
