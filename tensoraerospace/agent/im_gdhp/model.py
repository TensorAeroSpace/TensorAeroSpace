"""Incremental GDHP with an analytically differentiated scalar critic.

Sun & van Kampen (2021), Applied Soft Computing 103, 107153,
https://doi.org/10.1016/j.asoc.2021.107153, Eqs. (31)--(38), (51)--(67).
The public observation/reference adapter supplies tracking error; the actor
and critic see only this error. A configurable history identifies the local
error dynamics, including hidden plant and reference dynamics. A history
length alone does not establish observability or persistent excitation.
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

from tensoraerospace.optimization.agent import OptimizableAgent

from .incremental_model import IncrementalModelRLS
from .networks import GDHPActor, GDHPCritic


@dataclass
class IMGDHPConfig:
    """Configuration for the published GDHP equations.

    ``beta_lambda`` is the nonnegative derivative/scalar loss ratio:
    the article's beta is ``1 / (1 + beta_lambda)``. ``obs_scale`` scales
    network inputs and the tracking cost, while the identifier and costate
    use physical error coordinates. ``control_R`` adds the article's
    quadratic input cost. ``history_length`` is the measured window M;
    1 is the full-state special case, longer windows can represent hidden
    dynamics when the observation is informative enough.

    ``identifier_mode="tracking_error"`` retains the paper's unknown-reference
    formulation. With a known, possibly discontinuous command, ``"output"``
    identifies measured outputs and subtracts the next reference from the
    prediction. This explicit known-reference extension prevents command jumps
    from being fitted as plant dynamics. The policy still sees tracking error
    only; no additional controller or observation enters the networks.

    SGD is the paper's optimizer. Adam, gradient clipping, target smoothing,
    multiple critic iterations and a critic-only warmup are optional numerical
    extensions, disabled by the paper defaults where applicable. The former
    actor-only action-rate cost is rejected because it changed the objective.
    """

    gamma: float = 0.95
    actor_hidden: Sequence[int] = (32,)
    critic_hidden: Sequence[int] = (32,)
    actor_bias_input: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 5e-3
    actor_lr_decay: float = 1.0
    critic_lr_decay: float = 1.0
    actor_lr_min: float = 0.0
    critic_lr_min: float = 0.0
    weight_limit: float = 20.0
    beta_lambda: float = 1.0
    track_Q: Sequence[float] = (1.0,)
    control_R: Sequence[float] | None = None
    action_rate_penalty: float = 0.0
    history_length: int = 1
    identifier_mode: str = "tracking_error"
    forgetting: float = 0.9995
    cov_init: float | Sequence[float] = 1e2
    warmup_steps: int = 5
    critic_only_steps: int = 0
    critic_updates_per_step: int = 1
    target_update_tau: float = 0.0
    critic_weight_decay: float = 0.0
    obs_scale: Sequence[float] | None = None
    max_grad_norm: float = float("inf")
    exploration_noise_std: float = 0.0
    u_max: float = 25.0
    optimizer: str = "sgd"
    device: str = "cpu"
    seed: int | None = None
    history: dict = field(default_factory=dict)


class IMGDHPAgent(OptimizableAgent):
    """Online error-feedback IGDHP, with predict/step/learn interaction.

    ``n_obs`` describes the environment packet. Only ``tracking_indices``
    and the corresponding references enter the policy. Set a sufficient
    ``history_length`` when these errors do not form a full Markov state.
    The learner uses a model prediction made before assimilating the new
    transition, as in Algorithm 1. Reset between independent episodes.
    """

    def __init__(
        self,
        n_obs: int,
        n_action: int,
        reference_size: int = 1,
        tracking_indices: Sequence[int] | None = None,
        config: IMGDHPConfig | None = None,
    ) -> None:
        self.n_obs, self.n_action = int(n_obs), int(n_action)
        self.reference_size = int(reference_size)
        self.tracking_indices = list(
            [0] if tracking_indices is None else tracking_indices
        )
        self.cfg = IMGDHPConfig() if config is None else config
        self._validate_config()
        self._validate_learning_rates()
        self.device = torch.device(self.cfg.device)
        self._rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        self.augmented_size = len(self.tracking_indices)
        self._obs_scale_np = (
            np.ones(self.n_obs)
            if self.cfg.obs_scale is None
            else np.asarray(self.cfg.obs_scale, dtype=float)
        )
        if (
            self._obs_scale_np.shape != (self.n_obs,)
            or not np.isfinite(self._obs_scale_np).all()
            or np.any(self._obs_scale_np <= 0)
        ):
            raise ValueError("obs_scale must contain n_obs finite positive values")
        self._scale = self._obs_scale_np[self.tracking_indices]
        common: dict[str, Any] = dict(
            in_features=self.augmented_size, input_scale=self._scale
        )
        self.actor = GDHPActor(
            **common,
            n_u=self.n_action,
            hidden_sizes=self.cfg.actor_hidden,
            u_max=self.cfg.u_max,
            bias_input=self.cfg.actor_bias_input,
        ).to(self.device)
        self.critic = GDHPCritic(
            **common, n_y=self.augmented_size, hidden_sizes=self.cfg.critic_hidden
        ).to(self.device)
        self.target_critic = GDHPCritic(
            **common, n_y=self.augmented_size, hidden_sizes=self.cfg.critic_hidden
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        optimizer = optim.SGD if self.cfg.optimizer == "sgd" else optim.Adam
        self.actor_opt = optimizer(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = optimizer(
            self.critic.parameters(),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.critic_weight_decay,
        )
        self.incremental_model = IncrementalModelRLS(
            self.augmented_size,
            self.n_action,
            forgetting=self.cfg.forgetting,
            cov_init=self.cfg.cov_init,
            seed=self.cfg.seed,
            history_length=self.cfg.history_length,
        )
        # Section 5.2 prior: identity error-increment blocks and zero input map.
        self.incremental_model.theta[:] = 0
        self.incremental_model.theta[
            : self.cfg.history_length * self.augmented_size
        ] = np.tile(np.eye(self.augmented_size), (self.cfg.history_length, 1))
        self._R = (
            np.zeros(self.n_action)
            if self.cfg.control_R is None
            else np.asarray(self.cfg.control_R, dtype=float)
        )
        if (
            self._R.shape != (self.n_action,)
            or not np.isfinite(self._R).all()
            or np.any(self._R < 0)
        ):
            raise ValueError(
                "control_R must contain n_action nonnegative finite weights"
            )
        self._y_tm1: np.ndarray | None = None
        self._u_tm1: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        self._last_augmented: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None
        self._total_steps = 0
        self.history: dict[str, list[float]] = {
            key: []
            for key in (
                "episode_return",
                "critic_loss",
                "actor_loss",
                "rls_pred_error_norm",
            )
        }

    def _validate_config(self):
        if min(self.n_obs, self.n_action, self.reference_size) <= 0:
            raise ValueError("observation, action and reference sizes must be positive")
        if (
            not self.tracking_indices
            or len(set(self.tracking_indices)) != len(self.tracking_indices)
            or any(i < 0 or i >= self.n_obs for i in self.tracking_indices)
        ):
            raise ValueError(
                "tracking_indices must be unique valid observation indices"
            )
        if self.reference_size not in (1, len(self.tracking_indices)):
            raise ValueError(
                "reference_size must be one or the number of tracked channels"
            )
        q = np.asarray(self.cfg.track_Q, dtype=float)
        if (
            q.shape != (len(self.tracking_indices),)
            or not np.isfinite(q).all()
            or np.any(q < 0)
        ):
            raise ValueError(
                "track_Q must contain a nonnegative finite weight per tracked channel"
            )
        if (
            not 0 <= self.cfg.gamma <= 1
            or not np.isfinite(self.cfg.beta_lambda)
            or self.cfg.beta_lambda < 0
        ):
            raise ValueError("gamma must be in [0,1] and beta_lambda nonnegative")
        if self.cfg.action_rate_penalty != 0:
            raise ValueError(
                "action_rate_penalty is not part of the paper objective; use 0 and control_R for input cost"
            )
        if self.cfg.identifier_mode not in ("tracking_error", "output"):
            raise ValueError("identifier_mode must be tracking_error or output")
        if self.cfg.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be sgd or adam")
        if not 0 <= self.cfg.target_update_tau <= 1:
            raise ValueError("target_update_tau must be in [0,1]")

    def _validate_learning_rates(self):
        for name in ("actor", "critic"):
            rate = getattr(self.cfg, name + "_lr")
            floor = getattr(self.cfg, name + "_lr_min")
            decay = getattr(self.cfg, name + "_lr_decay")
            if (
                not np.isfinite([rate, floor, decay]).all()
                or not 0 <= floor <= rate
                or not 0 < decay <= 1
            ):
                raise ValueError(
                    f"{name} needs 0 <= lr_min <= lr and 0 < lr_decay <= 1, all finite"
                )
        if not np.isfinite(self.cfg.weight_limit) or self.cfg.weight_limit <= 0:
            raise ValueError("weight_limit must be finite and positive")

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _reference_at(self, reference_signal, time_step):
        reference = np.asarray(reference_signal, dtype=float)
        if reference.ndim == 1:
            reference = reference.reshape(1, -1)
        if (
            reference.ndim != 2
            or reference.shape[0] != self.reference_size
            or not reference.shape[1]
            or not np.isfinite(reference).all()
        ):
            raise ValueError("reference must have finite shape (reference_size, T)")
        return reference[:, int(np.clip(time_step, 0, reference.shape[1] - 1))].copy()

    def _augment(self, y, ref):
        """Adapt an environment packet to the physical tracking-error vector."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.n_obs or not np.isfinite(y).all():
            raise ValueError("observation must contain n_obs finite values")
        return y[self.tracking_indices] - np.asarray(ref).reshape(-1)

    def _augment_torch(self, y, ref):
        """Torch version of the physical tracking-error adapter."""
        return y[self.tracking_indices] - ref.reshape(-1)

    def reset(self):
        """Clear transition/history buffers while retaining learned parameters."""
        self._y_tm1 = self._u_tm1 = None
        self._last_action = self._last_augmented = self._last_obs = None
        self.incremental_model.reset()

    def retune_actor_inputs(
        self, *, feedback_gain: float = 1.0, bias_input: float | None = None
    ) -> None:
        """Retune a learned actor between experiments without adding a controller.

        ``feedback_gain`` scales the first-layer error weights, changing the
        policy's error sensitivity. Changing ``bias_input`` inversely rescales
        its first-layer weights, preserving the zero-error command while
        changing subsequent SGD sensitivity to that constant input. With gain
        one, the entire current policy is preserved (up to rounding).

        This is an explicit tuning operation, not an online step of the paper
        algorithm. The critic and identifier are unchanged. Optimizer moments
        for the transformed layer are cleared; reset before a new rollout.
        The new weights and bias scale are included in ordinary checkpoints.
        """
        bias = self.cfg.actor_bias_input if bias_input is None else float(bias_input)
        gain = float(feedback_gain)
        if not np.isfinite([gain, bias]).all() or min(gain, bias) <= 0:
            raise ValueError("feedback_gain and bias_input must be finite and positive")
        if self._last_action is not None:
            raise RuntimeError(
                "finish the pending predict/learn transition before retuning"
            )
        layer = self.actor.backbone[0] if len(self.actor.backbone) else self.actor.head
        weights = layer.weight.detach().clone()
        weights[:, :-1] *= gain
        weights[:, -1] *= self.actor.bias_input / bias
        if (
            not torch.isfinite(weights).all()
            or weights.abs().max() > self.cfg.weight_limit
        ):
            raise ValueError("retuned actor input weights exceed weight_limit")
        with torch.no_grad():
            layer.weight.copy_(weights)
        self.actor.bias_input = self.cfg.actor_bias_input = bias
        self.actor_opt.state.pop(layer.weight, None)

    def predict(self, obs, reference_signal, time_step, *, deterministic=False):
        """Compute a bounded action; learning happens after the environment step."""
        error = self._augment(obs, self._reference_at(reference_signal, time_step))
        with torch.no_grad():
            action = self.actor(self._tensor(error)).cpu().numpy().astype(float)
        if not deterministic:
            action += (
                self._rng.normal(size=self.n_action) * self.cfg.exploration_noise_std
            )
        action = np.clip(action, -self.cfg.u_max, self.cfg.u_max)
        self._last_action, self._last_augmented = action.copy(), error.copy()
        self._last_obs = np.asarray(obs, dtype=float).reshape(-1).copy()
        return action

    def learn(self, next_obs, reference_signal, time_step, *, applied_action=None):
        """Update the networks, then identify the just-measured transition.

        ``applied_action`` can report a clipped physical command in policy
        units. The history must describe the same input as the model mapping.
        """
        if (
            self._last_action is None
            or self._last_augmented is None
            or self._last_obs is None
        ):
            raise RuntimeError("learn() called before predict()")
        action = (
            self._last_action.copy()
            if applied_action is None
            else np.asarray(applied_action, dtype=float).reshape(-1)
        )
        if action.shape != (self.n_action,) or not np.isfinite(action).all():
            raise ValueError("applied_action must contain n_action finite values")
        error = self._last_augmented.copy()
        next_error = self._augment(
            next_obs, self._reference_at(reference_signal, time_step + 1)
        )
        model_current, model_next = error, next_error
        if self.cfg.identifier_mode == "output":
            model_current = self._last_obs[self.tracking_indices]
            model_next = np.asarray(next_obs, dtype=float).reshape(-1)[
                self.tracking_indices
            ]
        previous = model_current if self._y_tm1 is None else self._y_tm1
        previous_action = action if self._u_tm1 is None else self._u_tm1
        prediction = self.incremental_model.predict_next(
            model_current, previous, action, previous_action
        )
        if self.cfg.identifier_mode == "output":
            prediction -= self._reference_at(reference_signal, time_step + 1)
        cost = float(
            np.sum(np.asarray(self.cfg.track_Q) * (error * self._scale) ** 2)
            + np.sum(self._R * action**2)
        )
        metrics = dict(
            critic_loss=float("nan"),
            actor_loss=float("nan"),
            rls_pred_error_norm=float("nan"),
        )
        if self._total_steps >= self.cfg.warmup_steps:
            # Freeze the prior policy derivatives before either network changes.
            policy_jacobian = torch.autograd.functional.jacobian(
                self.actor, self._tensor(error)
            ).detach()
            # The input-cost derivative uses the same applied command as c,
            # including additive exploration (held fixed for the derivative).
            policy_action = self._tensor(action)
            actor_update = (
                self._total_steps >= self.cfg.warmup_steps + self.cfg.critic_only_steps
                and self._y_tm1 is not None
            )
            if actor_update:
                metrics["actor_loss"] = self._actor_update(
                    self._last_obs,
                    self._reference_at(reference_signal, time_step),
                    self._reference_at(reference_signal, time_step + 1),
                    previous_action,
                    previous,
                )
            for _ in range(max(1, int(self.cfg.critic_updates_per_step))):
                metrics["critic_loss"] = self._critic_update(
                    aug_t_np=error,
                    y_next_np=prediction,
                    ref_next_np=None,
                    c_now_value=cost,
                    err_now_np=error * self._scale,
                    policy_jacobian=policy_jacobian,
                    policy_action=policy_action,
                )
        if self._y_tm1 is not None and self._u_tm1 is not None:
            residual = self.incremental_model.update(
                self._y_tm1, model_current, model_next, self._u_tm1, action
            )
            metrics["rls_pred_error_norm"] = float(np.linalg.norm(residual))
        self._y_tm1, self._u_tm1 = model_current.copy(), action.copy()
        self._last_action = self._last_augmented = self._last_obs = None
        self._total_steps += 1
        return metrics

    def _soft_update_target(self):
        """Optional target smoothing, disabled in the paper profile."""
        if self.cfg.target_update_tau <= 0:
            return
        with torch.no_grad():
            for target, source in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.lerp_(source, self.cfg.target_update_tau)

    def _critic_update(
        self,
        aug_t_np,
        y_next_np,
        ref_next_np,
        c_now_value,
        err_now_np,
        *,
        policy_jacobian=None,
        policy_action=None,
    ):
        """Eqs. (54)--(61): Bellman residual and its full policy derivative."""
        error = self._tensor(aug_t_np)
        bootstrap = (
            self.target_critic if self.cfg.target_update_tau > 0 else self.critic
        )
        with torch.no_grad():
            j_next, lam_next = bootstrap(self._tensor(y_next_np))
        if policy_jacobian is None:
            policy_jacobian = torch.autograd.functional.jacobian(
                self.actor, error
            ).detach()
        with torch.no_grad():
            action = self.actor(error) if policy_action is None else policy_action
        p = self.augmented_size
        transition = torch.eye(p, device=self.device) + self._tensor(
            self.incremental_model.A[:, :p]
        )
        transition += (
            self._tensor(self.incremental_model.B[:, : self.n_action]) @ policy_jacobian
        )
        dc = self._tensor(2 * np.asarray(self.cfg.track_Q) * err_now_np * self._scale)
        dc += policy_jacobian.T @ (2 * self._tensor(self._R) * action)
        lambda_target = dc + self.cfg.gamma * transition.T @ lam_next
        j_target = c_now_value + self.cfg.gamma * j_next.squeeze()
        j, lam = self.critic(error)
        beta = 1 / (1 + self.cfg.beta_lambda)
        loss = 0.5 * (
            beta * (j.squeeze() - j_target).square()
            + (1 - beta) * (lam - lambda_target).square().sum()
        )
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_opt.step()
        self._finish_update(
            self.critic,
            self.critic_opt,
            self.cfg.critic_lr_decay,
            self.cfg.critic_lr_min,
        )
        self._soft_update_target()
        return float(loss.detach())

    def _actor_update(self, y_t_np, ref_now_np, ref_next_np, u_prev_np, y_prev_np):
        """Eqs. (65)--(67): minimize half the squared predicted cost-to-go."""
        error = self._augment(y_t_np, ref_now_np)
        action = self.actor(self._tensor(error))
        model = self.incremental_model
        model_current = error
        if self.cfg.identifier_mode == "output":
            model_current = np.asarray(y_t_np, dtype=float)[self.tracking_indices]
        base = model.predict_next(
            model_current, y_prev_np, np.zeros(self.n_action), u_prev_np
        )
        if self.cfg.identifier_mode == "output":
            base -= ref_next_np
        predicted = (
            self._tensor(base) + self._tensor(model.B[:, : self.n_action]) @ action
        )
        # Derivatives of J flow through the identified input map to the actor.
        j, _ = self.critic(predicted)
        loss = 0.5 * j.square().sum()
        self.actor_opt.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, tuple(self.actor.parameters()))
        for parameter, gradient in zip(self.actor.parameters(), gradients):
            parameter.grad = gradient
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()
        self._finish_update(
            self.actor, self.actor_opt, self.cfg.actor_lr_decay, self.cfg.actor_lr_min
        )
        return float(loss.detach())

    def _finish_update(self, network, optimizer, decay, minimum):
        """Section 5.2: bounded weights and configurable descending step sizes."""
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.clamp_(-self.cfg.weight_limit, self.cfg.weight_limit)
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"] * decay, minimum)

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
            if isinstance(value, np.ndarray):
                cfg_dict[key] = value.tolist()
            elif isinstance(value, tuple):
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
                "implementation_version": 2,
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
            * ``training_state.json`` — transition history, learning
              counter and agent-local exploration RNG state.
            * ``actor_optim.pth`` / ``critic_optim.pth`` — optimiser
              state dicts (only when ``save_gradients=True``).

        Args:
            path: Base directory. If ``None``, uses CWD.
            save_gradients: Persist optimiser states for continuation
                with the saved transition history and exploration stream.
                Reset the agent before starting a new environment episode.

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

        state: dict[str, Any] = {
            "total_steps": self._total_steps,
            "rng_state": self._rng.bit_generator.state,
            "model_dy_history": [value.tolist() for value in rls.dy_history],
            "model_du_history": [value.tolist() for value in rls.du_history],
        }
        for name in (
            "_y_tm1",
            "_u_tm1",
            "_last_action",
            "_last_augmented",
            "_last_obs",
        ):
            value = getattr(self, name)
            state[name] = value.tolist() if value is not None else None
        with open(run_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

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
        if policy.get("implementation_version") != 2:
            raise ValueError(
                "Legacy independent-costate IM-GDHP checkpoints require retraining with the paper implementation"
            )
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
                covariance = npz["cov_init"]
                agent.incremental_model.cov_init = (
                    float(covariance) if covariance.ndim == 0 else covariance.copy()
                )
                agent.incremental_model.num_updates = int(npz["num_updates"])

        agent._restore_training_state(folder_p)

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

    def _restore_training_state(self, folder: Path) -> None:
        """Restore transition history and exploration; accept legacy checkpoints."""
        state_path = folder / "training_state.json"
        if not state_path.exists():
            return
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._total_steps = int(state["total_steps"])
        self._rng.bit_generator.state = state["rng_state"]
        self.incremental_model.dy_history = [
            np.asarray(v, dtype=float) for v in state.get("model_dy_history", [])
        ]
        self.incremental_model.du_history = [
            np.asarray(v, dtype=float) for v in state.get("model_du_history", [])
        ]
        for name in (
            "_y_tm1",
            "_u_tm1",
            "_last_action",
            "_last_augmented",
            "_last_obs",
        ):
            value = state[name]
            setattr(
                self,
                name,
                np.asarray(value, dtype=np.float64) if value is not None else None,
            )

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
