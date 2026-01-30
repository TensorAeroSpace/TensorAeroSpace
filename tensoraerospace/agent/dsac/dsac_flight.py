# mypy: ignore-errors
# flake8: noqa
# ruff: noqa

"""DSAC implementation ported from dsac-flight.

Goal: keep the update equations and network definitions as close as possible to
the reference repo (`dsac-flight/src/agents/dsac/dsac_agent.py`), while
preserving TensorAeroSpace's public API (train/train_vector/save).
"""

from __future__ import annotations

import datetime
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from ..metrics import create_metric_writer
from ..sac.replay_memory import ReplayMemory
from ..sac.utils import soft_update
from .flight_actor import NormalPolicyNet
from .flight_critic import ZNet
from .risk_distortions import DistortionFn, distortion_functions

CLIP_GRAD = 1.0


def calculate_huber_loss(td_error: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Huber loss (element-wise) as used in dsac-flight."""
    return torch.where(
        td_error.abs() <= k,
        0.5 * td_error.pow(2),
        k * (td_error.abs() - 0.5 * k),
    )


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softplus: log(1 + exp(x)).

    We avoid relying on torch.nn.functional.softplus here because some type
    checkers in this repo mis-detect its callability.
    """
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp(x, min=0.0)


def quantile_huber_loss(
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss (element-wise) as used in dsac-flight.

    Shapes:
        target:     (B, N)
        prediction: (B, N)
        taus:       (B, N)
    """
    td_error = target - prediction
    huber_l = calculate_huber_loss(td_error=td_error, k=kappa)
    rho = (taus - (td_error.detach() < 0).float()).abs() * huber_l / float(kappa)
    return rho.sum(dim=1).mean()


class DSAC(BaseRLModel):
    """Distributional SAC (dsac-flight port)."""

    def __init__(
        self,
        env: Any,
        *,
        updates_per_step: int = 1,
        batch_size: int = 256,
        memory_capacity: int = 500_000,
        lr: float = 4.4e-4,
        policy_lr: Optional[float] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        policy_type: str = "Gaussian",
        target_update_interval: int = 1,
        automatic_entropy_tuning: bool = True,
        target_entropy_scale: float = 1.0,
        min_alpha: float = 0.0,
        exploration_noise_std: float = 0.0,
        max_grad_norm: Optional[float] = None,
        reward_clip: Optional[float] = None,
        hidden_size: int = 64,
        num_quantiles: int = 8,
        num_quantiles_exp: Optional[int] = None,
        embedding_dim: int = 64,
        hidden_layers: Optional[list] = None,
        layer_norm: bool = True,
        huber_threshold: float = 1.0,
        learning_starts: int = 10_000,
        warmup_action_scale: float = 1.0,
        caps_lambda_smoothness: float = 400.0,
        caps_lambda_temporal: float = 400.0,
        caps_noise_std: float = 0.05,
        risk_distortion: str = "neutral",
        risk_measure: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        verbose_histogram: bool = False,
        seed: int = 42,
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 1,
    ) -> None:
        super().__init__()
        self._global_train_vector_step = 0
        self.env = env
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device(device)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = create_metric_writer(self.log_dir)

        self.verbose_histogram = bool(verbose_histogram)
        self.log_every_updates = int(log_every_updates)
        self.updates_per_step = int(updates_per_step)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.warmup_action_scale = float(warmup_action_scale)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.reward_clip = None if reward_clip is None else float(reward_clip)

        self.caps_lambda_smoothness = float(caps_lambda_smoothness)
        self.caps_lambda_temporal = float(caps_lambda_temporal)
        self.caps_noise_std = float(caps_noise_std)

        self.num_quantiles = int(num_quantiles)
        self.num_quantiles_exp = (
            int(num_quantiles_exp)
            if num_quantiles_exp is not None
            else int(self.num_quantiles)
        )
        self.embedding_dim = int(embedding_dim)
        self.huber_threshold = float(huber_threshold)

        # Keep unused args for backward-compatibility with existing scripts/configs
        self.policy_type = str(policy_type)
        self.layer_norm = bool(layer_norm)
        self.hidden_size = int(hidden_size)
        self.exploration_noise_std = float(exploration_noise_std)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.min_alpha = float(min_alpha)
        self.target_entropy_scale = float(target_entropy_scale)

        # Replay buffer
        self.memory = ReplayMemory(int(memory_capacity), seed=self.seed)

        # Infer dims
        obs_dim = int(self.env.observation_space.shape[0])
        act_dim = int(self.env.action_space.shape[0])

        # Map "hidden_layers" list into dsac-flight's (n_hidden_layers, n_hidden_units)
        if hidden_layers is None:
            n_hidden_layers = 2
            n_hidden_units = int(self.hidden_size)
        else:
            if len(hidden_layers) < 1:
                raise ValueError("hidden_layers must be non-empty when provided")
            n_hidden_layers = int(len(hidden_layers))
            n_hidden_units = int(hidden_layers[0])
            if any(int(h) != n_hidden_units for h in hidden_layers):
                raise ValueError(
                    "dsac-flight architecture expects equal hidden units per layer; "
                    f"got hidden_layers={hidden_layers}"
                )

        # Actor
        self.policy = NormalPolicyNet(
            obs_dim=obs_dim,
            action_dim=act_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
        ).to(self.device)
        self.policy_optim = Adam(
            self.policy.parameters(),
            lr=float(lr if policy_lr is None else policy_lr),
        )

        # Critics (Z1, Z2) + targets
        self.Z1 = ZNet(
            n_states=obs_dim,
            n_actions=act_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_cos=self.embedding_dim,
            device=self.device,
        ).to(self.device)
        self.Z2 = ZNet(
            n_states=obs_dim,
            n_actions=act_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_cos=self.embedding_dim,
            device=self.device,
        ).to(self.device)
        self.Z1_target = ZNet(
            n_states=obs_dim,
            n_actions=act_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_cos=self.embedding_dim,
            device=self.device,
        ).to(self.device)
        self.Z2_target = ZNet(
            n_states=obs_dim,
            n_actions=act_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            n_cos=self.embedding_dim,
            device=self.device,
        ).to(self.device)
        self.Z1_target.load_state_dict(self.Z1.state_dict())
        self.Z2_target.load_state_dict(self.Z2.state_dict())
        self.Z1_optim = Adam(self.Z1.parameters(), lr=float(lr))
        self.Z2_optim = Adam(self.Z2.parameters(), lr=float(lr))

        # Risk distortion
        if risk_distortion not in distortion_functions:
            raise ValueError(
                f"Unknown risk_distortion={risk_distortion!r}. "
                f"Valid: {sorted(distortion_functions.keys())}"
            )
        self.risk_distortion = str(risk_distortion)
        self.risk_measure = float(risk_measure)
        self.risk_distortion_fn: DistortionFn = distortion_functions[
            self.risk_distortion
        ]

        # Temperature / entropy coefficient
        self.automatic_entropy_tuning = bool(automatic_entropy_tuning)
        self.alpha = float(alpha)
        self.log_alpha: Optional[torch.Tensor] = None
        self.alpha_optim: Optional[Adam] = None

        if self.automatic_entropy_tuning:
            base_target_entropy = -float(np.prod((act_dim,)))
            self.target_entropy = float(base_target_entropy) * float(
                self.target_entropy_scale
            )
            init_alpha = float(self.alpha)
            if init_alpha <= 0.0:
                init_alpha = 0.2
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device, dtype=torch.float32
            )
            with torch.no_grad():
                self.log_alpha.fill_(float(np.log(init_alpha)))
            self.alpha_optim = Adam([self.log_alpha], lr=float(lr))
            self.alpha = float(init_alpha)
        else:
            self.target_entropy = float("nan")

    @staticmethod
    def _clip_gradient(net: torch.nn.Module) -> None:
        for p in net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-CLIP_GRAD, CLIP_GRAD)

    def _sample(
        self, state: torch.Tensor, *, reparameterize: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and log_pi (dsac-flight formula)."""
        normal = self.policy(state)
        u = normal.rsample() if reparameterize else normal.sample()
        a = torch.tanh(u)
        log_pi = normal.log_prob(u) - (
            2 * (np.log(2) - u - _softplus_stable(-2 * u))
        ).sum(dim=1)
        return a, log_pi

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(
            1, -1
        )
        with torch.no_grad():
            if evaluate:
                mean = self.policy.get_mean(state_t)
                action_t = torch.tanh(mean)
            else:
                action_t, _ = self._sample(state_t, reparameterize=False)
            if self.exploration_noise_std > 0.0:
                action_t = action_t + torch.randn_like(action_t) * float(
                    self.exploration_noise_std
                )
        return cast(np.ndarray, action_t.clamp(-1.0, 1.0).cpu().numpy()[0])

    def select_action_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        *,
        evaluate: bool = False,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if torch.is_tensor(states):
            state_t = states.to(self.device, dtype=torch.float32)
        else:
            state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if state_t.ndim < 2:
            raise ValueError(f"states must be at least 2-D, got {tuple(state_t.shape)}")
        if state_t.ndim > 2:
            state_t = state_t.view(state_t.shape[0], -1)
        with torch.no_grad():
            if evaluate:
                mean = self.policy.get_mean(state_t)
                action_t = torch.tanh(mean)
            else:
                action_t, _ = self._sample(state_t, reparameterize=False)
            if self.exploration_noise_std > 0.0:
                action_t = action_t + torch.randn_like(action_t) * float(
                    self.exploration_noise_std
                )
        action_t = action_t.clamp(-1.0, 1.0)
        if return_tensor:
            return action_t
        return cast(np.ndarray, action_t.cpu().numpy())

    def update_parameters(
        self, memory: ReplayMemory, batch_size: int, updates: int
    ) -> Tuple[float, float, float, float, float]:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = memory.sample(batch_size=batch_size)

        s = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device).view(
            -1, 1
        )
        ns = torch.as_tensor(next_state_batch, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device).view(
            -1, 1
        )

        B = int(s.shape[0])
        N = int(self.num_quantiles)
        alpha_t = float(self.alpha)

        with torch.no_grad():
            next_action, next_log_pi = self._sample(ns, reparameterize=False)
            taus_i = ZNet.generate_taus(batch_size=B, n_taus=N, device=self.device)
            taus_j = ZNet.generate_taus(batch_size=B, n_taus=N, device=self.device)

            z1_target = self.Z1_target(ns, next_action, taus_i)
            z2_target = self.Z2_target(ns, next_action, taus_i)
            min_target = torch.min(z1_target, z2_target)

            entropy_term = (alpha_t * next_log_pi).unsqueeze(-1)
            target = r + self.gamma * (1.0 - d) * (min_target - entropy_term)

        # Critic predictions and losses
        z1_pred = self.Z1(s, a, taus_j)
        z2_pred = self.Z2(s, a, taus_j)
        z1_loss = quantile_huber_loss(
            target=target, prediction=z1_pred, taus=taus_j, kappa=self.huber_threshold
        )
        z2_loss = quantile_huber_loss(
            target=target, prediction=z2_pred, taus=taus_j, kappa=self.huber_threshold
        )

        # Optimize Z1
        self.Z1_optim.zero_grad()
        z1_loss.backward()
        self._clip_gradient(self.Z1)
        self.Z1_optim.step()

        # Optimize Z2
        self.Z2_optim.zero_grad()
        z2_loss.backward()
        self._clip_gradient(self.Z2)
        self.Z2_optim.step()

        # Freeze critic nets for actor update
        for p in self.Z1.parameters():
            p.requires_grad = False
        for p in self.Z2.parameters():
            p.requires_grad = False

        # Policy update
        new_action, log_pi = self._sample(s, reparameterize=True)

        # CAPS spatial smoothness (dsac-flight style: on mean, not tanh(action))
        a_det = self.policy.get_mean(s)
        a_near = self.policy.get_mean(torch.normal(mean=s, std=self.caps_noise_std))
        loss_spatial = torch.mean((a_det - a_near) ** 2)
        loss_spatial = loss_spatial * self.caps_lambda_smoothness / new_action.shape[0]

        # CAPS temporal smoothness
        loss_temporal = torch.mean((new_action - next_action) ** 2)
        loss_temporal = loss_temporal * self.caps_lambda_temporal / new_action.shape[0]

        # Risk-distorted expectation for actor objective
        taus_exp = ZNet.generate_taus(
            batch_size=B, n_taus=int(self.num_quantiles_exp), device=self.device
        )
        taus_dist = self.risk_distortion_fn(taus_exp, self.risk_measure)
        z1_r = self.Z1(s, new_action, taus_dist)
        z2_r = self.Z2(s, new_action, taus_dist)
        q1_r = z1_r.mean(dim=1)
        q2_r = z2_r.mean(dim=1)
        Q = torch.min(q1_r, q2_r)

        policy_loss = -torch.mean(Q - alpha_t * log_pi - loss_spatial - loss_temporal)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self._clip_gradient(self.policy)
        self.policy_optim.step()

        # Unfreeze critics
        for p in self.Z1.parameters():
            p.requires_grad = True
        for p in self.Z2.parameters():
            p.requires_grad = True

        # Temperature update
        if self.automatic_entropy_tuning and self.log_alpha is not None:
            alpha_loss = -(
                self.log_alpha * (log_pi + float(self.target_entropy)).detach()
            ).mean()
            assert self.alpha_optim is not None
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            if self.min_alpha > 0.0:
                with torch.no_grad():
                    self.log_alpha.data.clamp_(min=float(np.log(self.min_alpha)))
            self.alpha = float(self.log_alpha.exp().item())
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)

        # Soft update targets
        if updates % self.target_update_interval == 0:
            with torch.no_grad():
                soft_update(self.Z1_target, self.Z1, self.tau)
                soft_update(self.Z2_target, self.Z2, self.tau)

        if (updates % int(self.log_every_updates)) == 0:
            self.writer.add_scalar("Loss/Z1", float(z1_loss.item()), updates)
            self.writer.add_scalar("Loss/Z2", float(z2_loss.item()), updates)
            self.writer.add_scalar("Loss/Policy", float(policy_loss.item()), updates)
            self.writer.add_scalar("Loss/Alpha", float(alpha_loss.item()), updates)
            self.writer.add_scalar("Alpha/value", float(alpha_tlogs.item()), updates)
            try:
                self.writer.add_scalar("Train/Q_mean", float(Q.mean().item()), updates)
                self.writer.add_scalar(
                    "Train/LogPi_mean", float(log_pi.mean().item()), updates
                )
                self.writer.add_scalar(
                    "Train/CAPS_spatial", float(loss_spatial.item()), updates
                )
                self.writer.add_scalar(
                    "Train/CAPS_temporal", float(loss_temporal.item()), updates
                )
                self.writer.add_scalar(
                    "Train/ActionAbsMean",
                    float(new_action.abs().mean().item()),
                    updates,
                )
            except Exception:
                # Keep training robust to occasional logging failures.
                pass

        return (
            float(z1_loss.item()),
            float(z2_loss.item()),
            float(policy_loss.item()),
            float(alpha_loss.item()),
            float(alpha_tlogs.item()),
        )

    # -----------------------------
    # Training loops (same API)
    # -----------------------------
    def train(self, *args, **kwargs) -> None:
        num_episodes = (
            int(args[0]) if len(args) > 0 else int(kwargs.get("num_episodes", 1))
        )
        save_best = bool(kwargs.get("save_best", False))
        save_path = kwargs.get("save_path", None)
        save_best_with_gradients = bool(kwargs.get("save_best_with_gradients", False))

        total_numsteps = 0
        updates = 0
        best_reward = float("-inf")
        for i_episode in tqdm(range(num_episodes), desc="DSAC", unit="episode"):
            episode_reward = 0.0
            episode_steps = 0
            state, _ = self.env.reset()
            done = False

            while not done:
                if total_numsteps < self.learning_starts:
                    action = cast(np.ndarray, self.env.action_space.sample())
                    action = np.asarray(action, dtype=np.float32) * float(
                        self.warmup_action_scale
                    )
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = self.select_action(state)

                if len(self.memory) >= max(self.batch_size, self.learning_starts):
                    for _ in range(int(self.updates_per_step)):
                        self.update_parameters(self.memory, self.batch_size, updates)
                        updates += 1

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done_env = bool(terminated or truncated)
                done_bootstrap = float(bool(terminated))

                episode_steps += 1
                total_numsteps += 1

                r = float(reward)
                if self.reward_clip is not None and (not bool(terminated)):
                    r = float(np.clip(r, -self.reward_clip, self.reward_clip))

                episode_reward += r
                self.memory.push(state, action, r, next_state, done_bootstrap)
                state = next_state
                done = done_env

            self.writer.add_scalar("Performance/Reward", episode_reward, i_episode)
            self.writer.add_scalar(
                "Performance/EpisodeLength", episode_steps, i_episode
            )
            self.writer.add_scalar("Train/ReplaySize", len(self.memory), i_episode)
            self.writer.add_scalar("Train/Updates", updates, i_episode)
            self.writer.add_scalar("Train/TotalSteps", total_numsteps, i_episode)

            if save_best and episode_reward > best_reward:
                best_reward = episode_reward
                self.save(path=save_path, save_gradients=save_best_with_gradients)
                self.writer.add_scalar("Performance/BestReward", best_reward, i_episode)

    def train_vector(
        self,
        *,
        total_steps: int,
        warmup_steps: int = 10_000,
        log_every: int = 2_000,
        reward_window: int = 200,
        save_best: bool = False,
        save_path: Union[str, Path, None] = None,
        save_best_with_gradients: bool = False,
    ) -> None:
        total_steps = int(total_steps)
        warmup_steps = int(warmup_steps)
        log_every = int(log_every)
        reward_window = int(reward_window)
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        base_step = int(getattr(self, "_global_train_vector_step", 0))

        obs, _ = self.env.reset()
        if not torch.is_tensor(obs):
            raise TypeError(
                "train_vector expects env.reset() to return a torch Tensor observation"
            )
        if obs.ndim != 2:
            raise ValueError(
                f"train_vector expects obs of shape (N, obs_dim). Got {tuple(obs.shape)}"
            )

        num_envs = int(obs.shape[0])
        act_dim = int(getattr(self.env.action_space, "shape", (1,))[0])

        returns_window = np.zeros((max(1, reward_window),), dtype=np.float32)
        returns_ptr = 0
        episodes_done = 0

        ep_returns = np.zeros((num_envs,), dtype=np.float32)
        ep_lengths = np.zeros((num_envs,), dtype=np.int32)

        term_count = 0
        trunc_count = 0

        updates = 0
        best_mean_return = float("-inf")
        auto_reset = bool(getattr(self.env, "auto_reset", False))

        pbar = tqdm(range(total_steps), desc="DSAC train_vector", unit="step")
        for step in pbar:
            if step < warmup_steps:
                actions_t = (
                    (2.0 * torch.rand((num_envs, act_dim), device=self.device) - 1.0)
                    * float(self.warmup_action_scale)
                ).to(dtype=torch.float32)
            else:
                actions_t = cast(
                    torch.Tensor,
                    self.select_action_batch(obs, evaluate=False, return_tensor=True),
                )

            next_obs, reward, terminated, truncated, _info = self.env.step(actions_t)
            if not (torch.is_tensor(next_obs) and torch.is_tensor(reward)):
                raise TypeError(
                    "train_vector expects env.step() to return torch tensors"
                )

            obs_np = cast(np.ndarray, obs.detach().cpu().numpy())
            next_obs_np = cast(np.ndarray, next_obs.detach().cpu().numpy())
            actions_np = cast(np.ndarray, actions_t.detach().cpu().numpy())
            reward_np = cast(np.ndarray, reward.detach().cpu().numpy()).reshape(-1)
            terminated_np = (
                cast(np.ndarray, terminated.detach().cpu().numpy())
                .reshape(-1)
                .astype(bool)
            )
            truncated_np = (
                cast(np.ndarray, truncated.detach().cpu().numpy())
                .reshape(-1)
                .astype(bool)
            )

            if self.reward_clip is not None:
                reward_np = np.where(
                    terminated_np,
                    reward_np,
                    np.clip(reward_np, -self.reward_clip, self.reward_clip),
                )

            done_np = np.logical_or(terminated_np, truncated_np)
            term_count += int(np.sum(terminated_np))
            trunc_count += int(np.sum(truncated_np))

            done_bootstrap_np = (
                done_np.astype(np.float32)
                if auto_reset
                else terminated_np.astype(np.float32)
            )

            for i in range(num_envs):
                self.memory.push(
                    obs_np[i],
                    actions_np[i],
                    float(reward_np[i]),
                    next_obs_np[i],
                    float(done_bootstrap_np[i]),
                )

            if (
                len(self.memory) >= max(self.batch_size, self.learning_starts)
                and step >= warmup_steps
            ):
                for _ in range(int(self.updates_per_step)):
                    self.update_parameters(self.memory, self.batch_size, updates)
                    updates += 1

            ep_returns += reward_np
            ep_lengths += 1
            for i, done in enumerate(done_np):
                if done:
                    r_sum = float(ep_returns[i])
                    l = int(ep_lengths[i])
                    returns_window[returns_ptr % len(returns_window)] = r_sum
                    returns_ptr += 1
                    self.writer.add_scalar(
                        "Performance/EpisodeReward", r_sum, episodes_done
                    )
                    self.writer.add_scalar(
                        "Performance/EpisodeLength", l, episodes_done
                    )
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    episodes_done += 1

            if (step + 1) % log_every == 0:
                global_step = base_step + step + 1
                if returns_ptr == 0:
                    mean_r = 0.0
                else:
                    w = returns_window[: min(returns_ptr, len(returns_window))].astype(
                        np.float64
                    )
                    mean_r = float(np.mean(w))
                self.writer.add_scalar(
                    f"Performance/MeanReward{reward_window}", mean_r, global_step
                )
                self.writer.add_scalar(
                    "Train/ReplaySize", len(self.memory), global_step
                )
                self.writer.add_scalar("Train/Updates", updates, global_step)
                self.writer.add_scalar("Train/TotalSteps", step + 1, global_step)
                self.writer.add_scalar(
                    "Diagnostics/TerminatedCount", term_count, global_step
                )
                self.writer.add_scalar(
                    "Diagnostics/TruncatedCount", trunc_count, global_step
                )
                pbar.set_postfix(
                    {
                        "mean_R": f"{mean_r:.3f}",
                        "episodes": episodes_done,
                        "updates": updates,
                        "replay": len(self.memory),
                    }
                )
                term_count = 0
                trunc_count = 0

                if save_best and mean_r > best_mean_return and episodes_done > 0:
                    best_mean_return = mean_r
                    self.save(path=save_path, save_gradients=save_best_with_gradients)
                    self.writer.add_scalar(
                        "Performance/BestMeanReward", best_mean_return, global_step
                    )

            obs = next_obs

        self._global_train_vector_step = base_step + total_steps
        self.writer.flush()

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)

        policy_params = {
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": float(self.alpha),
            "updates_per_step": self.updates_per_step,
            "target_update_interval": self.target_update_interval,
            "batch_size": self.batch_size,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "target_entropy_scale": self.target_entropy_scale,
            "min_alpha": self.min_alpha,
            "exploration_noise_std": self.exploration_noise_std,
            "reward_clip": self.reward_clip,
            "num_quantiles": self.num_quantiles,
            "num_quantiles_exp": self.num_quantiles_exp,
            "embedding_dim": self.embedding_dim,
            "hidden_layers": [self.hidden_size] * 2,
            "huber_threshold": self.huber_threshold,
            "learning_starts": self.learning_starts,
            "warmup_action_scale": self.warmup_action_scale,
            "caps_lambda_smoothness": self.caps_lambda_smoothness,
            "caps_lambda_temporal": self.caps_lambda_temporal,
            "caps_noise_std": self.caps_noise_std,
            "risk_distortion": self.risk_distortion,
            "risk_measure": self.risk_measure,
            "device": self.device.type,
            "lr": self.Z1_optim.defaults["lr"],
            "policy_lr": self.policy_optim.defaults["lr"],
            "log_every_updates": self.log_every_updates,
            "seed": self.seed,
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
        }

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {
                "name": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "params": policy_params,
            },
        }

    def save(
        self, path: Union[str, Path, None] = None, save_gradients: bool = False
    ) -> Path:
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__

        run_dir = path / date_str
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        policy_path = run_dir / "policy.pth"
        critic_path = run_dir / "critic.pth"
        critic_target_path = run_dir / "critic_target.pth"
        log_alpha_path = run_dir / "log_alpha.pth"

        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        torch.save(self.policy.state_dict(), policy_path)
        torch.save(
            {"Z1": self.Z1.state_dict(), "Z2": self.Z2.state_dict()}, critic_path
        )
        torch.save(
            {
                "Z1_target": self.Z1_target.state_dict(),
                "Z2_target": self.Z2_target.state_dict(),
            },
            critic_target_path,
        )

        if self.automatic_entropy_tuning and self.log_alpha is not None:
            torch.save({"log_alpha": self.log_alpha.detach().cpu()}, log_alpha_path)

        if save_gradients:
            torch.save(self.policy_optim.state_dict(), run_dir / "policy_optim.pth")
            torch.save(self.Z1_optim.state_dict(), run_dir / "z1_optim.pth")
            torch.save(self.Z2_optim.state_dict(), run_dir / "z2_optim.pth")
            if self.automatic_entropy_tuning and self.alpha_optim is not None:
                torch.save(self.alpha_optim.state_dict(), run_dir / "alpha_optim.pth")
        return run_dir

    def to_device(self, device: Union[str, torch.device]) -> "DSAC":
        """Move all DSAC components (nets, optim states) to the target device."""
        new_device = torch.device(device)
        self.device = new_device

        # Move networks
        self.policy = self.policy.to(new_device)
        self.Z1 = self.Z1.to(new_device)
        self.Z2 = self.Z2.to(new_device)
        self.Z1_target = self.Z1_target.to(new_device)
        self.Z2_target = self.Z2_target.to(new_device)

        # Move log_alpha tensor if it exists
        if self.automatic_entropy_tuning and self.log_alpha is not None:
            self.log_alpha = self.log_alpha.to(new_device)

        def _move_optim_state(optim: Adam) -> None:
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(new_device)

        # Move optimizer states to the same device
        _move_optim_state(self.policy_optim)
        _move_optim_state(self.Z1_optim)
        _move_optim_state(self.Z2_optim)
        if self.automatic_entropy_tuning and self.alpha_optim is not None:
            _move_optim_state(self.alpha_optim)

        return self

    def eval(self) -> "DSAC":
        """Switch all DSAC networks to eval mode."""
        self.policy.eval()
        self.Z1.eval()
        self.Z2.eval()
        self.Z1_target.eval()
        self.Z2_target.eval()
        return self

    @staticmethod
    def _filter_kwargs_for_init(
        env_cls: type, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Drop unexpected kwargs so env construction is resilient to config changes."""
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
    def __load(cls, path: Union[str, Path], load_gradients: bool = False) -> "DSAC":
        path = Path(path)
        config_path = path / "config.json"
        policy_path = path / "policy.pth"
        critic_path = path / "critic.pth"
        critic_target_path = path / "critic_target.pth"
        log_alpha_path = path / "log_alpha.pth"
        policy_optim_path = path / "policy_optim.pth"
        z1_optim_path = path / "z1_optim.pth"
        z2_optim_path = path / "z2_optim.pth"
        alpha_optim_path = path / "alpha_optim.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        agent_name = f"{cls.__module__}.{cls.__name__}"
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch

        env_name = config["env"]["name"]
        if "tensoraerospace" in env_name:
            env_cls = get_class_from_string(env_name)
            env_params = dict(config["env"].get("params", {}) or {})
            env_params = cls._filter_kwargs_for_init(env_cls, env_params)

            if "device" in env_params:
                dev = str(env_params["device"])
                if dev == "cuda" and not torch.cuda.is_available():
                    env_params["device"] = "cpu"
                if dev == "mps" and not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    env_params["device"] = "cpu"

            env = env_cls(**env_params)
        else:
            env = get_class_from_string(env_name)()

        # --- rebuild policy params with safe device fallback
        policy_params = dict(config["policy"]["params"])
        if "device" in policy_params:
            dev = str(policy_params["device"])
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            if dev == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                dev = "cpu"
            policy_params["device"] = dev

        new_agent = cls(env=env, **policy_params)

        if new_agent.device.type == "cuda" and not torch.cuda.is_available():
            new_agent.device = torch.device("cpu")
        if new_agent.device.type == "mps":
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                new_agent.device = torch.device("cpu")

        policy_state = torch.load(
            policy_path, map_location=new_agent.device, weights_only=False
        )
        new_agent.policy.load_state_dict(policy_state)

        critic_state = torch.load(
            critic_path, map_location=new_agent.device, weights_only=False
        )
        if isinstance(critic_state, dict):
            if "Z1" in critic_state:
                new_agent.Z1.load_state_dict(critic_state["Z1"])
            if "Z2" in critic_state:
                new_agent.Z2.load_state_dict(critic_state["Z2"])

        critic_target_state = torch.load(
            critic_target_path, map_location=new_agent.device, weights_only=False
        )
        if isinstance(critic_target_state, dict):
            if "Z1_target" in critic_target_state:
                new_agent.Z1_target.load_state_dict(critic_target_state["Z1_target"])
            if "Z2_target" in critic_target_state:
                new_agent.Z2_target.load_state_dict(critic_target_state["Z2_target"])

        new_agent.policy = new_agent.policy.to(new_agent.device)
        new_agent.Z1 = new_agent.Z1.to(new_agent.device)
        new_agent.Z2 = new_agent.Z2.to(new_agent.device)
        new_agent.Z1_target = new_agent.Z1_target.to(new_agent.device)
        new_agent.Z2_target = new_agent.Z2_target.to(new_agent.device)

        if new_agent.automatic_entropy_tuning and log_alpha_path.exists():
            loaded_alpha = torch.load(
                log_alpha_path, map_location=new_agent.device, weights_only=False
            )
            if (
                isinstance(loaded_alpha, dict)
                and "log_alpha" in loaded_alpha
                and new_agent.log_alpha is not None
            ):
                new_agent.log_alpha.data.copy_(
                    loaded_alpha["log_alpha"].to(new_agent.device)
                )
                new_agent.alpha = float(new_agent.log_alpha.exp().item())

        if load_gradients:
            if policy_optim_path.exists():
                st = torch.load(
                    policy_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.policy_optim.load_state_dict(st)
            if z1_optim_path.exists():
                st = torch.load(
                    z1_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.Z1_optim.load_state_dict(st)
            if z2_optim_path.exists():
                st = torch.load(
                    z2_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.Z2_optim.load_state_dict(st)
            if (
                new_agent.automatic_entropy_tuning
                and new_agent.alpha_optim is not None
                and alpha_optim_path.exists()
            ):
                st = torch.load(
                    alpha_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.alpha_optim.load_state_dict(st)

        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "DSAC":
        """Load DSAC checkpoint from local dir or Hugging Face Hub."""
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                msg = (
                    f"Local directory not found: '{repo_name}'. Please check the path."
                )
                raise FileNotFoundError(msg)
            return cls.__load(p, load_gradients=load_gradients)

        folder_path = BaseRLModel.from_pretrained(
            repo_name, access_token=access_token, version=version
        )
        return cls.__load(folder_path, load_gradients=load_gradients)

    def push_to_hub(
        self,
        repo_name: str,
        access_token: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        include_gradients: bool = False,
    ) -> str:
        """Save model checkpoint and upload it to Hugging Face Hub."""
        base_path = Path.cwd() if save_path is None else Path(str(save_path))
        base_path.mkdir(parents=True, exist_ok=True)

        run_dir = self.save(path=base_path, save_gradients=include_gradients)
        BaseRLModel().publish_to_hub(
            repo_name=repo_name,
            folder_path=str(run_dir),
            access_token=access_token,
        )
        return str(run_dir)

    def close(self) -> None:
        try:
            self.writer.flush()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass
