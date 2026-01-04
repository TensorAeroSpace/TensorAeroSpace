"""Distributional Soft Actor-Critic (DSAC) implementation.

The agent reuses the SAC training loops and replay buffer, but swaps the
critics with distributional (quantile) heads and uses a quantile Huber
loss. Works with existing TensorAeroSpace environments (e.g., B747).
"""

import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import BaseRLModel, serialize_env
from ..sac.model import DeterministicPolicy, GaussianPolicy
from ..sac.replay_memory import ReplayMemory
from ..sac.utils import hard_update, soft_update
from .model import QuantileTwin


class DSAC(BaseRLModel):
    """Distributional Soft Actor-Critic with twin quantile critics."""

    def __init__(
        self,
        env: Any,
        *,
        updates_per_step: int = 1,
        batch_size: int = 32,
        memory_capacity: int = 1_000_000,
        lr: float = 3e-4,
        policy_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        policy_type: str = "Gaussian",
        target_update_interval: int = 1,
        automatic_entropy_tuning: bool = True,
        target_entropy_scale: float = 1.0,
        min_alpha: float = 0.0,
        exploration_noise_std: float = 0.0,
        max_grad_norm: Optional[float] = 1.0,
        reward_clip: Optional[float] = None,
        hidden_size: int = 256,
        num_quantiles: int = 32,
        embedding_dim: int = 32,
        hidden_layers: Optional[list] = None,
        layer_norm: bool = True,
        huber_threshold: float = 1.0,
        learning_starts: int = 1000,
        warmup_action_scale: float = 1.0,
        caps_lambda_smoothness: float = 400.0,
        caps_lambda_temporal: float = 400.0,
        caps_noise_std: float = 0.05,
        device: Union[str, torch.device] = "cpu",
        verbose_histogram: bool = False,
        seed: int = 42,
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 1,
    ) -> None:
        super().__init__()
        self.env = env
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha: float = float(alpha)
        self.verbose_histogram = verbose_histogram
        self.memory = ReplayMemory(memory_capacity, seed=seed)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy_type = policy_type
        self.updates_per_step = updates_per_step
        self.target_update_interval = int(target_update_interval)
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = torch.device(device)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = (
            SummaryWriter(log_dir=str(self.log_dir))
            if self.log_dir is not None
            else SummaryWriter()
        )
        self.log_every_updates = int(log_every_updates)
        if self.log_every_updates < 1:
            raise ValueError("log_every_updates must be >= 1")

        action_space = self.env.action_space
        num_inputs = int(self.env.observation_space.shape[0])
        num_actions = int(action_space.shape[0])
        # Cache action bounds for fast clipping during exploration
        try:
            self._action_low_t = torch.as_tensor(
                action_space.low, dtype=torch.float32, device=self.device
            ).view(1, -1)
            self._action_high_t = torch.as_tensor(
                action_space.high, dtype=torch.float32, device=self.device
            ).view(1, -1)
        except Exception:
            # Fallback for non-Box action spaces (should be rare here)
            self._action_low_t = torch.full((1, num_actions), -1.0, device=self.device)
            self._action_high_t = torch.full((1, num_actions), 1.0, device=self.device)

        # Quantile critics
        self.num_quantiles = int(num_quantiles)
        if self.num_quantiles < 1:
            raise ValueError("num_quantiles must be >= 1")
        self.huber_threshold = float(huber_threshold)
        self.embedding_dim = int(embedding_dim)
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
        self.layer_norm = layer_norm
        self.learning_starts = int(learning_starts)
        self.warmup_action_scale = float(warmup_action_scale)
        self.target_entropy_scale = float(target_entropy_scale)
        if self.target_entropy_scale <= 0:
            raise ValueError("target_entropy_scale must be > 0")
        self.min_alpha = float(min_alpha)
        if self.min_alpha < 0:
            raise ValueError("min_alpha must be >= 0")
        self.exploration_noise_std = float(exploration_noise_std)
        if self.exploration_noise_std < 0:
            raise ValueError("exploration_noise_std must be >= 0")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 or None")
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        if self.reward_clip is not None and self.reward_clip <= 0:
            raise ValueError("reward_clip must be > 0 or None")
        self.caps_lambda_smoothness = float(caps_lambda_smoothness)
        self.caps_lambda_temporal = float(caps_lambda_temporal)
        self.caps_noise_std = float(caps_noise_std)

        self.critic = QuantileTwin(
            num_inputs,
            num_actions,
            self.hidden_layers,
            embedding_dim=self.embedding_dim,
            layer_norm=self.layer_norm,
        ).to(device=self.device)
        self.critic_target = QuantileTwin(
            num_inputs,
            num_actions,
            self.hidden_layers,
            embedding_dim=self.embedding_dim,
            layer_norm=self.layer_norm,
        ).to(device=self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Policy (same as SAC)
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                base_target_entropy = -torch.prod(
                    torch.tensor(
                        action_space.shape, device=self.device, dtype=torch.float32
                    )
                ).item()
                # More exploration: increase magnitude (e.g., scale=2.0)
                self.target_entropy = float(base_target_entropy) * float(
                    self.target_entropy_scale
                )
                init_alpha = float(self.alpha) if self.alpha > 0 else 0.2
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                with torch.no_grad():
                    self.log_alpha.fill_(float(np.log(init_alpha)))
                self.alpha = float(init_alpha)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy: Union[GaussianPolicy, DeterministicPolicy] = GaussianPolicy(
                num_inputs, num_actions, hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)
        else:
            self.alpha = 0.0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, num_actions, hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        # Target actor (HybridRL-FlightControl keeps a full target_policy incl. actor).
        # This stabilizes targets by using a slowly-updated actor for bootstrap actions.
        self.policy_target = deepcopy(self.policy).to(self.device)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(
            1, -1
        )
        with torch.no_grad():
            if evaluate:
                _, _, action_t = self.policy.sample(state_t)
            else:
                action_t, _, _ = self.policy.sample(state_t)
                if self.exploration_noise_std > 0.0:
                    action_t = action_t + torch.randn_like(action_t) * float(
                        self.exploration_noise_std
                    )
                    action_t = torch.max(
                        torch.min(action_t, self._action_high_t), self._action_low_t
                    )
        return cast(np.ndarray, action_t.cpu().numpy()[0])

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
            raise ValueError(
                f"states must be at least 2-D. Got shape={tuple(state_t.shape)}"
            )
        if state_t.ndim > 2:
            state_t = state_t.view(state_t.shape[0], -1)
        with torch.no_grad():
            if evaluate:
                _, _, action_t = self.policy.sample(state_t)
            else:
                action_t, _, _ = self.policy.sample(state_t)
                if self.exploration_noise_std > 0.0:
                    action_t = action_t + torch.randn_like(action_t) * float(
                        self.exploration_noise_std
                    )
                    action_t = torch.max(
                        torch.min(action_t, self._action_high_t), self._action_low_t
                    )
        if return_tensor:
            return action_t
        return cast(np.ndarray, action_t.cpu().numpy())

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    @staticmethod
    def _generate_quantiles(
        batch_size: int, num_quantiles: int, device: torch.device
    ) -> torch.Tensor:
        # Uniform samples U(0,1), shape (B,Q,1)
        return torch.rand((batch_size, num_quantiles, 1), device=device)

    def _quantile_huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        taus: torch.Tensor,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """Quantile Huber regression loss (HybridRL style)."""
        # pred/target: (B,Q); taus: (B,Q,1)
        td_error = pred - target
        huber_loss = F.huber_loss(
            pred, target, reduction="none", delta=threshold
        )  # (B,Q)
        quantile_huber_loss = (
            torch.abs(taus.squeeze(-1) - (td_error.detach() < 0).float())
            * huber_loss
            / threshold
        )
        return quantile_huber_loss.sum(dim=1).mean()

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

        state_t = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(
            next_state_batch, dtype=torch.float32, device=self.device
        )
        action_t = torch.as_tensor(
            action_batch, dtype=torch.float32, device=self.device
        )
        reward_t = torch.as_tensor(
            reward_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        done_t = torch.as_tensor(
            done_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        mask_t = 1.0 - done_t

        batch_size = state_t.shape[0]
        tau_i = self._generate_quantiles(batch_size, self.num_quantiles, self.device)
        tau_j = self._generate_quantiles(batch_size, self.num_quantiles, self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy_target.sample(next_state_t)
            z1_next, z2_next = self.critic_target(next_state_t, next_action, tau_i)
            z_next = torch.min(z1_next, z2_next)
            # next_log_pi from our GaussianPolicy has shape (B, 1).
            # Broadcasting with z_next (B, Q) should produce (B, Q) — do NOT unsqueeze.
            target = reward_t + mask_t * self.gamma * (
                z_next - self.alpha * next_log_pi
            )

            # Safety: catch accidental broadcasting bugs early
            if target.ndim != 2 or target.shape != z_next.shape:
                raise RuntimeError(
                    f"DSAC target shape mismatch: target={tuple(target.shape)} z_next={tuple(z_next.shape)} "
                    f"log_pi={tuple(next_log_pi.shape)} reward={tuple(reward_t.shape)} done={tuple(done_t.shape)}"
                )

        z1, z2 = self.critic(state_t, action_t, tau_j)
        critic_loss = self._quantile_huber_loss(
            z1, target, tau_j, threshold=self.huber_threshold
        ) + self._quantile_huber_loss(z2, target, tau_j, threshold=self.huber_threshold)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.max_grad_norm
            )
        self.critic_optim.step()
        # Avoid carrying critic grads into actor backward and save compute
        self.critic_optim.zero_grad(set_to_none=True)

        # Policy update
        self._set_requires_grad(self.critic, False)
        pi, log_pi, _ = self.policy.sample(state_t)
        with torch.no_grad():
            a_tp1, _logp_tp1, _ = self.policy.sample(next_state_t)
        tau_actor = self._generate_quantiles(
            batch_size, self.num_quantiles, self.device
        )
        z1_pi, z2_pi = self.critic(state_t, pi, tau_actor)
        q_pi = torch.min(
            z1_pi.mean(dim=1, keepdim=True), z2_pi.mean(dim=1, keepdim=True)
        )

        # CAPS spatial smoothness
        lambda_smooth = self.caps_lambda_smoothness
        _, _, a_det = self.policy.sample(state_t)
        _, _, a_near = self.policy.sample(torch.normal(state_t, self.caps_noise_std))
        loss_spatial = F.mse_loss(a_det, a_near) * lambda_smooth / pi.shape[0]

        # CAPS temporal smoothness
        lambda_temporal = self.caps_lambda_temporal
        loss_temporal = F.mse_loss(pi, a_tp1) * lambda_temporal / pi.shape[0]

        policy_loss = (self.alpha * log_pi - q_pi + loss_spatial + loss_temporal).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.max_grad_norm
            )
        self.policy_optim.step()
        self._set_requires_grad(self.critic, True)

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            # Clamp alpha to keep some exploration (helps avoid "dead" deterministic policy)
            if self.min_alpha > 0.0:
                with torch.no_grad():
                    self.log_alpha.data.clamp_(min=float(np.log(self.min_alpha)))
            self.alpha = float(self.log_alpha.exp().item())
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

        if (updates % int(self.log_every_updates)) == 0:
            self.writer.add_scalar("Loss/Critic", critic_loss.item(), updates)
            self.writer.add_scalar("Loss/Policy", policy_loss.item(), updates)
            self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), updates)
            self.writer.add_scalar("Alpha/value", alpha_tlogs.item(), updates)
            # Diagnostics: decompose policy loss drivers
            try:
                self.writer.add_scalar("Train/Q_pi_mean", q_pi.mean().item(), updates)
                self.writer.add_scalar(
                    "Train/LogPi_mean", log_pi.mean().item(), updates
                )
                self.writer.add_scalar(
                    "Train/Entropy_mean", (-log_pi).mean().item(), updates
                )
                self.writer.add_scalar(
                    "Train/CAPS_spatial", loss_spatial.item(), updates
                )
                self.writer.add_scalar(
                    "Train/CAPS_temporal", loss_temporal.item(), updates
                )
                self.writer.add_scalar(
                    "Train/ActionAbsMean", pi.abs().mean().item(), updates
                )
            except Exception:
                pass

            if self.verbose_histogram:
                for name, param in self.critic.named_parameters():
                    self.writer.add_histogram(f"Critic/{name}", param, updates)
                for name, param in self.policy.named_parameters():
                    self.writer.add_histogram(f"Policy/{name}", param, updates)

        return (
            float(critic_loss.item()),
            float(critic_loss.item()),  # placeholder for compatibility (q1, q2)
            float(policy_loss.item()),
            float(alpha_loss.item()),
            float(alpha_tlogs.item()),
        )

    # Training loops follow SAC structure with calls to update_parameters
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
            done = False
            state, _ = self.env.reset()
            while not done:
                # Warmup exploration (HybridRL-FlightControl style)
                if total_numsteps < self.learning_starts:
                    # IMPORTANT:
                    # Uniform random actions in [-1, 1] can be too aggressive for some
                    # flight dynamics and flood the replay buffer with "crash" transitions.
                    # Scale them down during warmup for stability.
                    action = cast(np.ndarray, self.env.action_space.sample())
                    action = np.asarray(action, dtype=np.float32) * float(
                        self.warmup_action_scale
                    )
                    # Keep within action bounds
                    try:
                        action = np.clip(
                            action,
                            self.env.action_space.low,
                            self.env.action_space.high,
                        )
                    except Exception:
                        action = np.clip(action, -1.0, 1.0)
                else:
                    action = self.select_action(state)
                if len(self.memory) > max(self.batch_size, self.learning_starts):
                    for _ in range(self.updates_per_step):
                        self.update_parameters(self.memory, self.batch_size, updates)
                        updates += 1

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done_env = bool(terminated or truncated)
                done_bootstrap = float(bool(terminated))
                episode_steps += 1
                total_numsteps += 1
                r = float(reward)
                if self.reward_clip is not None and (not bool(terminated)):
                    # Clip only non-terminal rewards. Keep terminal penalties (e.g. -100)
                    # intact, otherwise the agent may learn to terminate early.
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

        # Keep a global step counter so successive train_vector calls continue TensorBoard steps.
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

        # Diagnostics counters between log points
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

            obs_np = cast(np.ndarray, obs.cpu().numpy())
            next_obs_np = cast(np.ndarray, next_obs.cpu().numpy())
            actions_np = cast(np.ndarray, actions_t.cpu().numpy())
            reward_np = cast(np.ndarray, reward.cpu().numpy()).reshape(-1)
            terminated_np = (
                cast(np.ndarray, terminated.cpu().numpy()).reshape(-1).astype(bool)
            )
            truncated_np = (
                cast(np.ndarray, truncated.cpu().numpy()).reshape(-1).astype(bool)
            )
            if self.reward_clip is not None:
                # Clip only non-terminal rewards (keep -100 termination penalties)
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
                    r = float(ep_returns[i])
                    l = int(ep_lengths[i])
                    returns_window[returns_ptr % len(returns_window)] = r
                    returns_ptr += 1
                    self.writer.add_scalar(
                        "Performance/EpisodeReward", r, episodes_done
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
                    median_r = 0.0
                    p10_r = 0.0
                    p90_r = 0.0
                    mean_len = 0.0
                else:
                    w = returns_window[: min(returns_ptr, len(returns_window))].astype(
                        np.float64
                    )
                    mean_r = float(np.mean(w))
                    median_r = float(np.median(w))
                    p10_r = float(np.percentile(w, 10))
                    p90_r = float(np.percentile(w, 90))
                    # Approx: mean current ep length for running envs (not just completed)
                    mean_len = float(np.mean(ep_lengths))
                self.writer.add_scalar(
                    f"Performance/MeanReward{reward_window}", mean_r, global_step
                )
                self.writer.add_scalar(
                    f"Performance/RewardMedian{reward_window}", median_r, global_step
                )
                self.writer.add_scalar(
                    f"Performance/RewardP10{reward_window}", p10_r, global_step
                )
                self.writer.add_scalar(
                    f"Performance/RewardP90{reward_window}", p90_r, global_step
                )
                self.writer.add_scalar(
                    f"Performance/MeanEpisodeLength{reward_window}",
                    mean_len,
                    global_step,
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

        # Persist updated global step for subsequent calls.
        self._global_train_vector_step = base_step + total_steps

        self.writer.flush()

    def close(self) -> None:
        try:
            self.writer.flush()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass

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
            "alpha": self.alpha,
            "verbose_histogram": self.verbose_histogram,
            "memory_capacity": self.memory.capacity,
            "policy_type": self.policy_type,
            "updates_per_step": self.updates_per_step,
            "target_update_interval": self.target_update_interval,
            "batch_size": self.batch_size,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "device": self.device.type,
            "lr": self.critic_optim.defaults["lr"],
            "target_entropy_scale": self.target_entropy_scale,
            "min_alpha": self.min_alpha,
            "exploration_noise_std": self.exploration_noise_std,
            "max_grad_norm": self.max_grad_norm,
            "reward_clip": self.reward_clip,
            "num_quantiles": self.num_quantiles,
            "embedding_dim": self.embedding_dim,
            "hidden_layers": self.hidden_layers,
            "layer_norm": self.layer_norm,
            "huber_threshold": self.huber_threshold,
            "learning_starts": self.learning_starts,
            "warmup_action_scale": self.warmup_action_scale,
            "caps_lambda_smoothness": self.caps_lambda_smoothness,
            "caps_lambda_temporal": self.caps_lambda_temporal,
            "caps_noise_std": self.caps_noise_std,
        }

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {
                "name": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "params": policy_params,
            },
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> None:
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__

        config_path = path / date_str / "config.json"
        policy_path = path / date_str / "policy.pth"
        critic_path = path / date_str / "critic.pth"
        critic_target_path = path / date_str / "critic_target.pth"
        policy_optim_path = path / date_str / "policy_optim.pth"
        critic_optim_path = path / date_str / "critic_optim.pth"
        alpha_optim_path = path / date_str / "alpha_optim.pth"
        log_alpha_path = path / date_str / "log_alpha.pth"

        policy_path.parent.mkdir(parents=True, exist_ok=True)
        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as outfile:
            import json

            json.dump(config, outfile)
        torch.save(self.policy, policy_path)
        torch.save(self.critic, critic_path)
        torch.save(self.critic_target, critic_target_path)

        if getattr(self, "automatic_entropy_tuning", False):
            torch.save({"log_alpha": self.log_alpha.detach().cpu()}, log_alpha_path)

        if save_gradients:
            try:
                torch.save(self.policy_optim.state_dict(), policy_optim_path)
                torch.save(self.critic_optim.state_dict(), critic_optim_path)
                if getattr(self, "automatic_entropy_tuning", False):
                    torch.save(self.alpha_optim.state_dict(), alpha_optim_path)
            except Exception as exc:
                raise RuntimeError(f"Error saving optimizer states: {exc}") from exc
