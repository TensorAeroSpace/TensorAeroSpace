"""Soft Actor-Critic (SAC) algorithm implementation module.

This module contains the SAC algorithm implementation for reinforcement learning,
including the main SAC agent class with automatic entropy tuning support
and various policy types for aerospace system control.
"""

import datetime
import inspect
import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from ..metrics import create_metric_writer
from .model import DeterministicPolicy, GaussianPolicy, QNetwork
from .replay_memory import ReplayMemory
from .utils import hard_update, soft_update


class SAC(BaseRLModel):
    """Soft Actor-Critic (SAC) algorithm for reinforcement learning.

    Args:
        env: Environment (Gym API compatible).
        updates_per_step (int): Updates per interaction step.
        batch_size (int): Mini-batch size.
        memory_capacity (int): Replay buffer capacity.
        lr (float): Learning rate.
        gamma (float): Discount coefficient.
        tau (float): Soft update coefficient for target network.
        alpha (float): Entropy coefficient (for policy).
        policy_type (str): Policy type ("Gaussian" or "Deterministic").
        target_update_interval (int): Target network update interval.
        automatic_entropy_tuning (bool): Automatic entropy tuning.
        hidden_size (int): Hidden layer size of networks.
        device (str | torch.device): Device for computations.
        verbose_histogram (bool): Histogram logging in TensorBoard.
        seed (int): Random number generator seed.

    Attributes:
        critic: Critic network.
        critic_optim: Optimizer for updating critic weights.
        critic_target: Target critic network.
        policy: Agent policy.
        policy_optim: Optimizer for updating policy weights.

    """

    def __init__(
        self,
        env: Any,
        updates_per_step: int = 1,
        batch_size: int = 32,
        memory_capacity: int = 10000000,
        lr: float = 0.0003,
        policy_lr: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        policy_type: str = "Gaussian",
        target_update_interval: int = 1,
        automatic_entropy_tuning: bool = False,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        verbose_histogram: bool = False,
        seed: int = 42,
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 1,
    ) -> None:
        """Initialize SAC agent, networks, replay buffer, and optimizers."""
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.alpha: float = float(alpha)
        self.verbose_histogram = verbose_histogram
        self.memory = ReplayMemory(memory_capacity, seed=seed)
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy_type = policy_type
        self.updates_per_step = updates_per_step
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.env = env
        action_space = self.env.action_space
        num_inputs = self.env.observation_space.shape[0]
        self.device = torch.device(device)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = create_metric_writer(self.log_dir)
        self.log_every_updates = int(log_every_updates)
        if self.log_every_updates < 1:
            raise ValueError("log_every_updates must be >= 1")
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Type annotation helps static checkers when assigning different policy classes
        self.policy: Union[GaussianPolicy, DeterministicPolicy]

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A)
            # (e.g., -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                # Initialize log_alpha from provided alpha for stable warm-start.
                # This prevents sudden jumps (e.g., alpha->1.0) on the first update.
                init_alpha = float(self.alpha)
                if not np.isfinite(init_alpha) or init_alpha <= 0.0:
                    init_alpha = 0.2
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                with torch.no_grad():
                    self.log_alpha.fill_(float(np.log(init_alpha)))
                self.alpha = float(init_alpha)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action based on current state.

        Args:
            state: Current state of the agent.
            evaluate (bool): Evaluation mode flag.

        Returns:
            action: Selected action.

        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if evaluate is False:
            action_t, _, _ = self.policy.sample(state_t)
        else:
            _, _, action_t = self.policy.sample(state_t)
        action_np = cast(np.ndarray, action_t.detach().cpu().numpy()[0])
        return action_np

    def select_action_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        *,
        evaluate: bool = False,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Select actions for a batch of states.

        This is the recommended API for vectorized environments.

        Args:
            states: Batch of states of shape (N, obs_dim). Can be numpy or torch.
            evaluate: If True, use deterministic evaluation action.
            return_tensor: If True, return a torch Tensor on agent device, else numpy.

        Returns:
            Actions with shape (N, act_dim).
        """
        if torch.is_tensor(states):
            state_t = states.to(self.device, dtype=torch.float32)
        else:
            state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if state_t.ndim != 2:
            raise ValueError(
                f"states must have shape (N, obs_dim). Got shape={tuple(state_t.shape)}"
            )

        with torch.no_grad():
            if evaluate:
                _, _, action_t = self.policy.sample(state_t)
            else:
                action_t, _, _ = self.policy.sample(state_t)

        if return_tensor:
            return action_t
        return cast(np.ndarray, action_t.detach().cpu().numpy())

    def update_parameters(
        self, memory: ReplayMemory, batch_size: int, updates: int
    ) -> Tuple[float, float, float, float, float]:
        """Update network parameters based on a mini-batch from memory.

        Args:
            memory: Memory for storing transitions.
            batch_size (int): Mini-batch size.
            updates (int): Number of updates.

        Returns:
            qf1_loss (float): Loss value for the first Q-network.
            qf2_loss (float): Loss value for the second Q-network.
            policy_loss (float): Loss value for the policy.
            alpha_loss (float): Loss value for the alpha coefficient.
            alpha_tlogs (float): Value of the alpha coefficient.

        """
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = memory.sample(batch_size=batch_size)

        state_batch_t = torch.as_tensor(
            state_batch, dtype=torch.float32, device=self.device
        )
        next_state_batch_t = torch.as_tensor(
            next_state_batch, dtype=torch.float32, device=self.device
        )
        action_batch_t = torch.as_tensor(
            action_batch, dtype=torch.float32, device=self.device
        )
        reward_batch_t = torch.as_tensor(
            reward_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        done_batch_t = torch.as_tensor(
            done_batch, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        mask_batch = 1.0 - done_batch_t

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch_t
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch_t, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch_t + mask_batch * self.gamma * (
                min_qf_next_target
            )
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch_t, action_batch_t)
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        # gradient clipping to prevent rare gradient spikes
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch_t)

        qf1_pi, qf2_pi = self.critic(state_batch_t, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = float(self.log_alpha.exp().item())
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha, device=self.device)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if (updates % int(self.log_every_updates)) == 0:
            self.writer.add_scalar("Loss/QF1", qf1_loss.item(), updates)
            self.writer.add_scalar("Loss/QF2", qf2_loss.item(), updates)
            self.writer.add_scalar("Loss/Policy", policy_loss.item(), updates)
            self.writer.add_scalar("Loss/Alpha", alpha_loss.item(), updates)
            self.writer.add_scalar("Alpha/value", alpha_tlogs.item(), updates)

            if self.verbose_histogram:
                for name, param in self.critic.named_parameters():
                    self.writer.add_histogram(f"Critic/{name}", param, updates)

                for name, param in self.policy.named_parameters():
                    self.writer.add_histogram(f"Policy/{name}", param, updates)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def train(self, *args, **kwargs) -> None:
        """Train SAC for the given number of episodes."""
        num_episodes = (
            int(args[0]) if len(args) > 0 else int(kwargs.get("num_episodes", 1))
        )
        save_best = bool(kwargs.get("save_best", False))
        save_path = kwargs.get("save_path", None)
        save_best_with_gradients = bool(kwargs.get("save_best_with_gradients", False))
        # Training Loop
        total_numsteps = 0
        updates = 0
        best_reward = float("-inf")
        for i_episode in tqdm(range(num_episodes)):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = self.env.reset()
            reward_per_step = []
            done = False
            while not done:
                action = self.select_action(state)
                if len(self.memory) > self.batch_size:
                    for _ in range(self.updates_per_step):
                        # Update parameters of all the networks
                        _c1, _c2, _pi, _ent, _a = self.update_parameters(
                            self.memory, self.batch_size, updates
                        )
                        updates += 1

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # Important: separate loop termination logic from bootstrap logic
                # - terminate loop when (terminated or truncated)
                # - for replay targets use done only when terminated
                done_env = bool(terminated or truncated)
                done_bootstrap = float(bool(terminated))
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                reward_per_step.append(reward)
                self.memory.push(
                    state, action, reward, next_state, done_bootstrap
                )  # Append transition to memory
                state = next_state
                done = done_env
            self.writer.add_scalar("Performance/Reward", episode_reward, i_episode)
            self.writer.add_scalar(
                "Performance/EpisodeLength", episode_steps, i_episode
            )
            if save_best and episode_reward > best_reward:
                best_reward = episode_reward
                self.save(
                    path=save_path,
                    save_gradients=save_best_with_gradients,
                )
                self.writer.add_scalar(
                    "Performance/BestReward",
                    best_reward,
                    i_episode,
                )

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
        """Train SAC on a vectorized (batched) environment.

        Expected env API (Gymnasium-like, batched):
            reset() -> (obs[N, obs_dim], info)
            step(action[N, act_dim]) -> (obs, reward[N], terminated[N], truncated[N], info)

        Notes:
            - If env has auto_reset=True, it may reset done envs internally.
              We still use terminated/truncated from the current step for episode accounting.
            - For replay bootstrap we use terminated only (not truncated), consistent with train().
        """
        total_steps = int(total_steps)
        warmup_steps = int(warmup_steps)
        log_every = int(log_every)
        reward_window = int(reward_window)
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

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

        # Episode accounting
        ep_returns = np.zeros((num_envs,), dtype=np.float32)
        ep_lengths = np.zeros((num_envs,), dtype=np.int32)
        returns_window: Deque[float] = deque(maxlen=max(1, reward_window))
        episodes_done = 0

        updates = 0
        best_mean_return = float("-inf")
        auto_reset = bool(getattr(self.env, "auto_reset", False))

        pbar = tqdm(range(total_steps), desc="SAC train_vector", unit="step")
        for step in pbar:
            # Action selection
            if step < warmup_steps:
                actions_t = (
                    2.0 * torch.rand((num_envs, act_dim), device=self.device) - 1.0
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

            # Convert tensors to numpy once per step for replay + metrics
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
            done_np = np.logical_or(terminated_np, truncated_np)
            # IMPORTANT:
            # - For plain (non-auto-reset) envs, time-limit bootstrapping is valid:
            #   use terminated only.
            # - For auto-reset vector envs, next_obs for done envs is already reset,
            #   so bootstrapping would mix episodes. Treat all done as terminal.
            done_bootstrap_np = (
                done_np.astype(np.float32)
                if auto_reset
                else terminated_np.astype(np.float32)
            )

            # Store transitions
            for i in range(num_envs):
                self.memory.push(
                    obs_np[i],
                    actions_np[i],
                    float(reward_np[i]),
                    next_obs_np[i],
                    float(done_bootstrap_np[i]),
                )

            # SAC updates
            if len(self.memory) >= self.batch_size and step >= warmup_steps:
                for _ in range(int(self.updates_per_step)):
                    self.update_parameters(self.memory, self.batch_size, updates)
                    updates += 1

            # Episode bookkeeping (based on current step's done flags)
            ep_returns += reward_np
            ep_lengths += 1
            for i, done in enumerate(done_np):
                if done:
                    r = float(ep_returns[i])
                    l = int(ep_lengths[i])
                    returns_window.append(r)
                    self.writer.add_scalar(
                        "Performance/EpisodeReward", r, episodes_done
                    )
                    self.writer.add_scalar(
                        "Performance/EpisodeLength", l, episodes_done
                    )
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    episodes_done += 1

            # Periodic summary
            if (step + 1) % log_every == 0:
                mean_r = float(np.mean(returns_window)) if len(returns_window) else 0.0
                self.writer.add_scalar(
                    f"Performance/MeanReward{reward_window}",
                    mean_r,
                    step + 1,
                )
                self.writer.add_scalar("Train/ReplaySize", len(self.memory), step + 1)
                self.writer.add_scalar("Train/Updates", updates, step + 1)
                pbar.set_postfix(
                    {
                        "mean_R": f"{mean_r:.3f}",
                        "episodes": episodes_done,
                        "updates": updates,
                        "replay": len(self.memory),
                    }
                )

                if save_best and mean_r > best_mean_return and episodes_done > 0:
                    best_mean_return = mean_r
                    self.save(path=save_path, save_gradients=save_best_with_gradients)
                    self.writer.add_scalar(
                        "Performance/BestMeanReward",
                        best_mean_return,
                        step + 1,
                    )

            obs = next_obs

        self.writer.flush()

    def close(self) -> None:
        """Flush and close TensorBoard writer."""
        try:
            self.writer.flush()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        """Return serializable configuration of environment and policy."""
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        if "tensoraerospace" in env_name:
            env_params = serialize_env(self.env)
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        # Get reference signal information if available
        try:
            ref_cls = self.env.ref_signal.__class__
            env_params["ref_signal"] = f"{ref_cls.__module__}.{ref_cls.__name__}"
        except AttributeError:
            pass

        # Add action space and observation space information
        try:
            action_space = str(self.env.action_space)
            env_params["action_space"] = action_space
        except AttributeError:
            pass

        try:
            observation_space = str(self.env.observation_space)
            env_params["observation_space"] = observation_space
        except AttributeError:
            pass

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
        }

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> None:
        """Save PyTorch model to the specified directory.

        Args:
            path (str | Path | None): Save path. If None, creates
                a folder with current date and time in the working directory.
            save_gradients (bool): Save optimizer states for
                continuing training (Adam moments, etc.).

        Returns:
            None
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)
        # Current date and time in format 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        date_str = date_str + "_" + self.__class__.__name__
        # Create path in current directory with date and time

        config_path = path / date_str / "config.json"
        policy_path = path / date_str / "policy.pth"
        critic_path = path / date_str / "critic.pth"
        critic_target_path = path / date_str / "critic_target.pth"
        policy_optim_path = path / date_str / "policy_optim.pth"
        critic_optim_path = path / date_str / "critic_optim.pth"
        alpha_optim_path = path / date_str / "alpha_optim.pth"
        log_alpha_path = path / date_str / "log_alpha.pth"

        # Create directory if it doesn't exist
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        # Save model
        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as outfile:
            json.dump(config, outfile)
        torch.save(self.policy, policy_path)
        torch.save(self.critic, critic_path)
        torch.save(self.critic_target, critic_target_path)

        # Save log_alpha if automatic entropy tuning is used
        if getattr(self, "automatic_entropy_tuning", False):
            torch.save(
                {"log_alpha": self.log_alpha.detach().cpu()},
                log_alpha_path,
            )

        # Optionally save optimizer states for resuming training
        if save_gradients:
            try:
                torch.save(self.policy_optim.state_dict(), policy_optim_path)
                torch.save(self.critic_optim.state_dict(), critic_optim_path)
                if getattr(self, "automatic_entropy_tuning", False):
                    # alpha_optim exists only when automatic entropy
                    # tuning is enabled
                    torch.save(
                        self.alpha_optim.state_dict(),
                        alpha_optim_path,
                    )
            except Exception as exc:  # protect against unexpected write errors
                raise RuntimeError(f"Error saving optimizer states: {exc}") from exc

    @staticmethod
    def _filter_kwargs_for_init(
        env_cls: type, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Filter kwargs to those accepted by env_cls.__init__.

        This makes checkpoint loading robust against extra fields stored in configs
        (e.g., action_space/observation_space metadata).
        """
        try:
            sig = inspect.signature(env_cls.__init__)
        except (TypeError, ValueError):
            return kwargs

        # If __init__ accepts **kwargs, keep everything
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
        load_gradients: bool = False,
    ) -> "SAC":
        """Load a SAC agent from checkpoint folder."""
        path = Path(path)
        config_path = path / "config.json"
        critic_path = path / "critic.pth"
        policy_path = path / "policy.pth"
        critic_target_path = path / "critic_target.pth"
        policy_optim_path = path / "policy_optim.pth"
        critic_optim_path = path / "critic_optim.pth"
        alpha_optim_path = path / "alpha_optim.pth"
        log_alpha_path = path / "log_alpha.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"

        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch
        if "tensoraerospace" in config["env"]["name"]:
            env_cls = get_class_from_string(config["env"]["name"])
            env_params = dict(config["env"].get("params", {}) or {})
            env_params = cls._filter_kwargs_for_init(env_cls, env_params)

            # Device fallback for env creation (avoid requesting cuda/mps when unavailable)
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
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])

        # If checkpoint was saved with CUDA but CUDA is not available now,
        # force CPU load to avoid torch.load failures.
        if new_agent.device.type == "cuda" and not torch.cuda.is_available():
            new_agent.device = torch.device("cpu")
        if new_agent.device.type == "mps":
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                new_agent.device = torch.device("cpu")

        # Load models
        new_agent.critic = torch.load(
            critic_path, map_location=new_agent.device, weights_only=False
        )
        new_agent.policy = torch.load(
            policy_path, map_location=new_agent.device, weights_only=False
        )
        new_agent.critic_target = torch.load(
            critic_target_path, map_location=new_agent.device, weights_only=False
        )

        # Ensure modules live on new_agent.device
        try:
            new_agent.critic = new_agent.critic.to(new_agent.device)
            new_agent.policy = new_agent.policy.to(new_agent.device)
            new_agent.critic_target = new_agent.critic_target.to(new_agent.device)
        except Exception:
            pass

        # Restore log_alpha if available
        if (
            getattr(new_agent, "automatic_entropy_tuning", False)
            and log_alpha_path.exists()
        ):
            loaded_alpha = torch.load(
                log_alpha_path, map_location=new_agent.device, weights_only=False
            )
            if isinstance(loaded_alpha, dict) and "log_alpha" in loaded_alpha:
                new_agent.log_alpha.data.copy_(
                    loaded_alpha["log_alpha"].to(new_agent.device)
                )
                new_agent.alpha = float(new_agent.log_alpha.exp().item())

        # Save current LR values before reinitializing optimizers
        critic_lr = new_agent.critic_optim.defaults.get("lr", 0.0003)
        policy_lr = new_agent.policy_optim.defaults.get("lr", 0.0003)
        alpha_lr = (
            new_agent.alpha_optim.defaults.get("lr", 0.0003)
            if getattr(new_agent, "automatic_entropy_tuning", False)
            else None
        )

        # Reinitialize optimizers for new parameters
        new_agent.critic_optim = Adam(new_agent.critic.parameters(), lr=critic_lr)
        new_agent.policy_optim = Adam(new_agent.policy.parameters(), lr=policy_lr)
        if (
            getattr(new_agent, "automatic_entropy_tuning", False)
            and alpha_lr is not None
        ):
            new_agent.alpha_optim = Adam([new_agent.log_alpha], lr=alpha_lr)

        # Optionally load optimizer states for continuing training
        if load_gradients:
            if policy_optim_path.exists():
                state = torch.load(
                    policy_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.policy_optim.load_state_dict(state)
            if critic_optim_path.exists():
                state = torch.load(
                    critic_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.critic_optim.load_state_dict(state)
            if (
                getattr(new_agent, "automatic_entropy_tuning", False)
                and alpha_optim_path.exists()
            ):
                state = torch.load(
                    alpha_optim_path, map_location=new_agent.device, weights_only=False
                )
                new_agent.alpha_optim.load_state_dict(state)
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "SAC":
        """Load pretrained model from local directory or Hugging Face.

        Args:
            repo_name: Path to local folder with weights or repository name
                in format "namespace/repo_name" on Hugging Face.
            access_token: Access token for private HF repository.
            version: Revision/branch/tag of HF repository.
            load_gradients: Load optimizer states for
                continuing training.

        Returns:
            SAC: Initialized agent.
        """
        # 1) Try local loading (absolute/relative path)
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        # 2) If path is explicitly specified (by prefix), but folder
        # doesn't exist - path error
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(
                    f"Local directory not found: '{repo_name}'."
                    " Please check the path."
                )
            return cls.__load(p, load_gradients=load_gradients)

        # 3) Otherwise - assume it's a repo id on Hugging Face (namespace/repo)
        folder_path = super().from_pretrained(repo_name, access_token, version)
        return cls.__load(folder_path, load_gradients=load_gradients)
