"""Soft Actor-Critic (SAC) algorithm implementation module.

This module contains the SAC algorithm implementation for reinforcement learning,
including the main SAC agent class with automatic entropy tuning support
and various policy types for aerospace system control.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
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
    ) -> None:
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
        self.writer = SummaryWriter()
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
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
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

    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
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
        print(policy_params)

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

    @classmethod
    def __load(
        cls,
        path: Union[str, Path],
        load_gradients: bool = False,
    ) -> "SAC":
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
            env = get_class_from_string(config["env"]["name"])(
                **config["env"]["params"]
            )
        else:
            env = get_class_from_string(config["env"]["name"])()
        new_agent = cls(env=env, **config["policy"]["params"])

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
