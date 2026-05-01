"""Generative Adversarial Imitation Learning (GAIL) agent.

This module contains the core GAIL implementation and supporting neural network
components used for imitation learning within TensorAeroSpace.
"""

import datetime
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from ..metrics import MetricWriter, create_metric_writer, schema

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def init_weights(m):
    """Initialize layer weights/biases with a normal distribution.

    Args:
        m (nn.Module): Layer/module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    """Combined policy/value network used by GAIL."""

    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        """Create actor-critic networks.

        Args:
            num_inputs (int): Input dimension.
            num_outputs (int): Output/action dimension.
            hidden_size (int): Hidden layer size.
            std (float, optional): Initial log-std scale. Defaults to 0.0.
        """
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: ``(action_distribution, value)``.
        """
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


def compute_gae(
    next_value: torch.Tensor,
    rewards: list[torch.Tensor],
    masks: list[torch.Tensor],
    values: list[torch.Tensor],
    gamma: float = 0.99,
    tau: float = 0.95,
) -> list[torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        next_value (Tensor): Value estimate for the last state.
        rewards (Tensor): Rewards.
        masks (Tensor): Terminal masks (0 for terminal, 1 otherwise).
        values (Tensor): Value estimates.
        gamma (float): Discount factor. Defaults to 0.99.
        tau (float): GAE parameter. Defaults to 0.95.

    Returns:
        list: Advantage-weighted returns (as a list of tensors).
    """
    values = values + [next_value]
    gae = torch.zeros_like(next_value)
    returns: list[torch.Tensor] = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    """Mini-batch iterator used by PPO updates."""
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :
        ], returns[rand_ids, :], advantage[rand_ids, :]


class Discriminator(nn.Module):
    """Binary classifier distinguishing expert vs. policy trajectories."""

    def __init__(self, num_inputs, hidden_size):
        """Create the discriminator network.

        Args:
            num_inputs (int): Input dimension.
            hidden_size (int): Hidden layer size.
        """
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        """Compute discriminator probability for state-action pairs."""
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        prob = torch.sigmoid(self.linear3(x))
        return prob


class GAIL:
    """Generative Adversarial Imitation Learning trainer."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        max_steps: int,
        mini_batch_size: int,
        epochs: int,
        data: np.ndarray,
        log_dir: Union[str, Path, None] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize the GAIL algorithm.

        Args:
            env: Environment instance.
            learning_rate (float): Learning rate.
            max_steps (int): Maximum steps per rollout/episode.
            mini_batch_size (int): Mini-batch size.
            epochs (int): Number of training epochs.
            data (Array): Expert demonstrations (states/actions).
            log_dir: Optional TensorBoard log directory. When provided, a
                ``MetricWriter`` is created and canonical metrics are
                emitted during :meth:`learn`. When ``None`` (default),
                no logging is performed.
        """
        self.env = env
        self.lr = learning_rate
        self.max_steps = max_steps
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.data = data

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if observation_shape is None or action_shape is None:
            raise TypeError("GAIL requires observation and action spaces with shapes.")

        self.num_inputs = int(observation_shape[0])
        self.num_outputs = int(action_shape[0])

        self.model = ActorCritic(self.num_inputs, self.num_outputs, 256).to(device)
        self.discriminator = Discriminator(self.num_inputs + self.num_outputs, 128).to(
            device
        )

        self.discrim_criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_discrim = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags
        self.wandb_config = wandb_config
        needs_writer = (
            log_dir is not None
            or wandb_project is not None
            or os.environ.get("WANDB_API_KEY") is not None
        )
        if needs_writer:
            self.writer: Optional[MetricWriter] = create_metric_writer(
                tb_log_dir=self.log_dir,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_run_name=wandb_run_name,
                wandb_tags=wandb_tags,
                wandb_config=wandb_config,
                algo="gail",
            )
        else:
            self.writer = None
        self.global_env_step = 0
        self.update_count = 0
        self.discrim_update_count = 0

    def expert_reward(self, state: torch.Tensor, action: np.ndarray) -> np.ndarray:
        """Compute imitation reward using the discriminator."""
        state_np = state.cpu().numpy()
        state_action = torch.FloatTensor(np.concatenate([state_np, action], 1)).to(
            device
        )
        return np.asarray(-np.log(self.discriminator(state_action).cpu().data.numpy()))

    def test_env(self) -> float:
        """Run one evaluation rollout and return total reward."""
        state = self.env.reset()[0].reshape(1, -1)
        done = False
        total_reward = 0.0
        for _ in range(self.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = self.model(state_tensor)
            next_state, reward, terminated, truncated, info = self.env.step(
                dist.sample().cpu().numpy()[0]
            )
            done = terminated or truncated
            next_state = next_state.reshape(1, -1)
            state = next_state
            total_reward += float(reward)
        return total_reward

    def ppo_update(
        self,
        ppo_epochs: int,
        mini_batch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        clip_param: float = 0.2,
        env_step: int = 0,
    ) -> None:
        """PPO update function.

        Args:
            ppo_epochs (int): Number of epochs.
            mini_batch_size (int): Mini-batch size.
            states (Tensor): Batch of states.
            actions (Tensor): Batch of actions.
            log_probs (Tensor): Batch of action log probabilities.
            returns (Tensor): Batch of discounted rewards.
            advantages (Tensor): Batch of advantage function values.
            clip_param (float, optional): Clipping constant.
            env_step (int, optional): Cumulative environment-step counter
                used as the x-axis for any TensorBoard scalars written
                from inside this update. Defaults to 0.
        """
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_count += 1
                if self.writer is not None:
                    self.writer.add_scalar(
                        schema.LOSS_ACTOR,
                        float(actor_loss.detach()),
                        env_step=env_step,
                    )
                    self.writer.add_scalar(
                        schema.LOSS_CRITIC,
                        float(critic_loss.detach()),
                        env_step=env_step,
                    )
                    self.writer.add_scalar(
                        schema.POLICY_ENTROPY,
                        float(entropy.detach()),
                        env_step=env_step,
                    )
                    self.writer.add_scalar(
                        schema.TRAIN_UPDATES,
                        int(self.update_count),
                        env_step=env_step,
                    )
                    self.writer.add_scalar(
                        schema.TRAIN_LR,
                        float(self.optimizer.param_groups[0]["lr"]),
                        env_step=env_step,
                    )

    def train(
        self,
        num_episodes: int = 100,
        *,
        max_steps: Optional[int] = None,
        save_best: bool = False,
        save_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Train GAIL (unified interface wrapper around :meth:`learn`).

        Args:
            num_episodes: Number of episodes. Combined with
                ``max_steps`` to produce a ``max_frames`` budget unless
                ``max_frames`` is supplied explicitly via ``**kwargs``.
            max_steps: Max steps per episode. Defaults to
                ``self.max_steps`` when not provided.
            save_best: Reserved for API consistency.
            save_path: Reserved for API consistency.
            verbose: Reserved for symmetry.
            **kwargs: GAIL-specific options:

                - ``max_frames`` (int): override the computed budget.
                - ``max_reward`` (float, default ``inf``): early-stop
                  threshold on mean test reward.

        Returns:
            dict: Empty dict (GAIL does not yet collect metrics).
        """
        _ = (save_best, save_path, verbose)
        per_ep = int(max_steps) if max_steps is not None else int(self.max_steps)
        max_frames = int(kwargs.pop("max_frames", int(num_episodes) * per_ep))
        max_reward = float(kwargs.pop("max_reward", float("inf")))
        self.learn(max_frames=max_frames, max_reward=max_reward)
        return {}

    def learn(self, max_frames: int, max_reward: float) -> None:
        """Agent training function.

        Args:
            max_frames (int): Maximum number of steps in the environment.
            max_reward (int): Reward threshold for stopping training.
        """
        self.max_frames = max_frames

        test_rewards = []
        frame_idx = 0

        i_update = 0
        state = self.env.reset()[0].reshape(1, -1)
        early_stop = False

        # Episode tracking — uses the ENVIRONMENT's reward (not the
        # discriminator-derived imitation reward). The canonical
        # rollout/episode_reward should reflect what the environment
        # actually returned so it is comparable across algorithms.
        self.global_env_step = 0
        ep_reward = 0.0
        ep_length = 0
        last_terminated = False
        last_truncated = False

        while frame_idx < max_frames and not early_stop:
            i_update += 1

            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            states: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            entropy = torch.tensor(0.0, device=device)

            for _ in range(self.max_steps):
                state = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    dist, value = self.model(state)

                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy += dist.entropy().mean()

                next_state, reward, terminated, truncated, info = self.env.step(
                    action.cpu().numpy()
                )
                self.global_env_step += 1
                ep_reward += float(np.asarray(reward).sum())
                ep_length += 1
                last_terminated = bool(terminated)
                last_truncated = bool(truncated)
                done = terminated or truncated
                next_state = next_state.reshape(1, -1)
                reward = self.expert_reward(state, action.cpu().numpy())

                log_probs.append(log_prob.detach())
                values.append(value.detach())
                rewards.append(torch.FloatTensor(reward).to(device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

                states.append(state.detach())
                actions.append(action.detach())

                state = next_state
                frame_idx += 1

                if done:
                    if self.writer is not None:
                        self.writer.log_episode(
                            reward=ep_reward,
                            length=ep_length,
                            env_step=self.global_env_step,
                            terminated=last_terminated,
                            truncated=last_truncated,
                        )
                    ep_reward = 0.0
                    ep_length = 0
                    # Many gymnasium envs do NOT auto-reset; reset so the
                    # next env.step works on a valid initial state.
                    state = self.env.reset()[0].reshape(1, -1)

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env() for _ in range(10)])
                    print(test_reward)
                    test_rewards.append(test_reward)
                    if test_reward > max_reward:
                        early_stop = True

            next_state = torch.FloatTensor(np.asarray(next_state)).to(device)
            with torch.no_grad():
                _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)

            returns_tensor = torch.cat(returns).detach()
            log_probs_tensor = torch.cat(log_probs).detach()
            values_tensor = torch.cat(values).detach()
            states_tensor = torch.cat(states)
            actions_tensor = torch.cat(actions)
            advantage = returns_tensor - values_tensor

            if i_update % 3 == 0:
                self.ppo_update(
                    4,
                    self.mini_batch_size,
                    states_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    returns_tensor,
                    advantage,
                    env_step=self.global_env_step,
                )

            expert_state_action_sample = self.data[
                np.random.randint(0, self.data.shape[0], 2 * self.max_steps * 16), :
            ]
            expert_state_action_tensor = torch.FloatTensor(
                expert_state_action_sample
            ).to(device)
            state_action = torch.cat([states_tensor, actions_tensor], 1)
            fake = self.discriminator(state_action)
            real = self.discriminator(expert_state_action_tensor)
            self.optimizer_discrim.zero_grad()
            # NOTE: Label convention here is inverted relative to the canonical
            # GAIL paper: policy (fake) -> 1, expert (real) -> 0. This is
            # intentional and consistent with `expert_reward`, which returns
            # `-log(D(s,a))`. Under this convention D ~ 0 for expert-like
            # state-action pairs, so `-log(D)` yields a HIGH reward when the
            # policy behaves like the expert. Flipping the labels without also
            # updating `expert_reward` would break training.
            discrim_loss = self.discrim_criterion(
                fake, torch.ones((states_tensor.shape[0], 1)).to(device)
            ) + self.discrim_criterion(
                real,
                torch.zeros((expert_state_action_tensor.size(0), 1)).to(device),
            )
            discrim_loss.backward()
            self.optimizer_discrim.step()

            self.discrim_update_count += 1
            if self.writer is not None:
                # Discriminator accuracy under the inverted-label convention:
                #   fake (policy) -> target 1, so D(policy) > 0.5 == "correct"
                #   real (expert) -> target 0, so D(expert) < 0.5 == "correct"
                with torch.no_grad():
                    policy_correct = (fake.detach() > 0.5).float().mean()
                    expert_correct = (real.detach() < 0.5).float().mean()
                self.writer.add_scalar(
                    schema.GAIL.LOSS_DISCRIMINATOR,
                    float(discrim_loss.detach()),
                    env_step=self.global_env_step,
                )
                self.writer.add_scalar(
                    schema.GAIL.POLICY_ACCURACY,
                    float(policy_correct),
                    env_step=self.global_env_step,
                )
                self.writer.add_scalar(
                    schema.GAIL.EXPERT_ACCURACY,
                    float(expert_correct),
                    env_step=self.global_env_step,
                )

        if self.writer is not None:
            self.writer.flush()
            self.writer.assert_contract_satisfied()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_param_env(self) -> dict:
        """Return serializable configuration of environment and policy."""
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        env_name = (
            f"{self.env.unwrapped.__class__.__module__}"
            f".{self.env.unwrapped.__class__.__name__}"
        )

        policy_params = {
            "learning_rate": self.lr,
            "max_steps": self.max_steps,
            "mini_batch_size": self.mini_batch_size,
            "epochs": self.epochs,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
        }

        return {
            "env": {"name": env_name, "params": {}},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> Path:
        """Save GAIL model to the specified directory.

        Saves actor-critic network, discriminator network, and configuration.
        Optionally saves optimizer states for resuming training.

        Args:
            path (str | Path | None): Base save directory.  If *None*, saves to
                the current working directory.
            save_gradients (bool): If True, also save optimizer state dicts.

        Returns:
            Path: The directory where the model was saved.
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        save_dir = path / f"{date_str}_{self.__class__.__name__}"
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.json"
        model_path = save_dir / "actor_critic.pth"
        discriminator_path = save_dir / "discriminator.pth"
        optimizer_path = save_dir / "optimizer.pth"
        optimizer_discrim_path = save_dir / "optimizer_discrim.pth"

        # Save configuration
        config = self.get_param_env()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save network weights
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

        # Optionally save optimizer states
        if save_gradients:
            torch.save(self.optimizer.state_dict(), optimizer_path)
            torch.save(self.optimizer_discrim.state_dict(), optimizer_discrim_path)

        return save_dir

    @classmethod
    def _load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        data: Optional[np.ndarray] = None,
        load_gradients: bool = False,
    ) -> "GAIL":
        """Load a GAIL agent from a checkpoint directory.

        Args:
            path: Directory containing saved model files.
            env: Environment instance.  Required because GAIL needs
                an environment to set up its networks.
            data: Expert demonstration data.  May be *None* if only
                inference is needed (a dummy array is created).
            load_gradients: If True, restore optimizer states.

        Returns:
            GAIL: Reconstructed agent.
        """
        path = Path(path)
        config_path = path / "config.json"
        model_path = path / "actor_critic.pth"
        discriminator_path = path / "discriminator.pth"
        optimizer_path = path / "optimizer.pth"
        optimizer_discrim_path = path / "optimizer_discrim.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        policy_params = config["policy"]["params"]

        if env is None:
            raise ValueError(
                "GAIL requires an 'env' argument when loading because "
                "the environment cannot be reconstructed from the config alone."
            )

        if data is None:
            # Create a small dummy array so the constructor does not fail.
            num_inputs = policy_params["num_inputs"]
            num_outputs = policy_params["num_outputs"]
            data = np.zeros((1, num_inputs + num_outputs), dtype=np.float32)

        new_agent = cls(
            env=env,
            learning_rate=policy_params["learning_rate"],
            max_steps=policy_params["max_steps"],
            mini_batch_size=policy_params["mini_batch_size"],
            epochs=policy_params["epochs"],
            data=data,
        )

        # Restore network weights
        map_loc = device
        new_agent.model.load_state_dict(
            torch.load(model_path, map_location=map_loc, weights_only=False)
        )
        new_agent.discriminator.load_state_dict(
            torch.load(discriminator_path, map_location=map_loc, weights_only=False)
        )

        # Optionally restore optimizer states
        if load_gradients:
            if optimizer_path.exists():
                new_agent.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=map_loc, weights_only=False)
                )
            if optimizer_discrim_path.exists():
                new_agent.optimizer_discrim.load_state_dict(
                    torch.load(
                        optimizer_discrim_path,
                        map_location=map_loc,
                        weights_only=False,
                    )
                )

        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        env: Optional[gym.Env] = None,
        data: Optional[np.ndarray] = None,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "GAIL":
        """Load pretrained model from a local directory or Hugging Face Hub.

        Args:
            repo_name: Path to a local folder **or** a Hugging Face repo id
                (e.g. ``"namespace/repo_name"``).
            env: Environment instance (required).
            data: Expert demonstration data (optional).
            access_token: Hugging Face access token for private repos.
            version: Revision / branch / tag on Hugging Face.
            load_gradients: Restore optimizer states for continued training.

        Returns:
            GAIL: Initialized agent.
        """
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load(p, env=env, data=data, load_gradients=load_gradients)

        # If it looks like an explicit filesystem path, raise immediately.
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            raise FileNotFoundError(f"Local directory not found: '{repo_name}'.")

        # Fall back to Hugging Face Hub download.
        from huggingface_hub import snapshot_download

        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls._load(folder_path, env=env, data=data, load_gradients=load_gradients)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Upload a saved model folder to Hugging Face Hub.

        Args:
            repo_name: Repository id on Hugging Face (e.g. ``"user/my-gail"``).
            folder_path: Local folder produced by :meth:`save`.
            access_token: Hugging Face access token.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )
