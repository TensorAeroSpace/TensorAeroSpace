"""Deep Q-Network (DQN) implementation and utilities.

This module implements a DQN agent and supporting data structures (e.g., replay
buffer helpers) used in TensorAeroSpace.
"""

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from ..metrics import create_metric_writer, schema

# Select device for computation
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """DQN with two hidden layers of 32.

    Only the number of actions is required to preserve the original
    signature. The first linear layer is lazily initialized to infer
    the input features at runtime from the first forward pass.

    Args:
        num_actions (int): Number of actions.
    """

    def __init__(self, num_actions: int) -> None:
        """Initialize network layers.

        Args:
            num_actions: Number of discrete actions.
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of observations."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Forward function. Returns Q-values for actions.

        Args:
            inputs: Batch of input data (numpy array [B, obs_dim]).

        Returns:
            numpy array [B, num_actions] with Q-values.
        """

        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(inputs).float().to(device)
            q = self.forward(x)
        return q.detach().cpu().numpy()

    def action_value(
        self, obs: np.ndarray
    ) -> Tuple[Union[np.ndarray, int], np.ndarray]:
        """Select greedy action and return Q-values for the first item.

        Args:
            obs: Batch of input data.

        Returns:
            best_action: Best action(s). If batch size is 1 -> int,
                else ndarray.
            q_values: Q-values of the first element in the batch for
                compatibility.
        """

        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return (
            best_action if best_action.shape[0] > 1 else best_action[0],
            q_values[0],
        )


def test_model():
    """Function to test model functionality."""

    env = gym.make("CartPole-v1")
    action_space = cast(Discrete, env.action_space)
    num_actions = action_space.n
    print("num_actions: ", num_actions)
    model = Model(num_actions)

    obs, _info = env.reset()
    print("obs_shape: ", obs.shape)

    best_action, q_values = model.action_value(obs[None])
    print("res of test model: ", best_action)
    print(q_values)


class SumTree:
    """Binary search tree class for prioritized replay buffer agent.

    Args:
        capacity (int): Buffer size.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize sum tree for prioritized replay.

        Args:
            capacity: Maximum number of transitions to store.
        """
        # buffer size; number of leaves in sum tree
        self.capacity = capacity
        # number of nodes in sum tree
        self.tree = np.zeros(2 * capacity - 1)
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):
        """Number of records in buffer.

        Returns:
            (int): Number of records in buffer.
        """

        return self.tree[0]

    def add(self, priority: float, transition: Any) -> None:
        """Function for adding object to buffer.

        Args:
            priority (int): Priority of added transition.
            transition: Transition vector S, A, R, S'.
        """

        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx: int, priority: float) -> None:
        """Function for updating object priority with given index.

        Args:
            idx (int): Transition index.
            priority (int): Priority of updated transition.
        """

        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)  # O(logn)

    def _propagate(self, idx: int, change: float) -> None:
        """Function for backward priority update in tree.

        Args:
            idx (int): Transition index.
            priority (int): Priority of updated transition.
        """

        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s: float) -> tuple[int, float, Any]:
        """Function for getting object by given priority.

        Args:
            s (int): Priority by which transition is selected.

        Returns:
            idx (int): Transition index.
            priority (int): Priority of updated transition.
            transitions: Required transition.
        """

        idx = self._retrieve(0, s)  # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx: int, s: float) -> int:
        """Function for searching object by given priority and index.

        Args:
            idx (int): Index where search is currently performed.
            s (int): Priority by which transition is selected.

        Returns:
            idx (int): Index of found transition.
        """

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


class DQNAgent:
    """DQN Agent.

    Args:
        model (torch.nn.Module): Deep Q-network model.
        target_model (torch.nn.Module): Target deep Q-network model.
        env (gym.Env): Gym/Gymnasium environment.
        learning_rate (float, optional): Learning rate.
        epsilon (float, optional): Environment exploration probability.
        epsilon_dacay (float, optional): Epsilon reduction coefficient per episode.
        min_epsilon (float, optional): Minimum epsilon value.
        gamma (float, optional): Discount coefficient.
        batch_size (int, optional): Mini-batch size.
        target_update_iter (int, optional): Target network update period (steps).
        train_nums (int, optional): Number of training steps.
        buffer_size (int, optional): Replay buffer size.
        replay_period (int, optional): Buffer sampling period.
        alpha (float, optional): Prioritization degree.
        beta (float, optional): Importance sampling coefficient.
        beta_increment_per_sample (float, optional): Beta increment per sample.
    """

    def __init__(
        self,
        model: Any,
        target_model: Any,
        env: Any,
        learning_rate: float = 0.0012,
        epsilon: float = 0.1,
        epsilon_dacay: float = 0.995,
        min_epsilon: float = 0.01,
        gamma: float = 0.9,
        batch_size: int = 8,
        target_update_iter: int = 400,
        train_nums: int = 5000,
        buffer_size: int = 200,
        replay_period: int = 20,
        alpha: float = 0.4,
        beta: float = 0.4,
        beta_increment_per_sample: float = 0.001,
        log_dir: str | None = None,
        verbose_histogram: bool = False,
        seed: int = 1,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize DQN agent and replay buffer.

        Args:
            model: Online Q-network.
            target_model: Target Q-network.
            env: Gym/Gymnasium environment.
            learning_rate: Optimizer learning rate.
            epsilon: Initial epsilon for exploration.
            epsilon_dacay: Multiplicative epsilon decay.
            min_epsilon: Minimum epsilon value.
            gamma: Discount factor.
            batch_size: Training batch size.
            target_update_iter: Steps between target updates.
            train_nums: Total training steps to run.
            buffer_size: Replay buffer capacity.
            replay_period: Sampling period from buffer.
            alpha: PER priority exponent.
            beta: PER importance sampling exponent.
            beta_increment_per_sample: Increment for beta per sample.
            log_dir: Directory for TensorBoard logs.
            verbose_histogram: Whether to log histograms extensively.
            seed: Random seed for numpy and torch. Defaults to 1.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Models and optimizer
        self.device = _DEVICE
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags
        self.wandb_config = wandb_config
        self.writer = create_metric_writer(
            tb_log_dir=self.log_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            algo="dqn",
        )

        # parameters
        self.env = env  # gym environment
        self.lr = learning_rate  # learning step
        self.epsilon = epsilon  # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay  # epsilon decay rate
        self.min_epsilon = min_epsilon  # minimum epsilon
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size  # minibatch k
        self.target_update_iter = target_update_iter  # target network update period
        self.train_nums = train_nums  # total training steps

        # replay buffer params [(s, a, r, ns, done), ...]
        self.b_obs = np.empty((self.batch_size,) + self.env.observation_space.shape)
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty(
            (self.batch_size,) + self.env.observation_space.shape
        )
        self.b_dones = np.empty(self.batch_size, dtype=np.bool_)

        self.replay_buffer = SumTree(buffer_size)  # sum-tree data structure
        self.buffer_size = buffer_size  # replay buffer size N
        self.replay_period = replay_period  # replay period K
        self.alpha = alpha  # priority parameter
        self.beta = beta  # importance sampling parameter
        self.beta_increment_per_sample = beta_increment_per_sample
        self.num_in_buffer = 0  # total number of transitions stored in buffer
        self.margin = 0.01  # pi = |td_error| + margin
        self.p1 = 1  # initialize priority for the first transition
        # self.is_weight = np.empty((None, 1))
        self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
        self.abs_error_upper = 1
        self.verbose_histogram = verbose_histogram
        self.global_step = 0
        self.global_env_step = 0
        self.episode_idx = 0

    def train(
        self,
        num_episodes: Optional[int] = None,
        *,
        max_steps: Optional[int] = None,
        save_best: bool = False,
        save_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Train the DQN agent (unified interface).

        DQN is frame-based internally. ``num_episodes`` is treated as an
        episode target that is multiplied by ``max_steps`` to derive a
        total training-step budget. When ``max_steps`` is omitted, the
        original ``self.train_nums`` budget set at construction is used.

        Args:
            num_episodes: Target number of episodes. Converted to a
                step budget via ``num_episodes * max_steps`` when
                ``max_steps`` is provided.
            max_steps: Approximate maximum steps per episode.
            save_best: Reserved for API consistency.
            save_path: Reserved for API consistency.
            verbose: Reserved for symmetry.
            **kwargs: Currently unused.

        Returns:
            dict: Empty dict for API compatibility.
        """
        _ = (save_best, save_path, verbose, kwargs)
        if num_episodes is not None and max_steps is not None:
            self.train_nums = int(num_episodes) * int(max_steps)
        elif max_steps is not None:
            # Allow overriding the step cap without touching num_episodes.
            self.train_nums = int(max_steps)
        obs, _info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        pbar = tqdm(range(1, self.train_nums), desc="DQNAgent Train", unit="step")
        recent_loss = None
        for t in pbar:
            self.global_env_step = t
            input_obs = obs.reshape([1, -1])
            best_action, _q_values = self.model.action_value(input_obs)
            # input the obs to the network model
            action = self.get_action(best_action)  # get the real action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            episode_length += 1
            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity :])
            self.store_transition(
                p, obs, action, reward, next_obs, done
            )  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.buffer_size:
                # if t % self.replay_period == 0:  # transition sampling and update
                recent_loss = self.train_step()
                if t % 200 == 0 and recent_loss is not None:
                    pbar.set_postfix(
                        {"loss": f"{recent_loss:.4f}", "eps": f"{self.epsilon:.3f}"}
                    )

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                # Episode end logging (canonical schema)
                self.writer.log_episode(
                    reward=float(episode_reward),
                    length=int(episode_length),
                    env_step=int(self.global_env_step),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                self.writer.add_scalar(
                    schema.DQN.EPSILON,
                    float(self.epsilon),
                    env_step=self.global_env_step,
                )
                self.episode_idx += 1
                episode_reward = 0.0
                episode_length = 0
                obs, info = self.env.reset()
            else:
                obs = next_obs
        self.writer.flush()
        # Note: don't assert_contract_satisfied() if max_steps is so small that the
        # warmup never ends and no train_step() ever runs. The smoke test uses a
        # small-but-sufficient budget that does cross the warmup threshold.
        self.writer.assert_contract_satisfied()
        return {"episodes": int(self.episode_idx)}

    def train_step(self) -> Any:
        """Function for training step.

        Returns:
            losses (float): Losses after one training step.
        """

        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)

        # Convert batch to tensors
        b_obs = torch.from_numpy(self.b_obs).float().to(self.device)
        b_next = torch.from_numpy(self.b_next_states).float().to(self.device)
        actions = torch.from_numpy(self.b_actions).long().to(self.device)
        rewards = torch.from_numpy(self.b_rewards).float().to(self.device)
        dones = (
            torch.from_numpy(self.b_dones.astype(np.float32)).float().to(self.device)
        )
        is_w = torch.from_numpy(self.is_weight).float().to(self.device).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_target = self.target_model(b_next)
            next_actions_online = self.model(b_next).argmax(dim=1)
            next_q_sa = next_q_target.gather(
                1, next_actions_online.unsqueeze(1)
            ).squeeze(1)
            q_target_sa = rewards + self.gamma * next_q_sa * (1.0 - dones)

        # Predicted Q for current states & chosen actions
        q_pred = self.model(b_obs)  # [B, A]
        q_pred_sa = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # PER weighted MSE loss
        loss = (is_w * (q_pred_sa - q_target_sa).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities based on TD-error
        with torch.no_grad():
            abs_td_error = (q_target_sa - q_pred_sa).abs().detach().cpu().numpy()
        clipped_error = np.where(
            abs_td_error < self.abs_error_upper,
            abs_td_error,
            self.abs_error_upper,
        )
        ps = np.power(clipped_error + self.margin, self.alpha)
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, float(p))

        # TensorBoard logging
        q_pred_sa_mean = float(q_pred_sa.detach().mean().item())
        q_target_sa_mean = float(q_target_sa.detach().mean().item())
        td_err_mean = float(np.mean(abs_td_error)) if abs_td_error.size > 0 else 0.0
        td_err_max = float(np.max(abs_td_error)) if abs_td_error.size > 0 else 0.0
        td_err_min = float(np.min(abs_td_error)) if abs_td_error.size > 0 else 0.0

        self.writer.add_scalar(
            schema.DQN.LOSS_Q,
            float(loss.item()),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.Q_PRED_SA_MEAN,
            q_pred_sa_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.Q_TARGET_SA_MEAN,
            q_target_sa_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MEAN,
            td_err_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MAX,
            td_err_max,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MIN,
            td_err_min,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.PER_BETA,
            float(self.beta),
            env_step=self.global_env_step,
        )

        self.global_step += 1
        self.writer.add_scalar(
            schema.TRAIN_UPDATES,
            self.global_step,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.TRAIN_LR,
            float(self.optimizer.param_groups[0]["lr"]),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.TRAIN_REPLAY_SIZE,
            int(self.num_in_buffer),
            env_step=self.global_env_step,
        )

        if (
            self.verbose_histogram
            and self.global_step > 0
            and (self.global_step % 1000 == 0)
        ):
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(
                    f"weights/q/{name}",
                    param,
                    env_step=self.global_env_step,
                )

        return float(loss.item())

    def sum_tree_sample(self, k: int):
        """Get batch for training.

        Args:
            k (int): Size of batch to get.

        Returns:
            idxes (int): Indices of objects from batch.
            is_weights (float): Priorities of objects from batch.
        """

        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        # Only consider filled leaves (buffer may not be full yet, so many leaves are 0).
        leaves = self.replay_buffer.tree[-self.replay_buffer.capacity :]
        filled_leaves = leaves[leaves > 0]
        if len(filled_leaves) == 0:
            min_prob = 1.0  # uniform fallback
        else:
            min_prob = np.min(filled_leaves) / self.replay_buffer.total_p
        min_prob = max(min_prob, 1e-8)  # guard against division by zero
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            (
                self.b_obs[i],
                self.b_actions[i],
                self.b_rewards[i],
                self.b_next_states[i],
                self.b_dones[i],
            ) = t
            # P(j)
            sampling_probabilities = (
                p / self.replay_buffer.total_p
            )  # where p = p ** self.alpha
            is_weights[i, 0] = (
                np.power(self.buffer_size * sampling_probabilities, -self.beta)
                / max_weight
            )
        return idxes, is_weights

    def evaluation(self, wrapped_env: Any, render: bool = False) -> float:
        """Get batch for training.

        Args:
            wrapped_env: Wrapped environment (for rendering/frame capture).
            render (bool, optional): Whether to visualize environment or not.

        Returns:
            ep_reward (float): Total reward per episode.
        """

        obs, _info = wrapped_env.env.reset()
        done = False
        # one episode until done
        ep_reward = 0
        ep_length = 0
        while not done:
            input_obs = obs.reshape([1, -1])
            action, _q_values = self.model.action_value(
                input_obs
            )  # Using [None] to extend its dimension (4,) -> (1, 4)
            next_obs, reward, terminated, truncated, _info = wrapped_env.env.step(
                action
            )
            done = bool(terminated or truncated)
            obs = next_obs
            ep_reward += reward
            ep_length += 1
            if render:  # visually show
                wrapped_env.env.render()
                wrapped_env.capture_frame()
            time.sleep(0.05)
        wrapped_env.close()
        # Log evaluation reward / length (canonical schema)
        self.writer.add_scalar(
            schema.EVAL_EPISODE_REWARD,
            float(ep_reward),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.EVAL_EPISODE_LENGTH,
            int(ep_length),
            env_step=self.global_env_step,
        )
        return ep_reward

    def close(self) -> None:
        """Close resources (e.g., SummaryWriter)."""
        self.writer.close()

    def store_transition(
        self,
        priority: float,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""

        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)

    # rank-based prioritization sampling
    def rand_based_sample(self, k):
        """Placeholder for rank-based prioritized sampling (not implemented)."""
        pass

    # e-greedy
    def get_action(self, best_action: int) -> int:
        """Epsilon-greedy action selection."""

        if np.random.rand() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(best_action)

    # assign the current network parameters to target network
    def update_target_model(self) -> None:
        """Target neural network update function."""

        self.target_model.load_state_dict(self.model.state_dict())
        # Log target update event (canonical schema)
        self.writer.add_scalar(
            schema.DQN.TARGET_UPDATE,
            1,
            env_step=self.global_env_step,
        )

    def get_target_value(self, obs: np.ndarray) -> np.ndarray:
        """Compute Q-values using the target network."""
        return cast(np.ndarray, self.target_model.predict(obs))

    def e_decay(self) -> None:
        """Function for reducing network exploration probability."""

        self.epsilon *= self.epsilon_decay

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> None:
        """Save PyTorch models to the specified directory.

        Args:
            path (str | Path | None): Save path. If None, saves into current cwd
                under folder "dqn_agent".
            save_gradients (bool): Save optimizer states to continue training.
        """
        if path is None:
            path = Path.cwd() / "dqn_agent"
        else:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # File paths
        model_path = path / "model.pth"
        target_model_path = path / "target_model.pth"
        optim_path = path / "optimizer.pth"
        config_path = path / "config.json"

        # Save models (full objects to preserve lazy layers)
        torch.save(self.model, model_path)
        torch.save(self.target_model, target_model_path)

        # Optionally save optimizer state
        if save_gradients:
            torch.save(self.optimizer.state_dict(), optim_path)

        # Minimal config for reload convenience
        config = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "batch_size": self.batch_size,
            "target_update_iter": self.target_update_iter,
            "train_nums": self.train_nums,
            "buffer_size": self.buffer_size,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Any,
        *,
        load_gradients: bool = False,
    ) -> "DQNAgent":
        """Load a DQNAgent from a directory created by :meth:`save`.

        Args:
            path: Directory that contains ``model.pth``, ``target_model.pth``
                and ``config.json`` (as written by :meth:`save`).
            env: A Gymnasium-compatible environment. The environment is
                required because it is not serialised alongside the weights.
            load_gradients: If ``True`` and ``optimizer.pth`` exists, also
                restore the optimiser state so training can be continued.

        Returns:
            DQNAgent: Fully initialised agent with loaded weights.
        """
        folder = Path(path)
        config_path = folder / "config.json"
        model_path = folder / "model.pth"
        target_model_path = folder / "target_model.pth"
        optim_path = folder / "optimizer.pth"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {str(folder)!r}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model.pth in {str(folder)!r}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Reconstruct online & target networks from saved full objects
        loaded_model = torch.load(model_path, map_location=_DEVICE, weights_only=False)
        loaded_target = (
            torch.load(target_model_path, map_location=_DEVICE, weights_only=False)
            if target_model_path.exists()
            else torch.load(model_path, map_location=_DEVICE, weights_only=False)
        )

        agent = cls(
            model=loaded_model,
            target_model=loaded_target,
            env=env,
            learning_rate=config.get("learning_rate", 0.0012),
            epsilon=config.get("epsilon", 0.1),
            epsilon_dacay=config.get("epsilon_decay", 0.995),
            min_epsilon=config.get("min_epsilon", 0.01),
            gamma=config.get("gamma", 0.9),
            batch_size=config.get("batch_size", 8),
            target_update_iter=config.get("target_update_iter", 400),
            train_nums=config.get("train_nums", 5000),
            buffer_size=config.get("buffer_size", 200),
            alpha=config.get("alpha", 0.4),
            beta=config.get("beta", 0.4),
        )

        # The constructor moves models to device and creates a *new* optimizer;
        # replace them with the loaded objects that are already on _DEVICE.
        agent.model = loaded_model
        agent.target_model = loaded_target

        if load_gradients and optim_path.exists():
            state = torch.load(optim_path, map_location=_DEVICE, weights_only=False)
            agent.optimizer.load_state_dict(state)

        return agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        env: Any,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        load_gradients: bool = False,
    ) -> "DQNAgent":
        """Load pretrained model from local directory or Hugging Face Hub.

        Args:
            repo_name: Path to local folder with weights **or** repository
                name in format ``namespace/repo_name`` on Hugging Face Hub.
            env: A Gymnasium-compatible environment (required for
                reconstruction).
            access_token: Access token for private HF repository.
            version: Revision / branch / tag of HF repository.
            load_gradients: If ``True``, also load optimiser states for
                continuing training.

        Returns:
            DQNAgent: Initialised agent with loaded weights.
        """
        # 1) Try local loading (absolute / relative path)
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.load(p, env=env, load_gradients=load_gradients)

        # 2) If it looks like a file-system path but doesn't exist -> error
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(
                    f"Local directory not found: '{repo_name}'."
                    " Please check the path."
                )
            return cls.load(p, env=env, load_gradients=load_gradients)

        # 3) Otherwise treat as a Hugging Face Hub repo id
        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version
        )
        return cls.load(folder_path, env=env, load_gradients=load_gradients)

    def publish_to_hub(
        self,
        repo_name: str,
        folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        """Publish saved model to Hugging Face Hub.

        Args:
            repo_name: Repository id on Hugging Face Hub
                (e.g. ``"my-org/dqn-cartpole"``).
            folder_path: Local folder that contains saved weights
                (as written by :meth:`save`).
            access_token: HF access token. If ``None``, the token cached
                by ``huggingface-cli login`` will be used.
        """
        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_name,
            repo_type="model",
            token=access_token,
        )


class PERNARXAgent:
    """DQN Agent with NARX training model.

    Args:
        model (torch.nn.Module): Deep Q-network model.
        target_model (torch.nn.Module): Target deep Q-network model.
        env (gym.Env): Gym environment.
        learning_rate (float, optional): Learning rate.
        epsilon (float, optional): Environment exploration probability.
        epsilon_dacay (float, optional): Epsilon reduction coefficient per episode.
        min_epsilon (float, optional): Minimum epsilon value.
        gamma (float, optional): Discount coefficient.
        batch_size (int, optional): Mini-batch size.
        target_update_iter (int, optional): Target network update period (steps).
        train_nums (int, optional): Number of training steps.
        buffer_size (int, optional): Replay buffer size.
        replay_period (int, optional): Buffer sampling period.
        alpha (float, optional): Prioritization degree.
        beta (float, optional): Importance sampling coefficient.
        beta_increment_per_sample (float, optional): Beta increment per sample.
    """

    def __init__(
        self,
        model: Any,
        target_model: Any,
        env: Any,
        learning_rate: float = 0.0012,
        epsilon: float = 0.1,
        epsilon_dacay: float = 0.995,
        min_epsilon: float = 0.01,
        gamma: float = 0.9,
        batch_size: int = 8,
        target_update_iter: int = 400,
        train_nums: int = 5000,
        buffer_size: int = 200,
        replay_period: int = 20,
        alpha: float = 0.4,
        beta: float = 0.4,
        beta_increment_per_sample: float = 0.001,
        log_dir: str | None = None,
        verbose_histogram: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize PER-NARX agent and buffers.

        Args:
            model: Online Q-network.
            target_model: Target Q-network.
            env: Gym/Gymnasium environment.
            learning_rate: Optimizer learning rate.
            epsilon: Initial epsilon for exploration.
            epsilon_dacay: Multiplicative epsilon decay.
            min_epsilon: Minimum epsilon value.
            gamma: Discount factor.
            batch_size: Training batch size.
            target_update_iter: Steps between target updates.
            train_nums: Total training steps to run.
            buffer_size: Replay buffer capacity.
            replay_period: Sampling period from buffer.
            alpha: PER priority exponent.
            beta: PER importance sampling exponent.
            beta_increment_per_sample: Increment for beta per sample.
            log_dir: Directory for TensorBoard logs.
            verbose_histogram: Whether to log histograms extensively.
        """
        self.device = _DEVICE
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags
        self.wandb_config = wandb_config
        self.writer = create_metric_writer(
            tb_log_dir=self.log_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            algo="dqn-narx",
        )

        # parameters
        self.env = env  # gym environment
        self.lr = learning_rate  # learning step
        self.epsilon = epsilon  # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay  # epsilon decay rate
        self.min_epsilon = min_epsilon  # minimum epsilon
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size  # minibatch k
        self.target_update_iter = target_update_iter  # target network update period
        self.train_nums = train_nums  # total training steps

        # replay buffer params [(s, a, r, ns, done), ...]
        self.b_obs = np.empty((self.batch_size,) + self.env.observation_space.shape)
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size,) + env.observation_space.shape)
        self.b_dones = np.empty(self.batch_size, dtype=np.bool_)

        self.replay_buffer = SumTree(buffer_size)  # sum-tree data structure
        self.buffer_size = buffer_size  # replay buffer size N
        self.replay_period = replay_period  # replay period K
        self.alpha = alpha  # priority parameter
        self.beta = beta  # importance sampling parameter
        self.beta_increment_per_sample = beta_increment_per_sample
        self.num_in_buffer = 0  # total number of transitions stored in buffer
        self.margin = 0.01  # pi = |td_error| + margin
        self.p1 = 1  # initialize priority for the first transition
        # self.is_weight = np.empty((None, 1))
        self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
        self.abs_error_upper = 1
        self.verbose_histogram = verbose_histogram
        self.global_step = 0
        self.global_env_step = 0
        self.episode_idx = 0

    def train(
        self,
        num_episodes: Optional[int] = None,
        *,
        max_steps: Optional[int] = None,
        save_best: bool = False,
        save_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Train the PERNARX variant (unified interface).

        See :meth:`DQNAgent.train` for semantics. Arguments mirror the
        canonical unified signature.
        """
        _ = (save_best, save_path, verbose, kwargs)
        if num_episodes is not None and max_steps is not None:
            self.train_nums = int(num_episodes) * int(max_steps)
        elif max_steps is not None:
            self.train_nums = int(max_steps)
        obs, _info = self.env.reset()
        prev_action = [0]
        episode_reward = 0.0
        episode_length = 0
        pbar = tqdm(range(1, self.train_nums), desc="PERNARXAgent Train", unit="step")
        recent_loss = None
        for t in pbar:
            self.global_env_step = t
            print(obs, prev_action)
            best_action, _q_values = self.model.action_value(obs[None])

            action = self.get_action(best_action)  # get the real action
            next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            episode_length += 1
            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity :])
            self.store_transition(
                p, obs, action, reward, next_obs, done
            )  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)
            prev_action = best_action
            if t > self.buffer_size:
                # if t % self.replay_period == 0:  # transition sampling and update
                recent_loss = self.train_step()
                if t % 200 == 0 and recent_loss is not None:
                    pbar.set_postfix(
                        {"loss": f"{recent_loss:.4f}", "eps": f"{self.epsilon:.3f}"}
                    )

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                # Episode end logging (canonical schema)
                self.writer.log_episode(
                    reward=float(episode_reward),
                    length=int(episode_length),
                    env_step=int(self.global_env_step),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                self.writer.add_scalar(
                    schema.DQN.EPSILON,
                    float(self.epsilon),
                    env_step=self.global_env_step,
                )
                self.episode_idx += 1
                episode_reward = 0.0
                episode_length = 0
                obs, _info = self.env.reset()  # one episode end
            else:
                obs = next_obs
        self.writer.flush()
        # Note: don't assert_contract_satisfied() if max_steps is so small that the
        # warmup never ends and no train_step() ever runs.
        self.writer.assert_contract_satisfied()
        return {"episodes": int(self.episode_idx)}

    def train_step(self) -> Any:
        """Function for training step.

        Returns:
            losses (float): Losses after one training step.
        """

        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)

        # Convert batch to tensors
        b_obs = torch.from_numpy(self.b_obs).float().to(self.device)
        b_next = torch.from_numpy(self.b_next_states).float().to(self.device)
        actions = torch.from_numpy(self.b_actions).long().to(self.device)
        rewards = torch.from_numpy(self.b_rewards).float().to(self.device)
        dones = (
            torch.from_numpy(self.b_dones.astype(np.float32)).float().to(self.device)
        )
        is_w = torch.from_numpy(self.is_weight).float().to(self.device).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_target = self.target_model(b_next)
            next_actions_online = self.model(b_next).argmax(dim=1)
            next_q_sa = next_q_target.gather(
                1, next_actions_online.unsqueeze(1)
            ).squeeze(1)
            q_target_sa = rewards + self.gamma * next_q_sa * (1.0 - dones)

        # Predicted Q for current states & chosen actions
        q_pred = self.model(b_obs)
        q_pred_sa = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # PER weighted MSE loss
        loss = (is_w * (q_pred_sa - q_target_sa).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities based on TD-error
        with torch.no_grad():
            abs_td_error = (q_target_sa - q_pred_sa).abs().detach().cpu().numpy()
        clipped_error = np.where(
            abs_td_error < self.abs_error_upper,
            abs_td_error,
            self.abs_error_upper,
        )
        ps = np.power(clipped_error + self.margin, self.alpha)
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, float(p))

        # TensorBoard logging
        q_pred_sa_mean = float(q_pred_sa.detach().mean().item())
        q_target_sa_mean = float(q_target_sa.detach().mean().item())
        td_err_mean = float(np.mean(abs_td_error)) if abs_td_error.size > 0 else 0.0
        td_err_max = float(np.max(abs_td_error)) if abs_td_error.size > 0 else 0.0
        td_err_min = float(np.min(abs_td_error)) if abs_td_error.size > 0 else 0.0

        self.writer.add_scalar(
            schema.DQN.LOSS_Q,
            float(loss.item()),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.Q_PRED_SA_MEAN,
            q_pred_sa_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.Q_TARGET_SA_MEAN,
            q_target_sa_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MEAN,
            td_err_mean,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MAX,
            td_err_max,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.VALUE_TD_ERROR_MIN,
            td_err_min,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.DQN.PER_BETA,
            float(self.beta),
            env_step=self.global_env_step,
        )

        self.global_step += 1
        self.writer.add_scalar(
            schema.TRAIN_UPDATES,
            self.global_step,
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.TRAIN_LR,
            float(self.optimizer.param_groups[0]["lr"]),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.TRAIN_REPLAY_SIZE,
            int(self.num_in_buffer),
            env_step=self.global_env_step,
        )

        if (
            self.verbose_histogram
            and self.global_step > 0
            and (self.global_step % 1000 == 0)
        ):
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(
                    f"weights/q/{name}",
                    param,
                    env_step=self.global_env_step,
                )

        return float(loss.item())

    def sum_tree_sample(self, k: int):
        """Get batch for training.

        Args:
            k (int): Size of batch to get.

        Returns:
            idxes (int): Indices of objects from batch.
            is_weights (float): Priorities of objects from batch.
        """

        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1.0, self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        # Only consider filled leaves (buffer may not be full yet, so many leaves are 0).
        leaves = self.replay_buffer.tree[-self.replay_buffer.capacity :]
        filled_leaves = leaves[leaves > 0]
        if len(filled_leaves) == 0:
            min_prob = 1.0  # uniform fallback
        else:
            min_prob = np.min(filled_leaves) / self.replay_buffer.total_p
        min_prob = max(min_prob, 1e-8)  # guard against division by zero
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            (
                self.b_obs[i],
                self.b_actions[i],
                self.b_rewards[i],
                self.b_next_states[i],
                self.b_dones[i],
            ) = t
            # P(j)
            sampling_probabilities = (
                p / self.replay_buffer.total_p
            )  # where p = p ** self.alpha
            is_weights[i, 0] = (
                np.power(self.buffer_size * sampling_probabilities, -self.beta)
                / max_weight
            )
        return idxes, is_weights

    def evaluation(self, env, render: bool = False) -> float:
        """Get batch for training.

        Args:
            env (_type_): Environment.
            render (bool, optional): Whether to visualize the environment or not.

        Returns:
            ep_reward (float): Total reward for the episode.
        """

        obs_info = env.reset()
        if isinstance(obs_info, tuple):
            obs, _info = obs_info
        else:
            # Fallback if environment doesn't follow Gymnasium API exactly
            obs = obs_info
        done, ep_reward = False, 0
        ep_length = 0
        # one episode until done
        while not done:
            action, _q_values = self.model.action_value(
                obs[None]
            )  # Using [None] to extend its dimension (4,) -> (1, 4)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _info = step_out
            obs = next_obs
            ep_reward += reward
            ep_length += 1
            if render:  # visually show
                env.render()
            time.sleep(0.05)
        env.close()
        # Log evaluation reward / length (canonical schema)
        self.writer.add_scalar(
            schema.EVAL_EPISODE_REWARD,
            float(ep_reward),
            env_step=self.global_env_step,
        )
        self.writer.add_scalar(
            schema.EVAL_EPISODE_LENGTH,
            int(ep_length),
            env_step=self.global_env_step,
        )
        return ep_reward

    def close(self) -> None:
        """Close resources (e.g., SummaryWriter)."""
        self.writer.close()

    def store_transition(
        self,
        priority: float,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer.

        Args:
            priority (int): Priority of the transition.
            obs (_type_): Current observation.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (_type_): Next observation.
            done: Whether the episode is finished or not.

        Returns:
            ep_reward (float): Total reward for the episode.
        """

        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)

    # rank-based prioritization sampling
    def rand_based_sample(self, k):
        """Placeholder for rank-based prioritized sampling (not implemented)."""
        pass

    # e-greedy
    def get_action(self, best_action: int) -> int:
        """Epsilon-greedy strategy function. Returns random action if exploration occurs.

        Args:
            best_action (int): Best action according to the policy.

        Returns:
            action (float): Action chosen according to the strategy.
        """

        if np.random.rand() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(best_action)

    # assign the current network parameters to target network
    def update_target_model(self) -> None:
        """Target neural network update function."""

        self.target_model.load_state_dict(self.model.state_dict())
        # Log target update event (canonical schema)
        self.writer.add_scalar(
            schema.DQN.TARGET_UPDATE,
            1,
            env_step=self.global_env_step,
        )

    def get_target_value(self, obs: np.ndarray) -> np.ndarray:
        """Get Q-values from target neural network.

        Returns:
            q_values (float): Q-values from the target network.
        """
        return cast(np.ndarray, self.target_model.predict(obs))

    def e_decay(self) -> None:
        """Function for reducing network exploration probability."""

        self.epsilon *= self.epsilon_decay

    def save(
        self,
        path: Union[str, Path, None] = None,
        save_gradients: bool = False,
    ) -> None:
        """Save PyTorch models to the specified directory.

        Args:
            path (str | Path | None): Save path. If None, saves into current cwd
                under folder "dqn_pernarx".
            save_gradients (bool): Save optimizer states to continue training.
        """
        if path is None:
            path = Path.cwd() / "dqn_pernarx"
        else:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # File paths
        model_path = path / "model.pth"
        target_model_path = path / "target_model.pth"
        optim_path = path / "optimizer.pth"
        config_path = path / "config.json"

        # Save models (full objects to preserve lazy layers)
        torch.save(self.model, model_path)
        torch.save(self.target_model, target_model_path)

        # Optionally save optimizer state
        if save_gradients:
            torch.save(self.optimizer.state_dict(), optim_path)

        # Minimal config for reload convenience
        config = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "batch_size": self.batch_size,
            "target_update_iter": self.target_update_iter,
            "train_nums": self.train_nums,
            "buffer_size": self.buffer_size,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
