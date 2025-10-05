import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Optional tqdm progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    # Fallback no-op tqdm if not available
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is None:
            class _Dummy:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, n=1):
                    pass

                def set_postfix(self, **kwargs):
                    pass

                def write(self, s):
                    print(s)

            return _Dummy()
        else:
            for x in iterable:
                yield x

# Optional TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass


class ReplayBuffer:
    """Class for ReplayBuffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Serialize replay buffer state for checkpointing."""
        return {
            "capacity": self.capacity,
            "buffer": self.buffer,
            "position": self.position,
        }

    def load_state_dict(self, state):
        """Restore replay buffer state from a checkpoint dict."""
        self.capacity = int(state.get("capacity", self.capacity))
        self.buffer = list(state.get("buffer", []))
        self.position = int(state.get("position", 0))


class OUNoise(object):
    """
    Класс для шума Орнштейна-Уленбека.
    """

    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = (
            self.theta * (self.mu - x)
            + self.sigma * np.random.randn(self.action_dim)
        )
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        )
        return np.clip(action + ou_state, self.low, self.high)

    def state_dict(self):
        """Serialize OU noise state for checkpointing."""
        return {
            "mu": self.mu,
            "theta": self.theta,
            "sigma": self.sigma,
            "max_sigma": self.max_sigma,
            "min_sigma": self.min_sigma,
            "decay_period": self.decay_period,
            "action_dim": self.action_dim,
            "low": self.low,
            "high": self.high,
            "state": self.state,
        }

    def load_state_dict(self, state):
        """Restore OU noise state from a checkpoint dict."""
        self.mu = float(state.get("mu", self.mu))
        self.theta = float(state.get("theta", self.theta))
        self.sigma = float(state.get("sigma", self.sigma))
        self.max_sigma = float(state.get("max_sigma", self.max_sigma))
        self.min_sigma = float(state.get("min_sigma", self.min_sigma))
        self.decay_period = int(state.get("decay_period", self.decay_period))
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.low = state.get("low", self.low)
        self.high = state.get("high", self.high)
        self.state = np.array(state.get("state", self.state))


class ValueNetwork(nn.Module):
    """
    Класс для Q функции.
    """

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Класс для функции стратегии.
    """

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state)
        return action.squeeze(0).cpu().numpy()


class DDPG:
    def __init__(self, env, value_lr, policy_lr, replay_buffer_size):
        """
        Инициализация агента DDPG
        Args:
            env: объект окружения, с которым будет взаимодействовать агент.
            value_lr (float): learning rate для Q функции.
            policy_lr (float): learning rate для функции стратеги.
            replay_buffer_size (int): размер буффера.
        """
        self.env = env
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        self.replay_buffer_size = replay_buffer_size

        self.ou_noise = OUNoise(self.env.action_space)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = 256

        self.value_net = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)
        self.policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)

        self.target_value_net = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)
        self.target_policy_net = PolicyNetwork(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(device)

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.value_lr
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.policy_lr
        )

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # TensorBoard writer (lazy init in learn to include run-time params)
        self.writer = None

    def ddpg_update(
        self,
        batch_size,
        gamma=0.99,
        min_value=-np.inf,
        max_value=np.inf,
        soft_tau=1e-2,
    ):
        """
        Функция обновления ddpg.
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size
        )

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(
            next_state, next_action.detach()
        )
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        except Exception:
            pass
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        except Exception:
            pass
        self.value_optimizer.step()

        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        # Log training metrics if writer is available
        if self.writer is not None:
            try:
                self.writer.add_scalar(
                    "loss/policy", float(policy_loss.item()), self.frame_idx
                )
                self.writer.add_scalar(
                    "loss/value", float(value_loss.item()), self.frame_idx
                )
                with torch.no_grad():
                    mean_action = self.policy_net(state).mean().item()
                self.writer.add_scalar(
                    "policy/mean_action", float(mean_action), self.frame_idx
                )
            except Exception:
                pass

    def learn(
        self,
        max_frames,
        max_steps,
        batch_size,
        gamma: float = 0.995,
        soft_tau: float = 5e-3,
        warmup_frames: int = 10_000,
        updates_per_step: int = 1,
        target_value_clip: tuple[float, float] | None = (-10.0, 10.0),
    ):
        """
        Функция обучения.
        """
        self.max_frames = max_frames
        self.max_steps = max_steps
        self.frame_idx = 0
        self.rewards = []
        self.batch_size = batch_size

        # Lazy init TensorBoard writer with a sensible logdir
        if self.writer is None:
            try:
                logdir = os.path.join("runs", "ddpg")
                os.makedirs(logdir, exist_ok=True)
                self.writer = SummaryWriter()
            except Exception:
                self.writer = None

        with tqdm(total=max_frames, desc="DDPG Training") as pbar:
            while self.frame_idx < max_frames:
                state = self.env.reset()[0]
                self.ou_noise.reset()
                episode_reward = 0

                for step in range(max_steps):
                    action = self.policy_net.get_action(state)
                    action = self.ou_noise.get_action(action, step)
                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = self.env.step(action)
                    done = terminated or truncated

                    self.replay_buffer.push(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )
                    # Warmup: collect transitions without updates
                    if self.frame_idx > warmup_frames and len(self.replay_buffer) > batch_size:
                        for _ in range(max(1, int(updates_per_step))):
                            if target_value_clip is None:
                                mn, mx = -np.inf, np.inf
                            else:
                                mn, mx = float(target_value_clip[0]), float(target_value_clip[1])
                            self.ddpg_update(
                                batch_size,
                                gamma=gamma,
                                min_value=mn,
                                max_value=mx,
                                soft_tau=soft_tau,
                            )

                    state = next_state
                    episode_reward += reward
                    self.frame_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        frame=self.frame_idx,
                        ep_reward=float(episode_reward),
                    )

                    if done:
                        break

                self.rewards.append(episode_reward)

                # Log per-episode reward
                if self.writer is not None:
                    try:
                        self.writer.add_scalar(
                            "Performance/Reward",
                            float(episode_reward),
                            len(self.rewards),
                        )
                    except Exception:
                        pass

    def _collect_grads(self, model):
        """Collect parameter gradients of a model as CPU tensors.

        Returns dict name -> Tensor or None.
        """
        # Use typing compatible with Python <3.10 to satisfy linters
        from typing import Dict, Optional

        grads: Dict[str, Optional[torch.Tensor]] = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                grads[name] = None
            else:
                grads[name] = param.grad.detach().cpu()
        return grads

    def save(self, filepath, include_grads=False):
        """Save full training state to a checkpoint.

        Args:
            filepath (str): Path to checkpoint file.
            include_grads (bool): Save parameter gradients for
                continuing an unfinished optimizer step.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        ckpt = {
            "value_net": self.value_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
            "target_policy_net": self.target_policy_net.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "ou_noise": self.ou_noise.state_dict(),
            "frame_idx": self.frame_idx,
            "rewards": self.rewards,
            "max_frames": getattr(self, "max_frames", None),
            "max_steps": getattr(self, "max_steps", None),
            "batch_size": getattr(self, "batch_size", None),
        }

        if include_grads:
            ckpt["value_net_grads"] = self._collect_grads(self.value_net)
            ckpt["policy_net_grads"] = self._collect_grads(self.policy_net)

        torch.save(ckpt, filepath)

    def load(
        self,
        filepath,
        map_location=None,
        load_optimizer=True,
        load_targets=True,
        load_replay=True,
        load_noise=True,
        load_grads=False,
        strict=True,
    ):
        """Load training state from a checkpoint.

        Args:
            filepath (str): Path to checkpoint file.
            map_location: torch.load map_location.
            load_optimizer (bool): Restore optimizer states.
            load_targets (bool): Restore target networks.
            load_replay (bool): Restore replay buffer.
            load_noise (bool): Restore OU noise state.
            load_grads (bool): Restore parameter gradients.
            strict (bool): Strictly match model keys.
        """
        ckpt = torch.load(filepath, map_location=map_location)

        self.value_net.load_state_dict(ckpt["value_net"], strict=strict)
        self.policy_net.load_state_dict(ckpt["policy_net"], strict=strict)
        if load_targets:
            self.target_value_net.load_state_dict(
                ckpt["target_value_net"], strict=strict
            )
            self.target_policy_net.load_state_dict(
                ckpt["target_policy_net"], strict=strict
            )

        if load_optimizer and "value_optimizer" in ckpt:
            self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        if load_optimizer and "policy_optimizer" in ckpt:
            self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])

        if load_replay and "replay_buffer" in ckpt:
            self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
        if load_noise and "ou_noise" in ckpt:
            self.ou_noise.load_state_dict(ckpt["ou_noise"])

        self.frame_idx = int(
            ckpt.get("frame_idx", getattr(self, "frame_idx", 0))
        )
        self.rewards = list(ckpt.get("rewards", []))
        self.max_frames = ckpt.get(
            "max_frames", getattr(self, "max_frames", None)
        )
        self.max_steps = ckpt.get(
            "max_steps", getattr(self, "max_steps", None)
        )
        self.batch_size = ckpt.get(
            "batch_size", getattr(self, "batch_size", None)
        )

        if load_grads:
            vgrads = ckpt.get("value_net_grads")
            pgrads = ckpt.get("policy_net_grads")
            if vgrads is not None:
                for name, param in self.value_net.named_parameters():
                    grad = vgrads.get(name)
                    if grad is None:
                        param.grad = None
                    else:
                        param.grad = grad.to(param.device).clone()
            if pgrads is not None:
                for name, param in self.policy_net.named_parameters():
                    grad = pgrads.get(name)
                    if grad is None:
                        param.grad = None
                    else:
                        param.grad = grad.to(param.device).clone()
