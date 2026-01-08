"""Adaptive Critic Design / ADP agent.

This implementation follows the Adaptive Critic Design (ACD) / Approximate
Dynamic Programming idea from:

Prokhorov D.V., Wunsch D.C. “Adaptive critic designs: A case study for
neurocontrol.” Neural Networks, 8(9), 1995, pp. 1367–1372.

We implement a practical ACD variant with:
  - deterministic actor  a = π(s)
  - adaptive critic      Q = Q(s, a)  (interpreted as *cost-to-go*)
  - online TD learning    Q(s,a) ≈ c(s,a) + γ Q(s', π(s'))
  - actor improvement     minimize Q(s, π(s)) w.r.t actor params

Notes:
  - The environment is assumed to follow Gymnasium API.
  - We treat the environment reward as "utility" and convert to cost via:
        cost = -reward
    so minimizing cost-to-go is equivalent to maximizing return.
"""

from __future__ import annotations

import datetime
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    get_class_from_string,
    serialize_env,
)
from ..metrics import create_metric_writer
from .networks import DeterministicActor, QCritic, polyak_update
from .replay import ReplayBuffer


def _as_flat_np(x: Any) -> np.ndarray:
    """Convert env observation to a flat float32 numpy array."""
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


class ADP(BaseRLModel):
    """Adaptive Critic Design (ADP) agent for continuous control."""

    def __init__(
        self,
        env: Any,
        *,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42,
        # Exploration (training-time only)
        exploration_std: float = 0.1,
        # Learning mode
        use_replay: bool = False,
        memory_capacity: int = 200_000,
        batch_size: int = 64,
        updates_per_step: int = 1,
        # Optional target networks (Polyak averaging). Disabled by default to
        # stay closer to classic online ACD; can be enabled for stability.
        use_target_networks: bool = False,
        tau: float = 0.01,
        # Logging
        log_dir: Union[str, Path, None] = None,
        log_every_updates: int = 100,
    ) -> None:
        super().__init__()
        self.env = env
        self.gamma = float(gamma)
        self.exploration_std = float(exploration_std)
        self.use_replay = bool(use_replay)
        self.batch_size = int(batch_size)
        self.updates_per_step = int(updates_per_step)
        self.use_target_networks = bool(use_target_networks)
        self.tau = float(tau)
        self.hidden_size = int(hidden_size)

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.updates_per_step < 1:
            raise ValueError("updates_per_step must be >= 1")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.exploration_std < 0.0:
            raise ValueError("exploration_std must be >= 0")
        if self.use_target_networks and not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1] when use_target_networks=True")

        self.device = torch.device(device)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Spaces
        obs_dim = int(getattr(self.env.observation_space, "shape", (0,))[0])
        act_dim = int(getattr(self.env.action_space, "shape", (0,))[0])
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(
                f"ADP expects Box-like spaces. Got obs_dim={obs_dim}, act_dim={act_dim}"
            )

        action_low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(
            -1
        )

        hidden_sizes = (int(self.hidden_size), int(self.hidden_size))
        self.actor = DeterministicActor(
            obs_dim,
            act_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.critic = QCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optim = Adam(self.critic.parameters(), lr=float(critic_lr))

        # Optional target networks
        self.actor_target: Optional[DeterministicActor]
        self.critic_target: Optional[QCritic]
        if self.use_target_networks:
            self.actor_target = DeterministicActor(
                obs_dim,
                act_dim,
                hidden_sizes=hidden_sizes,
                action_low=action_low,
                action_high=action_high,
            ).to(self.device)
            self.critic_target = QCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(
                self.device
            )
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.actor_target = None
            self.critic_target = None

        # Optional replay buffer
        self.memory: Optional[ReplayBuffer]
        if self.use_replay:
            self.memory = ReplayBuffer(int(memory_capacity), seed=self.seed)
        else:
            self.memory = None

        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = create_metric_writer(self.log_dir)
        self.log_every_updates = int(log_every_updates)
        if self.log_every_updates < 1:
            raise ValueError("log_every_updates must be >= 1")

        self._updates = 0

    # ---- common API ----
    def get_env(self):
        return self.env

    def select_action(self, state: np.ndarray, *, evaluate: bool = False) -> np.ndarray:
        """Select action for a single observation."""
        obs = _as_flat_np(state)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            act = self.actor(obs_t).squeeze(0).cpu().numpy()
        if not evaluate and self.exploration_std > 0.0:
            act = act + np.random.normal(0.0, self.exploration_std, size=act.shape).astype(
                np.float32
            )
        # Always clip to env bounds
        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        return np.clip(act, low, high).astype(np.float32)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Alias for compatibility with some agent APIs."""
        return self.select_action(state, evaluate=bool(deterministic))

    # ---- learning ----
    def _td_update_batch(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        rew_b: np.ndarray,
        next_obs_b: np.ndarray,
        done_bootstrap_b: np.ndarray,
    ) -> Tuple[float, float]:
        """One update step on a given batch. Returns (critic_loss, actor_loss)."""
        obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)
        # reward comes in as (B,1) from replay; accept (B,) too.
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=self.device).reshape(-1, 1)
        next_obs_t = torch.as_tensor(next_obs_b, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(
            done_bootstrap_b, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)

        # Convert reward to cost
        cost_t = -rew_t

        # Target networks if enabled, otherwise online networks
        actor_next = self.actor_target if self.actor_target is not None else self.actor
        critic_next = self.critic_target if self.critic_target is not None else self.critic

        with torch.no_grad():
            next_act_t = actor_next(next_obs_t)
            q_next = critic_next(next_obs_t, next_act_t)
            target_q = cost_t + (1.0 - done_t) * self.gamma * q_next

        # Critic update
        q = self.critic(obs_t, act_t)
        critic_loss_t = F.mse_loss(q, target_q)
        self.critic_optim.zero_grad()
        critic_loss_t.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # Actor update: minimize critic (cost-to-go)
        actor_act = self.actor(obs_t)
        actor_loss_t = self.critic(obs_t, actor_act).mean()
        self.actor_optim.zero_grad()
        actor_loss_t.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # Polyak updates (optional)
        if self.use_target_networks and self.actor_target is not None and self.critic_target is not None:
            polyak_update(self.actor_target, self.actor, tau=self.tau)
            polyak_update(self.critic_target, self.critic, tau=self.tau)

        return float(critic_loss_t.item()), float(actor_loss_t.item())

    def train(self, *args, **kwargs) -> None:
        """Train for a number of episodes.

        Args:
            num_episodes (int): Number of episodes.
            max_steps (int | None): Optional per-episode cap.
        """
        num_episodes = int(args[0]) if len(args) > 0 else int(kwargs.get("num_episodes", 1))
        max_steps = kwargs.get("max_steps", None)
        max_steps_i = int(max_steps) if max_steps is not None else None

        total_steps = 0
        for ep in range(num_episodes):
            obs, _info = self.env.reset()
            obs = _as_flat_np(obs)
            ep_reward = 0.0
            steps = 0
            done = False

            while not done:
                act = self.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, _info = self.env.step(act)
                next_obs = _as_flat_np(next_obs)

                done_env = bool(terminated or truncated)
                # Bootstrap stops only on true termination (not time limit)
                done_bootstrap = float(bool(terminated))

                ep_reward += float(reward)
                steps += 1
                total_steps += 1

                if self.use_replay and self.memory is not None:
                    self.memory.push(obs, act, float(reward), next_obs, done_bootstrap)
                    if len(self.memory) >= self.batch_size:
                        for _ in range(self.updates_per_step):
                            b = self.memory.sample(self.batch_size)
                            critic_loss, actor_loss = self._td_update_batch(*b)
                            self._updates += 1
                            if (self._updates % self.log_every_updates) == 0:
                                self.writer.add_scalar("loss/critic", critic_loss, self._updates)
                                self.writer.add_scalar("loss/actor", actor_loss, self._updates)
                else:
                    # Online update on the single transition
                    critic_loss, actor_loss = self._td_update_batch(
                        obs_b=obs.reshape(1, -1),
                        act_b=act.reshape(1, -1),
                        rew_b=np.asarray([[reward]], dtype=np.float32),
                        next_obs_b=next_obs.reshape(1, -1),
                        done_bootstrap_b=np.asarray([[done_bootstrap]], dtype=np.float32),
                    )
                    self._updates += 1
                    if (self._updates % self.log_every_updates) == 0:
                        self.writer.add_scalar("loss/critic", critic_loss, self._updates)
                        self.writer.add_scalar("loss/actor", actor_loss, self._updates)

                obs = next_obs
                done = done_env
                if max_steps_i is not None and steps >= max_steps_i:
                    break

            self.writer.add_scalar("performance/episode_reward", float(ep_reward), ep)
            self.writer.add_scalar("performance/episode_length", int(steps), ep)
            self.writer.add_scalar("train/total_steps", int(total_steps), ep)

        self.writer.flush()

    # ---- persistence (HF-style similar to SAC/DDPG) ----
    def get_param_env(self) -> Dict[str, Dict[str, Any]]:
        class_name = self.env.unwrapped.__class__.__name__
        module_name = self.env.unwrapped.__class__.__module__
        env_name = f"{module_name}.{class_name}"
        env_params: Dict[str, Any] = {}
        try:
            if "tensoraerospace" in env_name:
                env_params = serialize_env(self.env)
        except Exception:
            env_params = {}

        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        agent_name = f"{module_name}.{class_name}"

        policy_params = {
            "gamma": self.gamma,
            "exploration_std": self.exploration_std,
            "use_replay": self.use_replay,
            "memory_capacity": int(getattr(self.memory, "capacity", 0) or 0),
            "batch_size": self.batch_size,
            "updates_per_step": self.updates_per_step,
            "use_target_networks": self.use_target_networks,
            "tau": self.tau,
            "device": self.device.type,
            "seed": self.seed,
            "actor_lr": float(self.actor_optim.defaults.get("lr", 3e-4)),
            "critic_lr": float(self.critic_optim.defaults.get("lr", 3e-4)),
            "hidden_size": int(self.hidden_size),
        }

        return {
            "env": {"name": env_name, "params": env_params},
            "policy": {"name": agent_name, "params": policy_params},
        }

    def save(
        self, path: Union[str, Path, None] = None, *, save_gradients: bool = False
    ) -> str:
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path)

        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = path / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config.json"
        actor_path = run_dir / "actor.pth"
        critic_path = run_dir / "critic.pth"
        actor_target_path = run_dir / "actor_target.pth"
        critic_target_path = run_dir / "critic_target.pth"
        actor_optim_path = run_dir / "actor_optim.pth"
        critic_optim_path = run_dir / "critic_optim.pth"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)
        if self.use_target_networks and self.actor_target is not None and self.critic_target is not None:
            torch.save(self.actor_target, actor_target_path)
            torch.save(self.critic_target, critic_target_path)

        if save_gradients:
            torch.save(self.actor_optim.state_dict(), actor_optim_path)
            torch.save(self.critic_optim.state_dict(), critic_optim_path)

        return str(run_dir)

    @staticmethod
    def _filter_kwargs_for_init(env_cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
    def __load(
        cls,
        path: Union[str, Path],
        *,
        load_gradients: bool = False,
    ) -> "ADP":
        path = Path(path)
        config_path = path / "config.json"
        actor_path = path / "actor.pth"
        critic_path = path / "critic.pth"
        actor_target_path = path / "actor_target.pth"
        critic_target_path = path / "critic_target.pth"
        actor_optim_path = path / "actor_optim.pth"
        critic_optim_path = path / "critic_optim.pth"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        class_name = cls.__name__
        module_name = cls.__module__
        agent_name = f"{module_name}.{class_name}"
        if config["policy"]["name"] != agent_name:
            raise TheEnvironmentDoesNotMatch

        # Recreate env
        env_cfg = config.get("env", {})
        env_cls_path = env_cfg.get("name")
        env_params = dict(env_cfg.get("params", {}) or {})

        if env_cls_path and "tensoraerospace" in str(env_cls_path):
            env_cls = get_class_from_string(env_cls_path)
            env_params = cls._filter_kwargs_for_init(env_cls, env_params)
            env = env_cls(**env_params)
        else:
            env = get_class_from_string(env_cls_path)() if env_cls_path else None

        p = dict(config.get("policy", {}).get("params", {}) or {})
        # Device fallback
        dev = str(p.get("device", "cpu"))
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        if dev == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            dev = "cpu"
        p["device"] = dev

        new_agent = cls(env=env, **p)

        new_agent.actor = torch.load(
            actor_path, map_location=new_agent.device, weights_only=False
        ).to(new_agent.device)
        new_agent.critic = torch.load(
            critic_path, map_location=new_agent.device, weights_only=False
        ).to(new_agent.device)

        if new_agent.use_target_networks and actor_target_path.exists() and critic_target_path.exists():
            new_agent.actor_target = torch.load(
                actor_target_path, map_location=new_agent.device, weights_only=False
            ).to(new_agent.device)
            new_agent.critic_target = torch.load(
                critic_target_path, map_location=new_agent.device, weights_only=False
            ).to(new_agent.device)

        # Reinitialize optimizers
        actor_lr = float(p.get("actor_lr", 3e-4))
        critic_lr = float(p.get("critic_lr", 3e-4))
        new_agent.actor_optim = Adam(new_agent.actor.parameters(), lr=actor_lr)
        new_agent.critic_optim = Adam(new_agent.critic.parameters(), lr=critic_lr)

        if load_gradients:
            if actor_optim_path.exists():
                new_agent.actor_optim.load_state_dict(
                    torch.load(actor_optim_path, map_location=new_agent.device, weights_only=False)
                )
            if critic_optim_path.exists():
                new_agent.critic_optim.load_state_dict(
                torch.load(critic_optim_path, map_location=new_agent.device, weights_only=False)
                )
        return new_agent

    @classmethod
    def from_pretrained(
        cls,
        repo_name: str,
        access_token: Optional[str] = None,
        version: Optional[str] = None,
        *,
        load_gradients: bool = False,
    ) -> "ADP":
        # 1) local folder
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls.__load(p, load_gradients=load_gradients)

        # 2) explicit path-like but missing
        pathlike_prefixes = ("./", "../", "/", "~")
        if str(repo_name).startswith(pathlike_prefixes):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Local directory not found: '{repo_name}'.")
            return cls.__load(p, load_gradients=load_gradients)

        # 3) Hugging Face repo id
        folder_path = super().from_pretrained(repo_name, access_token, version)
        return cls.__load(folder_path, load_gradients=load_gradients)


