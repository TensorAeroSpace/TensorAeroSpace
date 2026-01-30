"""PyTorch-based A3C implementation.

This module contains the core A3C (Asynchronous Advantage Actor-Critic)
implementation, including network definitions and worker logic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

try:  # Prefer gymnasium when available for typing accuracy
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for older environments
    import gym

from typing import Callable, Optional, Tuple

from ..metrics import create_metric_writer
from .shared_optim import SharedAdam
from .utils import push_and_pull, record, set_init, v_wrap


class Net(nn.Module):
    """Neural network for policy and value function approximation in RL.

    Args:
        s_dim (int): State space dimension.
        a_dim (int): Action space dimension.

    Attributes:
        s_dim (int): State space dimension.
        a_dim (int): Action space dimension.
        a1 (nn.Linear): First policy layer.
        mu (nn.Linear): Mean layer of policy distribution.
        sigma (nn.Linear): Standard deviation layer of policy distribution.
        c1 (nn.Linear): First value function layer.
        v (nn.Linear): Value function output layer.
        distribution (torch.distributions.Distribution): Distribution for
            modeling agent actions.
    """

    def __init__(self, s_dim: int, a_dim: int) -> None:
        """Create network layers.

        Args:
            s_dim: State dimension.
            a_dim: Action dimension.
        """
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 256)
        self.mu = nn.Linear(256, a_dim)
        self.sigma = nn.Linear(256, a_dim)
        self.c1 = nn.Linear(s_dim, 256)
        self.v = nn.Linear(256, 1)
        self.softplus = nn.Softplus()
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one forward pass.

        Args:
            x (torch.Tensor): Input data, environment state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted mu, sigma,
            and value for the given state.
        """
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = self.softplus(self.sigma(a1)) + 0.001  # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s: torch.Tensor) -> np.ndarray:
        """Select agent action based on current state.

        Args:
            s (torch.Tensor): Current environment state.

        Returns:
            np.ndarray: Selected action.
        """
        self.eval()
        with torch.no_grad():
            mu, sigma, _ = self.forward(s)
            base = self.distribution(mu, sigma)
            dist = torch.distributions.Independent(base, 1) if self.a_dim > 1 else base
            a = dist.sample()
        return a.cpu().numpy().squeeze(0)

    def loss_func(
        self, s: torch.Tensor, a: torch.Tensor, v_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss function for network training.

        Args:
            s (torch.Tensor): States.
            a (torch.Tensor): Actions.
            v_t (torch.Tensor): Target state value function values.

        Returns:
            torch.Tensor: Loss function value.
        """
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        base = self.distribution(mu, sigma)
        dist = torch.distributions.Independent(base, 1) if self.a_dim > 1 else base
        log_prob = dist.log_prob(a)  # shape: [batch]
        entropy = dist.entropy()  # shape: [batch]
        exp_v = log_prob * td.detach().squeeze(-1) + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss.squeeze(-1)).mean()
        return total_loss


class Worker(mp.Process):
    """Worker process class for asynchronous agent training.

    Args:
        env (gym.Env): Environment for agent training.
        gnet (Net): Global model for shared training.
        opt (SharedAdam): Optimizer for global network.
        global_ep (mp.Value): Global episode counter.
        global_ep_r (mp.Value): Global total reward counter across all processes.
        res_queue (mp.Queue): Queue for storing results.
        name (int): Process number.
        num_actions (int): Number of possible actions in the environment.
        num_observations (int): Number of observations (state variables) in the environment.
        MAX_EP (int): Maximum number of episodes.
        MAX_EP_STEP (int): Maximum number of steps per episode.
        GAMMA (float): Discount factor for future rewards.
        update_global_iter (int): Frequency of global model updates.
        render (bool): Whether to render the environment. Defaults to False.
        writer (Optional[SummaryWriter]): TensorBoard writer. Defaults to None.
        global_step (Optional[mp.Value]): Global step counter. Defaults to None.

    Attributes:
        name (str): Unique process name.
        g_ep (mp.Value): Global episode counter.
        g_ep_r (mp.Value): Global total reward counter.
        res_queue (mp.Queue): Results queue.
        gnet (Net): Global neural network.
        opt (SharedAdam): Optimizer for updating global network.
        lnet (Net): Local neural network.
        env (gym.Env): OpenAI Gym environment.
        gamma (float): Discount factor.
        max_ep (int): Maximum number of episodes.
        max_ep_step (int): Maximum number of steps per episode.
        update_global_iter (int): Frequency of global network updates.
        render (bool): Whether to render the environment.
        writer (Optional[SummaryWriter]): TensorBoard writer.
        global_step (Optional[mp.Value]): Global step counter.
    """

    def __init__(
        self,
        env: gym.Env,
        gnet: Net,
        opt: SharedAdam,
        global_ep: mp.Value,
        global_ep_r: mp.Value,
        res_queue: mp.Queue,
        name: int,
        num_actions: int,
        num_observations: int,
        MAX_EP: int,
        MAX_EP_STEP: int,
        GAMMA: float,
        update_global_iter: int,
        render: bool = False,
        writer: Optional["torch.utils.tensorboard.SummaryWriter"] = None,
        global_step: Optional[mp.Value] = None,
    ) -> None:
        """Initialize worker process.

        Args:
            env: Environment instance.
            gnet: Global shared network.
            opt: Shared optimizer.
            global_ep: Shared episode counter.
            global_ep_r: Shared reward accumulator.
            res_queue: Queue for results.
            name: Worker id.
            num_actions: Action dimension.
            num_observations: Observation dimension.
            MAX_EP: Max episodes to run.
            MAX_EP_STEP: Max steps per episode.
            GAMMA: Discount factor.
            update_global_iter: Steps between syncs with global net.
            render: Whether to render (only worker 0 typically).
            writer: TensorBoard writer for metrics.
            global_step: Shared global step counter.
        """
        super(Worker, self).__init__()
        self.name = "w%i" % name
        self.g_ep, self.g_ep_r, self.res_queue = (
            global_ep,
            global_ep_r,
            res_queue,
        )
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(num_observations, num_actions)  # local network
        self.env = env
        self.gamma = GAMMA
        self.max_ep = MAX_EP
        self.max_ep_step = MAX_EP_STEP
        self.update_global_iter = update_global_iter
        self.render = render
        self.writer = writer
        self.global_step = global_step

    def run(self) -> None:
        """Execute worker process containing agent training."""
        total_step = 1
        # initial sync from global to local to avoid stale params
        self.lnet.load_state_dict(self.gnet.state_dict())
        while self.g_ep.value < self.max_ep:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                s = reset_out[0]
            else:
                s = reset_out
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.0
            for t in range(self.max_ep_step):
                if self.render and self.name == "w0" and hasattr(self.env, "render"):
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                if (
                    hasattr(self.env, "action_space")
                    and hasattr(self.env.action_space, "low")
                    and hasattr(self.env.action_space, "high")
                ):
                    low = self.env.action_space.low
                    high = self.env.action_space.high
                else:
                    low, high = -np.inf, np.inf
                a_clipped = np.clip(a, low, high)
                step_out = self.env.step(a_clipped)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    s_, r, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    s_, r, done, _ = step_out
                if t == self.max_ep_step - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                # use raw rewards
                # normalization strategy should be config-driven
                buffer_r.append(r)

                if (
                    total_step % self.update_global_iter == 0 or done
                ):  # update global and assign to local net
                    # sync and get metrics
                    metrics = push_and_pull(
                        self.opt,
                        self.lnet,
                        self.gnet,
                        done,
                        s_,
                        buffer_s,
                        buffer_a,
                        buffer_r,
                        self.gamma,
                    )

                    # Log training metrics to TensorBoard
                    if self.writer is not None and self.global_step is not None:
                        with self.global_step.get_lock():
                            step = self.global_step.value
                            self.global_step.value += 1
                        self.writer.add_scalar(
                            f"Loss/{self.name}/total",
                            metrics["loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/value",
                            metrics["value_loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/policy",
                            metrics["policy_loss"],
                            step,
                        )
                        self.writer.add_scalar(
                            f"Loss/{self.name}/entropy",
                            metrics["entropy"],
                            step,
                        )

                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and record episode
                        record(
                            self.g_ep,
                            self.g_ep_r,
                            ep_r,
                            self.res_queue,
                            self.name,
                            self.writer,
                        )
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


# Defaults for Agent wrapper
DEFAULT_MAX_EP = 10
DEFAULT_MAX_EP_STEP = 200
DEFAULT_GAMMA = 0.99
DEFAULT_UPDATE_GLOBAL_ITER = 10
DEFAULT_LR = 1e-4


def setup_global_params(
    *,
    max_episodes: int = DEFAULT_MAX_EP,
    max_ep_step: int = DEFAULT_MAX_EP_STEP,
    gamma: float = DEFAULT_GAMMA,
    update_global_iter: int = DEFAULT_UPDATE_GLOBAL_ITER,
    lr: float = DEFAULT_LR,
) -> None:
    """Update defaults used by Agent.

    This matches the previous TF API name to ease migration.
    """
    globals_dict = globals()
    globals_dict["DEFAULT_MAX_EP"] = max_episodes
    globals_dict["DEFAULT_MAX_EP_STEP"] = max_ep_step
    globals_dict["DEFAULT_GAMMA"] = gamma
    globals_dict["DEFAULT_UPDATE_GLOBAL_ITER"] = update_global_iter
    globals_dict["DEFAULT_LR"] = lr


class Agent:
    """Simple A3C Agent wrapper around multiprocessing Workers.

    Args:
        env_function: callable that returns a new env for a given worker id.
        gamma: discount factor.
        n_workers: number of worker processes.
        lr: learning rate for SharedAdam.
        max_episodes: total episodes to run per global counter.
        max_ep_step: max steps per episode.
        update_global_iter: frequency to push/pull.
        render: render from worker w0 (optional).

    Note: For unit tests or debugging, set run_in_main=True to avoid
    spawning processes. The single worker will run in the main process.
    """

    def __init__(
        self,
        env_function: Callable[[int], gym.Env],
        gamma: float = DEFAULT_GAMMA,
        n_workers: Optional[int] = None,
        lr: float = DEFAULT_LR,
        max_episodes: int = DEFAULT_MAX_EP,
        max_ep_step: int = DEFAULT_MAX_EP_STEP,
        update_global_iter: int = DEFAULT_UPDATE_GLOBAL_ITER,
        render: bool = False,
        run_in_main: bool = False,
        log_dir: str = "runs/a3c",
    ) -> None:
        """Configure A3C agent wrapper.

        Args:
            env_function: Factory returning an environment per worker id.
            gamma: Discount factor.
            n_workers: Number of worker processes; defaults to CPU count.
            lr: Learning rate for optimizer.
            max_episodes: Total episodes to run.
            max_ep_step: Max steps per episode.
            update_global_iter: Sync frequency for global net updates.
            render: Whether to render from worker 0.
            run_in_main: If True, run worker inline for debugging/tests.
            log_dir: TensorBoard log directory.
        """
        self.env_function = env_function
        self.gamma = gamma
        self.n_workers = (
            mp.cpu_count() if not n_workers or n_workers <= 0 else n_workers
        )
        self.lr = lr
        self.max_episodes = max_episodes
        self.max_ep_step = max_ep_step
        self.update_global_iter = update_global_iter
        self.render = render
        self.run_in_main = run_in_main

        # infer spaces
        probe_env = self.env_function(0)
        s_dim = int(probe_env.observation_space.shape[0])
        a_dim = int(probe_env.action_space.shape[0])
        probe_env.close()

        # global net and optimizer
        self.gnet = Net(s_dim, a_dim)
        self.gnet.share_memory()
        self.opt = SharedAdam(self.gnet.parameters(), lr=self.lr)

        # shared counters
        self.global_ep = mp.Value("i", 0)
        self.global_ep_r = mp.Value("d", 0.0)
        self.global_step = mp.Value("i", 0)
        # Queue type annotation kept as string to avoid requiring typing
        # extensions for multiprocessing types.
        self.res_queue: "mp.Queue" = mp.Queue()

        # TensorBoard writer
        self.writer: Optional["torch.utils.tensorboard.SummaryWriter"] = None
        try:
            self.writer = create_metric_writer(log_dir)
        except Exception:
            self.writer = None

    def train(self) -> None:
        """Launch training across worker processes (or single-process mode)."""
        workers = []
        if self.run_in_main:
            # run a single worker in current process (useful for tests)
            env = self.env_function(0)
            w = Worker(
                env=env,
                gnet=self.gnet,
                opt=self.opt,
                global_ep=self.global_ep,
                global_ep_r=self.global_ep_r,
                res_queue=self.res_queue,
                name=0,
                num_actions=self.gnet.a_dim,
                num_observations=self.gnet.s_dim,
                MAX_EP=self.max_episodes,
                MAX_EP_STEP=self.max_ep_step,
                GAMMA=self.gamma,
                update_global_iter=self.update_global_iter,
                render=self.render,
                writer=self.writer,
                global_step=self.global_step,
            )
            # directly call run without starting a new process
            w.run()
            env.close()
            return

        for i in range(self.n_workers):
            env = self.env_function(i)
            w = Worker(
                env=env,
                gnet=self.gnet,
                opt=self.opt,
                global_ep=self.global_ep,
                global_ep_r=self.global_ep_r,
                res_queue=self.res_queue,
                name=i,
                num_actions=self.gnet.a_dim,
                num_observations=self.gnet.s_dim,
                MAX_EP=self.max_episodes,
                MAX_EP_STEP=self.max_ep_step,
                GAMMA=self.gamma,
                update_global_iter=self.update_global_iter,
                render=self.render if i == 0 else False,
                writer=self.writer,
                global_step=self.global_step,
            )
            w.start()
            workers.append(w)

        # wait workers
        finished = 0
        while finished < self.n_workers:
            r = self.res_queue.get()
            if r is None:
                finished += 1

        for w in workers:
            w.join()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup resources."""
        if self.writer is not None:
            self.writer.close()
