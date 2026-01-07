import numpy as np
import torch

from tensoraerospace.agent.mpc.torch_mpc import (
    TorchMPC,
    TorchMPCAgent,
    TorchMPCConstraints,
    TorchMPCWeights,
)


def test_torch_mpc_shapes_and_constraints():
    torch.manual_seed(0)

    # Simple 2D system: x_next = x + [u, 0]
    def dyn(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + torch.cat([u, torch.zeros_like(u)], dim=-1)

    horizon = 6
    u_max = 0.5
    du_max = 0.1
    mpc = TorchMPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=horizon,
        weights=TorchMPCWeights(
            Q_diag=np.array([10.0, 0.0], dtype=np.float32),
            R_diag=np.array([0.1], dtype=np.float32),
            S_diag=np.array([0.5], dtype=np.float32),
            terminal_weight=1.0,
        ),
        constraints=TorchMPCConstraints(
            u_min=np.array([-u_max], dtype=np.float32),
            u_max=np.array([u_max], dtype=np.float32),
            du_min=np.array([-du_max], dtype=np.float32),
            du_max=np.array([du_max], dtype=np.float32),
        ),
        iters=40,
        lr=0.1,
        optimizer="adam",
        warm_start=False,
        seed=0,
    )

    x0 = np.array([1.0, 0.0], dtype=np.float32)
    x_ref = np.zeros((horizon + 1, 2), dtype=np.float32)
    res = mpc.solve(x0=x0, x_ref=x_ref, u_prev=np.array([0.0], dtype=np.float32))

    assert res.u0.shape == (1,)
    assert res.u_seq.shape == (horizon, 1)
    assert res.x_seq.shape == (horizon + 1, 2)

    assert np.all(res.u_seq <= u_max + 1e-6)
    assert np.all(res.u_seq >= -u_max - 1e-6)

    du = np.diff(res.u_seq[:, 0], axis=0)
    assert np.all(du <= du_max + 1e-6)
    assert np.all(du >= -du_max - 1e-6)


def test_torch_mpc_warm_start_reset():
    def dyn(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    mpc = TorchMPC(
        dynamics=dyn,
        state_dim=1,
        action_dim=1,
        horizon=3,
        weights=TorchMPCWeights(
            Q_diag=np.array([1.0], dtype=np.float32),
            R_diag=np.array([0.1], dtype=np.float32),
            S_diag=np.array([0.0], dtype=np.float32),
        ),
        iters=2,
        lr=0.01,
        warm_start=True,
        seed=0,
    )

    x0 = np.array([1.0], dtype=np.float32)
    x_ref = np.zeros((4, 1), dtype=np.float32)
    _ = mpc.solve(x0=x0, x_ref=x_ref, u_prev=np.array([0.0], dtype=np.float32))
    # Warm-start is populated
    assert mpc._u_warm is not None  # noqa: SLF001
    mpc.reset()
    assert mpc._u_warm is None  # noqa: SLF001


class _BoxSpace:
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)


class _SimpleEnv:
    """Tiny gym-like env for smoke-testing TorchMPCAgent."""

    def __init__(self) -> None:
        self.observation_space = _BoxSpace(
            low=np.array([-10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
        )
        self.action_space = _BoxSpace(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self.dt = 0.05
        self.number_time_steps = 25
        self.unwrapped = self
        self._t = 0
        self._x = np.zeros((2,), dtype=np.float32)

    def reset(self):
        self._t = 0
        self._x = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        return self._x.copy(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        u = float(a[0])
        # simple stable-ish dynamics
        self._x = 0.98 * self._x + np.array([u, -u], dtype=np.float32)
        self._t += 1
        terminated = bool(self._t >= int(self.number_time_steps))
        truncated = False
        reward = -float(np.sum(self._x**2))
        return self._x.copy(), reward, terminated, truncated, {}


def test_torch_mpc_agent_collect_and_train_smoke():
    env = _SimpleEnv()
    agent = TorchMPCAgent(
        env,
        horizon=6,
        iters=5,
        mpc_lr=0.05,
        memory_capacity=10_000,
        dynamics_lr=5e-3,
        grad_clip_norm=1.0,
        seed=0,
    )

    # Collect a bit of random data
    agent.collect_data(num_episodes=2, max_steps=20, exploration="random")
    assert len(agent.memory) > 0

    # Train dynamics (few steps)
    metrics = agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    assert "loss" in metrics

    # Select action with a trivial reference
    state, _ = env.reset()
    x_ref = np.zeros((agent.mpc.horizon + 1, agent.state_dim), dtype=np.float32)
    act = agent.select_action(state, x_ref=x_ref)
    assert act.shape == (agent.action_dim,)
