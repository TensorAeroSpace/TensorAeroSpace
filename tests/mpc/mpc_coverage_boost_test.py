"""Extra coverage tests for tensoraerospace/agent/mpc/mpc.py.

Targets uncovered lines: _to_serializable, _to_cpu_detached, _state_dict_cpu,
_optimizer_state_dict_cpu, _dtype_from_string, _safe_device_str, OneStepMLP
branches, MPCStandardScaler, MPCAgent.collect_data (signals mode),
MPCAgent.train_dynamics, MPCAgent.save/load/from_pretrained,
MPCAgent.reset, MPCAgent.to_device, MPCAgent.set_tracking_type,
cost computation paths, and various edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union, cast

import numpy as np
import pytest
import torch
import torch.nn as nn

from tensoraerospace.agent.mpc.mpc import (
    MPC,
    MPCAgent,
    MPCConstraints,
    MPCSolveResult,
    MPCStandardScaler,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
    OneStepMLP,
    _dtype_from_string,
    _optimizer_state_dict_cpu,
    _safe_device_str,
    _state_dict_cpu,
    _to_1d,
    _to_2d,
    _to_cpu_detached,
    _to_serializable,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BoxSpace:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)


class _SimpleEnv:
    """Tiny gym-like env for testing MPCAgent."""

    def __init__(self, state_dim=4, action_dim=1, max_steps=20):
        self.observation_space = _BoxSpace(
            low=-np.ones(state_dim, dtype=np.float32) * 10,
            high=np.ones(state_dim, dtype=np.float32) * 10,
        )
        self.action_space = _BoxSpace(
            low=-np.ones(action_dim, dtype=np.float32),
            high=np.ones(action_dim, dtype=np.float32),
        )
        self.dt = 0.1
        self.number_time_steps = max_steps
        self.unwrapped = self
        self._t = 0
        self._x = np.zeros(state_dim, dtype=np.float32)

    def reset(self):
        self._t = 0
        self._x = np.random.uniform(-0.5, 0.5, size=self._x.shape).astype(np.float32)
        return self._x.copy(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        state_dim = self._x.shape[0]
        act_dim = a.shape[0]
        # Simple dynamics: shift first act_dim components, decay rest
        self._x[:act_dim] = 0.95 * self._x[:act_dim] + a
        self._x[act_dim:] = 0.95 * self._x[act_dim:]
        self._t += 1
        terminated = self._t >= self.number_time_steps
        return self._x.copy(), -float(np.sum(self._x**2)), terminated, False, {}


def _make_mpc_agent(state_dim=4, action_dim=1, max_steps=20, **kwargs):
    env = _SimpleEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    defaults = dict(
        horizon=5,
        iters=5,
        mpc_lr=0.05,
        memory_capacity=5_000,
        dynamics_lr=5e-3,
        grad_clip_norm=1.0,
        seed=42,
        hidden_layers=(32, 32),
        device="cpu",
    )
    defaults.update(kwargs)
    agent = MPCAgent(env=env, **defaults)
    return agent, env


def _fill_agent_memory(agent, env, n_episodes=3):
    """Collect data via random exploration."""
    agent.collect_data(
        num_episodes=n_episodes,
        max_steps=env.number_time_steps,
        exploration="random",
    )


# ---------------------------------------------------------------------------
# Utility function tests (_to_serializable, _to_cpu_detached, etc.)
# Lines 178-183, 195-203, 208, 213-215, 222-231, 236-244, 254
# ---------------------------------------------------------------------------


def test_to_serializable_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = _to_serializable(t)
    assert result == [1.0, 2.0, 3.0]


def test_to_serializable_ndarray():
    a = np.array([1.0, 2.0])
    result = _to_serializable(a)
    assert result == [1.0, 2.0]


def test_to_serializable_dataclass():
    cfg = MPCTrackingExtraCostConfig(w_du=1.0, w_jerk=2.0)
    result = _to_serializable(cfg)
    assert isinstance(result, dict)
    assert result["w_du"] == 1.0


def test_to_serializable_nested():
    data = {"a": torch.tensor(1.0), "b": [np.array([2.0])], "c": (3,)}
    result = _to_serializable(data)
    assert result["a"] == 1.0
    assert result["b"] == [[2.0]]
    assert result["c"] == [3]


def test_to_serializable_plain():
    assert _to_serializable(42) == 42
    assert _to_serializable("hello") == "hello"


def test_to_cpu_detached_tensor():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    result = _to_cpu_detached(t)
    assert not result.requires_grad
    assert result.device.type == "cpu"


def test_to_cpu_detached_dict():
    d = {"k": torch.tensor([1.0])}
    result = _to_cpu_detached(d)
    assert isinstance(result, dict)
    assert result["k"].device.type == "cpu"


def test_to_cpu_detached_list_tuple():
    lst = [torch.tensor([1.0]), torch.tensor([2.0])]
    tup = (torch.tensor([3.0]),)
    r1 = _to_cpu_detached(lst)
    r2 = _to_cpu_detached(tup)
    assert isinstance(r1, list)
    assert isinstance(r2, tuple)


def test_to_cpu_detached_plain():
    assert _to_cpu_detached(42) == 42


def test_state_dict_cpu():
    m = nn.Linear(3, 2)
    sd = _state_dict_cpu(m)
    for v in sd.values():
        assert v.device.type == "cpu"


def test_optimizer_state_dict_cpu():
    m = nn.Linear(3, 2)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    # Run a step to populate state
    loss = m(torch.randn(1, 3)).sum()
    loss.backward()
    opt.step()
    sd = _optimizer_state_dict_cpu(opt)
    assert "state" in sd


# ---------------------------------------------------------------------------
# _dtype_from_string, _safe_device_str
# Lines 218-231, 236-244
# ---------------------------------------------------------------------------


def test_dtype_from_string():
    assert _dtype_from_string("torch.float32") == torch.float32
    assert _dtype_from_string("torch.float64") == torch.float64
    assert _dtype_from_string(None) == torch.float32
    assert _dtype_from_string("invalid_garbage") == torch.float32
    assert _dtype_from_string("float32") == torch.float32


def test_safe_device_str():
    assert _safe_device_str(None) == "cpu"
    assert _safe_device_str("cpu") == "cpu"
    # CUDA unavailable => "cpu"
    if not torch.cuda.is_available():
        assert _safe_device_str("cuda") == "cpu"
        assert _safe_device_str("cuda:0") == "cpu"


# ---------------------------------------------------------------------------
# _to_2d, _to_1d
# Lines 247-259
# ---------------------------------------------------------------------------


def test_to_2d_1d_input():
    x = np.array([1.0, 2.0, 3.0])
    result = _to_2d(x, dtype=torch.float32, device=torch.device("cpu"))
    assert result.shape == (1, 3)


def test_to_2d_3d_input():
    x = np.zeros((2, 3, 4))
    result = _to_2d(x, dtype=torch.float32, device=torch.device("cpu"))
    assert result.shape == (2, 12)


def test_to_1d():
    x = np.array([[1.0, 2.0]])
    result = _to_1d(x, dtype=torch.float32, device=torch.device("cpu"))
    assert result.shape == (2,)


# ---------------------------------------------------------------------------
# OneStepMLP branches
# Lines 724, 727, 730, 734-739, 750, 756
# ---------------------------------------------------------------------------


def test_one_step_mlp_tanh():
    mlp = OneStepMLP(input_dim=5, output_dim=3, hidden_layers=(16,), activation="tanh")
    out = mlp(torch.randn(2, 5))
    assert out.shape == (2, 3)


def test_one_step_mlp_gelu():
    mlp = OneStepMLP(input_dim=5, output_dim=3, hidden_layers=(16,), activation="gelu")
    out = mlp(torch.randn(2, 5))
    assert out.shape == (2, 3)


def test_one_step_mlp_bad_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        OneStepMLP(input_dim=5, output_dim=3, hidden_layers=(16,), activation="bad")


def test_one_step_mlp_bad_dims():
    with pytest.raises(ValueError, match="positive"):
        OneStepMLP(input_dim=0, output_dim=3, hidden_layers=(16,))


def test_one_step_mlp_empty_hidden():
    with pytest.raises(ValueError, match="non-empty"):
        OneStepMLP(input_dim=5, output_dim=3, hidden_layers=())


def test_one_step_mlp_negative_hidden():
    with pytest.raises(ValueError, match="positive"):
        OneStepMLP(input_dim=5, output_dim=3, hidden_layers=(-1,))


def test_one_step_mlp_3d_forward():
    mlp = OneStepMLP(input_dim=5, output_dim=3, hidden_layers=(16,))
    # 3D input should be reshaped
    out = mlp(torch.randn(2, 1, 5))
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# MPCStandardScaler
# Lines 660-701
# ---------------------------------------------------------------------------


def test_scaler_identity():
    sc = MPCStandardScaler.identity(4, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(3, 4)
    assert torch.allclose(sc.transform(x), x)
    assert torch.allclose(sc.inverse(x), x)


def test_scaler_fit_and_transform():
    data = torch.randn(100, 4) * 5 + 3
    sc = MPCStandardScaler.fit(data)
    xn = sc.transform(data)
    # mean should be ~0, std ~1
    assert abs(xn.mean().item()) < 0.5
    # inverse should recover
    recovered = sc.inverse(xn)
    assert torch.allclose(recovered, data, atol=1e-5)


def test_scaler_fit_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        MPCStandardScaler.fit(torch.randn(10))


def test_scaler_to_device():
    sc = MPCStandardScaler.identity(3, device=torch.device("cpu"), dtype=torch.float32)
    sc2 = sc.to(device=torch.device("cpu"), dtype=torch.float64)
    assert sc2.mean.dtype == torch.float64


# ---------------------------------------------------------------------------
# MPC solver edge cases
# Lines 296, 310, 318-336, 360, 365, 370, 404
# ---------------------------------------------------------------------------


def test_mpc_horizon_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=0,
            weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
        )


def test_mpc_best_check_every_zero_raises():
    with pytest.raises(ValueError, match="best_check_every"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
            best_check_every=0,
        )


def test_mpc_bad_Q_dim_raises():
    with pytest.raises(ValueError, match="Q_diag"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(3), R_diag=np.ones(1)),
        )


def test_mpc_bad_R_dim_raises():
    with pytest.raises(ValueError, match="R_diag"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(3)),
        )


def test_mpc_bad_S_dim_raises():
    with pytest.raises(ValueError, match="S_diag"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1), S_diag=np.ones(3)),
        )


def test_mpc_bad_constraint_dim_raises():
    with pytest.raises(ValueError, match="u_min"):
        MPC(
            dynamics=lambda x, u: x + u,
            state_dim=2,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
            constraints=MPCConstraints(u_min=np.ones(3)),
        )


def test_mpc_sgd_optimizer():
    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
        optimizer="sgd",
        iters=3,
        warm_start=False,
    )
    res = mpc.solve(x0=np.array([1.0, 0.0]), x_ref=np.zeros((4, 2)))
    assert res.u0.shape == (1,)


def test_mpc_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):

        def dyn(x, u):
            return x + u

        mpc = MPC(
            dynamics=dyn,
            state_dim=1,
            action_dim=1,
            horizon=3,
            weights=MPCWeights(Q_diag=np.ones(1), R_diag=np.ones(1)),
            optimizer="rmsprop",
            iters=2,
        )
        mpc.solve(x0=np.zeros(1))


def test_mpc_track_best_disabled():
    """Cover the branch when track_best=False (lines 632-642)."""

    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
        iters=3,
        track_best=False,
        warm_start=False,
    )
    res = mpc.solve(x0=np.array([1.0, 0.0]), x_ref=np.zeros((4, 2)))
    assert isinstance(res.final_cost, float)


def test_mpc_x_ref_horizon_length():
    """Cover x_ref that has length=horizon (not horizon+1) -- gets extended."""

    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=4,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
        iters=3,
        warm_start=False,
    )
    x_ref = np.zeros((4, 2))  # horizon, not horizon+1
    res = mpc.solve(x0=np.array([1.0, 0.0]), x_ref=x_ref)
    assert res.x_seq.shape == (5, 2)


def test_mpc_x_ref_bad_length_raises():
    def dyn(x, u):
        return x + u

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=2,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(2)),
        iters=2,
    )
    with pytest.raises(ValueError, match="x_ref must have length"):
        mpc.solve(x0=np.zeros(2), x_ref=np.zeros((10, 2)))


def test_mpc_x0_bad_dim_raises():
    def dyn(x, u):
        return x + u

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=2,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(2)),
        iters=2,
    )
    with pytest.raises(ValueError, match="x0 must have state_dim"):
        mpc.solve(x0=np.zeros(5))


def test_mpc_u_prev_bad_dim_raises():
    def dyn(x, u):
        return x + u

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=2,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(2)),
        iters=2,
    )
    with pytest.raises(ValueError, match="u_prev must have action_dim"):
        mpc.solve(x0=np.zeros(2), u_prev=np.zeros(5))


def test_mpc_no_x_ref_no_terminal_cost():
    """Solve without x_ref -- only control penalty applies."""

    def dyn(x, u):
        return x + u

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=2,
        horizon=3,
        weights=MPCWeights(
            Q_diag=np.ones(2),
            R_diag=np.ones(2),
            terminal_weight=0.0,
        ),
        iters=3,
        warm_start=False,
    )
    res = mpc.solve(x0=np.zeros(2))
    assert res.u0.shape == (2,)


def test_mpc_with_extra_cost_fn():
    """Cover extra_cost_fn branch (lines 509-514)."""
    calls = []

    def extra(x_seq, u_seq, x_ref):
        calls.append(1)
        return torch.tensor(0.01)

    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=3,
        weights=MPCWeights(Q_diag=np.ones(2), R_diag=np.ones(1)),
        extra_cost_fn=extra,
        iters=3,
        warm_start=False,
    )
    res = mpc.solve(x0=np.array([1.0, 0.0]), x_ref=np.zeros((4, 2)))
    assert len(calls) > 0


def test_mpc_cost_with_S_diag_and_u_prev():
    """Cover delta-u cost when u_prev is provided (lines 494-499)."""

    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=4,
        weights=MPCWeights(
            Q_diag=np.ones(2),
            R_diag=np.ones(1),
            S_diag=np.ones(1) * 0.5,
        ),
        iters=5,
        warm_start=False,
    )
    res = mpc.solve(
        x0=np.array([1.0, 0.0]),
        x_ref=np.zeros((5, 2)),
        u_prev=np.array([0.1]),
    )
    assert res.u0.shape == (1,)


def test_mpc_cost_with_S_diag_no_u_prev():
    """Cover delta-u cost when u_prev is None (lines 492-493)."""

    def dyn(x, u):
        return x + torch.cat([u, torch.zeros_like(x[..., :1])], dim=-1)

    mpc = MPC(
        dynamics=dyn,
        state_dim=2,
        action_dim=1,
        horizon=4,
        weights=MPCWeights(
            Q_diag=np.ones(2),
            R_diag=np.ones(1),
            S_diag=np.ones(1) * 0.5,
        ),
        iters=5,
        warm_start=False,
    )
    res = mpc.solve(
        x0=np.array([1.0, 0.0]),
        x_ref=np.zeros((5, 2)),
    )
    assert res.u0.shape == (1,)


# ---------------------------------------------------------------------------
# MPCAgent: collect_data random & signals modes
# Lines 1517-1525, 1531-1769, 1778-1802
# ---------------------------------------------------------------------------


def test_mpc_agent_collect_data_random():
    agent, env = _make_mpc_agent()
    agent.collect_data(num_episodes=2, max_steps=15, exploration="random")
    assert len(agent.memory) > 0


def test_mpc_agent_collect_data_signals():
    agent, env = _make_mpc_agent()
    agent.collect_data(
        num_episodes=1,
        max_steps=50,
        exploration="signals",
        signal_kinds=["unit_step", "sinusoid", "ramp"],
        dt=0.1,
        action_amplitude_frac=0.5,
    )
    assert len(agent.memory) > 0


def test_mpc_agent_collect_data_signals_many_kinds():
    """Cover many signal types in gen_1d by running each kind individually."""
    all_kinds = [
        "random_steps",
        "multi_step",
        "multisine",
        "chirp",
        "square_wave",
        "triangular_wave",
        "sawtooth",
        "doublet",
        "pulse",
        "gaussian_pulse",
        "exponential",
        "damped_sinusoid",
    ]
    for kind in all_kinds:
        agent, env = _make_mpc_agent(max_steps=30)
        agent.collect_data(
            num_episodes=1,
            max_steps=30,
            exploration="signals",
            signal_kinds=[kind],
            dt=0.1,
        )
        assert len(agent.memory) > 0, f"No data collected for signal kind: {kind}"


def test_mpc_agent_collect_data_bad_exploration():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="Unknown exploration"):
        agent.collect_data(num_episodes=1, exploration="bad_mode")


def test_mpc_agent_collect_data_bad_signal_kind():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="Unknown signal kind"):
        agent.collect_data(
            num_episodes=1,
            max_steps=10,
            exploration="signals",
            signal_kinds=["bogus_signal"],
        )


def test_mpc_agent_collect_data_empty_signal_kinds_raises():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="non-empty"):
        agent.collect_data(
            num_episodes=1,
            max_steps=10,
            exploration="signals",
            signal_kinds=[],
        )


def test_mpc_agent_collect_data_bad_num_episodes():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="num_episodes"):
        agent.collect_data(num_episodes=0)


# ---------------------------------------------------------------------------
# MPCAgent: train_dynamics
# Lines 1855-1941
# ---------------------------------------------------------------------------


def test_mpc_agent_train_dynamics_mse():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    metrics = agent.train_dynamics(
        epochs=1, batch_size=32, steps_per_epoch=2, loss="mse"
    )
    assert "loss" in metrics
    assert np.isfinite(metrics["loss"])


def test_mpc_agent_train_dynamics_huber():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    metrics = agent.train_dynamics(
        epochs=1, batch_size=32, steps_per_epoch=2, loss="huber"
    )
    assert "loss" in metrics


def test_mpc_agent_train_dynamics_no_normalize():
    agent, env = _make_mpc_agent(normalize=False)
    _fill_agent_memory(agent, env, n_episodes=2)
    metrics = agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    assert "loss" in metrics


def test_mpc_agent_train_dynamics_no_grad_clip():
    agent, env = _make_mpc_agent(grad_clip_norm=None)
    _fill_agent_memory(agent, env, n_episodes=2)
    metrics = agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    assert "loss" in metrics


def test_mpc_agent_train_dynamics_bad_loss():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    with pytest.raises(ValueError, match="Unknown loss"):
        agent.train_dynamics(epochs=1, batch_size=32, loss="bad")


def test_mpc_agent_train_dynamics_empty_buffer():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="Not enough samples"):
        agent.train_dynamics(epochs=1, batch_size=32)


def test_mpc_agent_train_dynamics_predict_absolute():
    agent, env = _make_mpc_agent(model_predict_delta=False)
    _fill_agent_memory(agent, env, n_episodes=2)
    metrics = agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# MPCAgent: fit_normalizers
# Lines 1824-1853
# ---------------------------------------------------------------------------


def test_mpc_agent_fit_normalizers():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.fit_normalizers(num_samples=50)
    # Scaler should have non-trivial mean now
    assert agent.x_scaler.mean.shape == (agent.state_dim,)


def test_mpc_agent_fit_normalizers_empty_raises():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="empty"):
        agent.fit_normalizers()


# ---------------------------------------------------------------------------
# MPCAgent: select_action
# Lines 1476-1488
# ---------------------------------------------------------------------------


def test_mpc_agent_select_action_with_ref():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)

    state, _ = env.reset()
    x_ref = np.zeros((agent.mpc.horizon + 1, agent.state_dim), dtype=np.float32)
    action = agent.select_action(state, x_ref=x_ref)
    assert action.shape == (agent.action_dim,)


def test_mpc_agent_select_action_without_ref():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)

    state, _ = env.reset()
    action = agent.select_action(state)
    assert action.shape == (agent.action_dim,)


def test_mpc_agent_select_action_bad_state_dim():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="state must have size"):
        agent.select_action(np.zeros(100))


# ---------------------------------------------------------------------------
# MPCAgent: reset
# Lines 1471-1474
# ---------------------------------------------------------------------------


def test_mpc_agent_reset():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    state, _ = env.reset()
    x_ref = np.zeros((agent.mpc.horizon + 1, agent.state_dim), dtype=np.float32)
    agent.select_action(state, x_ref=x_ref)
    agent.reset()
    assert agent._u_prev is None
    assert agent.mpc._u_warm is None


# ---------------------------------------------------------------------------
# MPCAgent: predict (wrapper)
# Lines 1347-1355
# ---------------------------------------------------------------------------


def test_mpc_agent_predict():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    state, _ = env.reset()
    action = agent.predict(state)
    assert action.shape == (agent.action_dim,)


def test_mpc_agent_predict_none_raises():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="requires an explicit"):
        agent.predict(None)


# ---------------------------------------------------------------------------
# MPCAgent: to_device
# Lines 1312-1338
# ---------------------------------------------------------------------------


def test_mpc_agent_to_device_cpu_noop():
    agent, env = _make_mpc_agent()
    agent2 = agent.to_device("cpu")
    assert agent2.device.type == "cpu"
    assert agent2 is agent  # same device returns self


def test_mpc_agent_to_device_with_optim_state():
    """Cover moving optimizer state (with populated tensors)."""
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)
    # to_device to same device is a no-op but we test at least it doesn't crash
    agent2 = agent.to_device("cpu")
    assert agent2.device.type == "cpu"


# ---------------------------------------------------------------------------
# MPCAgent: get_param_env
# Lines 1360-1409
# ---------------------------------------------------------------------------


def test_mpc_agent_get_param_env():
    agent, env = _make_mpc_agent()
    cfg = agent.get_param_env()
    assert "env" in cfg and "policy" in cfg
    pp = cfg["policy"]["params"]
    assert "state_dim" in pp
    assert "action_dim" in pp
    assert "horizon" in pp
    assert "model_class" in pp


# ---------------------------------------------------------------------------
# MPCAgent: set_tracking_type
# Lines 1128-1158
# ---------------------------------------------------------------------------


def test_mpc_agent_set_tracking_type_step_response():
    agent, env = _make_mpc_agent()
    step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
        tracked_idx=-1,
        rate_idx=-2,
        dt=0.1,
    )
    agent.set_tracking_type("step_response", step_response_config=step_cfg)
    assert agent.tracking_type == "step_response"


def test_mpc_agent_set_tracking_type_tracking():
    agent, env = _make_mpc_agent()
    tracking_cfg = MPCTrackingExtraCostConfig(w_du=1.0, w_jerk=2.0)
    agent.set_tracking_type("tracking", tracking_config=tracking_cfg)
    assert agent.tracking_type == "tracking"


def test_mpc_agent_set_tracking_type_no_config():
    """Cover branch where step_response_config is None => auto-build."""
    agent, env = _make_mpc_agent()
    agent.step_response_config = None
    agent.set_tracking_type("step_response")
    assert agent.step_response_config is not None


# ---------------------------------------------------------------------------
# MPCAgent: save / load roundtrip
# Lines 1971-2001, 2003-2147
# ---------------------------------------------------------------------------


def test_mpc_agent_save_load_roundtrip():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)

    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(path=d, save_gradients=True)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "dynamics_model.pth").exists()
        assert (run_dir / "normalizers.npz").exists()
        assert (run_dir / "dynamics_optim.pth").exists()

        loaded = MPCAgent.from_pretrained(str(run_dir), load_gradients=True)
        assert loaded.device.type == "cpu"
        assert loaded.state_dim == agent.state_dim
        assert loaded.action_dim == agent.action_dim

        state, _ = env.reset()
        a1 = agent.select_action(state)
        a2 = loaded.select_action(state)
        # They won't be identical (MPC is stochastic/optimizer) but shapes must match
        assert a1.shape == a2.shape


def test_mpc_agent_save_no_gradients():
    agent, env = _make_mpc_agent()
    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(path=d, save_gradients=False)
        assert not (run_dir / "dynamics_optim.pth").exists()


def test_mpc_agent_from_pretrained_bad_path():
    with pytest.raises(FileNotFoundError):
        MPCAgent.from_pretrained("./nonexistent_dir_abc_xyz")


# ---------------------------------------------------------------------------
# MPCAgent: _filter_kwargs_for_init
# Lines 1946-1969
# ---------------------------------------------------------------------------


def test_mpc_agent_filter_kwargs():
    class Dummy:
        def __init__(self, a, b=None):
            pass

    result = MPCAgent._filter_kwargs_for_init(Dummy, {"a": 1, "b": 2, "c": 3})
    assert "a" in result and "b" in result
    assert "c" not in result


def test_mpc_agent_filter_kwargs_varkw():
    class Open:
        def __init__(self, **kwargs):
            pass

    result = MPCAgent._filter_kwargs_for_init(Open, {"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# MPCAgent: _resolve_state_idx
# Lines 1112-1120
# ---------------------------------------------------------------------------


def test_resolve_state_idx_positive():
    agent, _ = _make_mpc_agent(state_dim=4)
    assert agent._resolve_state_idx(2) == 2


def test_resolve_state_idx_negative():
    agent, _ = _make_mpc_agent(state_dim=4)
    assert agent._resolve_state_idx(-1) == 3


def test_resolve_state_idx_invalid():
    agent, _ = _make_mpc_agent(state_dim=4)
    with pytest.raises(ValueError, match="Invalid state index"):
        agent._resolve_state_idx(10)


# ---------------------------------------------------------------------------
# MPCAgent: _update_u_limit_tensor edge cases
# Lines 1071-1110
# ---------------------------------------------------------------------------


def test_update_u_limit_no_constraints():
    agent, env = _make_mpc_agent()
    agent._mpc_constraints = None
    agent._update_u_limit_tensor()
    assert agent._u_lim is not None


def test_update_u_limit_only_u_max():
    agent, env = _make_mpc_agent()
    agent._mpc_constraints = MPCConstraints(u_max=np.array([2.0]))
    agent._update_u_limit_tensor()
    assert agent._u_lim is not None


def test_update_u_limit_only_u_min():
    agent, env = _make_mpc_agent()
    agent._mpc_constraints = MPCConstraints(u_min=np.array([-2.0]))
    agent._update_u_limit_tensor()
    assert agent._u_lim is not None


def test_update_u_limit_scalar_bound():
    """Cover the scalar broadcast path (line 1089-1090)."""
    agent, env = _make_mpc_agent()
    agent._mpc_constraints = MPCConstraints(
        u_min=np.array([-1.0]), u_max=np.array([1.0])
    )
    agent._update_u_limit_tensor()
    assert agent._u_lim is not None


# ---------------------------------------------------------------------------
# MPCAgent: _action adapters
# Lines 1433-1466
# ---------------------------------------------------------------------------


def test_action_env_from_internal_clips():
    agent, env = _make_mpc_agent()
    u = np.array([5.0], dtype=np.float32)  # way above bounds
    a = agent._action_env_from_internal(u)
    assert a[0] <= 1.0 + 1e-6


def test_action_internal_from_env():
    agent, env = _make_mpc_agent()
    a_env = np.array([0.5], dtype=np.float32)
    u = agent._action_internal_from_env(a_env)
    assert u.shape == (1,)


def test_state_from_obs():
    agent, env = _make_mpc_agent()
    obs = np.zeros(4, dtype=np.float32)
    state = agent._state_from_obs(obs)
    assert state.shape == (4,)


def test_state_from_obs_bad_dim():
    agent, env = _make_mpc_agent()
    with pytest.raises(ValueError, match="obs_to_state produced"):
        agent._state_from_obs(np.zeros(10))


# ---------------------------------------------------------------------------
# MPCAgent: MPC warm start across multiple calls
# ---------------------------------------------------------------------------


def test_mpc_agent_warm_start_behavior():
    agent, env = _make_mpc_agent()
    _fill_agent_memory(agent, env, n_episodes=2)
    agent.train_dynamics(epochs=1, batch_size=32, steps_per_epoch=2)

    state, _ = env.reset()
    x_ref = np.zeros((agent.mpc.horizon + 1, agent.state_dim), dtype=np.float32)

    a1 = agent.select_action(state, x_ref=x_ref)
    # Warm start should be populated after first call
    assert agent.mpc._u_warm is not None
    a2 = agent.select_action(state, x_ref=x_ref)
    assert a2.shape == a1.shape


# ---------------------------------------------------------------------------
# MPCAgent construction edge cases (lines 851-893)
# ---------------------------------------------------------------------------


def test_mpc_agent_bad_dims_raises():
    class _BadEnv:
        observation_space = type("O", (), {"shape": (0,)})()
        action_space = type("A", (), {"shape": (0,)})()
        unwrapped = None

    with pytest.raises(ValueError, match="could not infer"):
        MPCAgent(env=_BadEnv())


def test_mpc_agent_weights_infer_dims():
    """Cover dim inference from weights (lines 868-876)."""

    class _NoShapeEnv:
        observation_space = type("O", (), {"shape": (0,)})()
        action_space = type("A", (), {"shape": (0,)})()
        unwrapped = None

    agent = MPCAgent(
        env=_NoShapeEnv(),
        weights=MPCWeights(Q_diag=np.ones(3), R_diag=np.ones(2)),
    )
    assert agent.state_dim == 3
    assert agent.action_dim == 2


# ---------------------------------------------------------------------------
# MPCStepResponseExtraCostConfig.from_degrees
# Lines 87-131
# ---------------------------------------------------------------------------


def test_step_response_config_from_degrees():
    cfg = MPCStepResponseExtraCostConfig.from_degrees(
        tracked_idx=2,
        rate_idx=1,
        dt=0.05,
        overshoot_limit_deg=1.0,
        settle_band_deg=0.5,
    )
    assert cfg.tracked_idx == 2
    assert cfg.rate_idx == 1
    assert cfg.dt == 0.05
    assert cfg.overshoot_limit > 0
    assert cfg.settle_band > 0


# ---------------------------------------------------------------------------
# _make_step_response_extra_cost: with rate_idx
# Lines 1191-1307
# ---------------------------------------------------------------------------


def test_step_response_extra_cost_with_rate():
    agent, env = _make_mpc_agent(state_dim=4, action_dim=1)
    cfg = MPCStepResponseExtraCostConfig.from_degrees(
        tracked_idx=3,
        rate_idx=2,
        dt=0.1,
        ref_change_threshold_deg=0.01,
        min_step_amp_deg=0.01,
    )
    fn = agent._make_step_response_extra_cost(cfg)
    x_seq = torch.zeros((6, 4))
    u_seq = torch.zeros((5, 1))
    x_ref = torch.zeros((6, 4))
    x_ref[3:, 3] = 0.5  # step in reference
    x_seq[4:, 3] = 0.7  # overshoot
    cost = fn(x_seq, u_seq, x_ref)
    assert cost.item() > 0.0


def test_step_response_extra_cost_no_step():
    """Cover branch where no reference change is detected."""
    agent, env = _make_mpc_agent(state_dim=4, action_dim=1)
    cfg = MPCStepResponseExtraCostConfig.from_degrees(tracked_idx=3, dt=0.1)
    fn = agent._make_step_response_extra_cost(cfg)
    x_seq = torch.zeros((6, 4))
    u_seq = torch.zeros((5, 1))
    x_ref = torch.zeros((6, 4))  # flat reference
    cost = fn(x_seq, u_seq, x_ref)
    # Only jerk penalty, which is 0 for zero inputs
    assert cost.item() >= 0.0


def test_tracking_extra_cost_empty_u():
    """Cover branch with empty u_seq."""
    agent, env = _make_mpc_agent()
    cfg = MPCTrackingExtraCostConfig(w_du=1.0, w_jerk=1.0)
    fn = agent._make_tracking_extra_cost(cfg)
    cost = fn(torch.zeros((2, 4)), torch.zeros((0, 1)), None)
    assert cost.item() == 0.0
