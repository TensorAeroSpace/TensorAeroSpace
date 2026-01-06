import types

import numpy as np
import torch
from torch import nn

from tensoraerospace.agent.dsac.dsac_flight import (
    CLIP_GRAD,
    DSAC,
    _softplus_stable,
    calculate_huber_loss,
    quantile_huber_loss,
)


class _TinySpace:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class _TinyEnv:
    def __init__(self, obs_dim: int = 3, act_dim: int = 2):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))

    def reset(self):
        return np.zeros(self.observation_space.shape[0], dtype=np.float32), {}

    def step(self, action):
        del action
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self


def test_calculate_huber_loss_piecewise():
    td_error = torch.tensor([-2.0, -0.5, 0.2, 3.0])
    out = calculate_huber_loss(td_error, k=1.0)
    expected = torch.tensor(
        [
            1.5,  # 1 * (2 - 0.5*1)
            0.5 * 0.5**2,
            0.5 * 0.2**2,
            2.5,  # 1 * (3 - 0.5*1)
        ]
    )
    assert torch.allclose(out, expected)


def test_softplus_stable_matches_torch_softplus():
    x = torch.linspace(-50, 50, 100)
    stable = _softplus_stable(x)
    reference = torch.nn.functional.softplus(x)
    assert torch.allclose(stable, reference, atol=1e-6)


def test_quantile_huber_loss_reduces_and_finite():
    target = torch.zeros((4, 6))
    prediction = torch.ones((4, 6)) * 0.5
    taus = torch.rand((4, 6))
    loss = quantile_huber_loss(
        target=target, prediction=prediction, taus=taus, kappa=1.0
    )
    assert loss.dim() == 0
    assert float(loss) >= 0.0
    assert torch.isfinite(loss)


def test_clip_gradient_clamps_to_constant():
    lin = nn.Linear(2, 1, bias=False)
    for p in lin.parameters():
        p.grad = torch.tensor([[5.0, -5.0]])
    DSAC._clip_gradient(lin)
    for p in lin.parameters():
        assert torch.all(p.grad <= CLIP_GRAD)
        assert torch.all(p.grad >= -CLIP_GRAD)


def test_filter_kwargs_for_init_drops_unexpected():
    class Dummy:
        def __init__(self, a, *, b=None):
            self.a = a
            self.b = b

    kwargs = {"a": 1, "b": 2, "c": 3}
    filtered = DSAC._filter_kwargs_for_init(Dummy, kwargs)
    assert "a" in filtered and "b" in filtered
    assert "c" not in filtered


def test_invalid_risk_distortion_raises():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    with torch.random.fork_rng():
        with torch.no_grad():
            try:
                DSAC(env=env, risk_distortion="bad_distortion", device="cpu")
            except ValueError:
                return
    assert False, "Expected ValueError for invalid risk_distortion"


def test_sample_outputs_shapes_and_logpi():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    agent = DSAC(env=env, batch_size=2, hidden_size=8, device="cpu", learning_starts=0)
    state = torch.zeros((4, 3))
    action, log_pi = agent._sample(state, reparameterize=True)
    assert action.shape == (4, 2)
    assert log_pi.shape == (4,)
    assert torch.all(action <= 1.0) and torch.all(action >= -1.0)


def test_select_action_batch_rejects_rank1_input():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    agent = DSAC(env=env, batch_size=2, hidden_size=8, device="cpu", learning_starts=0)
    bad_states = torch.zeros((3,))
    try:
        agent.select_action_batch(bad_states, evaluate=False)
    except ValueError:
        return
    assert False, "Expected ValueError for rank-1 states"
