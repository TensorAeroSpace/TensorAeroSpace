import numpy as np
import torch
import torch.nn as nn

from tensoraerospace.agent.gail.model import (
    GAIL,
    ActorCritic,
    Discriminator,
    compute_gae,
    init_weights,
    ppo_iter,
)


class _DummySpace:
    def __init__(self, shape):
        self.shape = shape


class _FakeEnv:
    def __init__(self, obs_dim=3, act_dim=2):
        self.observation_space = _DummySpace((obs_dim,))
        self.action_space = _DummySpace((act_dim,))

    def reset(self):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return state, {}

    def step(self, action):
        # Keep state zero, reward simple and deterministic
        action = np.asarray(action, dtype=np.float32)
        next_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = float(1.0 - np.linalg.norm(action))
        terminated = False
        truncated = False
        info: dict = {}
        return next_state, reward, terminated, truncated, info


def test_init_weights_linear_sets_params():
    layer = nn.Linear(4, 3)
    init_weights(layer)
    with torch.no_grad():
        # Bias should be ~0.1 everywhere
        assert torch.allclose(layer.bias, torch.full_like(layer.bias, 0.1), atol=1e-4)
        # Weights should be finite and not zeros everywhere (normal init)
        assert torch.isfinite(layer.weight).all()
        assert not torch.all(layer.weight == 0)


def test_actorcritic_forward_shapes():
    obs_dim, act_dim, hidden = 3, 2, 16
    net = ActorCritic(obs_dim, act_dim, hidden, std=0.0)
    device = next(net.parameters()).device
    x = torch.randn(4, obs_dim, device=device)
    dist, value = net(x)
    assert value.shape == (4, 1)
    sample = dist.sample()
    assert sample.shape == (4, act_dim)


def test_compute_gae_returns_length_and_tensors():
    # simple sequence of three steps
    values = [torch.tensor([[0.1]]), torch.tensor([[0.2]]), torch.tensor([[0.3]])]
    rewards = [torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[0.0]])]
    masks = [torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]])]
    next_value = torch.tensor([[0.0]])
    returns = compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95)
    assert isinstance(returns, list)
    assert len(returns) == 3
    assert all(isinstance(r, torch.Tensor) for r in returns)


def test_ppo_iter_batches_count_and_shapes():
    N, S, A = 4, 3, 2
    mini = 2
    states = torch.randn(N, S)
    actions = torch.randn(N, A)
    log_probs = torch.randn(N, A)
    returns = torch.randn(N, 1)
    advantage = torch.randn(N, 1)

    count = 0
    for s, a, lp, ret, adv in ppo_iter(
        mini, states, actions, log_probs, returns, advantage
    ):
        count += 1
        assert s.shape == (mini, S)
        assert a.shape == (mini, A)
        assert lp.shape == (mini, A)
        assert ret.shape == (mini, 1)
        assert adv.shape == (mini, 1)
    assert count == N // mini


def test_discriminator_forward_range():
    disc = Discriminator(num_inputs=5, hidden_size=8)
    device = next(disc.parameters()).device
    x = torch.randn(4, 5, device=device)
    out = disc(x)
    assert out.shape == (4, 1)
    assert torch.ge(out, 0).all() and torch.le(out, 1).all()


def test_gail_expert_reward_and_test_env_smoke():
    env = _FakeEnv(obs_dim=3, act_dim=2)
    expert_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    expert_data = np.zeros((16, expert_dim), dtype=np.float32)
    gail = GAIL(
        env,
        learning_rate=1e-3,
        max_steps=2,
        mini_batch_size=2,
        epochs=1,
        data=expert_data,
    )

    # Get device from model
    device = next(gail.model.parameters()).device

    # expert_reward
    state = torch.zeros(1, env.observation_space.shape[0], device=device)
    action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
    r = gail.expert_reward(state, action)
    assert np.asarray(r).shape == (1, 1)

    # test_env
    total = gail.test_env()
    assert isinstance(total, (int, float))


def test_gail_env_step_unpacking_five_tuple():
    """Bug #11: GAIL should use 5-tuple env.step() API (terminated, truncated)."""

    class _TerminatingEnv(_FakeEnv):
        """Env that terminates after 2 steps to exercise done logic."""

        def __init__(self):
            super().__init__(obs_dim=3, act_dim=2)
            self._step_count = 0

        def reset(self):
            self._step_count = 0
            return super().reset()

        def step(self, action):
            self._step_count += 1
            next_state, reward, _, _, info = super().step(action)
            terminated = self._step_count >= 2
            truncated = False
            return next_state, reward, terminated, truncated, info

    env = _TerminatingEnv()
    expert_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    expert_data = np.zeros((16, expert_dim), dtype=np.float32)
    gail = GAIL(
        env,
        learning_rate=1e-3,
        max_steps=3,
        mini_batch_size=2,
        epochs=1,
        data=expert_data,
    )
    # test_env should run without crashing (previously used 4-tuple unpacking)
    total = gail.test_env()
    assert isinstance(total, (int, float))


def test_gail_ppo_update_runs_one_step():
    env = _FakeEnv(obs_dim=3, act_dim=2)
    expert_dim = env.observation_space.shape[0] + env.action_space.shape[0]
    expert_data = np.zeros((16, expert_dim), dtype=np.float32)
    gail = GAIL(
        env,
        learning_rate=1e-3,
        max_steps=2,
        mini_batch_size=2,
        epochs=1,
        data=expert_data,
    )

    # Get device from model
    device = next(gail.model.parameters()).device

    N, S, A = 4, env.observation_space.shape[0], env.action_space.shape[0]
    states = torch.randn(N, S, device=device)
    actions = torch.randn(N, A, device=device)
    # Create a simple Gaussian log_prob approximation (not exact but shape-correct)
    log_probs = torch.zeros(N, A, device=device)
    returns = torch.randn(N, 1, device=device)
    advantages = torch.randn(N, 1, device=device)

    # Should run without exceptions
    gail.ppo_update(
        ppo_epochs=1,
        mini_batch_size=2,
        states=states,
        actions=actions,
        log_probs=log_probs,
        returns=returns,
        advantages=advantages,
    )
