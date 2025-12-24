import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.mpc.stochastic import MPCAgent, Net


# Minimal dynamics model for unit tests (ignore action, pass-through state)
class _DummyDynamicsModel(torch.nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        # MPCAgent always builds an optimizer over model parameters; keep a
        # harmless parameter so the optimizer doesn't error on an empty list.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        _ = self._dummy  # keep parameter "used" for clarity
        return x[..., : self.obs_dim]


# Фикстура для создания среды и агента
@pytest.fixture
def setup_mpc_agent():
    def example_cost_function(state, action):
        theta = state[0, 0].item()
        theta_dot = state[0, 1].item()
        return theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)

    env = gym.make("Pendulum-v1")
    model = Net(env.action_space.shape[0], env.observation_space.shape[0])
    agent = MPCAgent(
        gamma=0.99,
        action_dim=env.action_space.shape[0],
        observation_dim=env.observation_space.shape[0],
        model=model,
        cost_function=example_cost_function,
        env=env,
    )
    return agent


def test_collect_data(setup_mpc_agent):
    agent = setup_mpc_agent
    states, actions, next_states = agent.collect_data(num_episodes=1)
    assert len(states) == len(actions) == len(next_states) == 200
    assert states.shape == (200, 3)
    assert actions.shape == (200, 1)
    assert next_states.shape == (200, 3)


def test_train_model(setup_mpc_agent):
    agent = setup_mpc_agent
    states, actions, next_states = agent.collect_data(num_episodes=1)
    agent.train_model(states, actions, next_states, epochs=1)
    assert agent.system_model_optimizer.param_groups[0]["lr"] == 1e-3


def test_choose_action(setup_mpc_agent):
    agent = setup_mpc_agent
    state = np.array([0.5, 0.3, -0.1])
    action = agent.choose_action(state, rollout=5, horizon=3)
    assert action.shape == (1, 1)


def test_choose_action_ref_minimizes_cost():
    """Regression: choose_action_ref must pick the *lowest* cost rollout."""
    torch.manual_seed(0)

    obs_dim = 1
    action_dim = 1
    model = _DummyDynamicsModel(obs_dim)

    def cost_fn(_next_state, action, _reference_signals, _step):
        # Positive cost: penalize magnitude of control.
        return (action**2).sum()

    agent = MPCAgent(
        gamma=0.99,
        action_dim=action_dim,
        observation_dim=obs_dim,
        model=model,
        cost_function=cost_fn,
        env=None,
        min_max_action_value=(-1.0, 1.0),
    )

    action, best_cost = agent.choose_action_ref(
        state=np.array([0.0], dtype=np.float32),
        rollout=256,
        horizon=1,
        reference_signals=np.zeros((1, 20), dtype=np.float32),
        step=0,
    )

    assert action.shape == (1, 1)
    # With enough rollouts, the best sampled action should be close to 0.
    assert abs(float(action[0, 0])) < 0.05
    assert best_cost >= 0.0


def test_choose_action_ref_advances_reference_step_across_horizon():
    torch.manual_seed(0)

    obs_dim = 1
    action_dim = 1
    model = _DummyDynamicsModel(obs_dim)
    called_steps: list[int] = []

    def cost_fn(_next_state, _action, _reference_signals, step):
        called_steps.append(int(step))
        return 0.0

    agent = MPCAgent(
        gamma=0.99,
        action_dim=action_dim,
        observation_dim=obs_dim,
        model=model,
        cost_function=cost_fn,
        env=None,
        min_max_action_value=(-1.0, 1.0),
    )

    agent.choose_action_ref(
        state=np.array([0.0], dtype=np.float32),
        rollout=1,
        horizon=4,
        reference_signals=np.zeros((1, 20), dtype=np.float32),
        step=7,
    )

    assert called_steps == [7, 8, 9, 10]


def test_test_model(setup_mpc_agent):
    agent = setup_mpc_agent
    rewards = agent.test_model(num_episodes=5, rollout=3, horizon=2)
    assert len(rewards) == 5


def test_test_network(setup_mpc_agent):
    agent = setup_mpc_agent
    states = np.random.random((100, 3))
    actions = np.random.random((100, 1))
    next_states = np.random.random((100, 3))

    agent.test_network(states, actions, next_states)
