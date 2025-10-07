"""Tests for A3C Agent class.

This module tests the Agent class functionality including:
- Agent initialization
- Environment function handling
- Global network setup
- Parameter configuration
"""

import numpy as np
import torch

from tensoraerospace.agent.a3c.pytorch import Agent, Net, setup_global_params


class _DummyEnv:
    """Minimal environment for testing."""

    def __init__(self):
        self.observation_space = type("S", (), {"shape": (4,)})()
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )()

    def reset(self, seed=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return (
            np.random.randn(4).astype(np.float32),
            0.0,
            False,
            False,
            {},
        )

    def close(self):
        pass


def _env_function(worker_id):
    """Environment function for Agent."""
    return _DummyEnv()


def test_agent_initialization():
    """Test Agent initialization."""
    agent = Agent(
        env_function=_env_function,
        gamma=0.99,
        n_workers=2,
        lr=1e-4,
        max_episodes=10,
        max_ep_step=100,
        update_global_iter=5,
        render=False,
        run_in_main=True,
    )

    assert agent.gamma == 0.99
    assert agent.n_workers == 2
    assert agent.lr == 1e-4
    assert agent.max_episodes == 10
    assert agent.max_ep_step == 100
    assert agent.update_global_iter == 5
    assert agent.render is False


def test_agent_global_network():
    """Test that Agent creates global network."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    assert isinstance(agent.gnet, Net)
    assert agent.gnet.s_dim == 4
    assert agent.gnet.a_dim == 2


def test_agent_optimizer():
    """Test that Agent creates optimizer."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        lr=1e-3,
        run_in_main=True,
    )

    assert agent.opt is not None
    assert len(agent.opt.param_groups) > 0


def test_agent_shared_counters():
    """Test that Agent creates shared counters."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    assert agent.global_ep.value == 0
    assert agent.global_ep_r.value == 0.0
    assert agent.global_step.value == 0


def test_agent_default_n_workers():
    """Test Agent with default number of workers."""
    agent = Agent(
        env_function=_env_function,
        run_in_main=True,
    )

    # Should use CPU count by default
    import torch.multiprocessing as mp

    assert agent.n_workers == mp.cpu_count()


def test_agent_custom_n_workers():
    """Test Agent with custom number of workers."""
    for n in [1, 2, 4, 8]:
        agent = Agent(
            env_function=_env_function,
            n_workers=n,
            run_in_main=True,
        )

        assert agent.n_workers == n


def test_agent_gamma_values():
    """Test Agent with different gamma values."""
    for gamma in [0.9, 0.95, 0.99, 0.999]:
        agent = Agent(
            env_function=_env_function,
            gamma=gamma,
            n_workers=2,
            run_in_main=True,
        )

        assert agent.gamma == gamma


def test_agent_learning_rate():
    """Test Agent with different learning rates."""
    for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
        agent = Agent(
            env_function=_env_function,
            lr=lr,
            n_workers=2,
            run_in_main=True,
        )

        assert agent.lr == lr


def test_agent_max_episodes():
    """Test Agent with different max episodes."""
    for max_ep in [5, 10, 50, 100]:
        agent = Agent(
            env_function=_env_function,
            max_episodes=max_ep,
            n_workers=2,
            run_in_main=True,
        )

        assert agent.max_episodes == max_ep


def test_agent_max_ep_step():
    """Test Agent with different max episode steps."""
    for max_step in [50, 100, 200, 500]:
        agent = Agent(
            env_function=_env_function,
            max_ep_step=max_step,
            n_workers=2,
            run_in_main=True,
        )

        assert agent.max_ep_step == max_step


def test_agent_update_global_iter():
    """Test Agent with different update frequencies."""
    for update_iter in [1, 5, 10, 20]:
        agent = Agent(
            env_function=_env_function,
            update_global_iter=update_iter,
            n_workers=2,
            run_in_main=True,
        )

        assert agent.update_global_iter == update_iter


def test_agent_env_function():
    """Test that Agent env_function works correctly."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    # Create environment using the function
    env = agent.env_function(0)

    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")
    env.close()


def test_agent_infers_dimensions():
    """Test that Agent correctly infers state and action dimensions."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    # Check that global network has correct dimensions
    assert agent.gnet.s_dim == 4  # from _DummyEnv
    assert agent.gnet.a_dim == 2  # from _DummyEnv


def test_agent_queue():
    """Test that Agent creates result queue."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    assert agent.res_queue is not None


def test_setup_global_params():
    """Test setup_global_params function."""
    from tensoraerospace.agent.a3c import pytorch

    # Save all original defaults
    saved_max_ep = pytorch.DEFAULT_MAX_EP
    saved_max_step = pytorch.DEFAULT_MAX_EP_STEP
    saved_gamma = pytorch.DEFAULT_GAMMA
    saved_update_iter = pytorch.DEFAULT_UPDATE_GLOBAL_ITER
    saved_lr = pytorch.DEFAULT_LR

    # Change defaults
    setup_global_params(
        max_episodes=50,
        max_ep_step=150,
        gamma=0.95,
        update_global_iter=15,
        lr=5e-4,
    )

    # Check that defaults were updated
    assert pytorch.DEFAULT_MAX_EP == 50
    assert pytorch.DEFAULT_MAX_EP_STEP == 150
    assert pytorch.DEFAULT_GAMMA == 0.95
    assert pytorch.DEFAULT_UPDATE_GLOBAL_ITER == 15
    assert pytorch.DEFAULT_LR == 5e-4

    # Restore all originals (for other tests)
    pytorch.DEFAULT_MAX_EP = saved_max_ep
    pytorch.DEFAULT_MAX_EP_STEP = saved_max_step
    pytorch.DEFAULT_GAMMA = saved_gamma
    pytorch.DEFAULT_UPDATE_GLOBAL_ITER = saved_update_iter
    pytorch.DEFAULT_LR = saved_lr


def test_agent_override_defaults():
    """Test that Agent can override default parameters."""
    # Create agent with explicit params
    agent = Agent(
        env_function=_env_function,
        max_episodes=42,
        max_ep_step=142,
        gamma=0.98,
        n_workers=2,
        run_in_main=True,
    )

    # Should use the explicitly provided values
    assert agent.max_episodes == 42
    assert agent.max_ep_step == 142
    assert agent.gamma == 0.98


def test_agent_run_in_main_flag():
    """Test Agent run_in_main flag."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    assert agent.run_in_main is True

    agent2 = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=False,
    )

    assert agent2.run_in_main is False


def test_agent_render_flag():
    """Test Agent render flag."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        render=True,
        run_in_main=True,
    )

    assert agent.render is True


def test_agent_log_dir():
    """Test Agent with custom log directory."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
        log_dir="custom_logs/test",
    )

    # Agent should be created without error
    assert agent is not None


def test_agent_shared_memory():
    """Test that global network is in shared memory."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        run_in_main=True,
    )

    # Check that parameters can be shared (have storage)
    for param in agent.gnet.parameters():
        assert param.storage() is not None


def test_agent_different_env_dimensions():
    """Test Agent with environment having different dimensions."""

    class _CustomEnv:
        def __init__(self, s_dim=6, a_dim=3):
            self.observation_space = type("S", (), {"shape": (s_dim,)})()
            self.action_space = type(
                "A",
                (),
                {
                    "shape": (a_dim,),
                    "low": np.full(a_dim, -1.0),
                    "high": np.full(a_dim, 1.0),
                },
            )()

        def reset(self, seed=None):
            return (
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                {},
            )

        def step(self, action):
            return (
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

        def close(self):
            pass

    def custom_env_fn(worker_id):
        return _CustomEnv(s_dim=6, a_dim=3)

    agent = Agent(
        env_function=custom_env_fn,
        n_workers=2,
        run_in_main=True,
    )

    assert agent.gnet.s_dim == 6
    assert agent.gnet.a_dim == 3


if __name__ == "__main__":
    # Run all tests
    test_agent_initialization()
    test_agent_global_network()
    test_agent_optimizer()
    test_agent_shared_counters()
    test_agent_default_n_workers()
    test_agent_custom_n_workers()
    test_agent_gamma_values()
    test_agent_learning_rate()
    test_agent_max_episodes()
    test_agent_max_ep_step()
    test_agent_update_global_iter()
    test_agent_env_function()
    test_agent_infers_dimensions()
    test_agent_queue()
    test_setup_global_params()
    test_agent_override_defaults()
    test_agent_run_in_main_flag()
    test_agent_render_flag()
    test_agent_log_dir()
    test_agent_shared_memory()
    test_agent_different_env_dimensions()
    print("All Agent tests passed!")
