"""Integration tests for A3C.

This module tests end-to-end A3C training including:
- Full training pipeline
- Worker-agent coordination
- Global network updates
"""

import numpy as np

from tensoraerospace.agent.a3c.pytorch import Agent


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
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 10
        reward = -np.sum(action**2) * 0.1  # Small negative reward
        return (
            np.random.randn(4).astype(np.float32),
            float(reward),
            done,
            False,
            {},
        )

    def close(self):
        pass


def _env_function(worker_id):
    """Environment function for Agent."""
    return _DummyEnv()


def test_agent_train_completes():
    """Test that agent training completes without errors."""
    agent = Agent(
        env_function=_env_function,
        gamma=0.99,
        n_workers=1,  # Single worker for simplicity
        lr=1e-3,
        max_episodes=2,  # Just 2 episodes
        max_ep_step=10,
        update_global_iter=5,
        render=False,
        run_in_main=True,  # Run in main process for testing
    )

    # Train should complete without raising
    agent.train()


def test_agent_updates_global_network():
    """Test that training updates the global network."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=2,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Save initial weights
    initial_weights = [p.data.clone() for p in agent.gnet.parameters()]

    # Train
    agent.train()

    # Check that at least some weights changed
    changed = False
    for initial, current in zip(initial_weights, agent.gnet.parameters()):
        if not initial.eq(current.data).all():
            changed = True
            break

    assert changed, "Global network should be updated during training"


def test_agent_updates_episode_counter():
    """Test that training updates episode counter."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=3,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Initially 0
    assert agent.global_ep.value == 0

    # Train
    agent.train()

    # Should have completed episodes
    assert agent.global_ep.value >= 3


def test_agent_collects_results():
    """Test that agent collects results in queue."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=2,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Train
    agent.train()

    # Queue should have results
    results = []
    while not agent.res_queue.empty():
        result = agent.res_queue.get()
        if result is not None:  # None is sentinel value
            results.append(result)

    # Should have collected some results
    assert len(results) > 0


def test_agent_with_multiple_workers():
    """Test agent with multiple workers (in main process)."""
    agent = Agent(
        env_function=_env_function,
        n_workers=2,
        max_episodes=4,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Should create and run without error
    agent.train()


def test_agent_with_different_update_frequencies():
    """Test agent with different update frequencies."""
    for update_iter in [1, 5, 10]:
        agent = Agent(
            env_function=_env_function,
            n_workers=1,
            max_episodes=2,
            max_ep_step=10,
            update_global_iter=update_iter,
            run_in_main=True,
        )

        # Should train without error
        agent.train()


def test_agent_with_short_episodes():
    """Test agent with very short episodes."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=2,
        max_ep_step=3,  # Very short
        update_global_iter=2,
        run_in_main=True,
    )

    agent.train()


def test_agent_with_long_episodes():
    """Test agent with longer episodes."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=1,
        max_ep_step=50,  # Longer
        update_global_iter=10,
        run_in_main=True,
    )

    agent.train()


def test_agent_moving_average_reward():
    """Test that moving average reward is updated."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=3,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Train
    agent.train()

    # Moving average should have been updated
    # (might be 0 if initial, but should be set)
    assert agent.global_ep_r.value != 1000.0  # Not initial impossible value


def test_agent_with_different_gamma():
    """Test agent training with different gamma values."""
    for gamma in [0.9, 0.95, 0.99]:
        agent = Agent(
            env_function=_env_function,
            gamma=gamma,
            n_workers=1,
            max_episodes=2,
            max_ep_step=10,
            update_global_iter=5,
            run_in_main=True,
        )

        agent.train()


def test_agent_with_different_learning_rates():
    """Test agent training with different learning rates."""
    for lr in [1e-4, 1e-3, 1e-2]:
        agent = Agent(
            env_function=_env_function,
            lr=lr,
            n_workers=1,
            max_episodes=2,
            max_ep_step=10,
            update_global_iter=5,
            run_in_main=True,
        )

        agent.train()


def test_agent_completes_max_episodes():
    """Test that agent stops after max_episodes."""
    max_ep = 5
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=max_ep,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    agent.train()

    # Should have completed exactly max_episodes
    assert agent.global_ep.value == max_ep


def test_agent_gradient_flow():
    """Test that gradients flow during training."""
    agent = Agent(
        env_function=_env_function,
        n_workers=1,
        max_episodes=1,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Save initial parameters
    initial_params = [p.data.clone() for p in agent.gnet.parameters()]

    # Train
    agent.train()

    # At least one parameter should have changed
    any_changed = any(
        not init.eq(curr.data).all()
        for init, curr in zip(initial_params, agent.gnet.parameters())
    )

    assert any_changed


def test_agent_handles_zero_rewards():
    """Test agent with environment giving zero rewards."""

    class _ZeroRewardEnv(_DummyEnv):
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 10
            return (
                np.random.randn(4).astype(np.float32),
                0.0,  # Always zero reward
                done,
                False,
                {},
            )

    def env_fn(worker_id):
        return _ZeroRewardEnv()

    agent = Agent(
        env_function=env_fn,
        n_workers=1,
        max_episodes=2,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    # Should handle zero rewards without error
    agent.train()


def test_agent_handles_negative_rewards():
    """Test agent with negative rewards."""

    class _NegativeRewardEnv(_DummyEnv):
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 10
            return (
                np.random.randn(4).astype(np.float32),
                -1.0,  # Always negative
                done,
                False,
                {},
            )

    def env_fn(worker_id):
        return _NegativeRewardEnv()

    agent = Agent(
        env_function=env_fn,
        n_workers=1,
        max_episodes=2,
        max_ep_step=10,
        update_global_iter=5,
        run_in_main=True,
    )

    agent.train()


def test_agent_with_variable_episode_lengths():
    """Test agent with episodes of variable length."""

    class _VariableLengthEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.episode_length = np.random.randint(5, 15)

        def reset(self, seed=None):
            self.step_count = 0
            self.episode_length = np.random.randint(5, 15)
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= self.episode_length
            return (
                np.random.randn(4).astype(np.float32),
                0.1,
                done,
                False,
                {},
            )

    def env_fn(worker_id):
        return _VariableLengthEnv()

    agent = Agent(
        env_function=env_fn,
        n_workers=1,
        max_episodes=3,
        max_ep_step=20,
        update_global_iter=5,
        run_in_main=True,
    )

    agent.train()


if __name__ == "__main__":
    # Run all tests
    test_agent_train_completes()
    test_agent_updates_global_network()
    test_agent_updates_episode_counter()
    test_agent_collects_results()
    test_agent_with_multiple_workers()
    test_agent_with_different_update_frequencies()
    test_agent_with_short_episodes()
    test_agent_with_long_episodes()
    test_agent_moving_average_reward()
    test_agent_with_different_gamma()
    test_agent_with_different_learning_rates()
    test_agent_completes_max_episodes()
    test_agent_gradient_flow()
    test_agent_handles_zero_rewards()
    test_agent_handles_negative_rewards()
    test_agent_with_variable_episode_lengths()
    print("All integration tests passed!")
