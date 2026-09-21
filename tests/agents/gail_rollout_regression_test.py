"""Exercise GAIL collection across shared buffers, evaluation and time limits."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.gail import model as gail_module


class CountingEnv(gym.Env):
    def __init__(self, terminal=False, horizon=10000):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,))
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (1,))
        self.state = np.zeros(1, dtype=np.float32)
        self.steps = 0
        self.horizon = horizon
        self.terminal = terminal
        self.before_step = 0.0

    def reset(self, **kwargs):
        self.steps = 0
        self.state[:] = 0.0
        return self.state, {}

    def step(self, action):
        self.before_step = float(self.state[0])
        self.state += 1.0
        self.steps += 1
        done = self.steps >= self.horizon
        return self.state, 1.0, done and self.terminal, done and not self.terminal, {}


def make_agent(env, max_steps=16):
    torch.manual_seed(11)
    return gail_module.GAIL(
        env, 1e-4, max_steps, 8, 1, np.zeros((16, 2), dtype=np.float32), device="cpu"
    )


def test_collection_preserves_states_and_exact_budget_across_evaluation(monkeypatch):
    env = CountingEnv()
    agent = make_agent(env, max_steps=64)
    collected = []
    evaluations = []
    gae = gail_module.compute_gae

    samples = []
    last_input = []
    handle = agent.model.register_forward_pre_hook(
        lambda module, args: last_input.append(args[0].detach().clone())
    )
    original_step = env.step

    def step(action):
        assert float(last_input[-1][0, 0]) == float(env.state[0])
        samples.append(float(env.state[0]))
        return original_step(action)

    def reward(state, action):
        observed = state[:, 0].cpu().tolist()
        assert observed == samples[len(collected) : len(collected) + len(observed)]
        collected.extend(observed)
        return np.ones((len(observed), 1), dtype=np.float32)

    env.step = step

    def evaluate():
        # Evaluation on self.env must only occur after the training rollout
        # is finalized, and training must reset before its next transition.
        evaluations.append(agent.global_env_step)
        env.state[:] = -1000.0
        return 0.0

    def capture(next_value, rewards, masks, values):
        return gae(next_value, rewards, masks, values)

    agent.expert_reward = reward
    agent.test_env = evaluate
    monkeypatch.setattr(gail_module, "compute_gae", capture)
    agent.learn(max_frames=1003, max_reward=float("inf"))
    handle.remove()
    assert agent.global_env_step == 1003
    assert len(collected) == 1003
    assert evaluations == [1000] * 10
    assert collected[1000:] == [0.0, 1.0, 2.0]
    assert all(torch.isfinite(p).all() for p in agent.model.parameters())


@pytest.mark.parametrize("terminal", [True, False])
@pytest.mark.parametrize(
    "final_key", [None, "final_observation", "terminal_observation", "empty_final"]
)
def test_time_limit_bootstraps_actual_final_state(monkeypatch, terminal, final_key):
    env = CountingEnv(terminal=terminal, horizon=1)
    if final_key is not None:
        step = env.step

        def autoreset(action):
            obs, reward, terminated, truncated, info = step(action)
            key = "terminal_observation" if final_key == "empty_final" else final_key
            info[key] = obs.copy()
            if final_key == "empty_final":
                info["final_observation"] = None
            env.state[:] = -50.0
            return env.state, reward, terminated, truncated, info

        env.step = autoreset

    agent = make_agent(env, max_steps=1)

    def model(state):
        return (
            torch.distributions.Normal(torch.zeros_like(state), torch.ones_like(state)),
            state.clone(),
        )

    agent.model.forward = model
    agent.ppo_update = lambda *args, **kwargs: None
    agent.expert_reward = lambda state, action: np.ones((1, 1), dtype=np.float32)
    captured = []
    original = gail_module.compute_gae

    def capture(next_value, rewards, masks, values):
        result = original(next_value, rewards, masks, values)
        captured.append(result[0].item())
        return result

    monkeypatch.setattr(gail_module, "compute_gae", capture)
    agent.learn(max_frames=1, max_reward=float("inf"))
    assert captured == pytest.approx([1.0 if terminal else 1.99])


def test_imitation_reward_is_finite_when_discriminator_saturates():
    agent = make_agent(CountingEnv())
    agent.discriminator.logits = lambda state: torch.full((state.shape[0], 1), -1000.0)
    reward = agent.expert_reward(torch.zeros((1, 1)), np.zeros((1, 1)))
    assert np.isfinite(reward).all()


def test_small_rollout_updates_policy_and_respects_epoch_count():
    agent = make_agent(CountingEnv(), max_steps=3)
    agent.epochs = 2
    agent.learn(max_frames=3, max_reward=float("inf"))
    assert agent.update_count == 2


def test_ppo_iterator_covers_all_samples_once_including_remainder():
    states = torch.arange(7).reshape(-1, 1)
    batches = list(gail_module.ppo_iter(4, states, states, states, states, states))
    ids = torch.cat([batch[0] for batch in batches]).flatten().sort().values
    torch.testing.assert_close(ids, torch.arange(7))


def test_discriminator_receives_executed_bounded_action():
    env = CountingEnv()
    env.action_space = gym.spaces.Box(-0.01, 0.01, (1,))
    agent = make_agent(env, max_steps=1)
    seen = []
    reward = agent.expert_reward

    def capture(state, action):
        assert np.max(np.abs(action)) <= 0.010001
        seen.append(action.copy())
        return reward(state, action)

    agent.expert_reward = capture
    agent.learn(max_frames=1, max_reward=float("inf"))
    assert len(seen) == 1


def test_ppo_uses_joint_probability_ratio_for_multidimensional_actions():
    from unittest.mock import Mock

    from tensoraerospace.agent.metrics import schema

    env = CountingEnv()
    env.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
    agent = make_agent(env)
    agent.writer = Mock()
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.0
    states = torch.zeros((2, 1))
    actions = torch.tensor([[1.0, 2.0], [-1.0, -0.5]])
    with torch.no_grad():
        dist, values = agent.model(states)
        # Both components have ratio 1.1, so their joint ratio is 1.21.
        old_log_probs = dist.log_prob(actions) - np.log(1.1)
    agent.ppo_update(
        1, 2, states, actions, old_log_probs, values, torch.tensor([[1.0], [-1.0]])
    )
    actor_losses = [
        call.args[1]
        for call in agent.writer.add_scalar.call_args_list
        if call.args[0] == schema.LOSS_ACTOR
    ]
    # PPO clips the positive advantage at 1.2; negative stays at -1.21.
    assert actor_losses == pytest.approx([0.005], abs=1e-6)
