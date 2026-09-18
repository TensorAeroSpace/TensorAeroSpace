"""DDPG replay must retain physical transitions and consistent coordinates."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ddpg.model import DDPG, ReplayBuffer
from tests.agents.sac_scalar_transition_regression_test import BufferEnv


def make_agent(normalize=False, final_key=None, terminal=False):
    torch.manual_seed(11)
    a = DDPG(
        BufferEnv(final_key, terminal),
        1e-3,
        1e-4,
        100,
        normalize_observations=normalize,
        device="cpu",
    )
    a.writer = Mock()
    a.writer.assert_contract_satisfied = Mock()
    return a


def test_replay_owns_all_arrays():
    replay = ReplayBuffer(3)
    state = np.array([1.0])
    action = np.array([0.2])
    next_state = np.array([2.0])
    replay.push(state, action, 1.0, next_state, False)
    state[:] = 9
    action[:] = 9
    next_state[:] = 9
    s, a, _, n, _ = replay.buffer[0]
    np.testing.assert_array_equal(s, [1])
    np.testing.assert_array_equal(a, [0.2])
    np.testing.assert_array_equal(n, [2])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "final_key", [None, "final_observation", "terminal_observation"]
)
def test_collection_preserves_raw_states_and_time_limit(normalize, final_key):
    a = make_agent(normalize, final_key)
    a.learn(max_frames=4, max_steps=5, batch_size=16, warmup_frames=10)
    states, _, _, next_states, terminals = map(np.stack, zip(*a.replay_buffer.buffer))
    np.testing.assert_array_equal(states.ravel(), [1, 2, 1, 2])
    np.testing.assert_array_equal(next_states.ravel(), [2, 3, 2, 3])
    np.testing.assert_array_equal(terminals, np.zeros(4))
    if normalize:
        assert a.obs_rms.mean[0] == pytest.approx(1.5, abs=1e-4)


def test_sampled_replay_uses_current_normalization():
    a = make_agent(True)
    a.obs_rms.mean[:] = 10.0
    a.obs_rms.var[:] = 4.0
    a.replay_buffer.push(
        np.array([12.0]), np.array([0.1]), 1.0, np.array([14.0]), False
    )
    a.obs_rms.mean[:] = 8.0  # Stats can change after collection, before sampling.
    seen = []
    h = a.value_net.register_forward_pre_hook(
        lambda m, args: seen.append(args[0].detach().clone())
    )
    next_seen = []
    h2 = a.target_value_net.register_forward_pre_hook(
        lambda m, args: next_seen.append(args[0].detach().clone())
    )
    a.frame_idx = 1
    a.ddpg_update(1)
    h.remove()
    h2.remove()
    assert all(torch.allclose(v, torch.tensor([[2.0]])) for v in seen)
    assert all(torch.allclose(v, torch.tensor([[3.0]])) for v in next_seen)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_restores_policy_and_compatible_replay(tmp_path, normalize, legacy):
    agent = make_agent(normalize)
    if normalize:
        agent.obs_rms.mean[:] = 10.0
        agent.obs_rms.var[:] = 4.0
    state = np.array([12.0])
    with torch.no_grad():
        expected = agent.policy_net(
            torch.tensor(agent._normalize_observation(state), dtype=torch.float32)
        )
    agent.replay_buffer.push(state, np.array([0.1]), 1.0, state + 1.0, False)
    path = tmp_path / "ddpg.pt"
    agent.save(path)
    if legacy:
        checkpoint = torch.load(path, weights_only=False)
        checkpoint.pop("replay_observation_format")
        torch.save(checkpoint, path)
    restored = make_agent(not normalize)
    restored.replay_buffer.push(state * 99.0, np.array([0.1]), 0.0, state, False)
    if normalize and legacy:
        with pytest.warns(UserWarning, match="Skipping incompatible DDPG replay"):
            restored.load(path)
        assert len(restored.replay_buffer) == 0
    else:
        restored.load(path)
        assert len(restored.replay_buffer) == 1
        np.testing.assert_array_equal(restored.replay_buffer.buffer[0][0], state)
    assert restored.normalize_observations == normalize
    with torch.no_grad():
        actual = restored.policy_net(
            torch.tensor(restored._normalize_observation(state), dtype=torch.float32)
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_noise_schedule_uses_current_step(monkeypatch):
    from tensoraerospace.agent.ddpg.model import OUNoise

    noise = OUNoise(
        BufferEnv().action_space, max_sigma=0.3, min_sigma=0.1, decay_period=4
    )
    monkeypatch.setattr(np.random, "randn", lambda n: np.ones(n))
    np.testing.assert_allclose(noise.get_action(np.zeros(1), t=4), [0.1])


def test_noise_schedule_does_not_restart_at_episode_boundary():
    agent = make_agent()
    calls = Mock(wraps=agent.ou_noise.get_action)
    agent.ou_noise.get_action = calls
    agent.learn(max_frames=4, max_steps=5, batch_size=16, warmup_frames=10)
    assert [call.args[1] for call in calls.call_args_list] == [0, 1, 2, 3]
