"""Public DDPG inference must use the same observation coordinates as training."""

import numpy as np
import pytest

from tests.agents.ddpg_transition_regression_test import make_agent


@pytest.mark.parametrize("normalize", [False, True])
def test_prediction_matches_training_coordinates_and_does_not_update_statistics(
    normalize,
):
    agent = make_agent(normalize)
    if normalize:
        agent.obs_rms.mean[:] = 10.0
        agent.obs_rms.var[:] = 4.0
    raw = np.array([[12.0], [7.0], [10.0]])
    expected = agent.policy_net.get_action(agent._normalize_observation(raw))
    stats = None if agent.obs_rms is None else agent.obs_rms.state_dict()
    noise = agent.ou_noise.state.copy()
    np.testing.assert_array_equal(agent.predict(raw), expected)
    np.testing.assert_array_equal(
        agent.predict(raw[0]),
        agent.policy_net.get_action(agent._normalize_observation(raw[0])),
    )
    assert stats == (None if agent.obs_rms is None else agent.obs_rms.state_dict())
    np.testing.assert_array_equal(agent.ou_noise.state, noise)
    if normalize:
        assert not np.allclose(agent.predict(raw), agent.policy_net.get_action(raw))


def test_checkpoint_prediction_preserves_raw_observation_contract(tmp_path):
    agent = make_agent(True)
    agent.obs_rms.mean[:] = 10.0
    agent.obs_rms.var[:] = 4.0
    observations = np.array([[12.0], [7.0]])
    expected = agent.predict(observations)
    path = tmp_path / "ddpg.pt"
    agent.save(path)
    loaded = make_agent(False)
    loaded.load(path)
    np.testing.assert_array_equal(loaded.predict(observations), expected)


@pytest.mark.parametrize(
    "state", [np.array(1.0), np.zeros(2), np.zeros((1, 1, 1)), np.array([np.nan])]
)
def test_prediction_rejects_invalid_observations(state):
    with pytest.raises(ValueError):
        make_agent().predict(state)
