"""AIDIAgent unit tests — API surface, save/load, single full step."""

import math

import numpy as np
import pytest

from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig
from tensoraerospace.agent.aidi.onboard_ce import LinearOnboardCE


def _make_obs(p=0.0, q=0.0, r=0.0, alpha=0.05, beta=0.0,
              theta=0.0, phi=0.0, V=200.0):
    return {
        "omega": np.array([p, q, r]),
        "alpha": alpha, "beta": beta,
        "theta": theta, "phi": phi, "V": V,
    }


def _toy_onboard_ce():
    B = np.array(
        [[ 0.10, -2.50,  0.00],
         [-3.00,  0.05,  0.00],
         [ 0.02,  0.00, -1.20]],
        dtype=np.float64,
    )
    return LinearOnboardCE(B)


def _refs():
    return {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}


def test_aidi_agent_predict_returns_correct_shape():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    u = agent.predict(_make_obs(), references=_refs(), time_step=0)
    assert u.shape == (3,)


def test_aidi_agent_full_step_records_metrics():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    agent.predict(_make_obs(), references=_refs(), time_step=0)
    metrics = agent.learn(_make_obs(p=0.01),
                          references=_refs(), time_step=0)
    assert {"residual_norm", "lambda_min", "G_norm", "frozen_axes"} <= set(metrics)


def test_aidi_agent_reset_clears_loop_state_keeps_theta():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    agent.predict(_make_obs(), references=_refs(), time_step=0)
    agent.learn(_make_obs(p=0.01), references=_refs(), time_step=0)
    theta_before = agent.rls.theta.copy()
    agent.reset()
    np.testing.assert_array_equal(agent.rls.theta, theta_before)
    np.testing.assert_array_equal(agent._u_prev, np.zeros(3))


def test_aidi_agent_predict_rejects_missing_keys():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    bad_obs = {"omega": np.zeros(3)}
    with pytest.raises(KeyError):
        agent.predict(bad_obs, references=_refs(), time_step=0)


def test_aidi_agent_save_load_roundtrip(tmp_path):
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01, seed=7))
    for k in range(20):
        agent.predict(_make_obs(), references=_refs(), time_step=k)
        next_obs = _make_obs(p=0.001 * (k + 1), q=0.002)
        agent.learn(next_obs, references=_refs(), time_step=k)

    run_dir = agent.save(path=str(tmp_path))
    loaded = AIDIAgent.from_pretrained(run_dir, onboard_ce=_toy_onboard_ce())
    np.testing.assert_array_equal(loaded.rls.theta, agent.rls.theta)
    np.testing.assert_array_equal(loaded.rls.P, agent.rls.P)
    np.testing.assert_array_equal(loaded._u_prev, agent._u_prev)
    np.testing.assert_array_equal(loaded._omega_dot_cached,
                                  agent._omega_dot_cached)
    np.testing.assert_array_equal(loaded._last_G_nominal,
                                  agent._last_G_nominal)


def test_aidi_agent_n_z_reconstruction_when_missing():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    obs = _make_obs(alpha=math.radians(2.0), V=200.0)  # n_z absent.
    u = agent.predict(obs, references=_refs(), time_step=0)
    assert u.shape == (3,)
    obs_with_nz = dict(obs); obs_with_nz["n_z"] = 1.5
    u2 = agent.predict(obs_with_nz, references=_refs(), time_step=0)
    assert u2.shape == (3,)
