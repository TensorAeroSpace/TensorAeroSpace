"""Coverage tests for NonlinearLongitudinalF16 Gym env."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensoraerospace.envs.f16.nonlinear_longitudinal import (
    MODEL_STATE_ORDER,
    NonlinearLongitudinalF16,
)


def _ref_sin(
    n: int, amp_deg: float = 2.0, freq_hz: float = 0.1, dt: float = 0.01
) -> np.ndarray:
    t = np.arange(n) * dt
    return (math.radians(amp_deg) * np.sin(2 * np.pi * freq_hz * t)).reshape(1, -1)


def test_defaults_match_linear_env_api():
    ref = _ref_sin(200)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=200,
    )
    assert env.tracking_states == ["alpha"]
    assert env.state_space == ["alpha", "wz"]
    assert env.control_space == ["stab"]
    assert env.output_space == ["alpha", "wz"]
    assert env.indices_tracking_states == [0]
    # action and observation spaces reflect the configured dims.
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (2,)
    # default control limit is 25 deg.
    assert env.max_action_value == 25.0


def test_reset_returns_selected_state_subset():
    ref = _ref_sin(100)
    # Provide a full 4-element model state to exercise the len-match branch.
    x0 = np.array([math.radians(1.5), math.radians(0.1), 0.0, 0.0])
    env = NonlinearLongitudinalF16(
        initial_state=x0,
        reference_signal=ref,
        number_time_steps=100,
    )
    obs, info = env.reset()
    assert obs.shape == (2,)
    assert info == {}
    # state_space=[alpha, wz] exposed as the first two model states.
    assert obs[0] == pytest.approx(x0[0], abs=1e-6)
    assert obs[1] == pytest.approx(x0[1], abs=1e-6)


def test_short_initial_state_is_padded_via_state_space():
    # Provide a 2-vector matching the default state_space [alpha, wz];
    # the env must pad it into the full 4-element model state.
    ref = _ref_sin(50)
    env = NonlinearLongitudinalF16(
        initial_state=np.array([math.radians(0.5), 0.0]),
        reference_signal=ref,
        number_time_steps=50,
    )
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(math.radians(0.5), abs=1e-9)
    assert obs[1] == 0.0


def test_step_runs_and_counts():
    ref = _ref_sin(80)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=80,
    )
    env.reset()
    for k in range(78):
        obs, r, done, truncated, info = env.step(np.array([0.0]))
        assert obs.shape == (2,)
        assert isinstance(r, float)
        assert truncated is False
        assert info == {}
    # Next call should flip done True (current_step >= number_time_steps - 1).
    obs, r, done, truncated, info = env.step(np.array([0.0]))
    assert done is True


def test_default_reward_matches_manual_computation():
    # Sanity check the closed form of the default tracking-error reward.
    state = np.array([0.05, 0.02])
    ref = np.array([[0.03, 0.04, 0.05]])
    got = float(NonlinearLongitudinalF16.default_reward(state, ref, 1))
    expected = -abs(0.05 - 0.04) - 0.1 * abs(0.02)
    assert got == pytest.approx(expected, abs=1e-12)


def test_default_reward_clips_time_step_at_end():
    # `ts` beyond the reference length should clip to the last column.
    state = np.array([0.1, 0.0])
    ref = np.array([[0.0, 0.05]])
    r_large = float(NonlinearLongitudinalF16.default_reward(state, ref, 999))
    r_last = float(NonlinearLongitudinalF16.default_reward(state, ref, 1))
    assert r_large == pytest.approx(r_last, abs=1e-12)


def test_use_reward_false_returns_constant_reward():
    ref = _ref_sin(20)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=20,
        use_reward=False,
    )
    env.reset()
    _, r, *_ = env.step(np.array([0.0]))
    assert r == 1.0


def test_custom_reward_is_called():
    calls = {"count": 0}

    def my_reward(state, ref_signal, ts):
        calls["count"] += 1
        return -3.14

    ref = _ref_sin(10)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
        reward_func=my_reward,
    )
    env.reset()
    _, r, *_ = env.step(np.array([0.0]))
    assert calls["count"] == 1
    assert r == pytest.approx(-3.14, abs=1e-12)


def test_control_bias_shifts_action_in_degrees():
    # With control_bias = env's bound, the agent's zero action still delivers
    # the full positive bound after clipping, while a large positive input is
    # still clipped to max.
    ref = _ref_sin(50)
    bias = 25.0
    env_biased = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=50,
        control_bias=bias,
    )
    env_baseline = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=50,
    )
    env_biased.reset()
    env_baseline.reset()
    # Compare single-step evolution: baseline with action=+25 must match
    # biased with action=0 (both effectively deliver +25° elevator, then clip).
    obs_b1, *_ = env_biased.step(np.array([0.0]))
    obs_b2, *_ = env_baseline.step(np.array([bias]))
    np.testing.assert_allclose(obs_b1, obs_b2, rtol=1e-10, atol=1e-10)


def test_action_clipping_bounds_effect_on_state():
    # A huge action is clipped to ±max_action_value, so overshooting the bound
    # should give the same next state as sending exactly ±max_action_value.
    ref = _ref_sin(10)
    env_big = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
    )
    env_bounded = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
    )
    env_big.reset()
    env_bounded.reset()
    s1, *_ = env_big.step(np.array([1e6]))
    s2, *_ = env_bounded.step(np.array([env_bounded.max_action_value]))
    np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-10)


def test_feedforward_fn_is_added_to_action():
    captured_steps: list[int] = []

    def ff(time_step: int, reference_signal: np.ndarray) -> float:
        captured_steps.append(int(time_step))
        return 5.0

    ref = _ref_sin(10)
    env_ff = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
        feedforward_fn=ff,
    )
    env_plain = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
    )
    env_ff.reset()
    env_plain.reset()
    # FF callback adds 5.0 deg on top of action -> action=+2 with FF matches
    # action=+7 without FF.
    s1, *_ = env_ff.step(np.array([2.0]))
    s2, *_ = env_plain.step(np.array([7.0]))
    np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-10)
    # feedforward_fn is called with the current step index (pre-increment => 0).
    assert captured_steps == [0]


def test_feedforward_fn_returning_array_is_accepted():
    def ff(_t, _ref):
        return np.asarray([3.0])

    ref = _ref_sin(5)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=5,
        feedforward_fn=ff,
    )
    env.reset()
    # Should not raise — the array return is reshaped inside step().
    env.step(np.array([0.0]))


def test_reset_with_seed_is_deterministic():
    ref = _ref_sin(20)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=20,
    )
    obs1, _ = env.reset(seed=123)
    # After running a few steps, resetting again should put us back at obs1.
    env.step(np.array([5.0]))
    env.step(np.array([-2.0]))
    obs2, _ = env.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2, atol=1e-12)


def test_close_is_noop():
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=_ref_sin(5),
        number_time_steps=5,
    )
    # close() is a noop — should not raise.
    env.close()


def test_rk4_integrator_option():
    ref = _ref_sin(30)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=30,
        integrator="rk4",
    )
    env.reset()
    obs, *_ = env.step(np.array([1.0]))
    assert obs.shape == (2,)
    assert np.all(np.isfinite(obs))


def test_get_init_args_returns_construction_kwargs():
    ref = _ref_sin(10)
    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=10,
        integrator="rk4",
    )
    args = env.get_init_args()
    # Self/class/model_x0 should be stripped, but user-facing kwargs remain.
    assert "self" not in args
    assert "__class__" not in args
    assert "model_x0" not in args
    assert args["number_time_steps"] == 10
    assert args["integrator"] == "rk4"


def test_model_state_order_constant():
    # Guard against accidental reordering of the agent-visible state vector —
    # the env relies on this exact order when padding short initial states.
    assert MODEL_STATE_ORDER == ["alpha", "wz", "stab", "dstab"]


def test_reward_func_override_with_use_reward_false_still_returns_one():
    ref = _ref_sin(5)

    def bad(*_a, **_kw):
        raise AssertionError("should not be called when use_reward=False")

    env = NonlinearLongitudinalF16(
        initial_state=np.zeros(4),
        reference_signal=ref,
        number_time_steps=5,
        reward_func=bad,
        use_reward=False,
    )
    env.reset()
    _, r, *_ = env.step(np.array([0.0]))
    assert r == 1.0
