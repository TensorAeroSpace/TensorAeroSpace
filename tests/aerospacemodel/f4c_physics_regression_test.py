"""Physical actuator, observation and history contracts for F4C."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f4c import LongitudinalF4C
from tensoraerospace.envs.f4c import F4CPitchEnvNormalized, LinearLongitudinalF4C


def test_first_step_obeys_elevator_rate_limit():
    model = LongitudinalF4C(np.zeros(4), 5, dt=0.01)
    model.run_step(np.array([np.deg2rad(20)]))
    assert np.rad2deg(model.store_input[0, 0]) == pytest.approx(0.6)


def test_initial_actuator_position_is_preserved_after_reset():
    model = LongitudinalF4C(np.zeros(4), 5, dt=0.01, initial_control=np.deg2rad(5))
    for _ in range(2):
        model.run_step(np.array([0]))
        assert np.rad2deg(model.store_input[0, 0]) == pytest.approx(4.4)
        model.initialise_system(np.zeros(4), 5)


def test_initial_state_is_not_aliased():
    initial = np.array([0.1, 0.2, 0, 0.01])
    model = LongitudinalF4C(initial, 5)
    expected = model.filt_A @ initial
    initial[:] = 100
    np.testing.assert_allclose(model.run_step([0]).ravel(), expected)


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [1, 2], []])
def test_bad_action_does_not_advance_model(action):
    model = LongitudinalF4C(np.zeros(4), 5)
    with pytest.raises(ValueError):
        model.run_step(action)
    assert model.time_step == 0
    np.testing.assert_array_equal(model.xt, np.zeros(4))


@pytest.mark.parametrize("convert", [{}, {"to_deg": True}, {"to_rad": True}])
def test_complete_output_history(convert):
    model = LongitudinalF4C([1, 0, 0, 0], 5)
    assert model.get_output("u", **convert).size == 0
    for _ in range(3):
        model.run_step([0])
    expected = model.store_states[0, :3].copy()
    if convert.get("to_deg"):
        expected = np.rad2deg(expected)
    if convert.get("to_rad"):
        expected = np.deg2rad(expected)
    np.testing.assert_array_equal(model.get_output("u", **convert), expected)


@pytest.mark.parametrize("lang", ["eng", "rus"])
def test_plot_velocity_does_not_convert_or_modify_history(lang):
    model = LongitudinalF4C([1, 0, 0, 0], 5)
    for _ in range(3):
        model.run_step([0])
    original = model.store_outputs.copy()
    fig = model.plot_output("u", np.arange(5) * 0.01, lang=lang)
    try:
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), original[0, :3])
        np.testing.assert_array_equal(model.store_outputs, original)
    finally:
        plt.close(fig)


def test_linear_reward_tracks_pitch_from_full_model_state():
    env = LinearLongitudinalF4C([0.1, 0.2, 0.01, 0.02], np.zeros((1, 4)), 4)
    obs, reward, _, _, _ = env.step([0])
    assert reward == pytest.approx(-abs(env.model.xt[3, 0]))
    np.testing.assert_allclose(obs, env.model.xt.ravel())


def test_linear_output_selection_and_multichannel_reward():
    env = LinearLongitudinalF4C(
        [0.1, 0.2, 0.01, 0.02],
        np.zeros((2, 4)),
        4,
        state_space=["u", "w"],
        output_space=["q", "theta"],
        tracking_states=["theta", "q"],
    )
    obs, _ = env.reset()
    np.testing.assert_allclose(obs, [0.01, 0.02])
    assert env.observation_space.contains(obs)
    obs, reward, _, _, _ = env.step([0])
    np.testing.assert_allclose(obs, env.model.xt.ravel()[[2, 3]])
    assert reward == pytest.approx(-np.mean(abs(env.model.xt.ravel()[[3, 2]])))


def test_linear_dt_reference_and_time_limit():
    env = LinearLongitudinalF4C(np.zeros(4), lambda t: t, 3, dt=0.02)
    np.testing.assert_allclose(env.reference_signal, [[0, 0.02, 0.04]])
    assert env.model.dt == 0.02
    assert env.step([0])[1] == pytest.approx(-0.02)
    assert env.step([0])[2:4] == (False, True)


def test_linear_reset_owns_initial_state_and_reference():
    initial = np.zeros(4)
    reference = np.zeros((1, 3))
    env = LinearLongitudinalF4C(initial, reference, 3)
    initial[:] = 99
    reference[:] = 99
    np.testing.assert_array_equal(env.reset()[0], np.zeros(4))
    assert env.step([0])[1] == pytest.approx(0)


def normalized(**kwargs):
    defaults = dict(
        initial_state=np.zeros(4),
        reference_signal=np.zeros((1, 20)),
        number_time_steps=20,
        dt=0.01,
        use_initial_action_on_first_step=False,
    )
    defaults.update(kwargs)
    return F4CPitchEnvNormalized(**defaults)


def test_normalized_observation_and_reward_use_applied_elevator():
    env = normalized()
    env.reset()
    env.step([0])
    obs, reward, _, _, info = env.step([1])
    applied = env.model.store_input[0, 1] / env.max_elevator_angle_rad
    assert obs[3] == pytest.approx(applied)
    assert env.previous_action == pytest.approx(applied)
    assert info["elevator_deg"] == pytest.approx(0.6)
    cost = (
        env.w_pitch * (env.state[3] / env.max_pitch_rad) ** 2
        + env.w_q * (env.state[2] / env.max_pitch_rate_rad_s) ** 2
        + (env.w_action + env.w_smooth + env.w_jerk) * applied**2
    )
    assert reward == pytest.approx(-env.reward_scale * cost)


def test_normalized_initial_elevator_sets_physical_actuator_state():
    env = normalized(initial_elevator_deg=5)
    for _ in range(2):
        assert env.reset()[0][3] == pytest.approx(0.25)
        env.step([0])
        assert np.rad2deg(env.model.store_input[0, 0]) == pytest.approx(4.4)


def test_normalized_pitch_rate_envelope_and_last_reward():
    env = normalized(initial_state=[0, 0, np.deg2rad(12), 0])
    _, reward, terminated, _, _ = env.step([0])
    assert terminated
    assert reward == -100
    assert env._last_reward == reward


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [0, 1], []])
@pytest.mark.parametrize("kind", ["linear", "normalized"])
def test_invalid_environment_action_does_not_advance(action, kind):
    env = (
        normalized()
        if kind == "normalized"
        else LinearLongitudinalF4C(np.zeros(4), np.zeros((1, 4)), 4)
    )
    with pytest.raises(ValueError):
        env.step(action)
    assert env.current_step == env.model.time_step == 0


@pytest.mark.parametrize("dt", [0, -0.1, np.nan, np.inf])
def test_invalid_clock_is_rejected(dt):
    with pytest.raises(ValueError):
        LongitudinalF4C(np.zeros(4), 4, dt=dt)


@pytest.mark.parametrize("initial", [[0, 0, 0], [0, 0, np.nan, 0]])
def test_invalid_initial_state_is_rejected(initial):
    with pytest.raises(ValueError):
        LongitudinalF4C(initial, 4)


@pytest.mark.parametrize("steps", [0, -1, 1.5])
def test_invalid_horizon_is_rejected(steps):
    with pytest.raises(ValueError):
        LongitudinalF4C(np.zeros(4), steps)


def test_complete_model_does_not_advance_past_history():
    model = LongitudinalF4C(np.zeros(4), 1)
    model.run_step([0])
    with pytest.raises(RuntimeError, match="complete"):
        model.run_step([0])
    assert model.time_step == 1


@pytest.mark.parametrize("initial_control", [np.nan, np.inf])
def test_invalid_initial_actuator_position_is_rejected(initial_control):
    with pytest.raises(ValueError):
        LongitudinalF4C(np.zeros(4), 4, initial_control=initial_control)


def test_normalized_clips_initial_actuator_consistently():
    env = normalized(initial_elevator_deg=100, use_initial_action_on_first_step=True)
    assert env.reset()[0][3] == 1
    env.step([0])
    assert np.rad2deg(env.model.store_input[0, 0]) == pytest.approx(20)
    assert env.previous_action == pytest.approx(1)


@pytest.mark.parametrize("kind", ["linear", "normalized"])
@pytest.mark.parametrize(
    "reference", [np.zeros((1, 0)), np.array([[np.nan]]), np.zeros(3)]
)
def test_invalid_reference_is_rejected(kind, reference):
    cls = LinearLongitudinalF4C if kind == "linear" else F4CPitchEnvNormalized
    with pytest.raises(ValueError):
        cls(np.zeros(4), reference, 4)
