"""UAV actuator, observation and reference contracts in radians."""

import numpy as np
import pytest
from scipy.linalg import expm

from tensoraerospace.aerospacemodel.uav import LongitudinalUAV
from tensoraerospace.envs.uav import LinearLongitudinalUAV


def env(**kwargs):
    args = dict(
        initial_state=np.zeros(4),
        reference_signal=np.zeros((1, 5)),
        number_time_steps=5,
    )
    args.update(kwargs)
    return LinearLongitudinalUAV(**args)


def test_first_elevator_command_obeys_rate_limit_from_rest():
    plant = LongitudinalUAV(np.zeros(4), 3)
    plant.run_step(np.array([1.0]))
    assert plant.store_input[0, 0] == pytest.approx(np.deg2rad(0.6))


def test_initial_actuator_position_is_preserved_after_reset():
    plant = LongitudinalUAV(np.zeros(4), 3, initial_control=np.deg2rad(5))
    for _ in range(2):
        plant.run_step(np.array([-1.0]))
        assert plant.store_input[0, 0] == pytest.approx(np.deg2rad(4.4))
        plant.initialise_system(np.zeros(4), 3)


def test_initial_array_cannot_mutate_the_live_plant():
    initial = np.zeros(4)
    plant = LongitudinalUAV(initial, 3)
    initial[3] = 0.2
    np.testing.assert_array_equal(plant.run_step(np.zeros(1)), np.zeros((4, 1)))


@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan])
def test_invalid_clock_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        LongitudinalUAV(np.zeros(4), 3, dt=dt)


@pytest.mark.parametrize("command", [[np.nan], [np.inf], [], [0.0, 0.0]])
def test_invalid_action_keeps_model_and_environment_clocks_unchanged(command):
    task = env()
    task.reset()
    with pytest.raises(ValueError, match="control"):
        task.step(command)
    assert task.current_step == task.model.time_step == 0
    np.testing.assert_array_equal(task.model.xt.reshape(-1), np.zeros(4))


def test_action_space_uses_the_models_radian_amplitude_bounds():
    task = env()
    np.testing.assert_allclose(task.action_space.high, np.deg2rad([25.0]))
    np.testing.assert_allclose(task.action_space.low, -np.deg2rad([25.0]))


def test_output_order_does_not_change_which_state_is_rewarded():
    task = env(
        initial_state=np.array([0.1, 0.2, 0.3, 0.4]), output_space=["q", "theta"]
    )
    task.reset()
    observation, reward, *_ = task.step([0.0])
    np.testing.assert_allclose(observation, task.model.xt.reshape(-1)[[2, 3]])
    assert reward == pytest.approx(-abs(task.model.xt.reshape(-1)[3]))


def test_tracking_can_use_a_state_that_is_not_observed():
    task = env(
        initial_state=np.array([0.0, 0.0, 0.1, 0.2]),
        state_space=["q"],
        output_space=["q"],
        tracking_states=["theta"],
    )
    obs, _ = task.reset()
    assert obs.shape == task.observation_space.shape == (1,)
    _, reward, *_ = task.step([0.0])
    assert reward == pytest.approx(-abs(task.model.xt.reshape(-1)[3]))


def test_observation_space_matches_explicit_selected_outputs():
    task = env(output_space=["u", "w", "theta"])
    obs, _ = task.reset()
    assert task.observation_space.contains(obs)
    obs, *_ = task.step([0.0])
    assert task.observation_space.contains(obs)


def test_state_space_is_the_default_observation_selection():
    task = env(initial_state=np.array([0.1, 0.2, 0.3, 0.4]), state_space=["u"])
    obs, _ = task.reset()
    np.testing.assert_allclose(obs, [0.1])


def test_multiple_reference_channels_reward_corresponding_tracked_states():
    task = env(
        initial_state=np.array([0.0, 0.0, 0.1, 0.2]),
        reference_signal=np.array([[0.3], [0.4]]),
    )
    task.reset()
    _, reward, *_ = task.step([0.0])
    tracked = task.model.xt.reshape(-1)[[3, 2]]
    assert reward == pytest.approx(-np.mean(abs(tracked - [0.3, 0.4])))


def test_callable_reference_and_nondefault_clock_match_sampled_reference():
    initial = np.array([0.1, 0.2, 0.03, 0.04])
    task = env(initial_state=initial, dt=0.02, reference_signal=lambda t: t)
    task.reset()
    obs, reward, *_ = task.step([0.0])
    expected = expm(task.model.A * 0.02) @ initial
    np.testing.assert_allclose(obs, expected[[3, 2]], rtol=1e-6)
    assert reward == pytest.approx(-abs(expected[3] - 0.02))


def test_environment_copies_initial_state_and_reference():
    initial, reference = np.zeros(4), np.zeros((1, 5))
    task = env(initial_state=initial, reference_signal=reference)
    initial[3] = 0.3
    reference[:] = 10
    obs, _ = task.reset()
    np.testing.assert_array_equal(obs, np.zeros(2))
    assert task.step([0.0])[1] == 0.0


def test_horizon_is_a_time_limit_and_reset_restores_the_trajectory():
    task = env(number_time_steps=3)
    initial, _ = task.reset()
    task.step([0.01])
    assert task.step([0.01])[2:4] == (False, True)
    np.testing.assert_array_equal(task.reset()[0], initial)


@pytest.mark.parametrize("conversion", [None, "to_deg", "to_rad"])
def test_output_history_contains_every_pretransition_sample(conversion):
    initial = np.array([0.1, 0.2, 0.03, 0.04])
    plant = LongitudinalUAV(initial, 3)
    assert plant.get_output("theta").size == 0
    plant.run_step(np.array([0.0]))
    second = float(plant.xt.reshape(-1)[3])
    plant.run_step(np.array([0.0]))
    expected = np.array([initial[3], second])
    kwargs = {} if conversion is None else {conversion: True}
    if conversion == "to_deg":
        expected = np.rad2deg(expected)
    if conversion == "to_rad":
        expected = np.deg2rad(expected)
    np.testing.assert_allclose(plant.get_output("theta", **kwargs), expected)
    np.testing.assert_allclose(plant.store_outputs[3, :2], [initial[3], second])


@pytest.mark.parametrize("lang", ["rus", "eng"])
def test_velocity_plot_keeps_si_units_and_does_not_mutate_the_history(lang):
    import matplotlib.pyplot as plt

    plant = LongitudinalUAV(np.array([0.2, 0.1, 0.03, 0.02]), 3)
    plant.run_step(np.array([0.0]))
    plant.run_step(np.array([0.0]))
    recorded = plant.store_outputs.copy()
    fig = plant.plot_output("u", np.arange(3) * plant.dt, lang=lang)
    try:
        np.testing.assert_array_equal(plant.store_outputs, recorded)
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), recorded[0, :2])
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [0, plant.dt])
    finally:
        plt.close(fig)
