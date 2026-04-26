"""Tests that verify super().__init__() is called in all legacy environments.

Each legacy environment should have np_random attribute (set by gym.Env.__init__)
and reset(seed=42) should work without error.
"""

import numpy as np
import pytest

from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
NUMBER_TIME_STEPS = 1000
REFERENCE_SIGNAL = np.reshape(
    unit_step(degree=5, tp=tp, time_step=0.1, output_rad=True), [1, -1]
)
REFERENCE_SIGNAL_SMALL = np.reshape(
    unit_step(degree=0.1, tp=tp, time_step=0.1, output_rad=True), [1, -1]
)


def _make_b747():
    from tensoraerospace.envs.b747 import LinearLongitudinalB747

    return LinearLongitudinalB747(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_x15():
    from tensoraerospace.envs.x15 import LinearLongitudinalX15

    return LinearLongitudinalX15(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_comsat():
    from tensoraerospace.envs.comsat import ComSatEnv

    return ComSatEnv(
        initial_state=[[6371.0], [0.0], [0.001]],
        reference_signal=REFERENCE_SIGNAL_SMALL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_geosat():
    from tensoraerospace.envs.geostat import GeoSatEnv

    return GeoSatEnv(
        initial_state=[[0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_uav():
    from tensoraerospace.envs.uav import LinearLongitudinalUAV

    return LinearLongitudinalUAV(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_f4c():
    from tensoraerospace.envs.f4c import LinearLongitudinalF4C

    return LinearLongitudinalF4C(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_lapan():
    from tensoraerospace.envs.lapan import LinearLongitudinalLAPAN

    return LinearLongitudinalLAPAN(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_ultrastick():
    from tensoraerospace.envs.ultrastick import LinearLongitudinalUltrastick

    return LinearLongitudinalUltrastick(
        initial_state=[[0], [0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_rocket():
    from tensoraerospace.envs.rocket import LinearLongitudinalMissileModel

    return LinearLongitudinalMissileModel(
        initial_state=[[0], [0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_elv():
    from tensoraerospace.envs.elv import LinearLongitudinalELVRocket

    return LinearLongitudinalELVRocket(
        initial_state=[[0], [0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )


def _make_f16():
    from tensoraerospace.envs.f16 import LinearLongitudinalF16

    return LinearLongitudinalF16(
        initial_state=[[0], [0]],
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        state_space=["theta", "q"],
        output_space=["theta", "q"],
        tracking_states=["theta"],
    )


ENV_FACTORIES = [
    pytest.param(_make_b747, id="b747"),
    pytest.param(_make_x15, id="x15"),
    pytest.param(_make_comsat, id="comsat"),
    pytest.param(_make_geosat, id="geosat"),
    pytest.param(_make_uav, id="uav"),
    pytest.param(_make_f4c, id="f4c"),
    pytest.param(_make_lapan, id="lapan"),
    pytest.param(_make_ultrastick, id="ultrastick"),
    pytest.param(_make_rocket, id="rocket"),
    pytest.param(_make_elv, id="elv"),
    pytest.param(_make_f16, id="f16"),
]


@pytest.mark.parametrize("factory", ENV_FACTORIES)
def test_has_np_random(factory):
    """Each legacy env should have np_random after construction (set by gym.Env.__init__)."""
    env = factory()
    assert hasattr(
        env, "np_random"
    ), f"{type(env).__name__} missing np_random attribute; super().__init__() likely not called"


@pytest.mark.parametrize("factory", ENV_FACTORIES)
def test_reset_with_seed(factory):
    """Each legacy env's reset(seed=42) should work without error."""
    env = factory()
    result = env.reset(seed=42)
    assert result is not None, f"{type(env).__name__}.reset(seed=42) returned None"
    assert (
        len(result) == 2
    ), f"{type(env).__name__}.reset(seed=42) should return (obs, info)"
