import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear._integrators import euler
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    default_parameters,
)


def test_dx_shape_and_finiteness():
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0])
    dx = f16_ode_long(x, u, t=0.0, params=p)
    assert dx.shape == (4,)
    assert np.all(np.isfinite(dx))


def test_actuator_dstab_rate_limit_clamps_dx_2():
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 10.0 * p.maxabsdstab])
    u = np.array([0.0])
    dx = f16_ode_long(x, u, t=0.0, params=p)
    assert dx[2] == pytest.approx(p.maxabsdstab)


def test_actuator_position_command_limit_is_applied_before_ode():
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u_huge = np.array([10.0 * p.maxabsstab])
    u_clip = np.array([p.maxabsstab])
    np.testing.assert_allclose(
        f16_ode_long(x, u_huge, 0.0, p),
        f16_ode_long(x, u_clip, 0.0, p),
    )


def test_actuator_step_response_settles_within_envelope():
    """Step input on stab_act drives stab to track within ~3*Tstab seconds."""
    p = default_parameters()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    target = math.radians(5.0)
    u = np.array([target])
    dt = 0.001
    for k in range(int(0.5 / dt)):
        x = euler(f16_ode_long, x, u, t=k * dt, dt=dt, params=p)
    assert x[2] == pytest.approx(target, abs=math.radians(0.5))


def test_pitch_rate_response_to_elevator_is_nonzero():
    p = default_parameters()
    x = np.array([math.radians(2.0), 0.0, math.radians(-5.0), 0.0])
    u = np.array([0.0])
    dx = f16_ode_long(x, u, 0.0, p)
    assert dx[1] != 0.0
