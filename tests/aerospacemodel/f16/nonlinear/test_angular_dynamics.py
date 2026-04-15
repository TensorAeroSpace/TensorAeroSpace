import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.python.angular.dynamics import (
    f16_ode_6dof,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.python.angular.params import (
    default_parameters,
)


def _zero_state():
    return np.zeros(14, dtype=np.float64)


def test_dx_shape_and_finite():
    p = default_parameters()
    dx = f16_ode_6dof(_zero_state(), np.zeros(3), 0.0, p)
    assert dx.shape == (14,)
    assert np.all(np.isfinite(dx))


def test_each_actuator_responds_to_its_own_command():
    p = default_parameters()
    x = _zero_state()
    dx_zero = f16_ode_6dof(x, np.zeros(3), 0.0, p)
    dx_stab = f16_ode_6dof(x, np.array([math.radians(5.0), 0.0, 0.0]), 0.0, p)
    dx_ail = f16_ode_6dof(x, np.array([0.0, math.radians(5.0), 0.0]), 0.0, p)
    dx_dir = f16_ode_6dof(x, np.array([0.0, 0.0, math.radians(5.0)]), 0.0, p)
    assert dx_stab[9] != dx_zero[9]  # ddstab
    assert dx_ail[11] != dx_zero[11]  # ddail
    assert dx_dir[13] != dx_zero[13]  # dddir


def test_actuator_command_clamping_3channels():
    p = default_parameters()
    x = _zero_state()
    huge = np.array([10 * p.maxabsstab, 10 * p.maxabsail, 10 * p.maxabsdir])
    clipped = np.array([p.maxabsstab, p.maxabsail, p.maxabsdir])
    np.testing.assert_allclose(
        f16_ode_6dof(x, huge, 0.0, p),
        f16_ode_6dof(x, clipped, 0.0, p),
    )


def test_zero_state_zero_command_no_nans():
    p = default_parameters()
    dx = f16_ode_6dof(_zero_state(), np.zeros(3), 0.0, p)
    assert np.all(np.isfinite(dx))
