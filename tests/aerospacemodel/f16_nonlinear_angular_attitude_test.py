"""Quaternion attitude integration of the angular F-16 through the vertical."""

import math

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.attitude import (
    euler_to_quaternion,
    quaternion_rate,
    quaternion_to_euler,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim


def _ned_from_euler(gamma, psi, theta):
    """Body (x fwd, y right, z down) to NED for model angles (yaw = -psi)."""
    cr, sr = math.cos(gamma), math.sin(gamma)
    cp, sp = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(-psi), math.sin(-psi)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _ned_from_quaternion(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _angle_deg(a, b):
    cosine = (np.trace(a.T @ b) - 1) / 2
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


@pytest.mark.parametrize(
    "angles",
    [
        (0.3, -1.2, 0.4),
        (-2.5, 3.0, -1.1),
        (1.0, 0.5, math.pi / 2),
        (0.2, 0.1, -math.pi / 2),
    ],
)
def test_conversions_describe_the_same_rotation(angles):
    q = euler_to_quaternion(*angles)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    np.testing.assert_allclose(
        _ned_from_quaternion(q), _ned_from_euler(*angles), atol=1e-12
    )
    back = quaternion_to_euler(q, reference=angles[:2])
    np.testing.assert_allclose(
        _ned_from_euler(*back), _ned_from_euler(*angles), atol=1e-12
    )
    assert -math.pi / 2 <= back[2] <= math.pi / 2


def test_heading_stays_continuous_across_pi():
    gamma, psi, _ = quaternion_to_euler(euler_to_quaternion(0.1, 7.0, 0.2), (0.1, 6.9))
    assert gamma == pytest.approx(0.1)
    assert psi == pytest.approx(7.0)


def test_quaternion_rate_matches_euler_rate_equations():
    gamma, psi, theta = 0.4, -0.7, 0.6
    wx, wy, wz = 0.3, -0.2, 0.5
    h = 1e-6
    q = euler_to_quaternion(gamma, psi, theta)
    stepped = quaternion_to_euler(q + h * quaternion_rate(q, wx, wy, wz), (gamma, psi))
    rates = (np.array(stepped) - [gamma, psi, theta]) / h
    expected = [
        wx
        - math.cos(gamma) * math.tan(theta) * wy
        + math.sin(gamma) * math.tan(theta) * wz,
        (math.cos(gamma) / math.cos(theta)) * wy
        - (math.sin(gamma) / math.cos(theta)) * wz,
        math.sin(gamma) * wy + math.cos(gamma) * wz,
    ]
    np.testing.assert_allclose(rates, expected, rtol=1e-5, atol=1e-6)


def _model(attitude, integrator="rk4", speed=250.0, altitude=5000.0):
    trim = find_trim(V_target=speed, h_target=altitude)
    model = AngularF16(
        x0=trim.x0,
        dt=1 / 120,
        integrator=integrator,
        track_altitude=True,
        thrust_mode="control",
        attitude=attitude,
    )
    return model, trim


def _rates_frd(state):
    return np.array([state[2], state[4], -state[3]])


def _quaternion_step(q, w0, w1, dt):
    w = 0.5 * (w0 + w1) * dt
    angle = np.linalg.norm(w)
    if angle < 1e-15:
        return q
    half = np.r_[math.cos(angle / 2), math.sin(angle / 2) * w / angle]
    a, b = q, half
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


@pytest.mark.parametrize("roll", [0.0, 1.2e-4, 1.5e-3, 0.02])
def test_loop_through_vertical_keeps_attitude_consistent(roll):
    """Attitude must follow the integrated body rates through the vertical.

    The legacy Euler integration jumps by up to about 1 degree on these loops
    (worse the closer the pass is to exactly 90 degrees).
    """
    model, trim = _model("quaternion")
    control = np.array(
        [-0.6 * model.param.maxabsstab, -roll * model.param.maxabsail, 0.0, 130000.0]
    )
    state = model.current_state
    q = euler_to_quaternion(state[5], state[6], state[7])
    max_pitch = 0.0
    worst = 0.0
    for _ in range(int(6 * 120)):
        before = model.current_state
        model.param.air_density_kg_m3 = 0.736
        model.run_step(control)
        after = model.current_state
        assert np.all(np.isfinite(after))
        assert -math.pi / 2 <= after[7] <= math.pi / 2
        q = _quaternion_step(q, _rates_frd(before), _rates_frd(after), model.dt)
        max_pitch = max(max_pitch, abs(after[7]))
        worst = max(
            worst,
            _angle_deg(
                _ned_from_euler(after[5], after[6], after[7]), _ned_from_quaternion(q)
            ),
        )
    assert math.degrees(max_pitch) > 89.0
    assert worst < 0.01


def test_exactly_vertical_state_integrates():
    model, _ = _model("quaternion")
    model.set_air_relative_state(
        altitude_m=5000.0, airspeed_mps=200.0, alpha_rad=0.05, beta_rad=0.0
    )
    state = model.current_state
    state[7] = math.pi / 2
    state[5] = 0.3
    model.x_history[-1] = state.reshape(-1, 1)
    model.param.air_density_kg_m3 = 0.736
    for _ in range(60):
        model.run_step(np.array([0.0, 0.0, 0.0, 60000.0]))
        assert np.all(np.isfinite(model.current_state))


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_quaternion_matches_legacy_away_from_vertical(integrator):
    models = [_model(mode, integrator)[0] for mode in ("quaternion", "euler")]
    for tick in range(600):
        t = tick / 120
        control = np.array(
            [
                -0.03 + 0.02 * math.sin(t),
                0.05 * math.sin(2 * t),
                0.01 * math.cos(t),
                40000.0,
            ]
        )
        for model in models:
            model.param.air_density_kg_m3 = 0.736
            model.run_step(control)
    a, b = (model.current_state for model in models)
    assert abs(math.degrees(a[7])) < 60
    # RK4 agrees to round-off. Forward Euler is first order in both
    # parameterizations, so they differ by O(dt) (here ~0.3 deg after 5 s).
    tolerance = 1e-7 if integrator == "rk4" else 2e-2
    np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def test_attitude_option_is_validated():
    with pytest.raises(ValueError, match="attitude"):
        AngularF16(x0=np.zeros(14), attitude="matrix")
