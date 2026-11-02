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


def test_body_velocity_equations_match_wind_axis_equations():
    """du, dv, dw equal the chain rule applied to dalpha, dbeta, dV."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )

    rng = np.random.default_rng(1)
    for _ in range(300):
        params = default_parameters()
        params.air_density_kg_m3 = rng.uniform(0.3, 1.2)
        params.T_active = rng.uniform(0, 130000)
        x = np.zeros(16)
        x[0], x[1] = rng.uniform(-0.3, 1.5), rng.uniform(-0.5, 0.5)
        x[2:5] = rng.normal(0, 1, 3)
        x[5:8] = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-1.4, 1.4)
        x[[8, 10, 12]] = rng.uniform(-0.4, 0.4, 3)
        x[14], x[15] = rng.uniform(500, 9000), rng.uniform(40, 450)
        u = rng.uniform(-0.3, 0.3, 3)
        wind = f16_ode_6dof(x, u, 0.0, params)
        body = f16_ode_6dof(x, u, 0.0, params, body_velocity=True)
        a, b, speed = x[0], x[1], x[15]
        da, db, dv = wind[0], wind[1], wind[15]
        chain = [
            dv * math.cos(a) * math.cos(b)
            - speed * math.sin(a) * math.cos(b) * da
            - speed * math.cos(a) * math.sin(b) * db,
            dv * math.sin(b) + speed * math.cos(b) * db,
            dv * math.sin(a) * math.cos(b)
            + speed * math.cos(a) * math.cos(b) * da
            - speed * math.sin(a) * math.sin(b) * db,
        ]
        np.testing.assert_allclose(body[[0, 1, 15]], chain, rtol=1e-11, atol=1e-9)
        others = [i for i in range(16) if i not in (0, 1, 15)]
        np.testing.assert_array_equal(body[others], wind[others])


def _ned_velocity(state):
    gamma, psi, theta = state[5], state[6], state[7]
    alpha, beta, speed = state[0], state[1], state[15]
    body = speed * np.array(
        [
            math.cos(alpha) * math.cos(beta),
            math.sin(beta),
            math.sin(alpha) * math.cos(beta),
        ]
    )
    return _ned_from_euler(gamma, psi, theta) @ body


@pytest.mark.parametrize("translation", ["body", "wind"])
def test_tumbling_without_air_falls_exactly_through_any_sideslip(translation):
    """With zero air density and thrust only gravity acts: v_ned = v0 + g t.

    A torque-free tumble sweeps the velocity through every body direction, so
    beta passes ±90 deg (and alpha ±180 deg). Body-axis integration follows
    the exact solution; the legacy wind-axis equations divide by cos(beta).
    """
    model, _ = _model("quaternion")
    if translation == "wind":
        model.translation = "wind"
    state = model.current_state
    # Mostly a yaw spin: the velocity sweeps sideways through beta = ±90 deg.
    state[2:5] = 0.3, 3.0, 0.2
    model.x_history[-1] = state.reshape(-1, 1)
    model.param.air_density_kg_m3 = 0.0
    v0 = _ned_velocity(state)
    worst = 0.0
    peak = 0.0
    tail_first = 0.0
    with np.errstate(all="ignore"):
        for tick in range(1, 361):
            model.run_step(np.array([0.0, 0.0, 0.0, 0.0]))
            current = model.current_state
            if not np.all(np.isfinite(current)):
                worst = math.inf
                break
            expected = v0 + np.array([0.0, 0.0, model.param.g]) * tick * model.dt
            worst = max(worst, np.max(np.abs(_ned_velocity(current) - expected)))
            peak = max(peak, abs(current[1]))
            tail_first = max(tail_first, abs(current[0]))
    if translation == "body":
        # Sampled at 120 Hz, beta peaks just below 90 deg; alpha near 180 deg
        # shows the flow came from behind after passing side-on.
        assert math.degrees(peak) > 85.0
        assert math.degrees(tail_first) > 170.0
        # RK4 in the rotating frame (3 rad/s): ~1e-5 m/s after 3 s at 250 m/s.
        assert worst < 1e-4
    else:
        # The wind-axis equations lose ~1e-2 m/s crossing beta = 90 deg here,
        # and divide by zero if a step lands on it.
        assert worst > 1e-3


def test_translation_option_is_validated():
    with pytest.raises(ValueError, match="translation"):
        AngularF16(x0=np.zeros(14), translation="inertial")
    with pytest.raises(ValueError, match="track_altitude"):
        AngularF16(x0=np.zeros(14), translation="body")
    assert AngularF16(x0=np.zeros(14)).translation == "wind"
    assert AngularF16(x0=np.zeros(14), track_altitude=True).translation == "body"
