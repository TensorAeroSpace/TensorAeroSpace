"""Outer-loop reference-model unit tests."""

import math

import numpy as np
import pytest

from tensoraerospace.agent.aidi.ref_models import (
    CStarController,
    LinearController,
    RollReferenceModel,
    SideslipCompensator,
    SpeedController,
)


def test_cstar_drives_q_des_toward_command():
    ctrl = CStarController(kp=2.0, ki=0.5, V_co=122.6, dt=0.01)
    q_des = ctrl.step(c_star_cmd=2.0, n_z=1.0, q=0.0, V=200.0, hedge=0.0)
    # C* error = 2 - (1 + (200/122.6)*0) = 1.0; kp*err = 2.0; ki·err·dt = 0.005
    assert q_des == pytest.approx(2.0 + 0.5 * 0.01, rel=1e-6)


def test_cstar_subtracts_hedge_before_integration():
    ctrl = CStarController(kp=0.0, ki=1.0, V_co=122.6, dt=0.1)
    ctrl.step(c_star_cmd=1.0, n_z=0.0, q=0.0, V=100.0, hedge=0.0)
    int_no_hedge = ctrl._int_err
    ctrl.reset()
    ctrl.step(c_star_cmd=1.0, n_z=0.0, q=0.0, V=100.0, hedge=0.1)
    int_with_hedge = ctrl._int_err
    assert int_with_hedge < int_no_hedge


def test_roll_reference_second_order_response():
    ref = RollReferenceModel(omega_n=2.0, zeta=0.7, dt=0.01)
    phi = 0.0
    for _ in range(500):
        p_des = ref.step(phi_cmd=math.radians(10.0), phi=phi, hedge=0.0)
        # Toy plant: phi tracks integrated p_des.
        phi += p_des * 0.01
    assert phi == pytest.approx(math.radians(10.0), abs=math.radians(0.5))


def test_sideslip_compensator_drives_beta_to_zero():
    comp = SideslipCompensator(kp=2.0, ki=0.1, dt=0.01)
    beta = math.radians(2.0)
    for _ in range(500):
        r = comp.step(beta_cmd=0.0, beta=beta, hedge=0.0)
        # Body-axis lateral plant: dβ/dt ≈ −r. Positive kp → r > 0 when β > 0,
        # which drives β back toward zero.
        beta -= r * 0.01
    assert abs(beta) < math.radians(0.05)


def test_speed_controller_no_op_when_disabled():
    ctrl = SpeedController(kp=0.0, ki=0.0, kd=0.0, dt=0.01, enabled=False)
    out = ctrl.step(V_cmd=200.0, V=180.0)
    assert out == 0.0


def test_linear_controller_passthrough_with_zero_gain():
    lin = LinearController(rate_kp=np.zeros(3))
    nu = lin.combine(
        omega_des=np.array([1.0, 2.0, 3.0]), omega=np.array([0.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(nu, np.array([1.0, 2.0, 3.0]))


def test_linear_controller_adds_rate_error_feedback():
    lin = LinearController(rate_kp=np.array([1.0, 0.0, 0.0]))
    nu = lin.combine(
        omega_des=np.array([1.0, 0.0, 0.0]), omega=np.array([0.5, 0.0, 0.0])
    )
    np.testing.assert_array_equal(nu, np.array([1.5, 0.0, 0.0]))
