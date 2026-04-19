"""Position reconstruction kinematic helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.visualization.kinematics import (
    reconstruct_position_6dof,
    reconstruct_position_longitudinal,
)


def _state_6dof(alpha=0.0, beta=0.0, gamma=0.0, psi=0.0, theta=0.0) -> np.ndarray:
    """Build a single-row 6-DoF state with the given attitude/aero angles.

    Layout: [alpha, beta, wx, wy, wz, gamma, psi, theta, ...6 actuator zeros]
    """
    row = np.zeros(14)
    row[0] = alpha
    row[1] = beta
    row[5] = gamma
    row[6] = psi
    row[7] = theta
    return row


def test_6dof_straight_and_level_grows_along_x():
    # alpha=beta=0, all Euler angles = 0 -> velocity along body x = inertial x
    state = np.tile(_state_6dof(), (10, 1))
    pos = reconstruct_position_6dof(state, airspeed=100.0, dt=0.1)
    assert pos.shape == (10, 3)
    np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-12)
    # x grows by 100 * 0.1 = 10 m per step
    np.testing.assert_allclose(pos[-1, 0], 100.0 * 0.1 * 9, atol=1e-9)  # 9 integration steps over T=10 rows


def test_6dof_climb_at_45_deg_pitch():
    # theta = pi/4, alpha=0 -> velocity vector in x-z plane at 45 deg above horizontal
    state = np.tile(_state_6dof(theta=np.pi / 4), (5, 1))
    pos = reconstruct_position_6dof(state, airspeed=100.0, dt=0.1)
    # x and z components equal magnitude (cos/sin of 45deg)
    np.testing.assert_allclose(pos[-1, 0], pos[-1, 2], atol=1e-9)
    # y stays zero
    np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-12)


def test_6dof_pure_roll_no_motion_change():
    # gamma (roll) alone with alpha=theta=0 doesn't deflect velocity
    state = np.tile(_state_6dof(gamma=np.pi / 4), (5, 1))
    pos = reconstruct_position_6dof(state, airspeed=100.0, dt=0.1)
    np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-12)


def test_6dof_initial_position_offset():
    state = np.tile(_state_6dof(), (3, 1))
    pos = reconstruct_position_6dof(
        state, airspeed=10.0, dt=1.0,
        initial_position=np.array([100.0, 200.0, 300.0]),
    )
    np.testing.assert_allclose(pos[0], [100.0, 200.0, 300.0])
    # 2 integration steps (rows 0->1 and 1->2), each adds 10 m along x
    np.testing.assert_allclose(pos[-1, 0], 100.0 + 10.0 * 2)


def test_longitudinal_returns_positions_and_attitudes():
    # state cols: alpha, wz, stab, dstab. Constant zero -> stays at origin pose.
    state = np.zeros((5, 4))
    pos, att = reconstruct_position_longitudinal(state, airspeed=100.0, dt=0.1)
    assert pos.shape == (5, 3)
    assert att.shape == (5, 3)
    np.testing.assert_allclose(att, 0.0)
    # roll + yaw stay zero
    np.testing.assert_allclose(att[:, 0], 0.0)
    np.testing.assert_allclose(att[:, 2], 0.0)


def test_longitudinal_pitch_integrated_from_wz():
    # constant wz = 0.1 rad/s, dt = 0.1 s, 10 integration steps
    # -> theta(end) = wz * dt * N = 0.1 * 0.1 * 10 = 0.1 rad
    state = np.zeros((11, 4))
    state[:, 1] = 0.1  # wz column
    pos, att = reconstruct_position_longitudinal(
        state, airspeed=100.0, dt=0.1, initial_pitch=0.0,
    )
    # theta is index 1 of attitude (roll, pitch, yaw)
    np.testing.assert_allclose(att[-1, 1], 0.1, atol=1e-9)
    # y stays zero
    np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-12)


def test_longitudinal_initial_pitch_applied():
    state = np.zeros((3, 4))  # wz = 0
    pos, att = reconstruct_position_longitudinal(
        state, airspeed=100.0, dt=0.1, initial_pitch=np.pi / 6,
    )
    np.testing.assert_allclose(att[:, 1], np.pi / 6)
