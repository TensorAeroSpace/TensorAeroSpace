"""AIDI helper tests — n_z reconstruction."""

import math

import pytest

from tensoraerospace.agent.aidi.utils import reconstruct_n_z


def test_n_z_unity_at_level_trim():
    nz = reconstruct_n_z(
        alpha=math.radians(2.0), alpha_dot=0.0, q=0.0, V=200.0, theta=0.0, phi=0.0
    )
    assert nz == pytest.approx(1.0, abs=1e-9)


def test_n_z_increases_with_pitch_rate():
    nz1 = reconstruct_n_z(
        alpha=math.radians(2.0), alpha_dot=0.0, q=0.0, V=200.0, theta=0.0, phi=0.0
    )
    nz2 = reconstruct_n_z(
        alpha=math.radians(2.0),
        alpha_dot=0.0,
        q=math.radians(5.0),
        V=200.0,
        theta=0.0,
        phi=0.0,
    )
    assert nz2 > nz1


def test_n_z_alpha_dot_subtracts():
    nz1 = reconstruct_n_z(
        alpha=math.radians(2.0),
        alpha_dot=0.0,
        q=math.radians(5.0),
        V=200.0,
        theta=0.0,
        phi=0.0,
    )
    nz2 = reconstruct_n_z(
        alpha=math.radians(2.0),
        alpha_dot=math.radians(2.0),
        q=math.radians(5.0),
        V=200.0,
        theta=0.0,
        phi=0.0,
    )
    assert nz2 < nz1


def test_n_z_inverted_flight_negative():
    nz = reconstruct_n_z(
        alpha=0.0, alpha_dot=0.0, q=0.0, V=200.0, theta=0.0, phi=math.pi
    )
    assert nz == pytest.approx(-1.0, abs=1e-9)
