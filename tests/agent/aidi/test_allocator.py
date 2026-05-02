"""MoorePenroseAllocator unit tests."""

import logging

import numpy as np
import pytest

from tensoraerospace.agent.aidi.allocator import MoorePenroseAllocator


def test_allocator_square_inverse():
    G = np.array([[1.0, 0.0], [0.0, 2.0]])
    alloc = MoorePenroseAllocator(rcond=1e-8, cond_threshold=1e8)
    nu = np.array([3.0, 4.0])
    omega_dot = np.array([1.0, 1.0])
    du = alloc.allocate(G, nu, omega_dot)
    np.testing.assert_allclose(du, np.array([2.0, 1.5]), atol=1e-9)


def test_allocator_redundant_min_norm():
    G = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    alloc = MoorePenroseAllocator()
    nu = np.array([1.0, 1.0])
    omega_dot = np.array([0.0, 0.0])
    du = alloc.allocate(G, nu, omega_dot)
    np.testing.assert_allclose(G @ du, nu, atol=1e-9)
    expected = np.linalg.pinv(G) @ nu
    np.testing.assert_allclose(du, expected, atol=1e-9)


def test_allocator_ill_conditioned_returns_zero(caplog):
    bad_G = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    alloc = MoorePenroseAllocator(cond_threshold=1e6)
    with caplog.at_level(logging.WARNING):
        du = alloc.allocate(bad_G, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    np.testing.assert_array_equal(du, np.zeros(2))
    assert any("ill-conditioned" in r.message.lower() for r in caplog.records)


def test_allocator_shape_validation():
    alloc = MoorePenroseAllocator()
    with pytest.raises(ValueError):
        alloc.allocate(np.eye(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        alloc.allocate(np.eye(3), np.zeros(3), np.zeros(2))
