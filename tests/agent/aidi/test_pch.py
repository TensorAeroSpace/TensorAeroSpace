"""PseudoControlHedge unit tests."""

import numpy as np

from tensoraerospace.agent.aidi.pch import PseudoControlHedge


def test_pch_zero_hedge_when_inner_loop_tracks():
    hedge = PseudoControlHedge(n_y=3, freeze_after=10)
    hedge.update(
        nu_des_prev=np.array([1.0, 0.0, 0.0]), omega_dot_meas=np.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(hedge.last_hedge, np.zeros(3), atol=1e-12)
    assert not hedge.is_frozen.any()


def test_pch_emits_hedge_when_plant_lags():
    hedge = PseudoControlHedge(n_y=3, freeze_after=5)
    hedge.update(
        nu_des_prev=np.array([1.0, 0.0, 0.0]), omega_dot_meas=np.array([0.4, 0.0, 0.0])
    )
    np.testing.assert_allclose(hedge.last_hedge, np.array([0.6, 0.0, 0.0]))


def test_pch_freezes_after_persistent_saturation():
    hedge = PseudoControlHedge(n_y=2, freeze_after=3)
    for _ in range(4):
        hedge.update(
            nu_des_prev=np.array([1.0, 0.0]), omega_dot_meas=np.array([0.0, 0.0])
        )
    assert bool(hedge.is_frozen[0]) is True
    assert bool(hedge.is_frozen[1]) is False


def test_pch_freeze_clears_when_gap_closes():
    hedge = PseudoControlHedge(n_y=1, freeze_after=2)
    for _ in range(3):
        hedge.update(nu_des_prev=np.array([1.0]), omega_dot_meas=np.array([0.0]))
    assert bool(hedge.is_frozen[0]) is True
    hedge.update(nu_des_prev=np.array([1.0]), omega_dot_meas=np.array([1.0]))
    assert bool(hedge.is_frozen[0]) is False


def test_pch_reset_clears_state():
    hedge = PseudoControlHedge(n_y=2, freeze_after=2)
    for _ in range(3):
        hedge.update(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    hedge.reset()
    np.testing.assert_array_equal(hedge.last_hedge, np.zeros(2))
    np.testing.assert_array_equal(hedge.is_frozen, np.zeros(2, dtype=bool))
