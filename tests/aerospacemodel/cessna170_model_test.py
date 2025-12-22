import numpy as np
import pytest

from tensoraerospace.aerospacemodel.cessna170 import LongitudinalCessna170


def test_cessna170_init_and_param_override():
    x0 = [55.0, 0.0, 0.0, 0.0]
    m = LongitudinalCessna170(x0=x0, number_time_steps=10, dt=0.05, aero_params={"m": 900.0})

    p = m.get_param()
    assert isinstance(p, dict)
    assert p["m"] == 900.0
    assert m.dt == pytest.approx(0.05)
    assert m.list_state == ["u", "w", "q", "theta"]
    assert m.control_list == ["ele", "throttle"]
    assert len(m.x_history) == 1
    assert m.x_history[0].shape == (4,)


def test_cessna170_run_step_returns_state_and_is_finite():
    x0 = [55.0, 0.0, 0.0, 0.0]
    m = LongitudinalCessna170(x0=x0, number_time_steps=10, dt=0.02)

    for _ in range(5):
        x = m.run_step([0.0, 0.5])
        assert isinstance(x, np.ndarray)
        assert x.shape == (4,)
        assert np.isfinite(x).all()

    assert len(m.u_history) == 5
    assert len(m.x_history) == 6


def test_cessna170_run_step_control_dim_mismatch_raises():
    x0 = [55.0, 0.0, 0.0, 0.0]
    m = LongitudinalCessna170(x0=x0, number_time_steps=10)

    with pytest.raises(ValueError, match="Размерность управляющего вектора"):
        m.run_step([0.0])  # missing throttle


def test_cessna170_input_magnitude_and_rate_limits_applied():
    # Make dt large to clearly see rate limiting
    x0 = [55.0, 0.0, 0.0, 0.0]
    m = LongitudinalCessna170(x0=x0, number_time_steps=10, dt=0.1)
    p = m.get_param()
    ele_lim = float(p["ele_lim"])
    ele_rate = float(p["ele_rate"])
    thr_rate = float(p["thr_rate"])

    # First step: magnitude limits should clip
    m.run_step([10.0, 5.0])  # absurd commands
    ele1, thr1 = m.u_history[-1]
    assert abs(ele1) <= ele_lim + 1e-12
    assert 0.0 <= thr1 <= 1.0

    # Second step: request opposite extreme; rate limit should apply
    m.run_step([-10.0, 0.0])
    ele2, thr2 = m.u_history[-1]

    max_ele_step = ele_rate * m.dt
    max_thr_step = thr_rate * m.dt
    assert abs(ele2 - ele1) <= max_ele_step + 1e-12
    assert abs(thr2 - thr1) <= max_thr_step + 1e-12


def test_cessna170_selected_state_output_returns_subset():
    x0 = [55.0, 0.0, 0.0, 0.0]
    # request subset by state names (ModelBase uses name->index mapping)
    m = LongitudinalCessna170(
        x0=x0,
        number_time_steps=10,
        selected_state_output=["theta", "q"],
        dt=0.02,
    )
    y = m.run_step([0.0, 0.5])
    assert y.shape == (2,)




