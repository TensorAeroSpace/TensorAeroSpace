import numpy as np

from tensoraerospace.aerospacemodel.b747 import LongitudinalB747


def test_b747_initialization_and_run_step():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 5
    model = LongitudinalB747(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    # One step with over-limit control to test clipping
    u = np.array([100.0])
    x1 = model.run_step(u)
    assert isinstance(x1, np.ndarray)
    assert x1.shape[0] == 4

    # Stored input must be within magnitude limits
    assert np.all(
        np.abs(model.store_input[:, 0]) <= np.array(model.input_magnitude_limits)
    )

    # Subsequent steps with zero control
    for _ in range(steps - 1):
        model.run_step(np.array([0.0]))

    # State and control histories have expected lengths
    theta_hist = model.get_state("theta")
    assert theta_hist.shape[0] == steps - 1

    ele_hist = model.get_control("ele")
    assert ele_hist.shape[0] == steps - 1

    # Output retrieval without unit conversion
    out_q = model.get_output("q")
    assert out_q.shape[0] == model.time_step - 1
