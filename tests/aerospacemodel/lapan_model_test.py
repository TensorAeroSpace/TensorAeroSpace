import numpy as np

from tensoraerospace.aerospacemodel.lapan import LAPAN


def test_lapan_initialization_run_and_histories():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 5
    dt = 0.01
    model = LAPAN(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=dt
    )

    x1 = model.run_step(np.array([10.0]))
    assert x1.shape[0] == 4

    assert model.get_state("theta").shape[0] == steps - 1
    assert model.get_control("ele").shape[0] == steps - 1
    assert model.get_output("q").shape[0] == model.time_step - 1

    # Plotting is guarded by a strict name check; skip to avoid false failures
