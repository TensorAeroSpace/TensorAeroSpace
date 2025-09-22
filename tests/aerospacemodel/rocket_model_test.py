import numpy as np

from tensoraerospace.aerospacemodel.rocket import MissileModel


def test_rocket_initialization_and_run_step():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 4
    model = MissileModel(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    x1 = model.run_step(np.array([100.0]))
    assert x1.shape[0] == 4

    assert model.get_state("theta").shape[0] == steps - 1
    assert model.get_control("ele").shape[0] == steps - 1
    assert model.get_output("q").shape[0] == model.time_step - 1
