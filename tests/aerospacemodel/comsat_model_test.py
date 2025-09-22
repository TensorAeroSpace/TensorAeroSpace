import numpy as np

from tensoraerospace.aerospacemodel.comsat import ComSat


def test_comsat_initialization_and_run_step():
    x0 = np.array([0.0, 0.0, 0.0])
    steps = 4
    model = ComSat(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    x1 = model.run_step(np.array([100.0]))
    assert x1.shape[0] == 3

    assert model.get_state("rho").shape[0] == steps - 1
    # Control retrieval is only defined for ['ele','ail','rud'] in current implementation
    # so verify output history instead
    assert model.get_output("omega").shape[0] == model.time_step - 1
