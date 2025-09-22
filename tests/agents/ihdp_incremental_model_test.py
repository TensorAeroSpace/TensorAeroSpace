import numpy as np

from tensoraerospace.agent.ihdp.Incremental_model import IncrementalModel


def test_incremental_model_identify_and_evaluate():
    selected_states = ["alpha"]
    selected_input = ["u"]
    number_time_steps = 5

    inc = IncrementalModel(
        selected_states,
        selected_input,
        number_time_steps,
        discretisation_time=0.1,
        input_magnitude_limits=1,
        input_rate_limits=10,
    )

    # seed previous values to avoid rate limiting corner-cases
    inc.xt_1 = np.array([[0.0]])
    inc.ut_1 = np.array([[0.0]])

    xt = np.array([[0.1]])
    ut = np.array([[0.05]])
    G = inc.identify_incremental_model_LS(xt, ut)
    assert G.shape == (1, 1)

    xt1_est = inc.evaluate_incremental_model()
    assert xt1_est.shape == (1, 1)
    assert np.isfinite(xt1_est).all()
