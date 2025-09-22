import numpy as np

from tensoraerospace.agent.ihdp.Actor import Actor


def test_actor_build_and_forward_basic():
    selected_inputs = ["u"]
    selected_states = ["alpha"]
    tracking_states = ["alpha"]
    indices_tracking_states = [0]
    number_time_steps = 3
    start_training = 10

    actor = Actor(
        selected_inputs,
        selected_states,
        tracking_states,
        indices_tracking_states,
        number_time_steps,
        start_training,
        layers=(4, 1),
        activations=("tanh", "tanh"),
        learning_rate=0.01,
        WB_limits=5,
        maximum_input=1,
        maximum_q_rate=1,
        cascaded_actor=False,
        NN_initial=1,
    )

    actor.build_actor_model()
    assert actor.model is not None

    xt = np.array([[0.1]], dtype=np.float32)
    xt_ref = np.array([[0.0]], dtype=np.float32)
    ut = actor.run_actor_online(xt, xt_ref)
    assert isinstance(ut, np.ndarray)
    assert ut.shape in ((1,), (1, 1))
    assert np.all(np.isfinite(ut))
    assert actor.dut_dWb is not None
    assert actor.dut_dWb is not None
