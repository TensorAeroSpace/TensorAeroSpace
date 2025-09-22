import numpy as np

from tensoraerospace.agent.ihdp.model import IHDPAgent


def test_ihdp_agent_predict_smoke():
    actor_settings = dict(
        start_training=10,
        layers=(4, 1),
        activations=("tanh", "tanh"),
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        type_PE="3211",
        amplitude_3211=1,
        pulse_length_3211=5,
        maximum_input=1,
        maximum_q_rate=1,
        WB_limits=5,
        NN_initial=1,
        cascade_actor=False,
        learning_rate_cascaded=0.01,
    )

    critic_settings = dict(
        Q_weights=[1.0],
        start_training=10,
        gamma=0.9,
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        layers=(4, 1),
        activations=("tanh", "linear"),
        indices_tracking_states=[0],
        WB_limits=5,
        NN_initial=1,
    )

    incremental_settings = dict(
        number_time_steps=5,
        dt=0.1,
        input_magnitude_limits=1,
        input_rate_limits=10,
    )

    tracking_states = ["alpha"]
    selected_states = ["alpha"]
    selected_input = ["u"]
    number_time_steps = 5
    indices_tracking_states = [0]

    agent = IHDPAgent(
        actor_settings,
        critic_settings,
        incremental_settings,
        tracking_states,
        selected_states,
        selected_input,
        number_time_steps,
        indices_tracking_states,
    )

    xt = np.array([[0.0]])
    reference = np.zeros((1, number_time_steps))
    ut = agent.predict(xt, reference, time_step=0)
    assert ut is not None
    # shape can be (1,) or (1,1) depending on path
    assert hasattr(ut, "shape")
