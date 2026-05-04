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


def test_ihdp_agent_predict_accepts_multiple_tracking_states():
    actor_settings = dict(
        start_training=-1,
        layers=(4, 1),
        activations=("tanh", "tanh"),
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        type_PE="3211",
        amplitude_3211=0.0,
        pulse_length_3211=1,
        maximum_input=1,
        maximum_q_rate=1,
        WB_limits=5,
        NN_initial=1,
        cascade_actor=False,
        learning_rate_cascaded=0.01,
    )

    critic_settings = dict(
        Q_weights=[1.0, 0.1],
        start_training=-1,
        gamma=0.9,
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        layers=(4, 1),
        activations=("tanh", "linear"),
        indices_tracking_states=[0, 1],
        WB_limits=5,
        NN_initial=1,
    )

    incremental_settings = dict(
        number_time_steps=6,
        dt=0.1,
        input_magnitude_limits=1,
        input_rate_limits=10,
    )

    agent = IHDPAgent(
        actor_settings,
        critic_settings,
        incremental_settings,
        tracking_states=["e_z", "w_b"],
        selected_states=["e_z", "w_b"],
        selected_input=["dT"],
        number_time_steps=6,
        indices_tracking_states=[0, 1],
    )

    reference = np.zeros((2, 6))
    for time_step in range(2):
        xt = np.array([[0.1 - 0.02 * time_step], [0.03]], dtype=float)
        ut = agent.predict(xt, reference, time_step=time_step)
        assert hasattr(ut, "shape")
        assert np.all(np.isfinite(ut))


def test_ihdp_agent_predict_accepts_multiple_actions():
    actor_settings = dict(
        start_training=-1,
        layers=(6, 2),
        activations=("tanh", "tanh"),
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        type_PE="3211",
        amplitude_3211=[0.0, 0.0],
        pulse_length_3211=1,
        maximum_input=[1.0, 0.5],
        maximum_q_rate=1,
        WB_limits=5,
        NN_initial=2,
        cascade_actor=False,
        learning_rate_cascaded=0.01,
    )

    critic_settings = dict(
        Q_weights=[1.0, 0.1],
        start_training=-1,
        gamma=0.9,
        learning_rate=0.01,
        learning_rate_exponent_limit=5,
        layers=(6, 1),
        activations=("tanh", "linear"),
        indices_tracking_states=[0, 1],
        WB_limits=5,
        NN_initial=2,
    )

    incremental_settings = dict(
        number_time_steps=8,
        dt=0.1,
        input_magnitude_limits=[1.0, 0.5],
        input_rate_limits=[10.0, 5.0],
    )

    agent = IHDPAgent(
        actor_settings,
        critic_settings,
        incremental_settings,
        tracking_states=["phi", "p"],
        selected_states=["phi", "p"],
        selected_input=["da", "dr"],
        number_time_steps=8,
        indices_tracking_states=[0, 1],
    )

    reference = np.zeros((2, 8))
    for time_step in range(3):
        xt = np.array([[0.05 - 0.01 * time_step], [0.02]], dtype=float)
        ut = agent.predict(xt, reference, time_step=time_step)
        assert ut.shape == (2, 1)
        assert np.all(np.isfinite(ut))
        assert np.all(np.abs(ut.flatten()) <= np.array([1.0, 0.5]) + 1e-12)
        assert agent.incremental_model.G.shape == (2, 2)
