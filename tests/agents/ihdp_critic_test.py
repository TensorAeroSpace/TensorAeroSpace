import numpy as np

from tensoraerospace.agent.ihdp.Critic import Critic


def test_critic_build_and_forward_basic():
    selected_states = ["alpha"]
    tracking_states = ["alpha"]
    indices_tracking_states = [0]
    number_time_steps = 3
    start_training = 10

    critic = Critic(
        Q_weights=[1.0],
        selected_states=selected_states,
        tracking_states=tracking_states,
        indices_tracking_states=indices_tracking_states,
        number_time_steps=number_time_steps,
        start_training=start_training,
        gamma=0.9,
        learning_rate=0.01,
        layers=(4, 1),
        activations=("tanh", "linear"),
        WB_limits=5,
        NN_initial=1,
    )
    critic.build_critic_model()

    xt = np.array([[0.2]], dtype=np.float32)
    xt_ref = np.array([[0.0]], dtype=np.float32)
    Jt, dJt_dxt = critic.evaluate_critic(xt, xt_ref)

    assert Jt is not None
    assert dJt_dxt is not None
    assert np.asarray(Jt).shape[0] >= 1
    assert np.isfinite(Jt).all()
