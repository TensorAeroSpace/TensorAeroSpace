"""Numerical checks for cascaded IHDP control and incremental predictions."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ihdp.Actor import Actor
from tensoraerospace.agent.ihdp.Incremental_model import IncrementalModel


def make_actor(activation="tanh", extra_state=False, **overrides):
    states = ["height", "alpha", "wz"] if extra_state else ["alpha", "wz"]
    settings = dict(
        selected_inputs=["u"],
        selected_states=states,
        tracking_states=["alpha"],
        indices_tracking_states=[states.index("alpha")],
        number_time_steps=8,
        start_training=0,
        layers=(1,),
        activations=(activation,),
        learning_rate=0.01,
        learning_rate_exponent_limit=3,
        type_PE=None,
        amplitude_3211=0,
        pulse_length_3211=1,
        maximum_input=2.0,
        maximum_q_rate=3.0,
        WB_limits=2.0,
        NN_initial=1,
        cascaded_actor=True,
        learning_rate_cascaded=0.02,
    )
    settings.update(overrides)
    actor = Actor(**settings)
    actor.build_actor_model()
    with torch.no_grad():
        actor.model[0].weight.fill_(0.4)
        actor.model[0].bias.zero_()
        actor.model_q[0].weight.fill_(0.7)
        actor.model_q[0].bias.zero_()
    return actor


def cascade_values(activation, outer_weight=0.4):
    """Independent scalar calculation for alpha error=.2 and measured q=.1."""
    if activation == "tanh":
        outer = np.tanh(outer_weight * 0.2)
        outer_prime = 1.0 - outer**2
        q_ref = 3.0 * outer
        inner = np.tanh(0.7 * (0.1 - q_ref))
        inner_prime = 1.0 - inner**2
        output = 2.0 * inner
    else:
        outer = 1.0 / (1.0 + np.exp(-outer_weight * 0.2))
        outer_prime = outer * (1.0 - outer)
        q_ref = 6.0 * outer - 3.0
        inner = 1.0 / (1.0 + np.exp(-0.7 * (0.1 - q_ref)))
        inner_prime = inner * (1.0 - inner)
        output = 4.0 * inner - 2.0
    return q_ref, output, outer_prime, inner_prime


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
@pytest.mark.parametrize("extra_state", [False, True])
def test_cascade_output_and_parameter_derivatives(activation, extra_state):
    actor = make_actor(activation, extra_state)
    state = np.array([[9.0], [0.3], [0.1]]) if extra_state else np.array([[0.3], [0.1]])
    ref = np.array([[0.1]])
    q_ref, output, outer_prime, inner_prime = cascade_values(activation)

    actual = actor.run_actor_online(state, ref)

    assert actual.shape == (1, 1)
    np.testing.assert_allclose(actual, [[output]], atol=3e-7)
    assert actor.store_q[0, 0] == pytest.approx(q_ref, abs=3e-7)
    np.testing.assert_allclose(actor.dq_ref_dWb[0], [[outer_prime * 0.2]], atol=1e-7)
    np.testing.assert_allclose(
        actor.dut_dWb[0], [[inner_prime * (0.1 - q_ref)]], atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(actor.dut_dq_ref).ravel(), [inner_prime * 0.7], atol=1e-7
    )
    for args in ((), (0,), (state, ref)):
        np.testing.assert_allclose(actor.evaluate_actor(*args), actual, atol=1e-7)


class QuadraticCritic:
    def evaluate_critic(self, state, reference):
        error = state[:1] - reference
        return error**2, 2.0 * error


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
@pytest.mark.parametrize(
    "method", ["train_actor_online_adam", "train_actor_online_alpha_decay"]
)
@pytest.mark.parametrize("training_started", [False, True])
def test_cascade_training_updates_both_controllers_only_after_warmup(
    activation, method, training_started
):
    actor = make_actor(activation)
    actor.time_step = 1 if training_started else 0
    actor.run_actor_online(np.array([[0.3], [0.1]]), np.array([[0.1]]))
    before = [
        p.detach().clone()
        for net in (actor.model, actor.model_q)
        for p in net.parameters()
    ]
    incremental = IncrementalModel(["alpha", "wz"], ["u"], 8)
    incremental.G = np.array([[0.2], [0.1]])
    kwargs = dict(
        Jt1=np.array([[0.5]]),
        dJt1_dxt1=np.array([[0.3]]),
        G=incremental.G,
        incremental_model=incremental,
        critic=QuadraticCritic(),
        xt_ref1=np.array([[0.1]]),
    )
    getattr(actor, method)(**kwargs)
    after = [p for net in (actor.model, actor.model_q) for p in net.parameters()]

    if not training_started:
        for old, new in zip(before, after):
            torch.testing.assert_close(new, old)
        assert actor.learning_rate == 0.01
        assert actor.learning_rate_cascaded == 0.02
    else:
        assert not torch.equal(before[0], after[0])
        assert not torch.equal(before[2], after[2])
        assert actor.learning_rate == pytest.approx(0.01 * 0.9995)
        assert actor.learning_rate_cascaded == pytest.approx(0.02 * 0.9995)
        for p in after:
            assert torch.isfinite(p).all()
            assert torch.max(torch.abs(p)) <= actor.WB_limits
        if method == "train_actor_online_alpha_decay":
            q_ref, _, _, inner_prime = cascade_values(activation)
            inner_scale = 2.0 if activation == "tanh" else 4.0
            # dE/du = .5 * .2 * .3 = .03; propagate through each controller.
            expected_inner = 0.7 - 0.02 * inner_scale * 0.03 * inner_prime * (
                0.1 - q_ref
            )
            # Differentiate the complete physical control output numerically,
            # independently of the implementation's intermediate Jacobians.
            epsilon = 1e-4
            derivative = (
                cascade_values(activation, 0.4 + epsilon)[1]
                - cascade_values(activation, 0.4 - epsilon)[1]
            ) / (2.0 * epsilon)
            expected_outer = 0.4 - 0.01 * 0.03 * derivative
            assert after[0].item() == pytest.approx(expected_outer, abs=1e-7)
            assert after[2].item() == pytest.approx(expected_inner, abs=1e-7)
    assert abs(actor.evaluate_actor().item()) <= 2.0


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 1)])
def test_incremental_prediction_respects_per_input_magnitude_and_rate(shape):
    model = IncrementalModel(
        ["x", "y"],
        ["u", "v"],
        12,
        discretisation_time=0.1,
        input_magnitude_limits=[1.0, 2.0],
        input_rate_limits=[2.0, 4.0],
    )
    model.time_step = 1
    model.F = np.diag([0.5, 0.25])
    model.G = np.diag([2.0, 3.0])
    model.xt_1 = np.array([[1.0], [2.0]])
    model.ut_1 = np.array([[0.9], [-1.8]])
    # Rate-limited [.9+.2, -1.8-.4] is magnitude-limited to [1, -2].
    actual = model.evaluate_incremental_model(
        np.array([2.0, 4.0]), np.array([10.0, -10.0]).reshape(shape)
    )
    np.testing.assert_allclose(actual, [[2.7], [3.9]])
    np.testing.assert_allclose(model.delta_ut, [[0.1], [-0.2]])
    np.testing.assert_allclose(model.evaluate_incremental_model(), actual)

    matrices = (model.F.copy(), model.G.copy())
    model.store_input.fill(3.0)
    model.store_delta_xt.fill(4.0)
    model.store_delta_ut.fill(5.0)
    model.restart_incremental_model()
    assert model.time_step == 0
    for value in (
        model.xt,
        model.xt_1,
        model.ut,
        model.ut_1,
        model.delta_xt,
        model.delta_ut,
        model.xt1_est,
        model.store_input,
        model.store_delta_xt,
        model.store_delta_ut,
    ):
        assert np.count_nonzero(value) == 0
    np.testing.assert_array_equal(model.F, matrices[0])
    np.testing.assert_array_equal(model.G, matrices[1])
    # First sample initializes history, so no incremental change is predicted.
    np.testing.assert_allclose(
        model.evaluate_incremental_model([3.0, 5.0], [0.1, -0.2]), [[3.0], [5.0]]
    )


def test_incremental_rejects_mismatched_control_and_state_dimensions():
    model = IncrementalModel(["x", "y"], ["u", "v"], 12)
    with pytest.raises(ValueError, match="input has shape"):
        model.evaluate_incremental_model([0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="xt has shape"):
        model.identify_incremental_model_LS(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="Unexpected number"):
        model.evaluate_incremental_model(1, 2, 3)
    with pytest.raises(ValueError, match="input_rate_limits must be scalar"):
        IncrementalModel(["x"], ["u", "v"], 12, input_rate_limits=[1, 2, 3])
