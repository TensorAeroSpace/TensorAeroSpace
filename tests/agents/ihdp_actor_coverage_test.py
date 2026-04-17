"""Additional targeted coverage for the IHDP Actor network.

Exercises the alternate training paths (adaptive alpha / Adam / alpha decay),
evaluate_actor branches, and WB-limit saturation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ihdp.Actor import Actor


def _mk(**overrides):
    kw = dict(
        selected_inputs=["u"],
        selected_states=["alpha"],
        tracking_states=["alpha"],
        indices_tracking_states=[0],
        number_time_steps=6,
        start_training=-1,
        layers=(4, 1),
        activations=("tanh", "tanh"),
        learning_rate=0.1,
        learning_rate_exponent_limit=3,
        type_PE="combined",
        amplitude_3211=1,
        pulse_length_3211=3,
        maximum_input=1,
        maximum_q_rate=1,
        WB_limits=2.0,
        NN_initial=1,
        cascaded_actor=False,
        learning_rate_cascaded=0.1,
    )
    kw.update(overrides)
    a = Actor(**kw)
    a.build_actor_model()
    return a


def test_build_actor_model_initialises_weights():
    a = _mk()
    assert a.model is not None
    # store_W1 is allocated by create_NN.
    assert "W1" in a.store_weights


def test_evaluate_actor_uses_stored_state_with_zero_args():
    a = _mk()
    a.xt = np.array([[0.1]])
    a.xt_ref = np.array([[0.0]])
    out = a.evaluate_actor()
    assert out is not None


def test_evaluate_actor_with_explicit_args():
    a = _mk()
    out = a.evaluate_actor(np.array([[0.1]]), np.array([[0.0]]))
    assert out is not None


def test_evaluate_actor_too_many_args_raises():
    a = _mk()
    with pytest.raises(Exception, match="THERE SHOULD BE AN OUTPUT"):
        a.evaluate_actor(1, 2, 3, 4)


def test_run_actor_online_populates_buffers():
    a = _mk()
    xt = np.array([[0.1]])
    xt_ref = np.array([[0.0]])
    ut = a.run_actor_online(xt, xt_ref)
    assert ut is not None
    assert a.dut_dWb is not None


def test_train_actor_online_adaptive_alpha_runs():
    a = _mk()
    a.run_actor_online(np.array([[0.1]]), np.array([[0.0]]))
    # Synthetic Jt1 / dJt1_dxt1 / G matching tracked-states shape.
    Jt1 = np.array([[0.05]])
    dJt1_dxt1 = np.array([[0.1]])
    G = np.array([[0.2]])

    class _StubIncModel:
        def evaluate_incremental_model(self, *_a, **_kw):
            return np.zeros((1, 1))

    class _StubCritic:
        def evaluate_critic(self, *_a, **_kw):
            return np.array([[0.1]]), np.array([[0.05]])

    # Time step must exceed start_training (=-1) so the training branch fires.
    a.time_step = 2
    a.train_actor_online_adaptive_alpha(
        Jt1=Jt1,
        dJt1_dxt1=dJt1_dxt1,
        G=G,
        incremental_model=_StubIncModel(),
        critic=_StubCritic(),
        xt_ref1=np.array([[0.0]]),
    )


def test_train_actor_online_adam_runs():
    a = _mk()
    a.run_actor_online(np.array([[0.1]]), np.array([[0.0]]))

    class _StubIncModel:
        def evaluate_incremental_model(self, *_a, **_kw):
            return np.zeros((1, 1))

    class _StubCritic:
        def evaluate_critic(self, *_a, **_kw):
            return np.array([[0.1]]), np.array([[0.05]])

    a.time_step = 2
    a.train_actor_online_adam(
        Jt1=np.array([[0.05]]),
        dJt1_dxt1=np.array([[0.1]]),
        G=np.array([[0.2]]),
        incremental_model=_StubIncModel(),
        critic=_StubCritic(),
        xt_ref1=np.array([[0.0]]),
    )


def test_train_actor_online_alpha_decay_runs():
    a = _mk()
    a.run_actor_online(np.array([[0.1]]), np.array([[0.0]]))

    class _StubIncModel:
        def evaluate_incremental_model(self, *_a, **_kw):
            return np.zeros((1, 1))

    class _StubCritic:
        def evaluate_critic(self, *_a, **_kw):
            return np.array([[0.1]]), np.array([[0.05]])

    a.time_step = 2
    a.ut = np.array([[0.0]])
    a.train_actor_online_alpha_decay(
        Jt1=np.array([[0.05]]),
        dJt1_dxt1=np.array([[0.1]]),
        G=np.array([[0.2]]),
        incremental_model=_StubIncModel(),
        critic=_StubCritic(),
        xt_ref1=np.array([[0.0]]),
    )


def test_check_WB_limits_clips_params():
    a = _mk(WB_limits=0.5)
    # Saturate parameters well above the limit.
    with torch.no_grad():
        for p in a.model.parameters():
            p.data.fill_(10.0)
    for p in a.model.parameters():
        a._check_WB_limits_param(p)
    for p in a.model.parameters():
        assert float(p.detach().abs().max()) <= 0.5 + 1e-6


def test_restart_actor_clears_state():
    a = _mk()
    a.run_actor_online(np.array([[0.1]]), np.array([[0.0]]))
    if hasattr(a, "restart_actor"):
        a.restart_actor()
        # The momentum buffers should be reset.
        assert all(v == 0 for v in a.momentum_dict.values())
