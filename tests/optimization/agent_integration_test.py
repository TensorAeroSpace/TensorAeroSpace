"""The same entry point constructs real agents with their native parameter APIs."""

import numpy as np
import pytest
import torch

from tensoraerospace.optimization import Float, Int


@pytest.mark.parametrize(
    "kind", ["aaindi", "aidi", "iadp", "imgdhp", "etdhp", "hdp", "ihdp", "mpc"]
)
def test_real_agent_optimization_rebuilds_independent_components(kind):
    if kind == "aaindi":
        from tensoraerospace.agent.aa_indi import (
            AAINDIAgent,
            AAINDIConfig,
            AircraftGeometry,
        )

        cls = AAINDIAgent
        kwargs = {
            "config": AAINDIConfig(
                AircraftGeometry(np.eye(3), 2.0, 2.0, 2.0), np.eye(3)
            )
        }
        space = {"config.rate_feedback.1": Float(3.0, 3.0)}

        def check(a):
            return a.cfg.rate_feedback[1]

    elif kind == "aidi":
        from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig
        from tensoraerospace.agent.aidi.onboard_ce import LinearOnboardCE

        cls = AIDIAgent
        kwargs = dict(
            n_state=3,
            n_control=3,
            onboard_ce=LinearOnboardCE(np.eye(3)),
            config=AIDIConfig(),
        )
        space = {"config.rate_kp.1": Float(3.0, 3.0)}

        def check(a):
            return a.cfg.rate_kp[1]

    elif kind == "iadp":
        from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

        cls = IADPAgent
        kwargs = dict(n_state=1, n_control=1, config=IADPConfig(R=np.eye(1)))
        space = {"config.R.0.0": Float(3.0, 3.0)}

        def check(a):
            return a.R[0, 0]

    elif kind == "imgdhp":
        from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig

        cls = IMGDHPAgent
        kwargs = dict(
            n_obs=1,
            n_action=1,
            config=IMGDHPConfig(actor_hidden=(4,), critic_hidden=(4,)),
        )
        space = {"config.actor_lr": Float(0.003, 0.003)}

        def check(a):
            return 1000 * a.actor_opt.param_groups[0]["lr"]

    elif kind == "etdhp":
        from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig

        cls = ETDHPAgent
        kwargs = dict(
            n_state=1,
            n_control=1,
            config=ETDHPConfig(
                Q=(1.0,),
                R=(1.0,),
                actor_hidden=(4,),
                critic_hidden=(4,),
                model_hidden=(4,),
            ),
        )
        space = {"config.actor_lr": Float(0.003, 0.003)}

        def check(a):
            return 1000 * a.actor_opt.param_groups[0]["lr"]

    elif kind == "hdp":
        from tensoraerospace.agent.hdp import HDP
        from tensoraerospace.envs.b747 import ImprovedB747Env

        cls = HDP

        def kwargs(seed):
            return dict(
                env=ImprovedB747Env(
                    initial_state=np.zeros(4),
                    reference_signal=np.zeros((1, 20)),
                    number_time_steps=20,
                ),
                hidden_size=8,
            )

        space = {"actor_lr": Float(0.003, 0.003)}

        def check(a):
            return 1000 * a.actor_optim.param_groups[0]["lr"]

    elif kind == "ihdp":
        from tensoraerospace.agent.ihdp import IHDPAgent

        cls = IHDPAgent
        kwargs = dict(
            actor_settings=dict(
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
            ),
            critic_settings=dict(
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
            ),
            incremental_settings=dict(
                number_time_steps=5,
                dt=0.1,
                input_magnitude_limits=1,
                input_rate_limits=10,
            ),
            tracking_states=["alpha"],
            selected_states=["alpha"],
            selected_input=["u"],
            number_time_steps=5,
            indices_tracking_states=[0],
        )
        space = {"actor_settings.learning_rate": Float(0.003, 0.003)}

        def check(a):
            return 1000 * a.actor.learning_rate

    else:
        from tensoraerospace.agent.mpc.mpc import MPC, MPCWeights

        cls = MPC
        kwargs = dict(
            dynamics=lambda x, u: 0.9 * x + 0.1 * u,
            state_dim=1,
            action_dim=1,
            weights=MPCWeights(Q_diag=[1.0], R_diag=[0.01]),
            horizon=2,
            iters=2,
        )
        space = {"horizon": Int(3, 3)}

        def check(a):
            return a.horizon

    seen = []

    def evaluate(agent, seed):
        seen.append(agent)
        try:
            assert check(agent) == pytest.approx(3.0)
            return float(check(agent))
        finally:
            if kind == "hdp":
                agent.env.close()

    result = cls.optimize(
        space, evaluate, agent_kwargs=kwargs, n_trials=1, seeds=[0, 1]
    )
    assert result.best_value == pytest.approx(3.0)
    assert len(seen) == 2 and seen[0] is not seen[1]
    if kind in ("imgdhp", "etdhp", "hdp"):
        first, second = next(seen[0].actor.parameters()), next(
            seen[1].actor.parameters()
        )
        assert first.data_ptr() != second.data_ptr()
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
