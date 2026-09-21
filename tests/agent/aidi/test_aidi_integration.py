"""Physical F-16 integration: measured servos, conventional axes, real baseline.

A command-gain fault changes actuator response, not deflection-to-moment CE.
Theta is therefore not asserted to equal the injected command multiplier.
"""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    stab_efficiency_step,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
from tensoraerospace.scripts.benchmark_aidi import _build_agent, _solve_trim

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("adapt", [True, False])
def test_aidi_runs_through_stab_command_loss(adapt):
    alpha, stab = _solve_trim()
    initial = np.zeros(14)
    initial[0] = initial[7] = alpha
    initial[8] = stab
    initial[1] = np.deg2rad(0.1)  # Exercise yaw/sideslip sign, not just pitch.
    env = NonlinearAngularF16(
        initial_state=initial,
        number_time_steps=2402,
        dt=0.01,
        integrator="rk4",
        airspeed=120.0,
        damage_profile=stab_efficiency_step(t_inject=5.0, mu=0.75),
    )
    agent = _build_agent("adaptive" if adapt else "frozen")
    x, _ = env.reset()
    agent.reset(initial_action=x[[8, 10, 12]])
    states, events = [], []
    for k in range(2400):

        def observe(state):
            return dict(
                omega=state[[2, 4, 3]] * [1, 1, -1],
                alpha=state[0],
                beta=state[1],
                theta=state[7],
                phi=state[5],
                V=120.0,
                state=state.copy(),
            )

        refs = dict(
            C_star=1.0,
            phi_cmd=np.deg2rad(2) * np.sin(k * 0.01 * np.pi / 20),
            beta_cmd=0.0,
            V_cmd=120.0,
        )
        command = agent.predict(observe(x), refs)
        old_actuators = x[[8, 10, 12]].copy()
        x, _, terminated, truncated, info = env.step(np.rad2deg(command))
        assert not terminated and not truncated
        events.extend(info.get("damage_events_triggered", []))
        agent.learn(
            observe(x),
            refs,
            applied_action=(old_actuators + x[[8, 10, 12]]) / 2,
            adapt=adapt,
        )
        states.append(x.copy())
    states = np.asarray(states)
    assert len(events) == 1
    assert np.isfinite(states).all()
    assert np.max(abs(np.rad2deg(states[:, 1]))) < 1.0
    assert np.max(abs(np.rad2deg(states[:, 5]))) < 5.0
    assert np.max(abs(np.rad2deg(states[:, 7]))) < 25.0
    assert np.isfinite(agent.rls.theta).all() and np.isfinite(agent.rls.P).all()
    if adapt:
        assert agent.rls.num_updates == 2399
        assert np.max(abs(agent.rls.theta - 1)) > 1e-5
    else:
        assert agent.rls.num_updates == 0
        np.testing.assert_array_equal(agent.rls.theta, np.ones((3, 3)))
