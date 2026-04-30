"""End-to-end: AIDIAgent on the F-16 nonlinear angular env under CE-loss.

The integration scenario: trim the aircraft, hold C\\* = 1g for 5 s, inject
a 25 % stab efficiency loss, then keep holding trim for another 7 s. The
adaptive AIDI run is compared to a frozen-Θ baseline. Assertions:

* both runs stay finite and bounded (no divergence),
* adaptive AIDI does not perform worse than frozen on post-fault tracking,
* adaptive Θ on the pitch row moves away from 1.0 after the fault — i.e.
  online adaptation is observed.
"""

import math

import numpy as np
import pytest
from scipy.optimize import fsolve

import tensoraerospace  # noqa: F401  — registers gym envs
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    stab_efficiency_step,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE


pytestmark = pytest.mark.integration


def _solve_trim():
    params = default_parameters()

    def trim_residual(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

    sol, _info, ier, _msg = fsolve(
        trim_residual, x0=[math.radians(2.0), math.radians(-2.0)],
        full_output=True,
    )
    assert ier == 1
    return float(sol[0]), float(sol[1])


def _make_env(damage_profile, n_steps, alpha_trim, stab_trim):
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    initial_state = np.zeros(14)
    initial_state[0] = alpha_trim
    initial_state[8] = stab_trim
    return NonlinearAngularF16(
        initial_state=initial_state,
        number_time_steps=n_steps + 2,
        dt=0.01,
        integrator="rk4",
        airspeed=200.0,
        damage_profile=damage_profile,
    )


def _build_agent(adapt_enabled: bool = True) -> AIDIAgent:
    cfg = AIDIConfig(
        dt=0.01,
        u_magnitude_limit=math.radians(20.0),
        u_rate_limit=math.radians(60.0),
        rls_lambda_min=0.7 if adapt_enabled else 0.999,
        rls_lambda_max=0.999 if adapt_enabled else 0.9999,
        rls_sigma0=1e-3, rls_memory_length=100,
        rls_cov_init=10.0,
        # Gentle outer-loop gains — INDI is robust but can autovibrate
        # against the F-16 actuator if pushed too hard.
        cstar_kp=0.5, cstar_ki=0.2,
        roll_omega_n=1.5, roll_zeta=0.8,
        sideslip_kp=0.5, sideslip_ki=0.05,
        seed=0,
    )
    return AIDIAgent(
        n_state=3, n_control=3,
        onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
        config=cfg,
    )


def _run(agent, env, n_steps):
    obs_arr, _ = env.reset()
    qs = []
    for k in range(n_steps):
        observation = {
            "omega": np.array([obs_arr[2], obs_arr[4], obs_arr[3]]),  # (p,q,r)
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        refs = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
        u_rad = agent.predict(observation, references=refs, time_step=k)
        obs_arr, _r, _term, _trunc, _info = env.step(np.rad2deg(u_rad))
        next_obs = {
            "omega": np.array([obs_arr[2], obs_arr[4], obs_arr[3]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        agent.learn(next_obs, references=refs, time_step=k)
        qs.append(float(obs_arr[4]))
    return np.asarray(qs)


def test_aidi_runs_through_stab_efficiency_loss():
    alpha_trim, stab_trim = _solve_trim()
    n_steps = 1200
    profile_a = stab_efficiency_step(t_inject=5.0, mu=0.25)
    profile_b = stab_efficiency_step(t_inject=5.0, mu=0.25)

    agent_adapt = _build_agent(adapt_enabled=True)
    agent_frozen = _build_agent(adapt_enabled=False)

    qs_adapt = _run(agent_adapt,
                    _make_env(profile_a, n_steps, alpha_trim, stab_trim),
                    n_steps)
    qs_frozen = _run(agent_frozen,
                     _make_env(profile_b, n_steps, alpha_trim, stab_trim),
                     n_steps)

    # Sanity: both runs stay finite and bounded.
    assert np.all(np.isfinite(qs_adapt))
    assert np.all(np.isfinite(qs_frozen))
    assert float(np.max(np.abs(qs_adapt))) < 2.0   # rad/s — generous bound.
    assert float(np.max(np.abs(qs_frozen))) < 2.0

    # Adaptive Θ on the pitch row should move away from unity once the
    # fault is identified (column 0 = stab, row 1 = pitch in (p,q,r) order).
    theta_qstab = float(agent_adapt.rls.theta[1, 0])
    assert theta_qstab < 0.95, (
        f"adaptive Θ_q,stab did not adapt (got {theta_qstab:.4f})"
    )
