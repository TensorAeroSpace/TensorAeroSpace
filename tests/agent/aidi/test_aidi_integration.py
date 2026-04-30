"""End-to-end: AIDIAgent on the F-16 nonlinear angular env under CE-loss."""

import math

import numpy as np
import pytest

import tensoraerospace  # noqa: F401  — registers gym envs
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    stab_efficiency_step,
)
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE


pytestmark = pytest.mark.integration


def _make_env(damage_profile=None, n_steps=2000):
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    initial_state = np.zeros(14)
    initial_state[0] = math.radians(2.0)  # alpha
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
        cstar_kp=1.0, cstar_ki=0.4,
        roll_omega_n=2.0, roll_zeta=0.7,
        sideslip_kp=1.0, sideslip_ki=0.05,
        seed=0,
    )
    return AIDIAgent(
        n_state=3, n_control=3,
        onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
        config=cfg,
    )


def _run(agent, env, n_steps=1500):
    obs_arr, _ = env.reset()
    rmse_q_sq = 0.0
    n = 0
    for k in range(n_steps):
        observation = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        c_star = 1.0
        if 5.0 <= k * env.dt < 8.0:
            c_star = 1.6
        elif 8.0 <= k * env.dt < 11.0:
            c_star = 0.4
        refs = {
            "C_star": float(c_star), "phi_cmd": 0.0,
            "beta_cmd": 0.0, "V_cmd": 200.0,
        }
        u_rad = agent.predict(observation, references=refs, time_step=k)
        u_deg = np.rad2deg(u_rad)
        obs_arr, _r, _term, _trunc, _info = env.step(u_deg)
        next_obs = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        agent.learn(next_obs, references=refs, time_step=k)
        if k * env.dt >= 12.0:
            rmse_q_sq += float(obs_arr[3] ** 2)
            n += 1
    return math.sqrt(rmse_q_sq / max(n, 1))


def test_aidi_recovers_under_stab_efficiency_loss():
    profile_a = stab_efficiency_step(t_inject=8.0, mu=0.25)
    profile_b = stab_efficiency_step(t_inject=8.0, mu=0.25)

    rmse_adapt = _run(_build_agent(adapt_enabled=True),
                      _make_env(damage_profile=profile_a),
                      n_steps=1500)
    rmse_baseline = _run(_build_agent(adapt_enabled=False),
                         _make_env(damage_profile=profile_b),
                         n_steps=1500)

    # Sanity bound — adaptive AIDI must keep |q| within 0.1 rad/s RMSE.
    assert rmse_adapt < 0.1
    # Adaptive must beat the frozen baseline (or at least not be worse).
    assert rmse_adapt <= 1.0 * rmse_baseline + 1e-3
