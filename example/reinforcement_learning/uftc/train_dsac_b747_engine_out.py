"""Pre-train the L4 D-SAC outer-loop actor for the B-747 engine-out demo.

The training environment is the **full nonlinear B-747 6-DoF model**
(``NonlinearB747Env`` with ``LEFT_OUTER_ENGINE_FAILURE`` damage profile);
all RL transitions come from this nonlinear simulator. The
``UFTC_F_INIT`` / ``UFTC_G_INIT`` matrices loaded below are
**topology-based warm-starts** for the L3 IADP RLS — they encode
kinematic relationships such as "theta_error integrates q" and "phi
responds to aileron", *not* a linearisation of the plant. RLS adapts
the matrices online to local nonlinear dynamics; the warm-start only
keeps them away from random/zero on the first 50–100 ticks.

Usage (default 5 episodes x 6000 steps @ dt=0.05):

    poetry run python example/reinforcement_learning/uftc/train_dsac_b747_engine_out.py \
        --episodes 5 --out artifacts/dsac/b747_engine_out_v1
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import fsolve

from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration, trim
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    LEFT_OUTER_ENGINE_FAILURE,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.damage.state import (
    B747DamageState,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.dynamics import b747_ode_6dof
from tensoraerospace.aerospacemodel.b747.nonlinear.params import default_parameters
from tensoraerospace.agent.aa_indi.model import AAINDIConfig
from tensoraerospace.agent.iadp.model import IADPConfig
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController
from tensoraerospace.agent.uftc.fdd.detector import FDDConfig
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --------------------------------------------------------------------------
# Scenario constants — must match the Phase 3 notebook so saved L4 weights
# are weight-compatible with the controller built there.
# --------------------------------------------------------------------------
DT = 0.05
TOTAL_TIME = 300.0
DAMAGE_TIME = 10.0
N_EP = int(TOTAL_TIME / DT)  # 6000 steps per episode
DAMAGE_STEP = int(DAMAGE_TIME / DT)

V_REF_FT_S = 674.0
ALT_REF_FT = 20_000.0
PSI_REF_DEG = 0.0


# --------------------------------------------------------------------------
# UFTC explicit regulation state and warm-start matrices.
#
# State layout (matches notebook):
#   [eV, eh, e_theta, e_psi, e_phi, e_beta, p, q, r]
# UFTC_F_INIT / UFTC_G_INIT are TOPOLOGY-BASED warm-starts that encode
# integrator relationships, not analytical linearisations.
# --------------------------------------------------------------------------
UFTC_STATE_SCALE = np.array(
    [
        25.0,
        250.0,
        math.radians(3.0),
        math.radians(8.0),
        math.radians(8.0),
        math.radians(2.0),
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float64,
)
N_UFTC_STATE = int(UFTC_STATE_SCALE.size)
UFTC_OMEGA_INDICES = [6, 7, 8]
UFTC_REFERENCE = np.zeros(N_UFTC_STATE, dtype=np.float64)

UFTC_F_INIT = np.eye(N_UFTC_STATE, dtype=np.float64) * 0.99
UFTC_F_INIT[2, 7] = DT / UFTC_STATE_SCALE[2]
UFTC_F_INIT[3, 8] = DT / UFTC_STATE_SCALE[3]
UFTC_F_INIT[4, 6] = DT / UFTC_STATE_SCALE[4]

UFTC_G_INIT = np.zeros((N_UFTC_STATE, 4), dtype=np.float64)
UFTC_G_INIT[0] = [0.0, 0.0, 0.0, 0.08]
UFTC_G_INIT[1] = [-0.02, 0.0, 0.0, 0.03]
UFTC_G_INIT[6] = [0.00, 0.30, 0.00, 0.00]
UFTC_G_INIT[7] = [0.40, 0.00, 0.00, 0.10]
UFTC_G_INIT[8] = [0.00, 0.05, 0.30, 0.00]
UFTC_G_INNER = UFTC_G_INIT[UFTC_OMEGA_INDICES].copy()

UFTC_RESIDUAL_SCALE = np.array(
    [
        0.0,
        math.radians(0.05),
        math.radians(0.05),
        0.0,
    ],
    dtype=np.float64,
)

L4_ACTION_SCALE = 0.05
L4_TRIM_FREE_INDICES = {"V_idx": 0, "gamma_idx": 1, "alpha_idx": 2, "q_idx": 7}

# IADP state-cost weighting — also used for shaping the L4 reward so that
# critic / actor see consistent regulation cost signals.
Q_REWARD = np.diag([20.0, 40.0, 20.0, 20.0, 20.0, 0.5, 2.0, 2.0, 2.0])

STATE_FEEDBACK_GAINS = {
    "de_h": 1.0e-4,
    "de_theta": 1.5,
    "de_q": 1.0,
    "throttle_v": -3.0e-3,
    "throttle_h": -1.0e-4,
    "da_phi": -2.5,
    "da_p": -1.5,
    "dr_r": 3.0,
    "dr_psi": 1.5,
    "dr_beta": -0.5,
}


# --------------------------------------------------------------------------
# Trim helpers — replicate the notebook so the envelope feed-forward and
# the engine-out trim solution match exactly.
# --------------------------------------------------------------------------
def _solve_trims():
    trim_result = trim(
        altitude_ft=ALT_REF_FT,
        V_ft_s=V_REF_FT_S,
        config=B747Configuration.NOMINAL,
    )
    assert trim_result.converged, "healthy trim did not converge"
    delta_e_trim_rad = float(trim_result.elevator_rad)
    throttle_trim = float(trim_result.throttle)
    healthy_trim_action = np.array(
        [
            delta_e_trim_rad,
            0.0,
            0.0,
            throttle_trim,
        ],
        dtype=np.float64,
    )

    engine_out_params = default_parameters(B747Configuration.NOMINAL)
    eo_state = B747DamageState.healthy()
    eo_state.engines_mu[1] = 0.0
    engine_out_params.damage_state = eo_state
    throttle_eo_estimate = min(throttle_trim * 4.0 / 3.0, 1.0)

    def state_from_vars(alpha_rad, beta_rad, theta_rad):
        return np.array(
            [
                V_REF_FT_S * math.cos(alpha_rad) * math.cos(beta_rad),
                V_REF_FT_S * math.sin(beta_rad),
                V_REF_FT_S * math.sin(alpha_rad) * math.cos(beta_rad),
                0.0,
                0.0,
                0.0,
                0.0,
                theta_rad,
                0.0,
                0.0,
                0.0,
                -ALT_REF_FT,
            ],
            dtype=np.float64,
        )

    def residual(z):
        alpha, beta, theta, de, da, dr, thr = z
        x = state_from_vars(alpha, beta, theta)
        u = np.array([de, da, dr, np.clip(thr, 0.0, 1.0)], dtype=np.float64)
        dx = b747_ode_6dof(x, u, 0.0, engine_out_params)
        return np.array(
            [dx[0], dx[1], dx[2], dx[3], dx[4], dx[5], dx[11]], dtype=np.float64
        )

    z0 = np.array(
        [
            trim_result.alpha_rad,
            math.radians(-0.44),
            trim_result.theta_rad,
            delta_e_trim_rad,
            math.radians(-4.4),
            math.radians(-2.64),
            throttle_eo_estimate,
        ],
        dtype=np.float64,
    )
    sol, info, ier, msg = fsolve(
        residual, z0, full_output=True, xtol=1e-10, maxfev=2_000
    )
    res_norm = float(np.linalg.norm(info["fvec"]))
    assert ier == 1 and res_norm < 1e-6, msg
    (alpha_eo, beta_eo, _, de_eo, da_eo, dr_eo, thr_eo) = sol
    thr_eo = float(np.clip(thr_eo, 0.0, 1.0))
    engine_out_action = np.array([de_eo, da_eo, dr_eo, thr_eo], dtype=np.float64)
    return healthy_trim_action, engine_out_action, float(beta_eo)


def _wrap_rad(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _true_airspeed(obs):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(obs[:3]))


def _altitude(obs):
    return float(-np.asarray(obs, dtype=np.float64).reshape(-1)[11])


def _body_sideslip(obs):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    v = max(_true_airspeed(obs), 1.0)
    return float(math.asin(np.clip(obs[1] / v, -1.0, 1.0)))


def _engine_loss_from_info(info, fallback=0.0):
    damage_state = info.get("damage_state", {}) if isinstance(info, dict) else {}
    engines_mu = (
        damage_state.get("engines_mu", {}) if isinstance(damage_state, dict) else {}
    )
    if not engines_mu:
        return float(fallback)
    mu1 = float(engines_mu.get(1, engines_mu.get("1", 1.0)))
    return float(np.clip(1.0 - mu1, 0.0, 1.0))


def _trim_action_for_loss(loss, healthy_act, eo_act):
    loss = float(np.clip(loss, 0.0, 1.0))
    return (1.0 - loss) * healthy_act + loss * eo_act


def _clip_physical(action):
    action = np.asarray(action, dtype=np.float64).reshape(4)
    return np.array(
        [
            np.clip(action[0], -math.radians(25.0), math.radians(25.0)),
            np.clip(action[1], -math.radians(20.0), math.radians(20.0)),
            np.clip(action[2], -math.radians(25.0), math.radians(25.0)),
            np.clip(action[3], 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def _make_initial_reference(obs):
    return {
        "V": _true_airspeed(obs),
        "h": _altitude(obs),
        "theta": float(np.asarray(obs).reshape(-1)[7]),
        "psi": float(np.asarray(obs).reshape(-1)[8]),
        "phi": float(np.asarray(obs).reshape(-1)[6]),
        "beta": _body_sideslip(obs),
    }


def _uftc_state_transform(obs, ref0, engine_loss, beta_engine_out_rad):
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    loss = float(np.clip(engine_loss, 0.0, 1.0))
    beta_target = (1.0 - loss) * ref0["beta"] + loss * beta_engine_out_rad
    raw = np.array(
        [
            _true_airspeed(obs) - ref0["V"],
            _altitude(obs) - ref0["h"],
            _wrap_rad(obs[7] - ref0["theta"]),
            _wrap_rad(obs[8] - ref0["psi"]),
            obs[6] - ref0["phi"],
            _body_sideslip(obs) - beta_target,
            obs[3],
            obs[4],
            obs[5],
        ],
        dtype=np.float64,
    )
    return raw / UFTC_STATE_SCALE


def _state_feedback_action(x_uftc, engine_loss, healthy_act, eo_act):
    x = np.asarray(x_uftc, dtype=np.float64).reshape(N_UFTC_STATE)
    v_err = x[0] * UFTC_STATE_SCALE[0]
    h_err = x[1] * UFTC_STATE_SCALE[1]
    theta_err = x[2] * UFTC_STATE_SCALE[2]
    psi_err = x[3] * UFTC_STATE_SCALE[3]
    phi_err = x[4] * UFTC_STATE_SCALE[4]
    beta_err = x[5] * UFTC_STATE_SCALE[5]
    p, q, r = x[6], x[7], x[8]
    action = _trim_action_for_loss(engine_loss, healthy_act, eo_act)
    action[0] += (
        STATE_FEEDBACK_GAINS["de_h"] * h_err
        + STATE_FEEDBACK_GAINS["de_theta"] * theta_err
        + STATE_FEEDBACK_GAINS["de_q"] * q
    )
    action[1] += (
        STATE_FEEDBACK_GAINS["da_phi"] * phi_err + STATE_FEEDBACK_GAINS["da_p"] * p
    )
    action[2] += (
        STATE_FEEDBACK_GAINS["dr_r"] * r
        + STATE_FEEDBACK_GAINS["dr_psi"] * psi_err
        + STATE_FEEDBACK_GAINS["dr_beta"] * beta_err
    )
    action[3] += (
        STATE_FEEDBACK_GAINS["throttle_v"] * v_err
        + STATE_FEEDBACK_GAINS["throttle_h"] * h_err
    )
    return _clip_physical(action)


def _compose_action(state_feedback, uftc_norm):
    residual = UFTC_RESIDUAL_SCALE * np.clip(
        np.asarray(uftc_norm, dtype=np.float64).reshape(4),
        -1.0,
        1.0,
    )
    return _clip_physical(
        np.asarray(state_feedback, dtype=np.float64).reshape(4) + residual
    )


def _make_env():
    return NonlinearB747Env(
        trim_at=(ALT_REF_FT, V_REF_FT_S),
        number_time_steps=N_EP + 5,
        dt=DT,
        integrator="rk4",
        action_space="virtual",
        config=B747Configuration.NOMINAL,
        damage_profile=LEFT_OUTER_ENGINE_FAILURE,
    )


def _make_controller(
    *, lr_actor: float, lr_critic: float, replay_capacity: int, batch_size: int
) -> UFTCController:
    cfg = UFTCConfig(
        dt=DT,
        fdd_warmup_steps=0,
        omega_indices=UFTC_OMEGA_INDICES,
        middle_lookahead_dt=0.3,
        trust_radius_nominal=0.03,
        trust_radius_fault=0.15,
        fdd_cfg=FDDConfig(
            process_noise=1e-4,
            measurement_noise=1e-3,
            adapt_Q=False,
            adapt_R=False,
            drift=6.0,
            h_alarm=15.0,
        ),
        inner_cfg=AAINDIConfig(
            dt=DT,
            ref_wn=3.0,
            ref_zeta=0.9,
            u_magnitude_limit=1.0,
            u_rate_limit=5.0,
            G_init=UFTC_G_INNER,
            ref_error_kp=2.0,
            ref_error_ki=0.05,
            seed=0,
        ),
        middle_cfg=IADPConfig(
            dt=DT,
            Q=np.diag([20.0, 40.0, 20.0, 20.0, 20.0, 0.5, 2.0, 2.0, 2.0]),
            R=np.diag([8.0, 8.0, 8.0, 20.0]),
            gamma=0.85,
            gamma_rls=0.995,
            u_magnitude_limit=1.0,
            u_rate_limit=4.0,
            policy_eval_every=80,
            policy_eval_blend=0.15,
        ),
        # Phase 3 — L4 D-SAC outer (training mode).
        enable_l4_outer=True,
        l4_action_scale=L4_ACTION_SCALE,
        l4_eval_mode=False,
        l4_seed=0,
        l4_trim_free=L4_TRIM_FREE_INDICES,
        l4_replay_capacity=int(replay_capacity),
        l4_batch_size=int(batch_size),
    )
    ctl = UFTCController(
        n_state=N_UFTC_STATE,
        n_control=4,
        nominal_F=UFTC_F_INIT,
        nominal_G=UFTC_G_INIT,
        config=cfg,
    )
    # Override learning rates on the fresh DSAC optimizers.
    for pg in ctl.l4.opt_actor.param_groups:
        pg["lr"] = float(lr_actor)
    for pg in ctl.l4.opt_c1.param_groups:
        pg["lr"] = float(lr_critic)
    for pg in ctl.l4.opt_c2.param_groups:
        pg["lr"] = float(lr_critic)
    return ctl


def _shaped_reward(
    x_next: np.ndarray,
    reference: np.ndarray,
    q_diag: np.ndarray,
    clip_low: float = -100.0,
    clip_high: float = 0.0,
) -> float:
    err = np.asarray(x_next, dtype=np.float64).reshape(-1) - np.asarray(
        reference, dtype=np.float64
    ).reshape(-1)
    cost = float(np.sum(q_diag * err * err))
    r = -cost
    if r < clip_low:
        r = clip_low
    elif r > clip_high:
        r = clip_high
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", type=str, default="artifacts/dsac/b747_engine_out_v1")
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--replay-capacity", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    healthy_act, eo_act, beta_eo_rad = _solve_trims()
    print(
        f"Healthy trim action: de={math.degrees(healthy_act[0]):+.3f} deg, "
        f"throttle={healthy_act[3]:.4f}"
    )
    print(
        f"Engine-out trim action: de/da/dr={math.degrees(eo_act[0]):+.3f}/"
        f"{math.degrees(eo_act[1]):+.3f}/{math.degrees(eo_act[2]):+.3f} deg, "
        f"throttle={eo_act[3]:.4f}, beta_eo={math.degrees(beta_eo_rad):+.3f} deg"
    )

    ctl = _make_controller(
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
    )
    print(
        f"DSAC outer: n_state={ctl.l4.cfg.n_state}, "
        f"n_action={ctl.l4.cfg.n_action}, "
        f"action_scale={ctl.l4.cfg.action_scale}, "
        f"replay_capacity={ctl.l4.cfg.replay_capacity}, "
        f"batch_size={ctl.l4.cfg.batch_size}"
    )
    print(
        f"target_entropy={ctl.l4.target_entropy:.3f}, "
        f"lr_actor={args.lr_actor:g}, lr_critic={args.lr_critic:g}"
    )

    q_diag = np.diag(Q_REWARD).astype(np.float64)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    start_wall = time.time()
    initial_beta = None
    initial_reward_mean = None
    initial_log_alpha = float(ctl.l4.log_alpha.detach().exp().item())

    for ep in range(int(args.episodes)):
        env = _make_env()
        obs, _ = env.reset()
        ctl.reset()
        ref0 = _make_initial_reference(obs)
        engine_loss = 0.0
        ep_rewards: list[float] = []
        ep_betas: list[float] = []
        ep_start = time.time()
        ep_fault_kind = "none"
        nan_seen = False

        for k in range(N_EP - 2):
            x_uftc = _uftc_state_transform(obs, ref0, engine_loss, beta_eo_rad)
            sf_action = _state_feedback_action(x_uftc, engine_loss, healthy_act, eo_act)
            u_norm = ctl.predict(x_uftc, UFTC_REFERENCE, time_step=k)
            action = _compose_action(sf_action, u_norm)

            obs, _, _, trunc, info = env.step(action)
            engine_loss = _engine_loss_from_info(info, fallback=engine_loss)
            x_next = _uftc_state_transform(obs, ref0, engine_loss, beta_eo_rad)

            if not np.all(np.isfinite(x_next)):
                print(f"  NaN in state at episode {ep+1}, step {k} — aborting episode")
                nan_seen = True
                break

            reward = _shaped_reward(
                x_next, UFTC_REFERENCE, q_diag, clip_low=-100.0, clip_high=0.0
            )
            # Patch the reward computed inside controller.learn() by setting
            # _last_r_eff first; controller.learn uses ||x - r_eff|| as reward.
            # Simplest: just call controller.learn() and let it record the
            # transition; then overwrite the reward of the most recent
            # transition so the shaped Q-form is what the critic trains on.
            ctl.learn(x_next, UFTC_REFERENCE, time_step=k)
            if len(ctl.l4.replay) > 0:
                ctl.l4.replay._buf[-1].reward = float(reward)

            ep_rewards.append(reward)
            ep_betas.append(float(ctl._last_beta))
            ep_fault_kind = str(ctl._last_fdd.fault_kind)
            total_steps += 1

            if total_steps % int(args.log_every) == 0:
                recent = ep_rewards[-int(args.log_every) :]
                la = float(ctl.l4.log_alpha.detach().exp().item())
                print(
                    f"  ep {ep+1:02d} step {k:5d} | total {total_steps:6d} | "
                    f"beta={ctl._last_beta:.3f} replay={len(ctl.l4.replay):5d} "
                    f"reward~{float(np.mean(recent)):+.3f} "
                    f"alpha={la:.3f} fault={ep_fault_kind}"
                )
                if not math.isfinite(la) or la > 50.0:
                    print(
                        f"  WARNING: log_alpha.exp() = {la:.3f} — possible saturation"
                    )

            if trunc:
                break

        ep_wall = time.time() - ep_start
        rmean = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        rmin = float(np.min(ep_rewards)) if ep_rewards else float("nan")
        rmax = float(np.max(ep_rewards)) if ep_rewards else float("nan")
        bmean = float(np.mean(ep_betas)) if ep_betas else 0.0
        la = float(ctl.l4.log_alpha.detach().exp().item())
        print(
            f"[ep {ep+1:02d}] steps={len(ep_rewards):5d} wall={ep_wall:6.1f}s "
            f"reward mean/min/max={rmean:+.3f}/{rmin:+.3f}/{rmax:+.3f} "
            f"beta_mean={bmean:.3f} alpha={la:.3f} replay={len(ctl.l4.replay):5d} "
            f"fault={ep_fault_kind}"
        )

        if ep == 0:
            initial_beta = bmean
            initial_reward_mean = rmean

        if nan_seen:
            print("  Episode aborted due to NaN — stopping training")
            break

    # Save weights.
    ctl.l4.save(out_dir)
    total_wall = time.time() - start_wall
    final_log_alpha = float(ctl.l4.log_alpha.detach().exp().item())
    print(f"\nSaved L4 D-SAC weights -> {out_dir}")
    print(f"Total wall-clock: {total_wall:.1f}s, total steps: {total_steps}")
    print(f"Initial alpha={initial_log_alpha:.3f} -> final alpha={final_log_alpha:.3f}")
    if initial_reward_mean is not None:
        print(
            f"Initial-episode reward mean: {initial_reward_mean:+.3f} "
            f"(beta_mean={initial_beta:.3f})"
        )


if __name__ == "__main__":
    main()
