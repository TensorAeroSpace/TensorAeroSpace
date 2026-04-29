"""iADP on the nonlinear F-16 with a real damage event at t=20s.

Demonstrates the iADP (Incremental Approximate Dynamic Programming) agent
adapting to in-flight aircraft damage. Unlike the older actuator-fault
example that fakes a failure by multiplying the agent's command, this
example uses the proper damage subsystem
(``tensoraerospace.aerospacemodel.f16.nonlinear.damage``):

  * At t=20 s, both wing tips lose 30% of their area simultaneously.
  * Effective wing area S, MAC, and inertias are recomputed via
    Huygens-Steiner; the longitudinal lift coefficient drops via
    strip-theory corrections.
  * The local plant gain G̃ shifts in real time. iADP's RLS catches it
    within a few tens of milliseconds and the closed-form policy keeps
    tracking.

Total simulation: 60 s. Damage trigger: 20 s.

Run with::

    poetry run python example/reinforcement_learning/example_iadp_damage_f16.py
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import fsolve

import tensoraerospace  # noqa: F401 — registers Gymnasium envs
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    default_parameters,
)
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

warnings.filterwarnings("ignore")

# === Simulation settings ===
DT = 0.01
TOTAL_TIME = 60.0
DAMAGE_TIME = 20.0
N_STEPS = int(TOTAL_TIME / DT)               # 6000 steps
DAMAGE_STEP = int(DAMAGE_TIME / DT)          # 2000

# === iADP hyper-parameters ===
Q_VAL = 30_000.0
R_VAL = 0.1
GAMMA = 0.9


def compute_trim() -> tuple[float, float]:
    """Solve for (alpha*, stab*) such that alpha-dot = wz-dot = 0 at wz=0."""
    params = default_parameters()

    def trim_residual(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        return list(f16_ode_long(x, np.array([stab]), 0.0, params)[:2])

    sol, _info, ier, msg = fsolve(
        trim_residual, x0=[math.radians(2.0), math.radians(-2.0)],
        full_output=True,
    )
    assert ier == 1, f"trim search failed: {msg}"
    return float(sol[0]), float(sol[1])


def make_env(n_steps: int, alpha_trim: float, stab_trim: float,
             damage_profile: DamageProfile | None = None):
    """Construct the longitudinal F-16 env at trim, optionally with damage."""
    env = gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=n_steps + 2,
        initial_state=[alpha_trim, 0.0, stab_trim, 0.0],
        reference_signal=np.full((1, n_steps + 2), alpha_trim),
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=DT,
        integrator="euler",
        control_bias=math.degrees(stab_trim),
        damage_profile=damage_profile,
    ).unwrapped
    env.reset()
    return env


def warm_start_FG(alpha_trim: float, stab_trim: float) -> tuple[np.ndarray, np.ndarray]:
    """PE excitation around trim → identify a scalar discrete-time (F, G).

    Returns the augmented-state F_init and G_init for iADP.
    """
    n_pe = 300
    env_pe = make_env(n_pe, alpha_trim, stab_trim)
    obs, _ = env_pe.reset()
    wz_hist = [float(obs[1])]
    u_hist = [0.0]
    for t in range(n_pe):
        u = (2.0 * math.sin(2 * math.pi * 0.7 * t * DT)
             + 1.0 * math.sin(2 * math.pi * 1.5 * t * DT))
        obs, *_ = env_pe.step(np.array([u]))
        wz_hist.append(float(obs[1]))
        u_hist.append(float(u))

    wz = np.asarray(wz_hist)
    us = np.asarray(u_hist)
    dwz = np.diff(wz)
    du = np.diff(us)
    A_pe = np.column_stack([dwz[:-1], du[:-1]])
    b_pe = dwz[1:]
    F_wz, G_wz = np.linalg.lstsq(A_pe, b_pe, rcond=None)[0]
    print(f"PE-identified F_wz = {F_wz:+.4f}")
    print(f"PE-identified G_wz = {G_wz:+.4f}  (per ° elevator, discrete-time)")

    # Augmented-state warm-start (block structure: [wz; wz_ref])
    F_init = np.array([[F_wz, 0.0], [0.0, 1.0]])
    G_init = np.array([[G_wz], [0.0]])
    return F_init, G_init


def dare_warm_start(F_init: np.ndarray, G_init: np.ndarray) -> np.ndarray:
    """DARE solution of the augmented-LQT problem → iADP P_init."""
    Q_aug = Q_VAL * np.array([[1.0, -1.0], [-1.0, 1.0]])
    R_aug = np.array([[R_VAL]])
    P = solve_discrete_are(
        np.sqrt(GAMMA) * F_init,
        np.sqrt(GAMMA) * G_init,
        Q_aug, R_aug,
    )
    return P


def build_damage_profile() -> DamageProfile:
    """Symmetric 30% wing-tip loss at t=20s.

    Symmetric loss is meaningful in the longitudinal model: it reduces
    Cy_α (lift-curve slope), which iADP's RLS observes as a change in
    the local plant gain G̃.
    """
    return DamageProfile(events=[
        DamageEvent(
            trigger_time=DAMAGE_TIME, event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 0.30},
            label="left_tip_30pct_loss",
        ),
        DamageEvent(
            trigger_time=DAMAGE_TIME, event_type="section_loss",
            payload={"section": "right_tip", "loss_fraction": 0.30},
            label="right_tip_30pct_loss",
        ),
    ])


def run_iadp(F_init: np.ndarray, G_init: np.ndarray, P_init: np.ndarray,
             alpha_trim: float, stab_trim: float,
             damage_profile: DamageProfile | None) -> dict[str, np.ndarray]:
    cfg = IADPConfig(
        dt=DT,
        Q=np.array([[Q_VAL]]),
        R=np.array([[R_VAL]]),
        gamma=GAMMA,
        gamma_rls=0.9999,
        phi_init=1.0,
        policy_eval_window=300,
        policy_eval_every=5,
        policy_eval_warmup_updates=20,
        policy_eval_regularization=1e-10,
        policy_eval_blend=0.10,
        F_init=F_init,
        G_init=G_init,
        P_init=P_init,
        u_magnitude_limit=8.0,
        u_rate_limit=200.0,
        model_learning_only_steps=0,
        seed=0,
    )
    agent = IADPAgent(n_state=1, n_control=1, config=cfg)
    env = make_env(N_STEPS, alpha_trim, stab_trim, damage_profile=damage_profile)
    obs, _ = env.reset()

    # Sinusoidal pitch-rate command, 0.8 °/s amplitude, period ~ 8.3 s
    t_arr = np.arange(N_STEPS) * DT
    wz_cmd = math.radians(0.8) * np.sin(2 * math.pi * 0.12 * t_arr)

    logs: dict[str, list[float]] = {
        k: [] for k in ("wz", "alpha", "u", "G_est", "P_norm", "residual",
                        "damage_active")
    }
    triggered_events: list[tuple[float, str]] = []
    for k in range(N_STEPS):
        u_agent = agent.predict(
            np.array([float(obs[1])]),  # wz
            np.array([wz_cmd[k]]),
            k,
        )
        obs, _, _, _, info = env.step(u_agent)
        metrics = agent.learn(
            np.array([float(obs[1])]),
            np.array([wz_cmd[k]]),
            k,
        )
        ev_labels = info.get("damage_events_triggered") or []
        for label in ev_labels:
            triggered_events.append((k * DT, label))

        logs["wz"].append(float(obs[1]))
        logs["alpha"].append(float(obs[0]))
        logs["u"].append(float(u_agent[0]))
        logs["G_est"].append(float(agent.G[0, 0]))
        logs["P_norm"].append(metrics["P_norm"])
        logs["residual"].append(metrics["rls_pred_error_norm"])
        logs["damage_active"].append(1.0 if k >= DAMAGE_STEP else 0.0)

    out = {k: np.asarray(v) for k, v in logs.items()}
    out["t"] = t_arr
    out["wz_cmd"] = wz_cmd
    out["triggered_events"] = triggered_events
    return out


def report(label: str, log: dict[str, np.ndarray]) -> None:
    """Print pre/post-damage tracking RMSE, and the damage events seen."""
    pre_window = np.arange(500, DAMAGE_STEP)         # 5 s ≤ t < 20 s
    post_window = np.arange(DAMAGE_STEP + 200, N_STEPS)  # 22 s ≤ t ≤ 60 s
    pre_rmse = math.degrees(np.sqrt(np.mean(
        (log["wz"][pre_window] - log["wz_cmd"][pre_window]) ** 2
    )))
    post_rmse = math.degrees(np.sqrt(np.mean(
        (log["wz"][post_window] - log["wz_cmd"][post_window]) ** 2
    )))
    print(f"\n=== {label} ===")
    print(f"Pre-damage RMSE  (5 s ≤ t < 20 s):  {pre_rmse:.4f} °/s")
    print(f"Post-damage RMSE (22 s ≤ t ≤ 60 s): {post_rmse:.4f} °/s")
    print(f"G̃ at t = 19.5 s: {log['G_est'][1950]:+.5f}")
    print(f"G̃ at t = 25.0 s: {log['G_est'][2500]:+.5f}")
    print(f"G̃ at t = end:    {log['G_est'][-1]:+.5f}")
    if log["triggered_events"]:
        print("Damage events triggered:")
        for t, label in log["triggered_events"]:
            print(f"  t={t:.2f}s : {label}")


def maybe_plot(baseline: dict[str, np.ndarray],
               damaged: dict[str, np.ndarray]) -> None:
    """Plot tracking, control, G̃ adaptation, and RLS residual if matplotlib
    is available. No-op otherwise."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
        return

    t = damaged["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(t, np.degrees(damaged["wz_cmd"]), "k--", label="command",
                 alpha=0.6)
    axes[0].plot(t, np.degrees(baseline["wz"]), label="baseline (no damage)",
                 alpha=0.6)
    axes[0].plot(t, np.degrees(damaged["wz"]), label="with damage", alpha=0.9)
    axes[0].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4,
                    label=f"damage @ t={DAMAGE_TIME:.0f}s")
    axes[0].set_ylabel("ω_z [°/s]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, baseline["u"], label="baseline", alpha=0.6)
    axes[1].plot(t, damaged["u"], label="with damage", alpha=0.9)
    axes[1].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Δδₑ [°]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, baseline["G_est"], label="baseline", alpha=0.6)
    axes[2].plot(t, damaged["G_est"], label="with damage", alpha=0.9)
    axes[2].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4)
    axes[2].set_ylabel("G̃ [rad/s / °]")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, baseline["residual"], label="baseline", alpha=0.6)
    axes[3].plot(t, damaged["residual"], label="with damage", alpha=0.9)
    axes[3].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4)
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("‖RLS residual‖")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        "iADP on nonlinear F-16: 30% bilateral wing-tip loss at t=20s "
        f"(total {TOTAL_TIME:.0f}s)"
    )
    plt.tight_layout()
    plt.savefig("iadp_damage_f16.png", dpi=120)
    print("\nPlot saved to iadp_damage_f16.png")
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    print(f"Total simulation time: {TOTAL_TIME:.0f} s ({N_STEPS} steps @ dt={DT})")
    print(f"Damage trigger:        t = {DAMAGE_TIME:.0f} s "
          f"(step {DAMAGE_STEP})")

    alpha_trim, stab_trim = compute_trim()
    print(f"\nGlobal trim:  α* = {math.degrees(alpha_trim):+.4f}°,  "
          f"δₑ* = {math.degrees(stab_trim):+.4f}°")

    print("\n--- Warm-start: PE excitation ---")
    F_init, G_init = warm_start_FG(alpha_trim, stab_trim)

    P_init = dare_warm_start(F_init, G_init)
    print(f"DARE-based P_init Frobenius norm: {np.linalg.norm(P_init):.1f}")

    print("\n--- Closed-loop simulation: BASELINE (no damage) ---")
    baseline = run_iadp(F_init, G_init, P_init, alpha_trim, stab_trim,
                        damage_profile=None)
    report("Baseline (no damage)", baseline)

    print("\n--- Closed-loop simulation: 30% BILATERAL WING-TIP LOSS at t=20s ---")
    profile = build_damage_profile()
    damaged = run_iadp(F_init, G_init, P_init, alpha_trim, stab_trim,
                       damage_profile=profile)
    report("With damage (30% bilateral wing-tip loss at t=20s)", damaged)

    maybe_plot(baseline, damaged)


if __name__ == "__main__":
    main()
