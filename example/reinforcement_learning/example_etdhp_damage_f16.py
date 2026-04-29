"""ET-DHP on the nonlinear F-16 with a real damage event at t=20s.

Demonstrates the **Event-Triggered Dual Heuristic Programming** agent
adapting online to in-flight aircraft damage. The setup mirrors
``example_etdhp_nonlinear_f16.ipynb`` (global trim + feedforward
elevator curve + sinusoidal α reference + plant-model pre-training)
but plugs the damage subsystem
(``tensoraerospace.aerospacemodel.f16.nonlinear.damage``) into the
final evaluation episode:

  * At t=20 s, both wing tips lose 30% of their area simultaneously.
  * Effective wing area S, MAC, and inertias are recomputed via
    Huygens-Steiner; the longitudinal lift coefficient drops via
    strip-theory corrections, so the actor/critic see a different
    plant from t=20 s onward.
  * Online ET-DHP learning is *kept active during the damaged
    evaluation episode* — when the tracking error breaches the
    Lipschitz event trigger, the actor and critic re-train. The plant
    NN is held fixed (trained offline on the healthy aircraft); the
    closed loop adapts purely through actor/critic policy improvement.

Total simulation: 60 s. Damage trigger: 20 s.

Run with::

    poetry run python example/reinforcement_learning/example_etdhp_damage_f16.py
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
from scipy.optimize import fsolve

import tensoraerospace.envs  # noqa: F401 — registers gym envs
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
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

warnings.filterwarnings("ignore")

# === Simulation settings ===
DT = 0.01
TOTAL_TIME = 60.0
DAMAGE_TIME = 20.0
DAMAGE_STEP = int(DAMAGE_TIME / DT)

# === ET-DHP training settings ===
NUM_TRAIN_EPISODES = 6        # episodes on healthy aircraft
LOOKAHEAD_SEC = 0.85
FF_GAIN = 1.55
WARMUP_SEC = 2.0
AMPLITUDE_DEG = 3.0
FREQ_HZ = 0.1


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


def stab_for_alpha(alpha_rad: float) -> float:
    """Elevator deflection (rad) producing zero pitching moment at given alpha."""
    params = default_parameters()

    def res(s):
        x = np.array([alpha_rad, 0.0, s[0], 0.0])
        return f16_ode_long(x, np.array([s[0]]), 0.0, params)[1]
    return float(fsolve(res, x0=[math.radians(-2.0)])[0])


def build_feedforward_table(
    alpha_trim_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_deg = np.linspace(-12.0, 17.0, 61)
    grid_rad = np.deg2rad(grid_deg)
    stabs_deg = np.array([math.degrees(stab_for_alpha(a)) for a in grid_rad])
    return grid_deg, stabs_deg


def make_reference(n_steps: int, alpha_trim_rad: float) -> np.ndarray:
    """Sinusoidal alpha reference centered on trim with a 2 s warm-up."""
    warmup_steps = int(WARMUP_SEC / DT)
    ref = np.full(n_steps, alpha_trim_rad)
    active_t = np.arange(n_steps - warmup_steps) * DT
    ref[warmup_steps:] = (
        alpha_trim_rad
        + math.radians(AMPLITUDE_DEG) * np.sin(2 * np.pi * FREQ_HZ * active_t)
    )
    return ref.reshape(1, -1)


def make_feedforward_fn(reference: np.ndarray, alpha_trim_deg: float,
                        alphas_grid_deg: np.ndarray, stabs_grid_deg: np.ndarray):
    lookahead_steps = int(LOOKAHEAD_SEC / DT)

    def feedforward_fn(time_step: int, ref_signal: np.ndarray) -> float:
        k = min(time_step + lookahead_steps, ref_signal.shape[1] - 1)
        alpha_ref_deg = math.degrees(ref_signal[0, k])
        alpha_eff_deg = (
            alpha_trim_deg + FF_GAIN * (alpha_ref_deg - alpha_trim_deg)
        )
        return float(np.interp(alpha_eff_deg, alphas_grid_deg, stabs_grid_deg))

    return feedforward_fn


def make_state_transform(alpha_trim_deg: float, alphas_grid_deg: np.ndarray,
                         stabs_grid_deg: np.ndarray):
    def state_transform(obs: np.ndarray, ref: np.ndarray | None,
                        ts: int) -> np.ndarray:
        obs_deg = np.degrees(np.asarray(obs, dtype=np.float64).reshape(-1))
        if ref is not None and int(ts) < ref.shape[1]:
            alpha_ref_deg_now = math.degrees(float(ref[0, int(ts)]))
        else:
            alpha_ref_deg_now = alpha_trim_deg
        stab_ref_deg_now = float(np.interp(
            alpha_ref_deg_now, alphas_grid_deg, stabs_grid_deg,
        ))
        return np.array([
            obs_deg[0] - alpha_ref_deg_now,
            obs_deg[1],
            obs_deg[2] - stab_ref_deg_now,
            obs_deg[3],
        ])

    return state_transform


def make_env(n_steps: int, reference: np.ndarray, alpha_trim_rad: float,
             stab_trim_rad: float, *, feedforward_fn=None,
             damage_profile: DamageProfile | None = None,
             control_bias: float | None = None):
    """Build the longitudinal F-16 env at trim."""
    kwargs = dict(
        number_time_steps=n_steps + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=reference,
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=DT,
        integrator="euler",
        damage_profile=damage_profile,
    )
    if feedforward_fn is not None:
        kwargs["feedforward_fn"] = feedforward_fn
    if control_bias is not None:
        kwargs["control_bias"] = control_bias
    return gym.make("NonlinearLongitudinalF16-v0", **kwargs).unwrapped


def collect_pe_data(alpha_trim_rad: float, stab_trim_rad: float,
                    state_transform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-train data: 30 s of multi-sine PE around trim with FF off."""
    n_id = 3000
    u_amp = 2.0
    env_id = gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=n_id + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, n_id + 2), alpha_trim_rad),
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=DT,
        integrator="euler",
        control_bias=math.degrees(stab_trim_rad),
    ).unwrapped

    rng = np.random.default_rng(0)
    states_buf, actions_buf, next_states_buf = [], [], []
    ref_at_trim = np.full((1, 1), alpha_trim_rad)
    obs, _ = env_id.reset()
    for t in range(n_id):
        u_t = u_amp * (
            0.6 * np.sin(2 * np.pi * 0.3 * t * DT)
            + 0.3 * np.sin(2 * np.pi * 0.9 * t * DT)
            + 0.1 * rng.normal()
        )
        x_curr = state_transform(obs, ref_at_trim, 0)
        obs_next, _, done, _, _ = env_id.step(np.array([u_t]))
        x_next = state_transform(obs_next, ref_at_trim, 0)
        states_buf.append(x_curr)
        actions_buf.append([u_t])
        next_states_buf.append(x_next)
        obs = obs_next
        if done:
            break
    return (np.asarray(states_buf, dtype=np.float32),
            np.asarray(actions_buf, dtype=np.float32),
            np.asarray(next_states_buf, dtype=np.float32))


def build_damage_profile() -> DamageProfile:
    """Symmetric 30% wing-tip loss at t=20s — matches the iADP example."""
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


def run_episode(agent: ETDHPAgent, env_factory, reference: np.ndarray, *,
                learn: bool, n_steps: int) -> dict:
    """Run a single episode. Returns logs and trigger statistics."""
    env = env_factory()
    obs, _ = env.reset()
    agent.reset()
    alpha_log, wz_log, u_log = [], [], []
    triggers_pre = 0
    triggers_post = 0
    triggered_events: list[tuple[float, str]] = []
    for k in range(n_steps):
        agent.predict(obs, reference, k)
        u_cmd = agent.last_action()
        obs_next, _, done, _, info = env.step(u_cmd)
        if learn:
            metrics = agent.learn(obs_next, reference, k, dt=DT)
            if metrics["triggered"]:
                if k < DAMAGE_STEP:
                    triggers_pre += 1
                else:
                    triggers_post += 1

        for label in info.get("damage_events_triggered", []) or []:
            triggered_events.append((k * DT, label))

        obs_arr = np.asarray(obs_next).reshape(-1)
        alpha_log.append(np.degrees(obs_arr[0]))
        wz_log.append(np.degrees(obs_arr[1]))
        u_log.append(float(u_cmd[0]))
        obs = obs_next
        if done:
            break
    return {
        "alpha": np.asarray(alpha_log),
        "wz": np.asarray(wz_log),
        "u": np.asarray(u_log),
        "triggers_pre": triggers_pre,
        "triggers_post": triggers_post,
        "triggered_events": triggered_events,
    }


def report_episode(label: str, log: dict, reference: np.ndarray,
                   n_plot: int) -> None:
    """Pre/post-damage RMSE and trigger counts."""
    ref_deg = np.rad2deg(reference[0, :n_plot])
    pre = np.arange(int(5.0 / DT), DAMAGE_STEP)
    post = np.arange(DAMAGE_STEP + int(2.0 / DT), n_plot)
    pre_rmse = float(np.sqrt(np.mean((log["alpha"][pre] - ref_deg[pre]) ** 2)))
    pre_mae = float(np.mean(np.abs(log["alpha"][pre] - ref_deg[pre])))
    post_rmse = float(np.sqrt(np.mean((log["alpha"][post] - ref_deg[post]) ** 2)))
    post_mae = float(np.mean(np.abs(log["alpha"][post] - ref_deg[post])))
    print(f"\n=== {label} ===")
    print(f"Pre-damage  (5–20 s):    MAE={pre_mae:.4f}°  RMSE={pre_rmse:.4f}°")
    print(f"Post-damage (22–60 s):   MAE={post_mae:.4f}°  RMSE={post_rmse:.4f}°")
    print(f"Triggers before damage:  {log['triggers_pre']}")
    print(f"Triggers after damage:   {log['triggers_post']}")
    if log["triggered_events"]:
        print("Damage events:")
        for t, lab in log["triggered_events"]:
            print(f"  t={t:.2f}s : {lab}")


def maybe_plot(baseline: dict, damaged: dict, reference: np.ndarray,
               tps: np.ndarray, n_plot: int) -> None:
    """Plot α tracking, residual u, ω_z if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
        return
    ref_deg = np.rad2deg(reference[0, :n_plot])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(tps[:n_plot], ref_deg, "k--", alpha=0.6, label="α_ref")
    axes[0].plot(tps[:n_plot], baseline["alpha"], alpha=0.6,
                 label="ET-DHP (no damage)")
    axes[0].plot(tps[:n_plot], damaged["alpha"], alpha=0.9,
                 label="ET-DHP (with damage @ t=20s)")
    axes[0].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4,
                    label=f"damage @ t={DAMAGE_TIME:.0f}s")
    axes[0].set_ylabel("α [°]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(tps[:n_plot], baseline["u"], alpha=0.6, label="no damage")
    axes[1].plot(tps[:n_plot], damaged["u"], alpha=0.9, label="with damage")
    axes[1].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("residual δₑ [°]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(tps[:n_plot], baseline["wz"], alpha=0.6, label="no damage")
    axes[2].plot(tps[:n_plot], damaged["wz"], alpha=0.9, label="with damage")
    axes[2].axvline(DAMAGE_TIME, color="red", linestyle="--", alpha=0.4)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("ω_z [°/s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        "ET-DHP on nonlinear F-16: 30% bilateral wing-tip loss at t=20s "
        f"(total {TOTAL_TIME:.0f}s)"
    )
    plt.tight_layout()
    plt.savefig("etdhp_damage_f16.png", dpi=120)
    print("\nPlot saved to etdhp_damage_f16.png")
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    print(f"Total simulation time: {TOTAL_TIME:.0f} s")
    print(f"Damage trigger:        t = {DAMAGE_TIME:.0f} s "
          f"(step {DAMAGE_STEP})")

    alpha_trim_rad, stab_trim_rad = compute_trim()
    alpha_trim_deg = math.degrees(alpha_trim_rad)
    stab_trim_deg = math.degrees(stab_trim_rad)
    print(f"\nGlobal trim:  α* = {alpha_trim_deg:+.4f}°,  "
          f"δₑ* = {stab_trim_deg:+.4f}°")

    print("\n--- Building feedforward elevator curve ---")
    alphas_grid_deg, stabs_grid_deg = build_feedforward_table(alpha_trim_deg)
    print(f"FF table: {len(alphas_grid_deg)} pts over "
          f"α ∈ [{alphas_grid_deg.min():+.1f}°, {alphas_grid_deg.max():+.1f}°]")

    feedforward_fn = make_feedforward_fn(
        None, alpha_trim_deg, alphas_grid_deg, stabs_grid_deg,
    )
    state_transform = make_state_transform(
        alpha_trim_deg, alphas_grid_deg, stabs_grid_deg,
    )

    print("\n--- Pre-training plant model on PE excitation ---")
    states_arr, actions_arr, next_states_arr = collect_pe_data(
        alpha_trim_rad, stab_trim_rad, state_transform,
    )
    print(f"PE buffer: {states_arr.shape[0]} transitions")

    cfg = ETDHPConfig(
        actor_hidden=(24, 24),
        critic_hidden=(24, 24),
        model_hidden=(24, 24),
        actor_lr=1e-3,
        critic_lr=1e-3,
        model_lr=5e-3,
        model_epochs=300,
        Q=[10.0, 0.1, 0.0, 0.0],
        R=[1.0],
        gamma=0.95,
        num_epochs_per_trigger=3,
        u_bound=2.0,
        rho=0.2,
        trigger_floor=0.1,
        weight_init_scale=0.2,
        seed=0,
    )
    agent = ETDHPAgent(
        n_state=4, n_control=1,
        state_transform=state_transform,
        config=cfg,
    )
    model_losses = agent.fit_plant_model(
        states_arr, actions_arr, next_states_arr,
        batch_size=128, verbose=False,
    )
    print(f"Plant-model final MSE: {model_losses[-1]:.3e}")

    # Reference, time grid for the FINAL eval episode (60 s)
    tp = generate_time_period(tn=int(TOTAL_TIME), dt=DT)
    tps = convert_tp_to_sec_tp(tp, dt=DT)
    n_steps = len(tp)
    reference = make_reference(n_steps, alpha_trim_rad)

    # ---- Training on healthy aircraft (multiple episodes) ----
    print(f"\n--- Training: {NUM_TRAIN_EPISODES} episodes on HEALTHY F-16 ---")

    def healthy_env():
        return make_env(
            n_steps, reference, alpha_trim_rad, stab_trim_rad,
            feedforward_fn=feedforward_fn,
        )

    for ep in range(NUM_TRAIN_EPISODES):
        log = run_episode(agent, healthy_env, reference,
                          learn=True, n_steps=n_steps - 2)
        ref_deg = np.rad2deg(reference[0, :len(log["alpha"])])
        late = np.arange(len(log["alpha"]) // 2, len(log["alpha"]))
        rmse = float(np.sqrt(np.mean(
            (log["alpha"][late] - ref_deg[late]) ** 2
        )))
        n_trig = log["triggers_pre"] + log["triggers_post"]
        print(f"  ep {ep+1}/{NUM_TRAIN_EPISODES}: "
              f"RMSE_late={rmse:.4f}°  triggers={n_trig}")

    # ---- Final eval: BASELINE (no damage), online learning enabled ----
    print("\n--- Final eval: BASELINE (no damage) ---")

    def baseline_env():
        return make_env(
            n_steps, reference, alpha_trim_rad, stab_trim_rad,
            feedforward_fn=feedforward_fn,
        )

    baseline_log = run_episode(agent, baseline_env, reference,
                               learn=True, n_steps=n_steps - 2)
    report_episode("Baseline (no damage)", baseline_log, reference,
                   len(baseline_log["alpha"]))

    # ---- Final eval: WITH DAMAGE at t=20s, online learning enabled ----
    print("\n--- Final eval: 30% BILATERAL WING-TIP LOSS at t=20s ---")
    profile = build_damage_profile()

    def damaged_env():
        return make_env(
            n_steps, reference, alpha_trim_rad, stab_trim_rad,
            feedforward_fn=feedforward_fn, damage_profile=profile,
        )

    damaged_log = run_episode(agent, damaged_env, reference,
                              learn=True, n_steps=n_steps - 2)
    report_episode("With damage (30% bilateral wing-tip loss at t=20s)",
                   damaged_log, reference, len(damaged_log["alpha"]))

    n_plot = min(len(baseline_log["alpha"]), len(damaged_log["alpha"]))
    maybe_plot(baseline_log, damaged_log, reference, tps, n_plot)


if __name__ == "__main__":
    main()
