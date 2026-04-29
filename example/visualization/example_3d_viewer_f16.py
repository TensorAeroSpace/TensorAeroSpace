"""3D viewer demo: ET-DHP pitch stabilization through wing-tip damage.

End-to-end scenario:

  1. Trains ET-DHP for 6 episodes on the healthy longitudinal F-16 to
     hold trim α (= pitch stabilization in level flight).
  2. Re-runs the trained agent on the **angular** 6-DoF env
     (NonlinearAngularF16) so the 3D WebGL viewer can render the full
     trajectory. The angular env is initialised at trim via find_trim();
     each step, the angular state's (α, ω_z, stab, dstab) are mapped
     to the longitudinal-shape vector the ET-DHP agent expects, the
     agent commands a stab deflection, and that deflection is fed to
     both stabilator halves of the angular env.
  3. At t = 20 s, both wing tips lose 30 % of their area (symmetric
     damage). Pre-damage: aircraft holds level cruise. Post-damage:
     the agent's online actor/critic adapts and the closed loop
     recovers within a few seconds, holding the aircraft level
     throughout the rest of the run.

Note on the operating point: the F-16 trim for the angular model is
computed at V=180 m/s, h=3000 m. At lower airspeeds (e.g. 120 m/s)
the F-16 NASA aero tables produce a negative drag coefficient, implying
that the minimum-thrust floor (8000 N) over-powers the aircraft and
causes a persistent acceleration that no elevator controller can arrest.
At V=180 m/s the calculated trim thrust (≈9183 N) exceeds the floor and
the aircraft holds altitude/speed open-loop. The longitudinal training
plant (which is speed-independent) is unaffected by this choice.

Run with::

    poetry run python example/visualization/example_3d_viewer_f16.py
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
from scipy.optimize import fsolve

import tensoraerospace.envs  # noqa: F401 — registers gym envs
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
    f16_ode_long,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
    default_parameters as long_default_params,
)
from tensoraerospace.agent.et_dhp import ETDHPAgent, ETDHPConfig
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

warnings.filterwarnings("ignore")

DT = 0.01
TOTAL_TIME = 60.0
DAMAGE_TIME = 20.0
NUM_TRAIN_EPISODES = 6

# Flight condition for the angular env eval.
# V=180 m/s is the minimum speed at h=3000 m where the angular model's
# trim thrust (≈9183 N) exceeds the 8000 N floor so the aircraft is in
# true equilibrium. V=120 m/s requires negative thrust and is untrimmable.
ANG_V_TARGET = 180.0
ANG_H_TARGET = 3000.0


# =====================================================================
#  ET-DHP setup on the longitudinal F-16 (training rig)
# =====================================================================

def _longitudinal_trim() -> tuple[float, float]:
    p = long_default_params()

    def res(z):
        alpha, stab = z
        x = np.array([alpha, 0.0, stab, 0.0])
        return list(f16_ode_long(x, np.array([stab]), 0.0, p)[:2])

    sol, _info, ier, _msg = fsolve(
        res, x0=[math.radians(2.0), math.radians(-2.0)], full_output=True,
    )
    assert ier == 1
    return float(sol[0]), float(sol[1])


def _build_state_transform(alpha_trim_deg: float, alphas_grid_deg, stabs_grid_deg):
    def state_transform(obs, ref, ts):
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


def _build_feedforward_table():
    """δ_e(α) inverse model — needed by the longitudinal env's
    feedforward channel during training."""
    p = long_default_params()
    grid_deg = np.linspace(-12.0, 17.0, 61)
    grid_rad = np.deg2rad(grid_deg)

    def stab_for_alpha(a):
        def res(s):
            return f16_ode_long(np.array([a, 0, s[0], 0]),
                                np.array([s[0]]), 0, p)[1]
        return float(fsolve(res, x0=[math.radians(-2.0)])[0])

    stabs_deg = np.array([math.degrees(stab_for_alpha(a)) for a in grid_rad])
    return grid_deg, stabs_deg


def _make_long_env(n_steps, alpha_trim_rad, stab_trim_rad,
                   feedforward_fn):
    return gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=n_steps + 2,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, n_steps + 2), alpha_trim_rad),
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=DT,
        integrator="euler",
        feedforward_fn=feedforward_fn,
    ).unwrapped


def _train_etdhp(
    n_steps_per_ep: int,
) -> tuple[ETDHPAgent, float, float, np.ndarray, np.ndarray]:
    """Pre-train an ET-DHP agent on the healthy longitudinal F-16,
    holding constant trim α (== pitch stabilization).

    Returns (agent, alpha_trim_rad, stab_trim_rad, grid_deg, stabs_deg).
    """
    print("=== Training ET-DHP for pitch stabilization (longitudinal F-16) ===")
    alpha_trim_rad, stab_trim_rad = _longitudinal_trim()
    alpha_trim_deg = math.degrees(alpha_trim_rad)
    stab_trim_deg = math.degrees(stab_trim_rad)
    print(f"  longitudinal trim:  α = {alpha_trim_deg:+.3f}°,  "
          f"stab = {stab_trim_deg:+.3f}°")

    grid_deg, stabs_deg = _build_feedforward_table()
    state_transform = _build_state_transform(alpha_trim_deg, grid_deg, stabs_deg)

    # Constant α reference at trim — pitch-stabilization benchmark.
    reference = np.full((1, n_steps_per_ep + 2), alpha_trim_rad)

    lookahead_steps = int(0.85 / DT)
    ff_gain = 1.55

    def feedforward_fn(t, ref):
        k = min(t + lookahead_steps, ref.shape[1] - 1)
        a_ref_deg = math.degrees(ref[0, k])
        a_eff = alpha_trim_deg + ff_gain * (a_ref_deg - alpha_trim_deg)
        return float(np.interp(a_eff, grid_deg, stabs_deg))

    # Plant-NN pre-training data
    print("  collecting PE data (3000 transitions, multi-sine around trim)")
    rng = np.random.default_rng(0)
    pe_env = gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=3002,
        initial_state=[alpha_trim_rad, 0.0, stab_trim_rad, 0.0],
        reference_signal=np.full((1, 3002), alpha_trim_rad),
        state_space=["alpha", "wz", "stab", "dstab"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False, dt=DT, integrator="euler",
        control_bias=stab_trim_deg,
    ).unwrapped
    states_buf, actions_buf, next_states_buf = [], [], []
    ref_at_trim = np.full((1, 1), alpha_trim_rad)
    obs, _ = pe_env.reset()
    for t in range(3000):
        u = 2.0 * (
            0.6 * np.sin(2 * np.pi * 0.3 * t * DT)
            + 0.3 * np.sin(2 * np.pi * 0.9 * t * DT)
            + 0.1 * rng.normal()
        )
        x_curr = state_transform(obs, ref_at_trim, 0)
        obs_next, _, done, _, _ = pe_env.step(np.array([u]))
        x_next = state_transform(obs_next, ref_at_trim, 0)
        states_buf.append(x_curr)
        actions_buf.append([u])
        next_states_buf.append(x_next)
        obs = obs_next
        if done:
            break
    states_arr = np.asarray(states_buf, dtype=np.float32)
    actions_arr = np.asarray(actions_buf, dtype=np.float32)
    next_states_arr = np.asarray(next_states_buf, dtype=np.float32)

    cfg = ETDHPConfig(
        actor_hidden=(24, 24), critic_hidden=(24, 24), model_hidden=(24, 24),
        actor_lr=1e-3, critic_lr=1e-3, model_lr=5e-3,
        model_epochs=300,
        Q=[10.0, 0.1, 0.0, 0.0], R=[1.0],
        gamma=0.95,
        # More learning per trigger and lower threshold so the agent
        # adapts quickly during and after the damage event.
        num_epochs_per_trigger=10,
        u_bound=2.0,
        rho=0.1,
        trigger_floor=0.05,
        weight_init_scale=0.2, seed=0,
    )
    agent = ETDHPAgent(n_state=4, n_control=1,
                       state_transform=state_transform, config=cfg)
    print("  fitting plant NN offline ...")
    agent.fit_plant_model(
        states_arr, actions_arr, next_states_arr,
        batch_size=128, verbose=False,
    )

    # Train actor/critic for several episodes on the healthy aircraft
    print(f"  training actor/critic for {NUM_TRAIN_EPISODES} healthy episodes")
    for ep in range(NUM_TRAIN_EPISODES):
        env = _make_long_env(n_steps_per_ep, alpha_trim_rad, stab_trim_rad,
                              feedforward_fn)
        obs, _ = env.reset()
        agent.reset()
        for k in range(n_steps_per_ep - 2):
            agent.predict(obs, reference, k)
            u_cmd = agent.last_action()
            obs_next, _, done, _, _ = env.step(u_cmd)
            agent.learn(obs_next, reference, k, dt=DT)
            obs = obs_next
            if done:
                break
        print(f"  ep {ep + 1}/{NUM_TRAIN_EPISODES} done")

    return agent, alpha_trim_rad, stab_trim_rad, grid_deg, stabs_deg


# =====================================================================
#  Apply trained agent to the angular env for 3D rendering
# =====================================================================

def _angular_to_long_obs(angular_state: np.ndarray) -> np.ndarray:
    """Map the angular env state (16 elements when track_altitude=True)
    to the 4-element longitudinal observation the ET-DHP agent expects.

    Angular state indices (track_altitude=True):
        0: alpha, 1: beta, 2: wx, 3: wy, 4: wz, 5: gamma, 6: psi,
        7: theta, 8: stab, 9: dstab, 10: ail, 11: dail, 12: dir,
        13: ddir, 14: h, 15: V

    Longitudinal obs expected by agent: [alpha, wz, stab, dstab]
    """
    s = angular_state.reshape(-1)
    return np.array([s[0], s[4], s[8], s[9]], dtype=np.float64)


def main() -> None:
    n_steps = int(TOTAL_TIME / DT)

    # 1. Train ET-DHP on longitudinal env
    agent, alpha_trim_rad, stab_trim_rad, grid_deg, stabs_deg = _train_etdhp(n_steps)

    # 2. Find angular trim at V=180 m/s, h=3000 m.
    #    At this speed, the required thrust (≈9183 N) exceeds the 8000 N
    #    floor so the trim solution is physically self-consistent.
    print("\n=== Finding angular F-16 trim (V=180 m/s, h=3000 m) ===")
    trim = find_trim(V_target=ANG_V_TARGET, h_target=ANG_H_TARGET)
    print(f"  α = {math.degrees(trim.alpha_rad):+.3f}°,  "
          f"stab = {math.degrees(trim.stab_rad):+.3f}°,  "
          f"T = {trim.T_thrust:.0f} N")

    # 3. Damage profile — symmetric 30 % bilateral wing-tip loss at t=20s
    profile = DamageProfile(events=[
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

    # 4. Build the angular env at the angular trim.
    print("\n=== Closed-loop simulation in angular env (3D viewer target) ===")
    env = NonlinearAngularF16(
        initial_state=trim.x0[:14],
        number_time_steps=n_steps + 10,
        dt=DT,
        airspeed=ANG_V_TARGET,
        split_stab=True,
        damage_profile=profile,
        render_mode="3d_web",
        track_altitude=True,
        thrust_mode="constant",
    )
    env.reset()

    # The angular model auto-pads V from the default (120 m/s) when only 14
    # state elements are supplied. Override the padded V in x_history so
    # the dynamics start at the correct trim airspeed (180 m/s).
    x = env.model.x_history[-1].reshape(-1).copy()
    x[14] = ANG_H_TARGET    # h
    x[15] = ANG_V_TARGET    # V
    env.model.x_history[-1] = x

    env.model.param.T_thrust = trim.T_thrust
    env.model.param.V = ANG_V_TARGET

    # 5. Re-attach the state transform to the agent, re-centred on the
    #    angular trim α (which differs slightly from the longitudinal trim).
    angular_alpha_trim_deg = math.degrees(trim.alpha_rad)
    angular_state_transform = _build_state_transform(
        angular_alpha_trim_deg, grid_deg, stabs_deg,
    )
    agent._state_transform = angular_state_transform

    # Reset agent for the eval episode, keeping the trained weights.
    agent.reset()
    reference = np.full((1, n_steps + 10), trim.alpha_rad)

    # 6. Closed-loop pitch stabilisation via the trained ET-DHP agent.
    #    At each step:
    #      a. Extract [alpha, wz, stab, dstab] from the angular state via
    #         _angular_to_long_obs().
    #      b. Feed it to agent.predict() / agent.learn().
    #      c. Interpret agent output as a stab residual (degrees) on top of
    #         the angular trim stab, clip to the actuator limit, and feed
    #         both stabilator halves.
    stab_trim_angular_deg = math.degrees(trim.stab_rad)
    n_triggers = 0

    for k in range(n_steps):
        # Map angular state → longitudinal-shaped observation (rad)
        long_obs = _angular_to_long_obs(env.model.current_state)

        agent.predict(long_obs, reference, k)
        u_long = agent.last_action()

        # u_long is a 1-element residual elevator deflection in degrees
        # (state_transform works in degrees; agent outputs matching units).
        # Add angular trim stab back.
        stab_residual_deg = float(u_long[0])
        stab_cmd_deg = stab_trim_angular_deg + stab_residual_deg
        stab_cmd_deg = max(-25.0, min(25.0, stab_cmd_deg))

        # Both stabilator halves get the same command (symmetric pitch ctrl)
        env.step(np.array([stab_cmd_deg, stab_cmd_deg, 0.0, 0.0]))

        metrics = agent.learn(long_obs, reference, k, dt=DT)
        if metrics["triggered"]:
            n_triggers += 1

    # 7. Reference signals for the 3D viewer charts. The ET-DHP agent
    #    holds level flight at the angular trim, so each commanded
    #    channel is constant. Values are in the SAME display units as the
    #    chart panel (deg for angles, m/s for airspeed, m for altitude).
    n_traj = len(env.time_history)
    alpha_ref_deg = math.degrees(trim.alpha_rad)
    env.reference_signals = {
        "alpha":  [alpha_ref_deg] * n_traj,
        "theta":  [alpha_ref_deg] * n_traj,
        "beta":   [0.0] * n_traj,
        "roll":   [0.0] * n_traj,
        "yaw":    [0.0] * n_traj,
        "wz":     [0.0] * n_traj,
        "h":      [ANG_H_TARGET] * n_traj,
        "V":      [ANG_V_TARGET] * n_traj,
    }

    # 8. Render
    out = env.render()
    print(f"\nRendered to: {out}")
    print(f"  triggers fired during eval: {n_triggers}")
    print(f"  damage events recorded: {len(env.damage_events_log)}")

    s0 = env.model.x_history[0].reshape(-1)
    sN = env.model.current_state
    pitch_init = math.degrees(s0[7])
    pitch_final = math.degrees(sN[7])
    pitch_target = math.degrees(trim.alpha_rad)
    print(
        f"  pitch:    {pitch_init:+.2f}° → {pitch_final:+.2f}° "
        f"(target ≈ {pitch_target:+.2f}°)\n"
        f"  altitude: {s0[14]:.0f} m → {sN[14]:.0f} m\n"
        f"  airspeed: {s0[15]:.1f} m/s → {sN[15]:.1f} m/s"
    )


if __name__ == "__main__":
    main()
