"""Example: 3D WebGL flight viewer for the nonlinear F-16 with damage.

Runs a 60-second episode of NonlinearAngularF16 with a left-wingtip
damage event at t=20s, then opens an interactive 3D viewer in the
default browser. Mouse rotates the camera (orbit), wheel zooms; the
timeline at the bottom scrubs through the flight; camera presets at
the bottom switch between Free / Chase / Top-down views.

Run with::

    poetry run python example/visualization/example_3d_viewer_f16.py
"""

from __future__ import annotations

import math

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageProfile,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

DT = 0.01
TOTAL_TIME = 60.0
DAMAGE_TIME = 20.0


def main() -> None:
    n_steps = int(TOTAL_TIME / DT)

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim
    trim = find_trim(V_target=120.0, h_target=3000.0)
    print(f"Trim:  α = {math.degrees(trim.alpha_rad):+.3f}°,  "
          f"stab = {math.degrees(trim.stab_rad):+.3f}°,  "
          f"T = {trim.T_thrust:.0f} N  (residuals: "
          f"{', '.join(f'{r:.2e}' for r in trim.residuals)})")

    ALPHA_TRIM_DEG = math.degrees(trim.alpha_rad)
    STAB_TRIM_DEG = math.degrees(trim.stab_rad)

    profile = DamageProfile(events=[
        DamageEvent(
            trigger_time=DAMAGE_TIME, event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 1.0},
            label="left_tip_total_loss",
        ),
    ])

    # Start at the computed trim point so level cruise actually holds
    # (lift = weight, thrust = drag, pitch moment = 0).
    # NonlinearAngularF16 expects a 14-element initial state; track_altitude=True
    # will auto-pad with the default altitude (Oy=3000 m) and airspeed (V=120 m/s),
    # which match the trim targets.
    x0 = trim.x0[:14]   # first 14 elements; h and V set via model params after reset

    env = NonlinearAngularF16(
        initial_state=x0,
        number_time_steps=n_steps + 10,
        dt=DT,
        airspeed=120.0,                    # match params.V trim
        split_stab=True,
        damage_profile=profile,
        render_mode="3d_web",
        # Energy-consistent altitude + airspeed dynamics. Aircraft now
        # actually climbs / descends as the physics integrates dh/dt and
        # dV/dt; HUD altimeter reads the simulation's altitude (not the
        # kinematic position). Thrust is constant from params.T_thrust.
        track_altitude=True,
        thrust_mode="constant",
    )
    env.reset()
    env.model.param.T_thrust = trim.T_thrust

    # Hold trim stab; for split_stab=True both halves get the same
    # commanded deflection (no roll). At t=20 s left_tip is lost — the
    # asymmetric loss then induces a roll the agent isn't there to
    # cancel, so the post-event trajectory dives + rolls.
    trim_cmd = np.array([STAB_TRIM_DEG, STAB_TRIM_DEG, 0.0, 0.0])
    for _ in range(n_steps):
        env.step(trim_cmd)

    # In a script: opens the rendered HTML in the default browser.
    # In Jupyter: returns IPython.display.HTML for inline display.
    out = env.render()
    print(f"Rendered to: {out}")
    print(
        f"  damage events: {len(env.damage_events_log)}\n"
        f"  damage snapshots: {len(env.damage_state_log)}"
    )
    # Show the altitude / airspeed evolution since it's the new feature
    # this script demonstrates.
    s0 = env.model.x_history[0].reshape(-1)
    sN = env.model.current_state
    print(
        f"  altitude: {s0[14]:.0f} m → {sN[14]:.0f} m "
        f"(Δ {sN[14] - s0[14]:+.0f} m)\n"
        f"  airspeed: {s0[15]:.1f} m/s → {sN[15]:.1f} m/s "
        f"(Δ {sN[15] - s0[15]:+.1f} m/s)"
    )


if __name__ == "__main__":
    main()
