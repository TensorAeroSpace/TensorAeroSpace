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

# F-16 angular trim point at V=120 m/s, h=3000 m
ALPHA_TRIM_DEG = 5.0
STAB_TRIM_DEG = -4.45


def main() -> None:
    n_steps = int(TOTAL_TIME / DT)

    profile = DamageProfile(events=[
        DamageEvent(
            trigger_time=DAMAGE_TIME, event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 1.0},
            label="left_tip_total_loss",
        ),
    ])

    # Start at the trim point so level cruise holds without input.
    # γ_path = θ − α; for level flight γ_path = 0, so θ = α at trim.
    x0 = np.zeros(14)
    x0[0] = math.radians(ALPHA_TRIM_DEG)   # alpha = 5°
    x0[7] = math.radians(ALPHA_TRIM_DEG)   # theta = 5° (pitch up to keep γ=0)
    x0[8] = math.radians(STAB_TRIM_DEG)    # stab  = -4.45°

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
    # Without an active trim controller, the angular model isn't in true
    # equilibrium even from this state — lift/thrust/drag don't perfectly
    # balance. The aircraft descends gradually as the integrator runs
    # the real altitude / airspeed dynamics. The point of this demo is
    # exactly to show that the HUD altimeter / KIAS now read live physics
    # values rather than constants — so the descent is feature, not bug.
    #
    # Bump T_thrust modestly so drag wins less aggressively in the dive.
    env.model.param.T_thrust = 12000.0

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
