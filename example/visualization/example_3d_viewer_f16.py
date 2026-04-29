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

    profile = DamageProfile(events=[
        DamageEvent(
            trigger_time=DAMAGE_TIME, event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 1.0},
            label="left_tip_total_loss",
        ),
    ])

    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps + 10,
        dt=DT,
        airspeed=200.0,
        split_stab=True,
        damage_profile=profile,
        render_mode="3d_web",
    )
    env.reset()
    for _ in range(n_steps):
        # Zero stick command — let the dynamics tell the story.
        env.step(np.zeros(4))

    # In a script: opens the rendered HTML in the default browser.
    # In Jupyter: returns IPython.display.HTML for inline display.
    out = env.render()
    print(f"Rendered to: {out}")
    print(
        f"  damage events: {len(env.damage_events_log)}\n"
        f"  damage snapshots: {len(env.damage_state_log)}"
    )


if __name__ == "__main__":
    main()
