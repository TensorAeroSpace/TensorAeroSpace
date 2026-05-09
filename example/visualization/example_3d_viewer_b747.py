"""3D WebGL viewer demo for the nonlinear Boeing 747 model.

Run with::

    poetry run python example/visualization/example_3d_viewer_b747.py

The script flies a trimmed B-747 cruise condition, triggers a left-outer
engine flameout, and writes a self-contained HTML viewer.
"""

from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.b747.nonlinear import B747Configuration, trim
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile,
    EngineFailureEvent,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env
from tensoraerospace.visualization.three_d import render

DT = 0.05
TOTAL_TIME = 80.0
DAMAGE_TIME = 20.0
ALTITUDE_FT = 20_000.0
AIRSPEED_FT_S = 674.0


def main() -> None:
    trim_result = trim(
        altitude_ft=ALTITUDE_FT,
        V_ft_s=AIRSPEED_FT_S,
        config=B747Configuration.NOMINAL,
    )
    if not trim_result.converged:
        raise RuntimeError(f"B-747 trim failed: residual={trim_result.residual:.3e}")

    env = NonlinearB747Env(
        trim_at=(ALTITUDE_FT, AIRSPEED_FT_S),
        number_time_steps=int(TOTAL_TIME / DT),
        dt=DT,
        integrator="rk4",
        render_mode="3d_web",
        damage_profile=DamageProfile(
            events=[
                EngineFailureEvent(
                    trigger_time=DAMAGE_TIME,
                    engine_id=1,
                    thrust_fraction=0.0,
                    label="left_outer_engine_flameout",
                )
            ]
        ),
    )
    env.reset()
    action = np.array(
        [trim_result.elevator_rad, 0.0, 0.0, trim_result.throttle],
        dtype=np.float64,
    )
    for _ in range(env.number_time_steps):
        _, _, _, truncated, _ = env.step(action)
        if truncated:
            break

    # Publish reference signals for the 3D viewer's chart overlays — the
    # same interface the F-16 demo uses. Holds trim values throughout.
    n_pub = int(env.number_time_steps) + 10
    env.reference_signal = np.array(
        [
            np.full(n_pub, ALTITUDE_FT * 0.3048),  # h target (m)
            np.full(n_pub, AIRSPEED_FT_S * 0.3048),  # V target (m/s)
        ],
        dtype=np.float64,
    )
    env.tracking_states = ["h", "V"]

    out = render(
        env,
        open_in_browser=False,
        inline=False,
        save_to="example/visualization/b747_3d_viewer.html",
        title="B-747 3D flight viewer",
    )
    print(out)


if __name__ == "__main__":
    main()
