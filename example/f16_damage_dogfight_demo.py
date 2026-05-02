"""Demo: F-16 with left-wingtip damage at t=10s.

Simulates 20 seconds of level flight; at t=10s the left wingtip is
fully lost. Without active stabilisation, a roll moment develops and
the aircraft trajectory diverges from healthy baseline.

Run with:
    poetry run python example/f16_damage_dogfight_demo.py
"""

from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


def main() -> None:
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=2000,
        dt=0.01,
        airspeed=200.0,
        damage_profile=WING_STRIKE_LEFT_TIP,
        split_stab=True,
    )
    obs, _ = env.reset()
    wx_history: list[float] = [float(obs[2])]
    for k in range(2000):
        obs, _, _, _, info = env.step(np.zeros(4))
        wx_history.append(float(obs[2]))
        triggered = info.get("damage_events_triggered")
        if triggered:
            print(f"t={k*0.01:.2f}s: {triggered}")
    print(f"Final wx (roll rate, rad/s): {wx_history[-1]:.4f}")
    print(f"Max |wx|: {max(abs(w) for w in wx_history):.4f}")


if __name__ == "__main__":
    main()
