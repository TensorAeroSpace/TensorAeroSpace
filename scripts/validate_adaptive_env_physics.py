"""Replay fixed actions through native environments and cross-check F-16 RK4.

F-16 reference integration shares the aerodynamic RHS, but uses SciPy DOP853.
This verifies integration and environment wiring, not aerodynamic identification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import root
    from tensoraerospace.envs.b747 import LinearLongitudinalB747
    from tensoraerospace.envs.lapan import LinearLongitudinalLAPAN
    from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.params import (
        default_parameters,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.dynamics import (
        f16_ode_long,
    )

    p = default_parameters()

    def residual(z):
        return f16_ode_long(np.array([z[0], 0, z[1], 0]), [z[1]], 0, p)[:2]

    trim = root(residual, np.deg2rad([2, -2]))
    assert trim.success and np.max(abs(trim.fun)) < 1e-10
    a, u0 = trim.x
    result = dict(
        repo=str(args.repo.resolve()), f16_trim=[float(a), float(u0)], cases=[]
    )
    for plant in ["b747", "lapan", "f16"]:
        for dt in [0.02, 0.01, 0.005]:
            n = round(2 / dt)
            if plant == "f16":
                initial = np.array([a, 0, u0, 0])
                env = NonlinearLongitudinalF16(
                    initial,
                    np.full((1, n + 1), a),
                    n + 1,
                    state_space=["alpha", "wz", "stab", "dstab"],
                    tracking_states=["alpha"],
                    control_space=["stab"],
                    use_reward=False,
                    dt=dt,
                    integrator="rk4",
                    control_bias=float(np.rad2deg(u0)),
                )
            else:
                cls = (
                    LinearLongitudinalB747
                    if plant == "b747"
                    else LinearLongitudinalLAPAN
                )
                initial = np.zeros(4)
                env = cls(
                    initial,
                    np.zeros((1, n + 1)),
                    n + 1,
                    dt=dt,
                    tracking_states=["q"],
                    output_space=["q", "theta"],
                )
            env.reset()
            expected = initial.copy()
            trace, errors = [], []
            for k in range(n):
                command = 0.1 * np.sin(2 * np.pi * 0.7 * k * dt)
                obs, reward, terminated, truncated, _ = env.step(np.array([command]))
                if plant == "f16":
                    actual = env.model.current_state.copy()
                    reference = solve_ivp(
                        lambda t, x: f16_ode_long(x, [u0 + np.deg2rad(command)], t, p),
                        (0, dt),
                        expected,
                        method="DOP853",
                        rtol=1e-11,
                        atol=1e-12,
                    )
                    assert reference.success
                    expected = reference.y[:, -1]
                    errors.append(abs(actual - expected))
                    assert abs(actual[2]) <= p.maxabsstab + 1e-12
                    assert abs(actual[3]) <= p.maxabsdstab + 1e-12
                else:
                    actual = env.model.xt.reshape(-1).copy()
                assert np.isfinite(actual).all()
                trace.append(
                    [
                        *actual.tolist(),
                        *np.asarray(obs).reshape(-1).tolist(),
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                    ]
                )
            result["cases"].append(
                dict(
                    plant=plant,
                    dt=dt,
                    steps=n,
                    trace=trace,
                    max_DOP853_error=float(np.max(errors)) if errors else None,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in c.items() if k != "trace"} for c in result["cases"]],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
