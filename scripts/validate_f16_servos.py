"""Check nonlinear F-16 servos against an independent hybrid analytic solution.

The reference uses the closed-form underdamped second-order response between
rate saturation and hard-stop events, and constant velocity while rate limited.
It imports no implementation code for actuator derivatives or projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def analytic_step(state, command, dt, tau, damping, limit, rate_limit):
    """Integrate a held command exactly between physical constraint events."""
    x, v = map(float, state)
    command = float(np.clip(command, -limit, limit))
    elapsed = 0.0
    frequency = np.sqrt(1 - damping**2) / tau
    decay = damping / tau
    for _ in range(20):
        remaining = dt - elapsed
        if remaining < 1e-13:
            return np.array([x, v])
        if abs(x) >= limit - 1e-13 and x * v >= 0:
            x, v = np.copysign(limit, x), 0.0
            if abs(command - x) < 1e-13:
                return np.array([x, v])
        acceleration = (command - x - 2 * tau * damping * v) / tau**2
        if abs(v) >= rate_limit - 1e-12 and v * acceleration > 1e-10:
            v = np.copysign(rate_limit, v)
            release_position = command - 2 * tau * damping * v
            interval = min(remaining, (release_position - x) / v)
            assert interval > 0
            x += v * interval
            elapsed += interval
            continue

        def free(t):
            error = x - command
            sine, cosine = np.sin(frequency * t), np.cos(frequency * t)
            factor = np.exp(-decay * t)
            return np.array(
                [
                    command
                    + factor
                    * (error * cosine + (v + decay * error) / frequency * sine),
                    factor
                    * (v * cosine - (decay * v + error / tau**2) / frequency * sine),
                ]
            )

        grid = np.linspace(0, remaining, max(3, int(remaining / 0.0005) + 2))
        free_states = free(grid)
        crossings = []
        for axis, bound in [(0, limit), (1, rate_limit)]:
            for sign in [-1, 1]:
                values = sign * free_states[axis] - bound
                hits = np.flatnonzero((values[:-1] < -1e-13) & (values[1:] >= 0))
                if hits.size:
                    index = hits[0]
                    root = brentq(
                        lambda t: sign * free(t)[axis] - bound,
                        grid[index],
                        grid[index + 1],
                        xtol=1e-14,
                    )
                    crossings.append((root, axis, sign, bound))
        if not crossings:
            return free(remaining)
        interval, axis, sign, bound = min(crossings)
        x, v = free(interval)
        if axis == 0:
            x, v = sign * bound, 0.0
        else:
            v = sign * bound
        elapsed += interval
    raise RuntimeError("Too many servo constraint events")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
    from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal import (
        LongitudinalF16,
    )

    result = {"repo": str(args.repo.resolve()), "cases": []}
    configs = [
        ("longitudinal", LongitudinalF16, 4, 0, 2, "stab"),
        ("angular", AngularF16, 14, 0, 8, "stab"),
        ("angular", AngularF16, 14, 1, 10, "ail"),
        ("angular", AngularF16, 14, 2, 12, "dir"),
    ]
    for kind, cls, size, channel, index, name in configs:
        for dt in [0.02, 0.01, 0.005, 0.0025]:
            model = cls(np.zeros(size), dt=dt, integrator="rk4")
            p = model.param
            limit, rate_limit = getattr(p, "maxabs" + name), getattr(
                p, "maxabsd" + name
            )
            reference = np.zeros(2)
            actual, expected = [reference.copy()], [reference.copy()]
            for step in range(round(1.8 / dt)):
                command = np.zeros(model.action_space_length)
                command[channel] = limit if step < round(0.8 / dt) else -limit
                reference = analytic_step(
                    reference,
                    command[channel],
                    dt,
                    getattr(p, "T" + name),
                    getattr(p, "Xi" + name),
                    limit,
                    rate_limit,
                )
                model.run_step(command)
                actual.append(model.current_state[index : index + 2])
                expected.append(reference)
            actual, expected = np.array(actual), np.array(expected)
            assert np.isfinite(actual).all()
            result["cases"].append(
                {
                    "model": kind,
                    "surface": name,
                    "dt": dt,
                    "max_angle_deg": float(np.rad2deg(abs(actual[:, 0])).max()),
                    "max_rate_deg_s": float(np.rad2deg(abs(actual[:, 1])).max()),
                    "max_step_rate_deg_s": float(
                        np.rad2deg(abs(np.diff(actual[:, 0])) / dt).max()
                    ),
                    "angle_limit_deg": float(np.rad2deg(limit)),
                    "rate_limit_deg_s": float(np.rad2deg(rate_limit)),
                    "max_angle_error_deg": float(
                        np.rad2deg(abs(actual[:, 0] - expected[:, 0])).max()
                    ),
                    "max_rate_error_deg_s": float(
                        np.rad2deg(abs(actual[:, 1] - expected[:, 1])).max()
                    ),
                    "actual": actual.tolist(),
                    "reference": expected.tolist(),
                }
            )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in case.items() if k not in ["actual", "reference"]}
                for case in result["cases"]
            ]
        )
    )


if __name__ == "__main__":
    main()
