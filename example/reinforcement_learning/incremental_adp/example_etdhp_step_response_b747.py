"""Reproducible commanded-step tests for the nonlinear B747 ET-DHP/PID example.

Each trial continues a common, independently warmed-up controller/plant state.
Only setpoints change: neither the fault schedule nor a reset is sent to a
controller. ET-DHP retains its native predict/learn/held-command timing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_vs_pid_b747 as demo,
)

CHANNELS = ("heading_deg", "roll_deg", "speed_m_s", "height_m")
NOMINAL_REFERENCE = np.array([0.0, 0.0, demo.SPEED * 0.3048, demo.ALTITUDE * 0.3048])
STEP_AMPLITUDES = np.array([1.0, 0.5, 1.0, 10.0])
TRACE_COLUMNS = (
    "time_s",
    "flight_time_s",
    *CHANNELS,
    *("ref_" + name for name in CHANNELS),
    "beta_deg",
    "aileron_deg",
    "rudder_deg",
    "elevator_deg",
    "throttle",
    "triggered",
)


@dataclass(frozen=True)
class Protocol:
    warmup_s: float = 1000.0
    step_time_s: float = 20.0
    response_s: float = 500.0
    dt: float = 0.05
    fault_time_s: float = 137.0
    tail_s: float = 50.0
    confirmation_s: float = 30.0
    seed: int = 11

    def __post_init__(self):
        values = (
            self.warmup_s,
            self.step_time_s,
            self.response_s,
            self.dt,
            self.fault_time_s,
            self.tail_s,
            self.confirmation_s,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Protocol times must be finite and positive")
        if self.fault_time_s >= self.warmup_s:
            raise ValueError("The fault must occur during warmup")
        if max(self.tail_s, self.confirmation_s) > self.response_s:
            raise ValueError("Tail/confirmation must fit within response")
        for value in values[:5]:
            if not np.isclose(
                value / self.dt, round(value / self.dt), rtol=0, atol=1e-8
            ):
                raise ValueError("Protocol times must align with control steps")

    @property
    def trial_s(self):
        return self.step_time_s + self.response_s


def step_metrics(
    time,
    output,
    *,
    step_time,
    initial_reference,
    final_reference,
    tail_seconds=50.0,
    confirmation_seconds=30.0,
):
    """Metrics relative to the *command*, not the observed terminal value.

    Rise thresholds use the commanded amplitude (10–90%); settling bands are
    2%/5% of it around the final command. Settling must persist to the end, with
    at least ``confirmation_seconds`` of observations. None means unconfirmed.
    Tail bias is a finite-horizon residual estimate, not an asymptotic proof.
    Integrals use trapezoidal quadrature, including the exact step boundary.
    """
    t, y = np.asarray(time, dtype=float), np.asarray(output, dtype=float)
    if (
        t.ndim != 1
        or y.shape != t.shape
        or len(t) < 3
        or not np.isfinite(t).all()
        or not np.isfinite(y).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Finite, matching vectors with strictly increasing time required"
        )
    constants = [
        step_time,
        initial_reference,
        final_reference,
        tail_seconds,
        confirmation_seconds,
    ]
    if (
        not np.isfinite(constants).all()
        or tail_seconds <= 0
        or confirmation_seconds <= 0
    ):
        raise ValueError("Finite references and positive metric windows required")
    delta = final_reference - initial_reference
    if delta == 0 or not t[0] < step_time < t[-1]:
        raise ValueError("A nonzero step with pre/post observations is required")
    if max(tail_seconds, confirmation_seconds) > t[-1] - step_time:
        raise ValueError("Metric windows exceed the available response")
    after = t > step_time
    tau = np.r_[0.0, t[after] - step_time]
    response = np.r_[np.interp(step_time, t, y), y[after]]
    normalized = (response - initial_reference) / delta
    error = response - final_reference

    def crossing(level):
        reached = np.flatnonzero(normalized >= level)
        if not len(reached):
            return None
        i = reached[0]
        if i == 0:
            return 0.0
        fraction = (level - normalized[i - 1]) / (normalized[i] - normalized[i - 1])
        return float(tau[i - 1] + fraction * (tau[i] - tau[i - 1]))

    def settling(fraction):
        outside = np.flatnonzero(np.abs(error) > fraction * abs(delta))
        i = outside[-1] + 1 if len(outside) else 0
        if i == len(tau) or tau[-1] - tau[i] < confirmation_seconds - 1e-9:
            return None
        return float(tau[i])

    t10, t90 = crossing(0.1), crossing(0.9)
    tail = tau >= tau[-1] - tail_seconds
    before = (t < step_time) & (t >= max(t[0], step_time - 10))
    peak = int(np.argmax(normalized))
    return {
        "initial_reference": float(initial_reference),
        "final_reference": float(final_reference),
        "amplitude": float(delta),
        "observed_response_s": float(tau[-1]),
        "delay_10_s": t10,
        "time_90_s": t90,
        "rise_10_90_s": None if t90 is None else float(t90 - t10),
        "settling_2pct_s": settling(0.02),
        "settling_5pct_s": settling(0.05),
        "overshoot_pct": float(max(0.0, normalized[peak] - 1) * 100),
        "undershoot_pct": float(max(0.0, -np.min(normalized)) * 100),
        "peak_value": float(response[peak]),
        "peak_time_s": float(tau[peak]),
        "tail_bias": float(error[tail].mean()),
        "tail_bias_pct": float(error[tail].mean() / abs(delta) * 100),
        "tail_std": float(error[tail].std()),
        "tail_peak_to_peak": float(np.ptp(error[tail])),
        "tail_drift_per_s": float(
            np.polyfit(tau[tail] - tau[tail][0], error[tail], 1)[0]
        ),
        "prestep_bias": float(np.mean(y[before] - initial_reference)),
        "prestep_std": float(np.std(y[before])),
        "rmse": float(np.sqrt(np.trapz(error**2, tau) / tau[-1])),
        "iae": float(np.trapz(np.abs(error), tau)),
        "ise": float(np.trapz(error**2, tau)),
        "itae": float(np.trapz(tau * np.abs(error), tau)),
    }


def reference_at(time_s, channel, protocol):
    ref = NOMINAL_REFERENCE.copy()
    if channel is not None and time_s >= protocol.step_time_s - 1e-9:
        ref[channel] += STEP_AMPLITUDES[channel]
    return ref


class TrialState:
    """Complete continuation state, cloned before branching step experiments."""

    def __init__(self, protocol, algorithm, *, fault, template):
        self.protocol, self.algorithm = protocol, algorithm
        if algorithm not in ("etdhp", "pid"):
            raise ValueError("Unknown algorithm")
        self.cfg = demo.Experiment(
            duration=protocol.warmup_s + protocol.trial_s,
            dt=protocol.dt,
            fault_time=protocol.fault_time_s,
            seed=protocol.seed,
        )
        self.env = demo.make_env(self.cfg, fault=fault)
        self.obs, _ = self.env.reset(seed=protocol.seed)
        self.lon = demo.LongitudinalHold(protocol.dt)
        self.pid = demo.LateralPID(demo.NOMINAL_PID_GAINS, protocol.dt)
        self.agent = copy.deepcopy(template) if algorithm == "etdhp" else None
        if self.agent is not None:
            self.agent.reset()
        self.integrals = np.zeros(2)
        self.step = 0
        self.triggers = 0
        self.action = np.zeros(2)
        self.elevator, self.throttle = (
            demo.nominal_trim().elevator_rad,
            demo.nominal_trim().throttle,
        )
        self.triggered = 0

    def advance(self, ref):
        heading, roll, speed, height = ref
        kwargs = dict(roll_ref_deg=roll, heading_ref_deg=heading)
        x = demo.lateral_state(self.obs, self.integrals, **kwargs)
        self.action = (
            self.agent.predict(x, time_step=self.step)
            if self.agent is not None
            else self.pid.command(self.obs, **kwargs)
        )
        self.elevator, self.throttle = self.lon.command(
            self.obs,
            speed_ref_ft_s=speed / 0.3048,
            height_ref_ft=height / 0.3048,
        )
        control = np.r_[self.elevator, np.deg2rad(self.action), self.throttle]
        self.obs, _, terminated, truncated, _ = self.env.step(control)
        angles = demo.lateral_state(self.obs, **kwargs)[3:5]
        self.integrals = np.clip(self.integrals + self.protocol.dt * angles, -100, 100)
        learning = (
            self.agent.learn(
                demo.lateral_state(self.obs, self.integrals, **kwargs),
                time_step=self.step,
                dt=self.protocol.dt,
            )
            if self.agent is not None
            else {"triggered": 0}
        )
        self.triggered = int(learning["triggered"])
        self.triggers += self.triggered
        self.step += 1
        measured = demo.lateral_state(self.obs)
        if (
            not np.isfinite(self.obs).all()
            or not np.isfinite(self.action).all()
            or abs(measured[3]) > 30
            or abs(measured[4]) > 45
            or abs(measured[0]) > 15
            or abs(np.linalg.norm(self.obs[:3]) - demo.SPEED) > 100
            or abs(-self.obs[11] - demo.ALTITUDE) > 1500
        ):
            raise RuntimeError("Nonfinite state or numerical flight-envelope guard")
        if terminated or (truncated and self.step < self.cfg.steps):
            raise RuntimeError("Unexpected early end of the environment")

    def record(self, local_time, ref):
        measured = demo.lateral_state(self.obs)
        return [
            local_time,
            self.step * self.protocol.dt,
            measured[4],
            measured[3],
            np.linalg.norm(self.obs[:3]) * 0.3048,
            -self.obs[11] * 0.3048,
            *ref,
            measured[0],
            *self.action,
            np.rad2deg(self.elevator),
            self.throttle,
            self.triggered,
        ]

    def weights(self):
        return (
            [
                demo.parameter_vector(net)
                for net in (self.agent.actor, self.agent.critic, self.agent.plant_model)
            ]
            if self.agent is not None
            else []
        )


def source_digest():
    root = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        Path(demo.__file__),
        root / "tensoraerospace/envs/b747_nonlinear.py",
    ]
    for directory in ("agent/et_dhp", "aerospacemodel/b747/nonlinear"):
        files += sorted((root / "tensoraerospace" / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_suite(output_dir, protocol=Protocol()):
    """Run four warmups and 20 continuations (16 steps + four no-step controls)."""
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template, initialization = demo.make_agent(dt=protocol.dt, seed=protocol.seed)
    result = {
        "protocol": asdict(protocol),
        "source_digest": source_digest(),
        "trace_columns": TRACE_COLUMNS,
        "channels": CHANNELS,
        "amplitudes": STEP_AMPLITUDES.tolist(),
        "initialization": initialization,
        "runs": {},
        "warmups": {},
    }
    for fault in (False, True):
        for algorithm in ("etdhp", "pid"):
            arm = ("fault" if fault else "healthy") + "_" + algorithm
            print(f"Warmup {arm}: {protocol.warmup_s:g} s", flush=True)
            state = TrialState(protocol, algorithm, fault=fault, template=template)
            for k in range(round(protocol.warmup_s / protocol.dt)):
                state.advance(NOMINAL_REFERENCE)
                if (k + 1) % round(250 / protocol.dt) == 0:
                    print(f"  {arm}: {(k+1)*protocol.dt:g} s", flush=True)
            result["warmups"][arm] = {
                "events": state.env.damage_events_log,
                "triggers": state.triggers,
                "last_row": state.record(0, NOMINAL_REFERENCE),
            }
            baseline = None
            for channel in (None, 0, 1, 2, 3):
                name = (
                    arm + "_" + ("baseline" if channel is None else CHANNELS[channel])
                )
                run = copy.deepcopy(state)
                start_weights = run.weights()
                start_triggers = run.triggers
                rows = [run.record(0, reference_at(0, channel, protocol))]
                for k in range(round(protocol.trial_s / protocol.dt)):
                    run.advance(reference_at(k * protocol.dt, channel, protocol))
                    local_time = (k + 1) * protocol.dt
                    rows.append(
                        run.record(
                            local_time, reference_at(local_time, channel, protocol)
                        )
                    )
                trace = np.asarray(rows, dtype=float)
                if channel is None:
                    baseline = trace
                else:
                    # Reference at the boundary has changed, physical history has not.
                    prefix = trace[:, 0] <= protocol.step_time_s
                    np.testing.assert_array_equal(
                        trace[prefix, :6], baseline[prefix, :6]
                    )
                    np.testing.assert_array_equal(
                        trace[prefix, 10:], baseline[prefix, 10:]
                    )
                data = {
                    "status": "complete",
                    "algorithm": algorithm,
                    "condition": "fault" if fault else "healthy",
                    "channel": channel,
                    "events": run.env.damage_events_log,
                    "triggers_during_trial": run.triggers - start_triggers,
                    "weight_change_during_trial": [
                        float(np.linalg.norm(a - b))
                        for a, b in zip(run.weights(), start_weights)
                    ],
                }
                if channel is not None:
                    data["metrics"] = step_metrics(
                        trace[:, 0],
                        trace[:, 2 + channel],
                        step_time=protocol.step_time_s,
                        initial_reference=NOMINAL_REFERENCE[channel],
                        final_reference=NOMINAL_REFERENCE[channel]
                        + STEP_AMPLITUDES[channel],
                        tail_seconds=protocol.tail_s,
                        confirmation_seconds=protocol.confirmation_s,
                    )
                post = trace[:, 0] >= protocol.step_time_s
                err = trace[post, 2:6] - trace[post, 6:10]
                # Scales: 1 degree, 1 degree, 1 m/s, 1 m (as in the comparison notebook).
                data["all_channel_iae"] = np.trapz(
                    np.abs(err), trace[post, 0], axis=0
                ).tolist()
                data["all_channel_rmse"] = np.sqrt(np.mean(err**2, axis=0)).tolist()
                data["normalized_total_iae"] = float(sum(data["all_channel_iae"]))
                np.savetxt(
                    output_dir / f"{name}.csv",
                    trace,
                    delimiter=",",
                    header=",".join(TRACE_COLUMNS),
                    comments="",
                )
                result["runs"][name] = data
                print(name, data.get("metrics", {}).get("settling_2pct_s"), flush=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/b747-etdhp-vs-pid/step_response")
    )
    args = parser.parse_args()
    run_suite(args.output)
