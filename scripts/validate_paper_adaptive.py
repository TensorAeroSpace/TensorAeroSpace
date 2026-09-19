"""Reproducible paper-architecture checks on known physical/algebraic plants.

This is not a reproduction of Flying-V or PH-LAB flight-test data. Faults are
injected into the plant/sensors only; neither controller receives fault times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensoraerospace.agent.aa_indi import (
    AAINDIAgent,
    AAINDIConfig,
    AircraftGeometry,
    FlightMeasurement,
    ObserverConfig,
)
from tensoraerospace.agent.aa_indi.kinematics import body_to_ned, rk4
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig


def iadp_trial(duration=80.0, dt=0.001, record_trace=False):
    time = np.arange(round(duration / dt)) * dt
    initial_time = np.arange(round(20 / dt)) * dt
    excitation = (
        0.15 * np.sin(2 * np.pi * 0.7 * initial_time)
        + 0.05 * np.sin(2 * np.pi * 1.7 * initial_time)
    )[:, None]
    persistent_time = np.arange(round(10 / dt)) * dt
    persistent = (0.015 * np.sin(2 * np.pi * 0.7 * persistent_time))[:, None]
    config = IADPConfig.paper(
        excitation_signal=excitation,
        dt=dt,
        Q=np.array([[100.0]]),
        R=np.array([[0.0001]]),
        gamma=0.95,
        gamma_rls=0.999,
        phi_init=1e6,
        u_magnitude_limit=0.5,
        u_rate_limit=2.0,
        continuous_excitation_signal=persistent,
    )
    agent = IADPAgent(1, 1, config)
    state = np.zeros(1)
    states = []
    trace = []
    a = np.exp(-2 * dt)
    b = (1 - a) / 2
    for k, t in enumerate(time):
        reference = np.array([0.05])
        action = agent.predict(state, reference, k)
        gain = 1.0 if t < 60 else 0.7
        state = a * state + b * gain * action
        agent.learn(state, reference, k, applied_action=action)
        if not np.isfinite(state).all() or not np.isfinite(agent.P).all():
            raise FloatingPointError("iADP produced nonfinite state/critic")
        states.append(state[0])
        if record_trace and k % 100 == 0:
            trace.append(
                [
                    float((k + 1) * dt),
                    float(reference[0]),
                    float(state[0]),
                    float(action[0]),
                    float(agent.G[0, 0]),
                    b * gain,
                ]
            )
    states = np.array(states)

    def rmse(start, stop):
        selected = (time >= start) & (time < stop)
        return (
            float(np.sqrt(np.mean((states[selected] - 0.05) ** 2)))
            if selected.any()
            else None
        )

    return dict(
        plant="exact discretization of rate_dot=-2*rate+gain*input",
        duration=duration,
        dt=dt,
        nominal_tracking_rmse=rmse(40, 60),
        faulty_tracking_rmse=rmse(65, duration),
        max_abs_state=float(np.max(np.abs(states))),
        final_G=float(agent.G[0, 0]),
        true_final_G=b * (0.7 if duration > 60 else 1.0),
        critic_min_eigenvalue=float(np.linalg.eigvalsh(agent.P).min()),
        final_rls_updates=agent.rls.num_updates,
        critic_window=len(agent._window),
        **(
            {
                "trace": trace,
                "trace_columns": [
                    "time_s",
                    "reference",
                    "state",
                    "action",
                    "estimated_G",
                    "true_G",
                ],
            }
            if record_trace
            else {}
        ),
    )


def aaindi_trial(
    duration=60.0,
    dt=0.01,
    seed=17,
    sensor_fault=True,
    correct_sensor_faults=True,
    rate_limit_deg=180.0,
    cutoff_hz=10.0,
    rate_feedback=4.0,
    record_trace=False,
):
    rng = np.random.default_rng(seed)
    geometry = AircraftGeometry(np.diag([100.0, 150.0, 200.0]), 10.0, 8.0, 1.5)
    density, airspeed = 1.2, 40.0
    desired_G = np.diag([4.0, 3.0, 2.0])
    derivatives = (geometry.inertia @ desired_G) / geometry.moment_scale(
        density, airspeed
    )[:, None]
    config = AAINDIConfig(
        geometry,
        derivatives,
        observer=ObserverConfig(dt=dt),
        enable_sensor_correction=correct_sensor_faults,
        rate_limit=np.deg2rad(rate_limit_deg),
        acceleration_cutoff_hz=cutoff_hz,
        rate_feedback=np.full(3, rate_feedback),
    )
    agent = AAINDIAgent(config)
    plant = np.zeros(6)  # body rates, Euler attitude; no hidden aircraft model.
    applied = np.zeros(3)
    errors = []
    fault_errors = []
    attitudes = []
    commands = []
    last_metrics = {}
    trace = []
    failure = None
    time = np.arange(round(duration / dt) + 1) * dt

    def measurement(t):
        true_fault = np.zeros(6)
        if sensor_fault and t >= 20:
            true_fault[4] = 0.02
        imu = np.r_[
            body_to_ned(plant[3:]).T @ np.array([0.0, 0.0, -9.80665]), plant[:3]
        ]
        imu += true_fault + rng.normal(size=6) * config.observer.imu_std
        # Independent navigation; GPS is available at 10 Hz.
        velocity = (
            np.array([airspeed, 0.0, 0.0]) + rng.normal(scale=0.03, size=3)
            if round(t / dt) % 10 == 0
            else None
        )
        attitude = plant[3:] + rng.normal(size=3) * config.observer.navigation_std[3:]
        return (
            FlightMeasurement(
                t, imu[3:], imu[:3], velocity, attitude, applied, airspeed, density
            ),
            true_fault,
        )

    sample, true_fault = measurement(0.0)
    for k, t in enumerate(time[:-1]):
        reference = 0.04 * np.sin(2 * np.pi * np.array([0.1, 0.13, 0.17]) * t)
        command = agent.predict(sample, reference, k)
        # Separate-surface doublets, scheduled independently of the fault.
        # The controller sees their actual surfaces through actuator feedback.
        if t < 6:
            axis = int(t // 2)
            local = t % 2
            command[axis] += 0.025 if local < 0.4 else (-0.025 if local < 0.8 else 0.0)
        command = np.clip(
            command, applied - config.rate_limit * dt, applied + config.rate_limit * dt
        )
        command = np.clip(command, -config.magnitude_limit, config.magnitude_limit)
        applied = command.copy()
        effective = derivatives.copy()
        if t >= 20:
            effective[:, 1] *= 0.7
        moment = geometry.moment_scale(density, airspeed) * (effective @ applied)

        def dynamics(x):
            omega = x[:3]
            p, q, r = omega
            phi, theta, _ = x[3:]
            rates = np.array(
                [
                    p
                    + q * np.sin(phi) * np.tan(theta)
                    + r * np.cos(phi) * np.tan(theta),
                    q * np.cos(phi) - r * np.sin(phi),
                    (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta),
                ]
            )
            return np.r_[
                np.linalg.solve(
                    geometry.inertia, moment - np.cross(omega, geometry.inertia @ omega)
                ),
                rates,
            ]

        plant = rk4(dynamics, plant, dt)
        sample, true_fault = measurement(time[k + 1])
        try:
            last_metrics = agent.learn(sample, applied_action=applied)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure = {"time": float(sample.time), "reason": str(exc)}
            break
        if not np.isfinite(plant).all():
            raise FloatingPointError("AA-INDI plant diverged")
        errors.append(plant[:3] - reference)
        fault_errors.append(agent.observer.faults - true_fault)
        attitudes.append(plant[3:].copy())
        commands.append(command.copy())
        if record_trace and k % 10 == 0:
            trace.append(
                np.r_[
                    sample.time,
                    reference,
                    plant[:3],
                    command,
                    agent.observer.faults[4],
                    true_fault[4],
                ].tolist()
            )
    errors = np.asarray(errors)
    fault_errors = np.asarray(fault_errors)
    selected = time[1 : len(errors) + 1] >= min(30.0, duration * 0.75)
    return dict(
        plant="rigid-body Euler moment dynamics with independent navigation",
        duration=duration,
        dt=dt,
        seed=seed,
        sensor_fault=sensor_fault,
        completed_steps=len(errors),
        failure=failure,
        rate_limit_deg=rate_limit_deg,
        cutoff_hz=cutoff_hz,
        rate_feedback=rate_feedback,
        correct_sensor_faults=correct_sensor_faults,
        tracking_rmse_after_transient=(
            np.sqrt(np.mean(errors[selected] ** 2, axis=0)).tolist()
            if selected.any()
            else None
        ),
        fault_rmse_after_transient=(
            np.sqrt(np.mean(fault_errors[selected] ** 2, axis=0)).tolist()
            if selected.any()
            else None
        ),
        max_abs_attitude=np.max(np.abs(attitudes), axis=0).tolist(),
        max_abs_action=np.max(np.abs(commands), axis=0).tolist(),
        estimated_derivatives=agent.identifier.derivatives.tolist(),
        true_final_derivatives=effective.tolist(),
        identifier_updates=[e.num_updates for e in agent.identifier.estimators],
        saturation_fraction=np.mean(
            np.isclose(np.abs(commands), config.magnitude_limit), axis=0
        ).tolist(),
        final_metrics=last_metrics,
        **(
            {
                "trace": trace,
                "trace_columns": [
                    "time_s",
                    "reference_p",
                    "reference_q",
                    "reference_r",
                    "p",
                    "q",
                    "r",
                    "u_roll",
                    "u_pitch",
                    "u_yaw",
                    "estimated_gyro_q_fault",
                    "true_gyro_q_fault",
                ],
            }
            if record_trace
            else {}
        ),
        min_state_covariance_eigenvalue=float(
            np.linalg.eigvalsh(agent.observer.filter.state_covariance).min()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["iadp", "aaindi"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rate-limit-deg", type=float, default=180.0)
    parser.add_argument("--cutoff-hz", type=float, default=10.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--no-sensor-correction",
        action="store_true",
        help="AA-INDI ablation: run the observer but use uncorrected rates",
    )
    args = parser.parse_args()
    if args.duration is not None and (
        not np.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be finite and positive")
    result = (
        iadp_trial(args.duration or 80.0, record_trace=args.trace)
        if args.agent == "iadp"
        else aaindi_trial(
            args.duration or 60.0,
            seed=args.seed,
            correct_sensor_faults=not args.no_sensor_correction,
            rate_limit_deg=args.rate_limit_deg,
            cutoff_hz=args.cutoff_hz,
            record_trace=args.trace,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("failure") is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
