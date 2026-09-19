"""Reproducible B747 engine-failure benchmark built from public SDK components.

The protocol fixes healthy initialization, baseline tuning, limits and metrics.
It is a local simulator comparison, not universal tuning or flight validation.
"""

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from tensoraerospace.aerospacemodel.b747.nonlinear import NonlinearB747, trim
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    DamageProfile,
    EngineFailureEvent,
)
from tensoraerospace.agent.aa_indi import (
    AAINDIAgent,
    AAINDIConfig,
    AircraftGeometry,
    FlightMeasurement,
    ObserverConfig,
)
from tensoraerospace.agent.lqr import LQRAgent
from tensoraerospace.agent.pid import B747LongitudinalHold, LateralAircraftPID
from tensoraerospace.agent.pid.aircraft import B747_LATERAL_PID_GAINS
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

from .bench import ControlBenchmark


@dataclass(frozen=True)
class B747EngineFailureBenchmark:
    duration: float = 90.0
    dt: float = 0.02
    fault_time: float = 30.0
    engine_fraction: float = 0.0
    engine_id: int = 1
    substeps: int = 1
    initial_heading_deg: float = 1.0
    initial_roll_deg: float = 0.3
    seed: int = 11
    altitude_ft: float = 20000.0
    airspeed_ft_s: float = 674.0
    surface_limit_deg: float = 8.0
    surface_rate_deg_s: float = 20.0

    algorithms = ("AA-INDI", "PID", "LQR", "LQI")

    def __post_init__(self):
        vals = (
            self.altitude_ft,
            self.airspeed_ft_s,
            self.surface_limit_deg,
            self.surface_rate_deg_s,
            self.duration,
            self.dt,
            self.fault_time,
            self.engine_fraction,
            self.initial_heading_deg,
            self.initial_roll_deg,
        )
        if (
            not all(np.isfinite(v) for v in vals)
            or min(
                self.dt,
                self.duration,
                self.airspeed_ft_s,
                self.surface_limit_deg,
                self.surface_rate_deg_s,
            )
            <= 0
        ):
            raise ValueError("Finite positive duration/dt required")
        if (
            not 0 <= self.fault_time < self.duration
            or not 0 <= self.engine_fraction <= 1
        ):
            raise ValueError("Invalid fault time/effectiveness")
        if (
            self.engine_id not in (1, 2, 3, 4)
            or self.substeps < 1
            or not isinstance(self.substeps, (int, np.integer))
        ):
            raise ValueError("Invalid engine/substeps")
        if not np.isclose(self.duration / self.dt, round(self.duration / self.dt)):
            raise ValueError("Duration must align with control steps")

    @property
    def steps(self):
        return round(self.duration / self.dt)

    def nominal_trim(self):
        """Healthy cruise equilibrium used once for controller initialization."""
        result = trim(self.altitude_ft, self.airspeed_ft_s)
        if not result.converged or result.residual > 1e-7:
            raise RuntimeError("B747 nominal trim did not converge")
        return result

    def nominal_model(self):
        """Fresh healthy model for nominal local analysis."""
        return NonlinearB747(self.nominal_trim().to_state(), dt=self.dt)

    def make_env(self, *, fault):
        """Create the native SDK environment with a plant-only damage profile."""
        initial = self.nominal_trim().to_state()
        initial[[6, 8]] = np.deg2rad([self.initial_roll_deg, self.initial_heading_deg])
        profile = (
            DamageProfile(
                events=[
                    EngineFailureEvent(
                        trigger_time=self.fault_time,
                        engine_id=self.engine_id,
                        thrust_fraction=self.engine_fraction,
                    )
                ]
            )
            if fault
            else None
        )
        return NonlinearB747Env(
            initial_state=initial,
            dt=self.dt / self.substeps,
            number_time_steps=self.steps * self.substeps,
            action_space="virtual",
            damage_profile=profile,
        )

    def make_aaindi(self):
        """AA-INDI with a healthy derivative prior and documented cruise gains."""
        model = self.nominal_model()
        tr = self.nominal_trim()
        action = np.array([tr.elevator_rad, 0.0, 0.0, tr.throttle])
        geometry = AircraftGeometry.from_parameters(model.param)
        sample = FlightMeasurement.from_model(
            model, applied_action=action, surface_indices=(1, 2)
        )
        _, B = model.linearize(model.current_state, action)
        derivatives = np.column_stack(
            [
                geometry.coefficients(
                    np.zeros(3), B[3:6, channel], sample.density, sample.airspeed
                )
                for channel in (1, 2)
            ]
        )
        return AAINDIAgent(
            AAINDIConfig(
                geometry=geometry,
                nominal_derivatives=derivatives,
                observer=ObserverConfig(
                    dt=self.dt,
                    gravity=model.param.g_ft_s2 * 0.3048,
                    drift_scales=np.array([1.0] * 3 + [1e-4] * 3),
                ),
                covariance_init=0.001,
                acceleration_cutoff_hz=5.0,
                magnitude_limit=np.deg2rad(self.surface_limit_deg),
                rate_limit=np.deg2rad(self.surface_rate_deg_s),
                rate_feedback=np.full(3, 0.5),
            )
        )

    def make_lqr(self, *, integral=False, input_weight=0.2, angle_weight=1.0):
        """SDK discrete LQR, optionally including measured angle integrals."""
        tr = self.nominal_trim()
        action = np.array([tr.elevator_rad, 0.0, 0.0, tr.throttle])
        A, B = self.nominal_model().lateral_linearization(action)
        weights = np.array([0.1, 0.2, 0.2, 2 * angle_weight, 5 * angle_weight, 1, 2])
        if not integral:
            A, B, weights = A[:5, :5], B[:5], weights[:5]
        return LQRAgent(A, B, np.diag(weights), np.eye(2) * input_weight)

    def limit_lateral(self, requested, previous):
        """Apply the protocol's common magnitude and per-second slew limits."""
        rate = np.deg2rad(self.surface_rate_deg_s) * self.dt
        requested = np.clip(requested, previous - rate, previous + rate)
        return np.clip(
            requested,
            -np.deg2rad(self.surface_limit_deg),
            np.deg2rad(self.surface_limit_deg),
        )

    def evaluate(self, states, actions, *, start, end):
        """Library tracking metrics plus physical flight-envelope summaries."""
        states, actions = np.asarray(states, dtype=float), np.asarray(
            actions, dtype=float
        )
        if (
            states.ndim != 2
            or states.shape[1] != 12
            or actions.shape != (len(states) - 1, 4)
            or not np.isfinite(states).all()
        ):
            raise ValueError(
                "Expected complete finite 12-state and four-action trajectories"
            )
        result = ControlBenchmark().tracking_metrics(
            0.0,
            np.rad2deg(states[:, [6, 8]]),
            self.dt,
            start=start,
            end=end,
            tolerance=0.05,
            actions=np.rad2deg(actions[:, 1:3]),
        )
        times = np.arange(len(states)) * self.dt
        mask = (times > start) & (times <= end)
        return dict(
            roll_rmse_deg=float(result["rmse"][0]),
            heading_rmse_deg=float(result["rmse"][1]),
            combined_rmse_deg=result["combined_rmse"],
            angle_iae_deg_s=result["iae"],
            peak_roll_deg=float(result["peak_error"][0]),
            peak_heading_deg=float(result["peak_error"][1]),
            recovery_s=result["recovery_time"],
            surface_rms_deg=result["control_rms"],
            surface_peak_deg=result["control_peak"],
            surface_total_variation_deg=result["control_variation"],
            peak_speed_error_ft_s=float(
                abs(np.linalg.norm(states[mask, :3], axis=1) - self.airspeed_ft_s).max()
            ),
            peak_height_error_ft=float(abs(-states[mask, 11] - self.altitude_ft).max()),
        )

    def run(self, algorithm, *, fault, pid_gains=None, lqr_settings=None):
        """Run one fresh controller and environment for the complete protocol."""
        if algorithm not in self.algorithms:
            raise ValueError(f"unknown controller: {algorithm}")
        env = self.make_env(fault=fault)
        state, _ = env.reset(seed=self.seed)
        hold = B747LongitudinalHold(
            self.dt,
            trim_result=self.nominal_trim(),
            altitude_ft=self.altitude_ft,
            airspeed_ft_s=self.airspeed_ft_s,
        )
        tr = self.nominal_trim()
        initial_action = np.array([tr.elevator_rad, 0, 0, tr.throttle])
        agent = self.make_aaindi() if algorithm == "AA-INDI" else None
        pid = (
            LateralAircraftPID(
                pid_gains if pid_gains is not None else B747_LATERAL_PID_GAINS,
                self.dt,
                magnitude_limit_deg=self.surface_limit_deg,
                rate_limit_deg_s=self.surface_rate_deg_s,
            )
            if algorithm == "PID"
            else None
        )
        lqr = (
            self.make_lqr(integral=algorithm == "LQI", **(lqr_settings or {}))
            if algorithm in ("LQR", "LQI")
            else None
        )
        measurement = (
            FlightMeasurement.from_model(
                env.model, applied_action=initial_action, surface_indices=(1, 2)
            )
            if agent
            else None
        )
        initial_derivatives = agent.identifier.derivatives.copy() if agent else None
        integrals, previous = np.zeros(2), np.zeros(2)
        states, actions, learning = [state.copy()], [], []
        try:
            for k in range(self.steps):
                if agent:
                    phi, theta = state[6:8]
                    # Application outer loop: roll and heading hold, no event input.
                    euler_rate = -0.15 * state[[6, 8]]
                    desired_rate = np.array(
                        [
                            euler_rate[0] - np.sin(theta) * euler_rate[1],
                            0.0,
                            np.cos(theta) * np.cos(phi) * euler_rate[1],
                        ]
                    )
                    requested = agent.predict(measurement, desired_rate)
                elif pid:
                    requested = np.deg2rad(
                        pid.command(state, applied_output=np.rad2deg(previous))
                    )
                else:
                    x = env.model.lateral_state(state, integrals)
                    requested = np.deg2rad(
                        cast(LQRAgent, lqr).predict(x if algorithm == "LQI" else x[:5])
                    )
                lateral = self.limit_lateral(requested, previous)
                elevator, throttle = hold.command(state)
                command = np.r_[elevator, lateral, throttle]
                for substep in range(self.substeps):
                    state, _, terminated, truncated, _ = env.step(command)
                applied = env.model.u_history[-1].ravel().copy()
                if agent:
                    measurement = FlightMeasurement.from_model(
                        env.model,
                        applied_action=applied,
                        time=(k + 1) * self.dt,
                        surface_indices=(1, 2),
                    )
                    diagnostics = agent.learn(measurement, applied_action=applied[1:3])
                    learning.append(
                        [
                            np.linalg.norm(
                                agent.identifier.derivatives - initial_derivatives
                            ),
                            diagnostics["moment_residual_norm"],
                            *np.rad2deg(agent.observer.faults[3:]),
                            np.linalg.cond(agent.G),
                        ]
                    )
                integrals = np.clip(
                    integrals + self.dt * np.rad2deg(state[[6, 8]]), -100, 100
                )
                previous = applied[1:3]
                states.append(state.copy())
                actions.append(applied)
                beta = np.arcsin(np.clip(state[1] / np.linalg.norm(state[:3]), -1, 1))
                if (
                    not np.isfinite(state).all()
                    or abs(state[6]) > np.deg2rad(30)
                    or abs(state[8]) > np.deg2rad(45)
                    or abs(beta) > np.deg2rad(15)
                    or abs(np.linalg.norm(state[:3]) - self.airspeed_ft_s) > 100
                    or abs(-state[11] - self.altitude_ft) > 1500
                ):
                    raise RuntimeError(
                        f"{algorithm}: flight envelope exceeded at {(k+1)*self.dt:g} s"
                    )
                if terminated or (truncated and k != self.steps - 1):
                    raise RuntimeError("incomplete episode")
        finally:
            env.close()
        state_array, action_array = np.asarray(states), np.asarray(actions)
        return {
            "algorithm": algorithm,
            "fault": fault,
            "states": state_array,
            "actions": action_array,
            "learning": np.asarray(learning),
            "events": env.damage_events_log,
            "updates": (
                [e.num_updates for e in agent.identifier.estimators] if agent else []
            ),
            "before": (
                self.evaluate(state_array, action_array, start=0, end=self.fault_time)
                if self.fault_time > 0
                else None
            ),
            "after": self.evaluate(
                state_array, action_array, start=self.fault_time, end=self.duration
            ),
            "whole": self.evaluate(
                state_array, action_array, start=0, end=self.duration
            ),
        }

    @staticmethod
    def healthy_cost(run):
        angles = np.rad2deg(run["states"][1:, [6, 8]])
        surfaces = np.rad2deg(run["actions"][:, 1:3])
        # Same angle/effort objective for every baseline; no fault trajectory here.
        return float(
            np.mean(np.sum(angles**2, axis=1) + 0.002 * np.sum(surfaces**2, axis=1))
        )

    def tune_baselines(self):
        """Small reproducible nominal-only search, including the existing tuned PID.

        Use a separate 60 s healthy regulation episode with larger initial errors.
        Every candidate is evaluated; no trial is removed from the returned table.
        """
        training = replace(
            self, duration=60, fault_time=30, initial_roll_deg=1, initial_heading_deg=2
        )
        records, selected = [], {}
        base = np.asarray(B747_LATERAL_PID_GAINS)
        candidates = [base]
        # Include faster integral recovery and less aggressive roll P gain.
        for p_scale, i_scale, d_scale in [
            (0.5, 1, 1),
            (1, 3, 1),
            (0.5, 3, 1),
            (0.5, 10, 1),
            (0.25, 3, 0.75),
            (1, 10, 1),
        ]:
            candidates.append(base * np.tile([p_scale, i_scale, d_scale], 2))
        for algorithm in ("PID", "LQR", "LQI"):
            best = float("inf")
            settings: list[dict[str, Any]] = (
                [{"pid_gains": g} for g in candidates]
                if algorithm == "PID"
                else [
                    {"lqr_settings": {"input_weight": r, "angle_weight": q}}
                    for r, q in [(0.2, 1), (0.05, 1), (0.8, 1), (0.2, 4), (0.2, 0.25)]
                ]
            )
            for setting in settings:
                run = training.run(algorithm, fault=False, **setting)
                cost = self.healthy_cost(run)
                records.append(
                    {"algorithm": algorithm, "healthy_cost": cost, "settings": setting}
                )
                if cost < best:
                    best, selected[algorithm] = cost, setting
        return selected, records

    def validate_additional_cases(self, settings):
        """Same controllers, with no retuning on the three extra fault scenarios."""
        cases = {
            "Right outer engine out at 20 s": replace(self, engine_id=4, fault_time=20),
            "Half thrust on engine 1 at 45 s": replace(
                self, engine_fraction=0.5, fault_time=45
            ),
            "500 s, left outer engine out": replace(self, duration=500),
        }
        rows = []
        for name, case in cases.items():
            for algorithm in self.algorithms:
                run = case.run(algorithm, fault=True, **settings.get(algorithm, {}))
                rows.append(
                    {
                        "case": name,
                        "algorithm": algorithm,
                        "after": run["after"],
                        "final_roll_heading_deg": np.rad2deg(
                            run["states"][-1, [6, 8]]
                        ).tolist(),
                    }
                )
        return rows
