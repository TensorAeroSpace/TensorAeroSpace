"""Reusable nonlinear B737 pitch-step benchmark with library environments."""

from dataclasses import dataclass

import numpy as np

from tensoraerospace.aerospacemodel.b737.nonlinear import (
    B737Configuration,
    ElevatorEffectiveness,
    trim,
)
from tensoraerospace.envs.b737_nonlinear import NonlinearB737Env

from .bench import ControlBenchmark


@dataclass(frozen=True)
class B737PitchStepBenchmark:
    """Reproducible B737 pitch-step protocol, assessment and Matplotlib plots."""

    elevator_fault: ElevatorEffectiveness | None = None
    integration_substeps: int = 1
    duration: float = 60.0
    dt: float = 0.02
    step_time: float = 15.0
    step_deg: float = 1.0
    altitude_ft: float = 20000.0
    airspeed_ft_s: float = 650.0
    config: B737Configuration = B737Configuration.B737_800
    seed: int = 47

    def __post_init__(self):
        if (
            not isinstance(self.integration_substeps, (int, np.integer))
            or self.integration_substeps < 1
        ):
            raise ValueError("integration_substeps must be a positive integer")
        if self.elevator_fault is not None and not isinstance(
            self.elevator_fault, ElevatorEffectiveness
        ):
            raise TypeError("elevator_fault must be ElevatorEffectiveness or None")
        values = [
            self.duration,
            self.dt,
            self.step_time,
            self.step_deg,
            self.altitude_ft,
            self.airspeed_ft_s,
        ]
        if (
            not np.isfinite(values).all()
            or min(self.duration, self.dt, self.airspeed_ft_s) <= 0
        ):
            raise ValueError(
                "finite configuration and positive duration, dt and speed required"
            )
        if not 0 < self.step_time < self.duration or self.step_deg == 0:
            raise ValueError("a nonzero step must occur strictly inside the episode")
        for value in (self.duration, self.step_time):
            if not np.isclose(value / self.dt, round(value / self.dt)):
                raise ValueError(
                    "duration and step time must be integer multiples of dt"
                )

    @property
    def steps(self):
        return round(self.duration / self.dt)

    @property
    def time(self):
        return np.arange(self.steps + 1) * self.dt

    @property
    def reference(self):
        """Pitch deviation [rad], including the final observation timestamp."""
        result = np.zeros(self.steps + 1)
        result[round(self.step_time / self.dt) :] = np.deg2rad(self.step_deg)
        return result

    def validate_transition(self, state, terminated, truncated, step):
        """Fail visibly on an incomplete run or departure from the demo envelope."""
        if not np.isfinite(state).all():
            raise RuntimeError("Non-finite B737 state")
        speed = np.linalg.norm(state[:3])
        if (
            abs(state[7]) > np.deg2rad(20)
            or abs(state[6]) > np.deg2rad(20)
            or (not 300 < speed < 900)
            or (not 0 < -state[11] < 40000)
        ):
            raise RuntimeError("B737 left this example's operating envelope")
        if terminated or (truncated and step != self.steps - 1):
            raise RuntimeError("Episode ended before the requested horizon")
        if step == self.steps - 1 and (not truncated):
            raise RuntimeError("Environment did not report the configured horizon")

    def evaluate(self, states, actions):
        """Evaluate only a complete run; radians in the benchmark, degrees in tables."""
        states, actions = (np.asarray(states), np.asarray(actions))
        if (
            states.shape != (self.steps + 1, 12)
            or actions.shape != (self.steps, 4)
            or (not np.isfinite(states).all())
            or (not np.isfinite(actions).all())
        ):
            raise ValueError("Step assessment requires a complete finite trajectory")
        time, reference = (self.time, self.reference)
        output = states[:, 7] - states[0, 7]
        error_deg = np.rad2deg(reference - output)
        post = time >= self.step_time
        windows = {}
        for label, end in (
            ("First 15 s after step", min(self.duration, self.step_time + 15)),
            ("Full post-step interval", self.duration),
        ):
            mask = (time >= max(0, self.step_time - 1)) & (time <= end)
            metrics = ControlBenchmark().benchmarking_step_response(
                reference[mask],
                output[mask],
                signal_val=0.0,
                dt=self.dt,
            )
            windows[label] = metrics
        physical = {
            "Completed time [s]": float(time[-1]),
            "Post-step pitch RMSE [deg]": float(np.sqrt(np.mean(error_deg[post] ** 2))),
            "Post-step pitch MAE [deg]": float(np.mean(np.abs(error_deg[post]))),
            "Final pitch error [deg]": float(error_deg[-1]),
            "Peak |q| [deg/s]": float(np.rad2deg(np.abs(states[:, 4])).max()),
            "Peak |elevator| [deg]": float(np.rad2deg(np.abs(actions[:, 0])).max()),
            "Altitude change [ft]": float(states[0, 11] - states[-1, 11]),
            "Airspeed change [ft/s]": float(
                np.linalg.norm(states[-1, :3]) - np.linalg.norm(states[0, :3])
            ),
        }
        return (windows, physical)

    @staticmethod
    def metric_table(windows):
        import pandas as pd

        rows = {}
        for label, m in windows.items():
            rows[label] = {
                "Overshoot vs. final output [%]": m["overshoot"],
                "Settling around final output, 5% [s]": m["settling_time"],
                "Overshoot vs. command [%]": m["command_overshoot"],
                "Settling around command, 5% [s]": m["command_settling_time"],
                "Rise time, 10–90% [s]": m["rise_time"],
                "Peak time [s]": m["peak_time"],
                "Static error [deg]": np.rad2deg(m["static_error"]),
                "IAE [deg·s]": np.rad2deg(m["iae"]),
                "ISE [deg²·s]": m["ise"] * (180 / np.pi) ** 2,
                "ITAE [deg·s²]": np.rad2deg(m["itae"]),
            }
        return pd.DataFrame(rows)

    def plot_reference(self, theta_trim):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.step(
            self.time,
            np.rad2deg(theta_trim + self.reference),
            where="post",
            color="#bd6230",
            linewidth=2,
            label="Pitch reference",
        )
        ax.axvline(self.step_time, color="#64748b", linestyle=":", label="Step applied")
        ax.set(
            title=f"Commanded pitch: trim hold followed by a {self.step_deg:+g}° step",
            xlabel="Time [s]",
            ylabel="Pitch angle [deg]",
        )
        ax.legend(loc="upper left")
        return fig

    def plot_response(self, states, actions, q_reference, title):
        import matplotlib.pyplot as plt

        time = self.time
        pitch = np.rad2deg(states[:, 7])
        reference = np.rad2deg(states[0, 7] + self.reference)
        error = reference - pitch
        fig, ax = plt.subplots(
            3, 2, figsize=(14, 10), sharex=True, constrained_layout=True
        )
        ax = ax.ravel()
        ax[0].step(
            time,
            reference,
            where="post",
            color="#bd6230",
            linestyle="--",
            label="Reference",
        )
        ax[0].plot(time, pitch, color="#176b87", label="B737 pitch")
        ax[0].set(title="Pitch tracking", ylabel="Angle [deg]")
        ax[1].plot(time, error, color="#176b87", label="Reference − pitch")
        ax[1].axhline(0, color="#64748b", linewidth=1)
        ax[1].set(title="Tracking error", ylabel="Error [deg]")
        ax[2].step(
            time[:-1],
            np.rad2deg(q_reference),
            where="post",
            color="#bd6230",
            linestyle="--",
            label="Outer-loop rate command",
        )
        ax[2].plot(time, np.rad2deg(states[:, 4]), color="#176b87", label="Measured q")
        ax[2].set(title="Pitch rate", ylabel="Rate [deg/s]")
        ax[3].step(
            time,
            np.rad2deg(np.r_[actions[:, 0], actions[-1, 0]]),
            where="post",
            color="#176b87",
            label="Applied elevator",
        )
        ax[3].set(title="Physical control surface", ylabel="Deflection [deg]")
        ax[4].plot(
            time,
            -states[:, 11] + states[0, 11],
            color="#176b87",
            label="Altitude change",
        )
        ax[4].set(
            title="Vertical motion (fixed trim throttle)", ylabel="Altitude change [ft]"
        )
        post = time >= self.step_time
        integral = np.cumsum(abs(error[post])) * self.dt
        ax[5].plot(
            time[post], integral, color="#8064a2", label="Post-step cumulative IAE"
        )
        ax[5].set(title="Accumulated pitch tracking error", ylabel="IAE [deg·s]")
        for axis in ax:
            axis.axvline(self.step_time, color="#64748b", linestyle=":", linewidth=1)
            axis.grid(alpha=0.22)
            axis.legend(fontsize=9, loc="best")
            axis.set_xlabel("Time [s]")
        fig.suptitle(title, fontsize=17)
        return fig

    def plot_step(self, states, windows):
        import matplotlib.pyplot as plt

        time = self.time - self.step_time
        response = np.rad2deg(states[:, 7] - states[0, 7])
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        ax.step(
            time,
            np.rad2deg(self.reference),
            where="post",
            color="#bd6230",
            linestyle="--",
            label="Pitch step",
        )
        ax.plot(time, response, color="#176b87", label="Pitch deviation")
        lower, upper = sorted([0.95 * self.step_deg, 1.05 * self.step_deg])
        ax.fill_between(
            [0, self.duration - self.step_time],
            lower,
            upper,
            color="#4b9b75",
            alpha=0.15,
            label="±5% of the commanded step",
        )
        settling = windows["Full post-step interval"]["command_settling_time"]
        if settling is not None:
            ax.axvline(
                settling,
                color="#4b9b75",
                linestyle=":",
                label=f"Settling: {settling:.2f} s",
            )
        ax.set(
            xlim=(-1, min(15, self.duration - self.step_time)),
            xlabel="Time since step [s]",
            ylabel="Pitch deviation [deg]",
            title="Step response: the shaded band is relative to the command",
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        return fig

    def make_env(self):
        """Construct a fresh SDK environment, converged trim and trim action."""
        result = trim(self.altitude_ft, self.airspeed_ft_s, config=self.config)
        if not result.converged:
            raise RuntimeError(f"B737 trim failed: residual={result.residual:g}")
        env = NonlinearB737Env(
            initial_state=result.to_state(),
            number_time_steps=self.steps,
            dt=self.dt,
            integrator="rk4",
            action_space="virtual",
            config=self.config,
            elevator_fault=self.elevator_fault,
            integration_substeps=self.integration_substeps,
        )
        action = np.array([result.elevator_rad, 0.0, 0.0, result.throttle])
        return env, result, action
