# Active-Adaptive Incremental Nonlinear Dynamic Inversion (AA-INDI)

Use `AAINDIAgent(AAINDIConfig(...))` for physical moment identification and independent-navigation OTSEKF–HOSM fault estimation. This is the only AA-INDI implementation. Its configuration requires aircraft geometry and nominal surface derivatives; control accepts `FlightMeasurement` packets. The old rate-only agent and reintegration bias heuristic have been removed.

Sources: [Atmaca et al., AA-INDI, 2026](https://doi.org/10.2514/6.2026-1743), and the authors' detailed [OTSEKF–HOSM paper, 2025](https://doi.org/10.2514/1.G009147).

## Start with a complete SDK example

This example creates a native B737 environment, initializes AA-INDI from a
healthy trim, commands a +1° pitch step at 15 s and loses 50% elevator authority
at 30 s. Execute the block with the installed `tensoraerospace` package. It runs
all 3,000 transitions, plots the reference and response, and prints library metrics.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensoraerospace.agent.aa_indi import (
    AAINDIAgent,
    AAINDIConfig,
    AircraftGeometry,
    FlightMeasurement,
    ObserverConfig,
)
from tensoraerospace.aerospacemodel.b737.nonlinear import ElevatorEffectiveness
from tensoraerospace.benchmark import B737PitchStepBenchmark

experiment = B737PitchStepBenchmark(
    duration=60.0,
    dt=0.02,
    step_time=15.0,
    step_deg=1.0,
    elevator_fault=ElevatorEffectiveness(time=30.0, effectiveness=0.5),
)
env, trim, trim_action = experiment.make_env()
state, _ = env.reset(seed=experiment.seed)
theta_trim = state[7]
geometry = AircraftGeometry.from_parameters(env.model.param)
measurement = FlightMeasurement.from_model(
    env.model,
    applied_action=trim_action,
    surface_indices=(0,),
)
_, B = env.model.linearize(state, trim_action)
nominal_derivatives = geometry.coefficients(
    np.zeros(3),
    B[3:6, 0],
    measurement.density,
    measurement.airspeed,
)[:, None]
agent = AAINDIAgent(
    AAINDIConfig(
        geometry=geometry,
        nominal_derivatives=nominal_derivatives,
        observer=ObserverConfig(
            dt=experiment.dt, gravity=env.model.param.g_ft_s2 * 0.3048
        ),
        covariance_init=1.0,
        rate_feedback=np.full(3, 3.0),
        acceleration_cutoff_hz=5.0,
        magnitude_limit=env.model.param.elevator_max_rad,
        rate_limit=np.deg2rad(20.0),
        enable_sensor_correction=True,
    )
)
states, actions, rate_commands = [state.copy()], [], []
try:
    for k in range(experiment.steps):
        q_ref = np.clip(
            0.8 * (theta_trim + experiment.reference[k] - state[7]),
            -np.deg2rad(3.0),
            np.deg2rad(3.0),
        )
        command = agent.predict(measurement, np.array([0.0, q_ref, 0.0]))
        action = trim_action.copy()
        action[0] = command[0]
        state, _, terminated, truncated, _ = env.step(action)
        experiment.validate_transition(state, terminated, truncated, k)
        applied = env.model.applied_action
        measurement = FlightMeasurement.from_model(env.model, surface_indices=(0,))
        agent.learn(measurement, applied_action=applied[:1])
        states.append(state.copy())
        actions.append(applied)
        rate_commands.append(q_ref)
finally:
    env.close()
states, actions, rate_commands = map(np.asarray, (states, actions, rate_commands))
experiment.plot_response(states, actions, rate_commands, "AA-INDI: B737 elevator fault")
plt.show()
windows, physical_metrics = experiment.evaluate(states, actions)
print(experiment.metric_table(windows).to_string())
print(physical_metrics)
```

`predict` returns an absolute elevator angle in radians; trim is not added again.
`FlightMeasurement.from_model` supplies SI measurements and actual surface feedback.
Learning continues at every interval. Set `elevator_fault=None` for a fresh healthy
comparison. The speed/altitude response belongs to a fixed-throttle pitch task.

### Choose the next example

- [Full B737 walkthrough](../example/agent/aa_indi/example_aaindi_nonlinear.md).
- [Actuator and gyro faults, correction enabled/disabled](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_sensor_actuator_faults.ipynb).
- [AA-INDI versus PID, LQR and LQI on B747](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).

## Paper architecture

| Block | Implementation | Source |
| --- | --- | --- |
| Kinematic state estimation with independent navigation | `OTSEKFHOSMObserver`, `OptimalTwoStageEKF` | 2026 Section III.A; 2025 Eqs. (20)–(42) |
| Four-state nonrecursive differentiator | `HOSMDifferentiator` | 2025 Eqs. (47)–(50) |
| Rigid-body moment reconstruction | `AircraftGeometry.coefficients` | 2026 Eqs. (7)–(16) |
| Surface-derivative identification, one scalar VFF-RLS per moment axis | `MomentIdentifier` | 2026 Eqs. (50)–(57) |
| Incremental inversion from virtual angular acceleration | `AAINDIAgent.predict_acceleration` | 2026 Eq. (9) |

The measured moment is
\[
M = J\dot\omega + \omega\times J\omega,
\quad C_M = \frac{M}{\bar q S [b,c,b]^T},
\quad G=J^{-1}\bar q S\operatorname{diag}(b,c,b) C_\delta.
\]
Division in the coefficient expression is componentwise. Identification uses **absolute measured surface positions**, paired with reconstructed moment coefficients, and separate forgetting for roll, pitch and yaw. The published settings `sigma0=15`, `forgetting_min=0.25`, and maximum forgetting factor 1 are supplied by default. The input and moment filters share a time constant and are initialized from the same measured interval.

The primary control interface receives virtual angular acceleration from an outer controller. `predict(measurement, rate_reference)` supplies a configurable proportional rate-feedback adapter. It does not reproduce the Flying-V C*/roll/sideslip guidance system.

## Measurements and units

`FlightMeasurement` contains one timestamp:

| Field | Meaning / units |
| --- | --- |
| `angular_rate` | Body `[p,q,r]`, rad/s; may contain sensor faults |
| `specific_force` | Body accelerometer `[Ax,Ay,Az]`, m/s²: specific force `R.T @ (a_NED - g_NED)` |
| `ground_velocity` | Independent NED velocity, m/s |
| `attitude` | Independent `[roll,pitch,yaw]`, radians, 3-2-1 Euler convention |
| `surface_position` | Actual control surface angles over the preceding interval, radians; ZOH value or measured interval average |
| `airspeed`, `density` | Air-relative speed in m/s, air density in kg/m³ |

Body axes are forward/right/down; navigation axes are north/east/down. In level unaccelerated flight the body accelerometer reads `[0,0,-g]`. Geometry uses kg·m² for the full inertia tensor, m² for area and metres for span/chord. Convert environment actions in degrees explicitly.

The first packet requires both navigation velocity and attitude. Later packets may use `None` for an unavailable navigation channel, e.g. 10 Hz GPS with a 100 Hz IMU. Timestamps must advance exactly `observer.dt`; the next `predict` reuses the packet already passed to `learn`. Navigation must contain independent information: integrating the same faulty gyro to create “attitude” does not make its bias observable. Airspeed must not be replaced with GPS speed in wind.

## Interpretation and remaining differences

The two-stage covariance factorization is checked against an independent augmented Kalman filter, including random bias and correlated process noise. HOSM uses the published simultaneous updates and exponents `3/4`, `2/3`, `1/2`, and `sign`.

**Drift-coordinate interpretation:** the article calls the second filter coordinate state drift, while its propagation uses the integrated input-noise matrix. Here the two-stage coordinate has input-bias units. HOSM therefore differentiates accumulated *physical state drift*: the integrated non-exact kinematic trajectory minus its navigation-corrected estimate. The derivative is converted to body IMU faults using the kinematic channel gains. This is an explicit dimensional interpretation of the architecture, not a verified reproduction of the authors' unpublished implementation. Differentiating an already rate-valued bias would estimate its change, losing a constant fault.

Observer process-noise tuning, HOSM gains/scales, filter cutoff and initial parameter covariance are explicit configuration choices. The papers do not supply a complete executable setup. Tests and synthetic rollouts validate this implementation, not the published flight-test results. The default `sigma0=15` can yield slow actuator identification when dimensionless residuals are small; rate tracking alone does not prove derivative convergence. Euler kinematics also restrict operation away from pitch ±90° and ill-conditioned fault-reconstruction attitudes.

`agent.save(path)` and `AAINDIAgent.from_pretrained(folder)` retain the observer, HOSM, RLS, input/output filters and any pending transition. Save/load is tested for identical continuation. Old rate-only checkpoints cannot supply missing geometry or independent navigation; create a new configuration and checkpoint.

## Paper API reference

::: tensoraerospace.agent.aa_indi.model.AAINDIAgent

::: tensoraerospace.agent.aa_indi.model.AAINDIConfig

::: tensoraerospace.agent.aa_indi.observer.ObserverConfig

::: tensoraerospace.agent.aa_indi.kinematics.FlightMeasurement

## Nonlinear B737 notebook

[Run the B737 pitch-step example](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_nonlinear_b737.ipynb): the same cruise trim and +1° step as the IHDP notebook, with an explicit pitch-to-rate outer loop, continuous adaptation, saved plots and `ControlBenchmark` metrics. The example documents its nominal-model initialization and reduced-model assumptions.

## Fault examples

- [B737: 50% elevator-authority loss](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb), with continuous learning, actual surface feedback and post-fault error metrics.
- [B747: AA-INDI vs PID, LQR and LQI after an engine failure](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_vs_pid_lqr_b747.ipynb), with healthy-only baseline tuning and separate validation up to 500 s.

[Comparison with plots and metrics](../comparison/aaindi_vs_pid_lqr_lqi_b747.md).

## Simulator measurements through the public API

For the native nonlinear B737/B747 models, use
`AircraftGeometry.from_parameters(model.param)` for the SI inertia/geometry and
`FlightMeasurement.from_model(model, surface_indices=(0,))` for elevator-only
feedback, or `(1, 2)` for aileron/rudder. Before the first transition, supply
`applied_action=trim_action` explicitly; later packets use `model.applied_action`
and `model.current_time`. Surface indices refer to the four-channel physical
input `[elevator, aileron, rudder, throttle]`.

The adapter simulates ideal IMU and independent navigation using the native
model's `dynamics` API. It removes gravity and body-frame transport terms from
accelerometer output and converts US units to SI. It assumes no wind and adds no
sensor noise or faults. For hardware/noisy sensors, construct `FlightMeasurement`
from the actual sensor streams. Neither adapter call advances the model.

`model.linearize(state, trim_action)` supplies continuous native-unit A/B
Jacobians for a nominal prior. Calculate the prior on a healthy model before the
flight; do not replace it with post-failure derivatives during adaptation.
