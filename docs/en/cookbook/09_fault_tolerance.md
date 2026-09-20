# Recipe 09 — Evaluate fault tolerance on a common aircraft

**Goal:** tune classical controllers on a healthy B747, inject a physical engine
failure, and compare AA-INDI, PID, LQR and LQI using the same flight and actuator
constraints. You will run paired healthy/faulty episodes, verify causality, draw
reference/response plots and measure recovery and control effort.

**Notebook:** [recipe_09_fault_tolerance.ipynb](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/cookbook/recipe_09_fault_tolerance.ipynb).
It contains executable cells and saved figures. Run the following Python blocks
in order in an environment with this version of `tensoraerospace` installed. The complete
[comparison report](../comparison/aaindi_vs_pid_lqr_lqi_b747.md) includes additional
physical and adaptation plots.

## 1. Define what actually fails

The native nonlinear 12-state B747 has four engines. At 30 s, the left outer
engine loses all thrust. At unchanged throttle and flight condition, total thrust
falls to 75% of its healthy value and an asymmetric yaw moment appears. The native
`EngineFailureEvent` changes propulsion inside the physical model.

Aileron and rudder authority remain intact. Multiplying a measured heading or
adding an arbitrary state jump would describe a different experiment.

| Condition | Common value |
|---|---|
| Altitude / airspeed | 20,000 ft / 674 ft/s |
| Initial roll / heading | 0.3° / 1° |
| Roll / heading commands | 0° / 0° throughout |
| Duration / control interval | 90 s / 0.02 s |
| Aileron and rudder bounds | ±8° and 20°/s |
| Longitudinal hold | Same measured-state PI/PD speed/altitude controller |
| Sensors | Ideal IMU, navigation and actual surface feedback |


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensoraerospace.benchmark import B747EngineFailureBenchmark

experiment = B747EngineFailureBenchmark(
    duration=90.0, dt=0.02, fault_time=30.0,
    engine_fraction=0.0, engine_id=1,
    initial_heading_deg=1.0, initial_roll_deg=0.3, seed=11,
)
COLORS = {"AA-INDI": "#176b87", "PID": "#c77835",
          "LQR": "#8064a2", "LQI": "#4b9b75"}
```


`experiment` configures the simulator/evaluator. The controller constructors do
not receive the fault schedule. `B747EngineFailureBenchmark` composes the SDK models, AA-INDI sensor
adapter, `PID`-based lateral loops, `LQRAgent` and `ControlBenchmark`. It manages
healthy initialization, common limits, complete-episode checks and cleanup. Read [its implementation](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/tensoraerospace/benchmark/engine_failure.py)
when adapting the example to another aircraft.

## 2. Select the baselines on healthy data

PID has separate roll/heading loops and anti-windup. LQR uses the five lateral
states; LQI adds two bounded angle-error integrals, allowing integral disturbance
rejection. Their parameters are selected on a separate **healthy 60 s episode**
with larger initial errors: 1° roll and 2° heading.

The common objective is the mean of
`roll_error² + heading_error² + 0.002*(aileron² + rudder²)`, with angles in degrees.
Seven PID candidates and five weight choices each for LQR and LQI are assessed.


```python
settings, trials = experiment.tune_baselines()
print(pd.DataFrame([
    {"Controller": row["algorithm"], "Healthy cost": row["healthy_cost"]}
    for row in trials
]).to_string(index=False))
for name, selected in settings.items():
    print(name, selected)
```


Keep the selected settings for every subsequent fault scenario. This finite
candidate search is reproducible but does not establish a globally optimal PID,
LQR or LQI. AA-INDI uses the nominal derivatives, observer and gains documented
in the comparison page; it starts fresh for each episode and adapts throughout.

## 3. Run healthy and faulty aircraft independently

Each call constructs a new environment and controller. It returns the complete
states, actual actions, metrics, event log and AA-INDI diagnostics. Independent
initialization matters: reusing an already adapted agent would bias the second run.


```python
runs = {}
for name in experiment.algorithms:
    for failed in (False, True):
        runs[(name, failed)] = experiment.run(name, fault=failed, **settings.get(name, {}),
        )
        assert len(runs[(name, failed)]["actions"]) == experiment.steps
    event_index = round(experiment.fault_time / experiment.dt)
    np.testing.assert_array_equal(
        runs[(name, False)]["states"][:event_index + 1],
        runs[(name, True)]["states"][:event_index + 1],
    )
    np.testing.assert_array_equal(
        runs[(name, False)]["actions"][:event_index],
        runs[(name, True)]["actions"][:event_index],
    )
assert runs[("AA-INDI", True)]["updates"] == [experiment.steps] * 3
print("Eight complete trajectories; pre-failure histories agree")
```


The equality checks include the state at 30 s and every action applied before
that time. They catch an event leaking into earlier integration stages or a
controller using different pre-failure settings. The equality is expected because
this case uses deterministic ideal sensors. With noisy sensors, use paired seeds
and compare matching noise streams.

AA-INDI performs 4,500 identification updates per moment axis. No algorithm is
reset, switched or frozen when the failure occurs. The rollout rejects nonfinite
states, departures from the example envelope and incomplete episodes.

## 4. Plot the reference as well as the response


```python
time = np.arange(experiment.steps + 1) * experiment.dt
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey="row", constrained_layout=True)
for column, failed in enumerate((False, True)):
    for algorithm in experiment.algorithms:
        state = runs[(algorithm, failed)]["states"]
        for row, index in enumerate((6, 8)):
            axes[row, column].plot(time, np.rad2deg(state[:, index]),
                                   color=COLORS[algorithm], label=algorithm)
    for row, name in enumerate(("Roll", "Heading")):
        ax = axes[row, column]
        ax.axhline(0, color="#333333", linestyle="--", label="Command: 0°")
        ax.axvline(experiment.fault_time, color="#b44040", linestyle=":")
        ax.set(title=f"{name} · {'engine 1 out' if failed else 'healthy aircraft'}",
               ylabel=f"{name} [deg]", xlabel="Time [s]")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9, loc="best")
fig.suptitle("Same task and actuator limits; nominal design before the fault", fontsize=16)
plt.show()
```


![Healthy and engine-out B747 responses with explicit zero references](../../assets/images/aaindi_b747_engine_failure_attitude.png)

Compare matching rows: roll above, heading below. Shared row scales keep the
healthy and failed-engine panels comparable. A line at 30 s is an evaluation
annotation; it is not a signal supplied to the controller.

## 5. Measure both error and control effort

The post-failure window is **(30, 90] s**. `ControlBenchmark.tracking_metrics`, called by the protocol, computes:

- Combined RMSE: `sqrt(mean(roll_error² + heading_error²))`, in degrees.
- Combined IAE: `dt*sum(abs(roll_error) + abs(heading_error))`, in degree-seconds.
- Recovery: time since failure until both errors stay within ±0.05° to the end.
- Surface RMS and total variation across aileron/rudder in the same window.


```python
columns = ["roll_rmse_deg", "heading_rmse_deg", "combined_rmse_deg",
           "angle_iae_deg_s", "recovery_s", "surface_rms_deg",
           "surface_total_variation_deg"]
post_fault = pd.DataFrame({name: runs[(name, True)]["after"]
                          for name in experiment.algorithms}).T[columns]
print(post_fault.to_string(float_format=lambda value: f"{value:.6f}",
                          na_rep="Not reached"))
healthy = pd.DataFrame({name: runs[(name, False)]["whole"]
                       for name in experiment.algorithms}).T[columns]
print("Healthy aircraft, full 90 s:")
print(healthy.to_string(float_format=lambda value: f"{value:.6f}",
                       na_rep="Not reached"))
```


| Controller | Combined RMSE, ° | Combined IAE, °·s | Recovery after failure, s |
|---|---:|---:|---:|
| AA-INDI | 0.032433 | 2.03693 | 8.68 |
| PID | 1.086181 | 67.48585 | Not reached |
| LQR | 0.359953 | 26.10714 | Not reached |
| LQI | 0.300430 | 24.43503 | Not reached |

AA-INDI has lower angular error in this configuration. Its surface RMS is 3.0300°,
compared with 2.1685° for LQR and 2.3486° for LQI: improved accuracy requires more
control activity here. Total variation measures command changes, not physical
energy use or calibrated actuator wear.

![Combined accumulated error and actual surface activity](../../assets/images/aaindi_b747_engine_failure_effort.png)

This is disturbance rejection at a zero reference, so a percentage overshoot
normalized by the command is undefined. For a commanded pitch step, use the B737
`ControlBenchmark` workflow in [Recipe 14](14_aaindi.md). Keep response-to-step and
response-to-failure windows separate when both events occur in one flight.

## 6. Check whether slow recovery is mistaken for divergence

Enable the following block to run another 12 trajectories: right outer engine
failure at 20 s, 50% remaining thrust on engine 1 from 45 s, and the original
failure with a **500 s** horizon. Settings remain unchanged.


```python
RUN_EXTENDED_VALIDATION = False
if RUN_EXTENDED_VALIDATION:
    validation = experiment.validate_additional_cases(settings)
    table = pd.DataFrame([
        {"Case": row["case"], "Controller": row["algorithm"], **row["after"]}
        for row in validation
    ])
    print(table.to_string(index=False, na_rep="Not reached"))
```


At 500 s, recovery times after the failure are **8.68 s for AA-INDI, 186.10 s for
PID and 88.34 s for LQI**. Standard LQR retains a nonzero offset. Thus, “not reached”
in the 90 s run does not imply that PID or LQI diverged. Inspect the full trajectory
and the required horizon before drawing that conclusion.

Set `RUN_EXTENDED_VALIDATION=True` above to recompute the extended cases through the SDK and inspect their metrics locally.

## 7. Extend the experiment to other kinds of failure

| Question | Appropriate example |
|---|---|
| Can the controller reject asymmetric propulsion? | This B747 comparison. |
| What happens when the elevator loses aerodynamic authority? | [iADP on B737](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_iadp_fault_b737.ipynb) and [AA-INDI on B737](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/develop/example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb). |
| Can independent navigation reconstruct a gyro bias? | [AA-INDI sensor/actuator experiment](../example/agent/aa_indi/example_aaindi_nonlinear.md). |

The B737 fault scales the elevator angle **inside the aerodynamic calculation**;
the encoder still reports the physical angle. Its 60 s examples use a +1° pitch
step at 15 s and 50% authority loss at 30 s. Post-failure pitch RMSE is 0.129777°
for the current iADP configuration and 0.003756° for AA-INDI. iADP retains an offset;
those results should not be described as universal successful zero-error recovery.

## Interpretation and troubleshooting

| Observation | How to investigate |
|---|---|
| Healthy/faulty histories differ before the event | Check resets, seeds, scheduled gains and event integration timing. |
| Small angle error but speed/height deteriorate | Inspect the shared longitudinal loop, available thrust and flight envelope. |
| Low error but fitted derivatives drift | The AA-INDI regression omits other aerodynamic/engine moments; coefficients can absorb them. |
| All controls hit limits | Check trim, units, gain signs and actuator authority before tuning learning rates. |
| No recovery by the last sample | Report “not reached”; extend the horizon and distinguish offset, slow convergence and divergence. |

The B747 model has simplified aerodynamics, ideal sensors, zero-order-held
surfaces and quasi-steady thrust. Servo lag and engine spool dynamics are absent.
The advantage belongs to the complete controller/observer/adaptation architecture;
this comparison does not isolate online parameter learning as its cause or
establish physical derivative convergence.

**Next:** [Recipe 14 — Physical AA-INDI measurements](14_aaindi.md) ·
[Recipe 08 — Save and resume](08_huggingface.md).
