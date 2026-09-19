# Adaptive aircraft fault examples: validation

## Scope and reproduction

Three executed English notebooks accompany this report:

- [iADP, B737 elevator-authority loss](../example/reinforcement_learning/incremental_adp/example_iadp_fault_b737.ipynb)
- [AA-INDI, B737 elevator-authority loss](../example/reinforcement_learning/incremental_adp/example_aaindi_fault_b737.ipynb)
- [AA-INDI / PID / LQR / LQI, B747 engine failure](../example/reinforcement_learning/incremental_adp/example_aaindi_vs_pid_lqr_b747.ipynb)

Run all cells in the B747 notebook to reproduce its healthy-only tuning and paired 90 s runs. Set `RUN_EXTENDED_VALIDATION = True` in its last cell to repeat all 12 additional scenarios below. The comparison implementation is in [the public SDK protocol](../tensoraerospace/benchmark/engine_failure.py). No fault time, engine identifier or remaining effectiveness enters a controller constructor or control law.

## B737: loss of elevator aerodynamic authority

60 s at dt=0.02 s; +1° pitch command at 15 s; 50% authority loss at 30 s. The native B737 model with `ElevatorEffectiveness` feeds eta times the physical elevator angle into the original aerodynamic table, recomputing lift, drag and pitch moment consistently. Encoder feedback remains the physical angle. This is a parametric fault, not a validated model of structural damage. The fault is configured through `NonlinearB737Env(elevator_fault=...)`; agent equations are unchanged by the SDK extraction.

| Controller | Post-fault pitch RMSE [deg] | Peak error [deg] | Final error [deg] |
|---|---:|---:|---:|
| iADP | 0.129777 | 0.233743 | 0.233743 |
| AA-INDI | 0.003756 | 0.012846 | -0.002334 |

Both completed 3,000 transitions. iADP performed 2,999 RLS updates and 176 critic-matrix changes; AA-INDI performed 3,000 VFF-RLS updates per moment axis. The iADP example retains its nominal configuration and exposes the nonzero post-fault pitch offset. It is not presented as successful zero-error recovery. AA-INDI keeps pitch close to the command, but its fitted control-only coefficient must not be equated with the true physical effectiveness.

## B747: full loss of left outer engine at 30 s

90 s at dt=0.02 s; initial roll 0.3°, heading 1°; both commands are zero. Native `EngineFailureEvent` physics removes one engine contribution and creates the corresponding yaw moment. At fixed throttle/flight condition, total thrust becomes 75% of the healthy value. The engine event is applied at the start of its physical interval using the native model scheduler, which also splits off-grid events exactly.

All four controllers use the same ±8° lateral limits, 20°/s slew limit, initial condition and measured-state longitudinal hold. PID/LQR/LQI candidates are ranked on a separate healthy 60 s run (1° roll, 2° heading initial error). Seven PID candidates and five weight configurations each for LQR and LQI are shown in the notebook. This is a finite candidate search, not a claim of optimal baseline design.

AA-INDI starts from nominal healthy surface derivatives; its observer and identifier run continuously. Its application tuning is outer gain 0.15 1/s, rate gain 0.5 1/s, acceleration filter 5 Hz, parameter covariance 0.001, and angular HOSM drift scales 1e-4 rad. The smaller angular scale reduces finite-step HOSM chattering in this ideal-sensor experiment. Sensor fault identification is not assessed.

Post-fault window: (30,90] s. Combined RMSE is sqrt(mean(roll²+heading²)); combined IAE is dt*sum(|roll|+|heading|). Recovery requires both angles to stay within ±0.05° to the end of the run.

| Controller | Roll RMSE [deg] | Heading RMSE [deg] | Combined RMSE [deg] | IAE [deg·s] | Recovery [s] | Surface RMS [deg] | Surface variation [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|
| AA-INDI | 0.018095 | 0.026916 | 0.032433 | 2.03693 | 8.68 | 3.0300 | 16.8900 |
| PID | 0.116834 | 1.079879 | 1.086181 | 67.48585 | Not reached | 3.1658 | 19.6881 |
| LQR | 0.157841 | 0.323500 | 0.359953 | 26.10714 | Not reached | 2.1685 | 9.7475 |
| LQI | 0.218396 | 0.206304 | 0.300430 | 24.43503 | Not reached | 2.3486 | 11.2320 |

AA-INDI reduces combined post-fault RMSE by 97.0% relative to PID, 91.0% relative to LQR, 89.2% relative to LQI. It uses more surface RMS and total variation than LQR/LQI in this case. This is a tracking advantage with an explicit control-effort tradeoff, not dominance on every metric.

## Additional cases without retuning

| Case | Controller | Combined RMSE [deg] | IAE [deg·s] | Recovery [s] | Final roll [deg] | Final heading [deg] |
|---|---|---:|---:|---:|---:|---:|
| Right outer engine out at 20 s | AA-INDI | 0.019028 | 1.07074 | 5.72 | 0.001866 | -0.005141 |
| Right outer engine out at 20 s | PID | 0.978606 | 72.03219 | Not reached | -0.128868 | 0.779582 |
| Right outer engine out at 20 s | LQR | 0.389380 | 33.77799 | Not reached | -0.240722 | 0.402397 |
| Right outer engine out at 20 s | LQI | 0.188419 | 14.97806 | Not reached | -0.074955 | 0.010635 |
| Half thrust on engine 1 at 45 s | AA-INDI | 0.013813 | 0.71294 | 0.02 | 0.002413 | -0.000838 |
| Half thrust on engine 1 at 45 s | PID | 0.495463 | 23.19187 | Not reached | 0.076202 | -0.592124 |
| Half thrust on engine 1 at 45 s | LQR | 0.138020 | 7.50692 | Not reached | 0.074458 | -0.165683 |
| Half thrust on engine 1 at 45 s | LQI | 0.161126 | 9.86384 | Not reached | 0.117014 | -0.066487 |
| 500 s, left outer engine out | AA-INDI | 0.011691 | 2.58084 | 8.68 | 0.000096 | -0.000009 |
| 500 s, left outer engine out | PID | 0.472303 | 132.07947 | 186.10 | -0.000334 | 0.001436 |
| 500 s, left outer engine out | LQR | 0.418473 | 262.18788 | Not reached | 0.197511 | -0.376866 |
| 500 s, left outer engine out | LQI | 0.113656 | 33.57150 | 88.34 | 0.000266 | -0.000179 |

All 12 additional runs reached their requested horizon. The 500 s run contains 25,000 controller/physics transitions per controller. AA-INDI recovers sooner; PID and LQI also converge on the longer interval, so their missing recovery in the 90 s experiment must not be read as instability. LQR retains a constant disturbance offset.

## Physical and numerical checks

- Before the event, healthy and faulty runs have exactly matching states/actions for each B747 controller; no event information is fed to any policy.
- B737 tests cover zero/unit effectiveness, the encoder/aerodynamic distinction, off-grid event splitting, endpoint causality, reset reproducibility and SI accelerometer consistency.
- B747 tests cover engine moment direction, specific-force/kinematic consistency, identical actuator limits, nominal LQR/LQI closed-loop eigenvalues and RK4 substep refinement.
- The broader example/aircraft regression run passed 134 tests. After the final observer tuning, all 25 directly affected tests passed again.
- The notebook loops reject non-finite states, flight-envelope exits and incomplete episodes rather than silently assessing a partial trajectory.

## Interpretation limits

The B737-800 model reuses B737-100 aerodynamic derivatives with changed geometry/inertia. Both aircraft use simplified aerodynamics, ideal sensors and zero-order-held surfaces; servo lag and engine spool dynamics are absent. These scenarios support the reported simulator results, not flight qualification.

The engine failure changes propulsion and airspeed, not the physical aileron/rudder derivatives. The AA-INDI control-only moment regression omits other aerodynamic and propulsion terms, so fitted coefficients can drift. The observed advantage belongs to the complete acceleration-feedback/observer/adaptive controller; it does not isolate online parameter learning as its cause. No controller is frozen at a known failure time.

## SDK extraction verification

The examples now import only public `tensoraerospace` components. The B737 support
modules were removed. Shared capabilities live in nonlinear model analysis,
`FlightMeasurement.from_model`, `AircraftGeometry.from_parameters`, SDK PID/LQR
controllers and `ControlBenchmark`; named aircraft benchmark protocols compose
these components. The companion B747 Python file is a short executable example.

After extraction, all eight 90 s B747 trajectories were compared against arrays
saved before the refactor. AA-INDI, LQR and LQI states and actions are bitwise
identical. PID differs by at most 5.69e-14 in state and 4.17e-17 in actions; its
reported metrics are unchanged at the displayed precision. The B737 healthy
AA-INDI run retains post-step RMSE 0.1354742997 degrees, final error -0.0050902798
degrees, altitude change +397.7534333 ft and speed change -18.8399339 ft/s.

Six notebooks were executed with their plots and tables refreshed: four B737
healthy/fault notebooks, the B747 comparison and cookbook recipe 09. The installed
wheel was also exercised from `/tmp`, with a Python import hook explicitly
rejecting any `example` import: B737 analysis/sensors and all four B747 controllers
completed successfully. The targeted regression suite passed 314 tests, covering
model physics, causal faults, SI sensors, PID/LQR behavior, adaptive-agent contracts,
benchmark metrics and the existing ET-DHP examples. The additional 500 s scenarios
above are the previously recorded validation; this extraction's direct before/after
trajectory comparison uses the paired 90 s runs.
