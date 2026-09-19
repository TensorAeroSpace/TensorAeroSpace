# iADP and AA-INDI: replacement implementations and validation

Date: 2026-09-19. Branch: `fix/agent-runtime-bugs`.

## Outcome

There is now **one implementation per algorithm**. AA-INDI's rate-only
controller and reintegration bias heuristic were removed. iADP's ridge,
PSD projection, critic blending and alternate policy pseudoinverse were
removed. Existing AIDI work was preserved; its still-needed low-pass
measurement differentiator was moved into the AIDI package.

This implements the published algorithm blocks with documented initialization,
tuning and observer-coordinate choices. It does **not** establish identical
behavior to the authors' unpublished flight-control software or universal
closed-loop convergence. Long-horizon failures are included below.

[Machine-readable results](adaptive-paper-implementation.json).
[Earlier audit](adaptive-paper-conformance.md) describes the previous code.

## Sources and equation mapping

- [Konatala et al., AIAA 2024-2402](https://doi.org/10.2514/6.2024-2402):
  [official full text](https://research.tudelft.nl/files/173498220/konatala_et_al_2024_flight_testing_reinforcement_learning_based_online_adaptive_flight_control_laws_on_cs_25_class.pdf).
- [Atmaca et al., AIAA 2026-1743](https://doi.org/10.2514/6.2026-1743):
  [official full text](https://repository.tudelft.nl/file/File_ee9931f5-cf45-45a5-b5a3-0225b0f35da2).
- [Atmaca et al., JGCD 2025, 10.2514/1.G009147](https://doi.org/10.2514/1.G009147):
  [detailed observer equations](https://pure.tudelft.nl/ws/portalfiles/portal/249563022/atmaca-et-al-2025-online-inertial-measurement-unit-fault-identification-and-active-fault-tolerant-flight-control.pdf).

| Component | Source location | Current code / check |
| --- | --- | --- |
| Independent plant/reference dimensions, output tracking cost | Konatala (5)–(6) | `IADPConfig.n_reference`, `output_matrix`, `reference_output_matrix`; unequal-dimension and checkpoint tests |
| Incremental identification and model-based Bellman target | Konatala (9)–(10), Fig. 2 | Fixed-forgetting RLS, actual actuator feedback, correctly paired state increments |
| Unregularized critic and policy improvement | Konatala Fig. 2, (11) | SVD batch least squares, symmetric kernel; direct linear solve for the control increment |
| CLA/SLA scheduling | Konatala III.B | Initial identification; continuous adaptation or explicit sequential model/training/assessment phases |
| Non-exact aircraft kinematics and independent navigation | Atmaca 2026 III.A | Body/NED transformation, RK4 kinematics, independent velocity/attitude packets |
| Two-stage Kalman algebra | Atmaca 2025 (20)–(42) | Conditional covariance factorization; independent augmented-Kalman oracle |
| HOSM differentiator | Atmaca 2025 (47)–(50) | Simultaneous four-state update; published powers 3/4, 2/3, 1/2 and sign |
| Physical moment reconstruction and inversion | Atmaca 2026 (7)–(16) | Full inertia tensor and gyroscopic term; dynamic-pressure scaling of the identified derivatives |
| Per-axis surface-derivative estimation | Atmaca 2026 (50)–(57) | Absolute actual surfaces, matched moment/input filters, three scalar VFF-RLS recursions |

### Explicit implementation choices

The iADP paper does not publish its complete initial kernel. The default here
is a positive tracking-shaped seed, coupling the plant and reference outputs.
The `paper()` factory supplies 20 s identification, a 20 s critic window,
1 kHz control/model updates and 20 Hz critic updates. Its default is CLA;
adaptation does not stop when a simulated fault occurs. SLA remains an explicit
published experiment option, with default critic fitting in the 55–60 s interval.
Direct `IADPConfig` construction changes settings, not the underlying update law.

**Observer drift interpretation:** the observer article calls its second
coordinate state drift while propagating it through an integrated input-noise
matrix. The two-stage implementation uses a dimensionally explicit input-bias
coordinate. HOSM differentiates accumulated physical state drift—the integrated
non-exact kinematic trajectory minus the navigation-corrected state—and divides
by the primary kinematic channel gains. This is an engineering interpretation,
not a verified match to unpublished author code. Differentiating an already
rate-valued constant input bias would incorrectly return zero fault.

Observer process-noise tuning, HOSM gains/scales and the initial identification
covariance remain explicit settings. The papers do not provide a complete
executable tuning set. The primary AA-INDI API accepts virtual angular
acceleration. Its optional proportional rate adapter is not the Flying-V
C*/roll/sideslip guidance system.

## Public interface and migration

```python
from tensoraerospace.agent.iadp import IADPAgent, IADPConfig
from tensoraerospace.agent.aa_indi import AAINDIAgent, AAINDIConfig

# iADP: one update law; paper() is an experiment-schedule convenience.
iadp = IADPAgent(n_state, n_control, IADPConfig.paper(excitation_signal=excitation))

# AA-INDI: geometry, derivatives and real measurement packets are required.
aaindi = AAINDIAgent(AAINDIConfig(geometry, nominal_derivatives))
command = aaindi.predict_acceleration(measurement, virtual_acceleration)
# Apply command and collect the next FlightMeasurement.
metrics = aaindi.learn(next_measurement)
```

AA-INDI receives body rates, accelerometer specific force, independent NED
velocity and Euler attitude, actual surface positions, airspeed and density.
Units are SI; actuator positions are radians. Supplying attitude integrated
from the same faulty gyro does not create independent information. Initial
navigation is mandatory; later packets may omit slower-rate channels.

Old rate-only AA-INDI checkpoints lack the required state and geometry.
iADP configurations containing removed critic modifiers also require migration;
there is no hidden compatibility implementation. New checkpoints preserve
observer and controller history, including a pending transition.

## Independent numerical and physical checks

The scoped suite passed **3169 tests** after legacy removal:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest \
  tests/agent tests/agents tests/envs tests/aerospacemodel tests/examples \
  tests/mpc tests/scripts/test_benchmark_aidi.py \
  -p pytest_mock -p pytest_timeout -o addopts= \
  --strict-markers --strict-config --import-mode=importlib -q
```

Tests for the removed heuristic and critic modifiers were replaced or removed;
this count is not a coverage percentage. Meaningful checks include:

- Two-stage mean, covariance and cross-covariance agree with an independent
  augmented Kalman filter over 100 random updates, including correlated noises.
- Known accelerometer and gyro offsets are reconstructed from independent
  navigation; HOSM is checked against the published simultaneous update and a ramp.
- Reconstructed moments obey rotational energy balance with off-diagonal inertia;
  doubling airspeed multiplies control effectiveness by four.
- Independent VFF/RLS fits recover known surface derivatives; nonzero initial
  surfaces do not create an artificial first-sample effectiveness loss.
- CLA/SLA phase behavior, unequal reference dimensions and exact save/load
  continuation are covered.
- The LAPAN LQR oracle gives relative kernel error **9.62e-16** and feedback-gain
  error **1.89e-14**. This is an algebraic fixed-point check with privileged model
  knowledge, not a trained flight policy.

## Closed-loop results

### iADP: 80-second known-model trial

The exact discrete plant represents `rate_dot = -2*rate + gain*input`, with
30% effectiveness loss at 60 s. Identification starts without an injected
true model; the agent receives initial and ongoing excitation, not fault time.

| Metric | Result |
| --- | ---: |
| Tracking RMSE, 40–60 s | 2.762e-5 rad/s |
| Tracking RMSE, 65–80 s | 1.153e-5 rad/s |
| Final identified input gain | 0.000699300466138 |
| True damaged input gain | 0.000699300466433 |
| RLS updates | 79,999 |
| Final minimum critic eigenvalue | 0.007975 |

The two RMSE windows have different adaptation histories; their ordering does
not mean damage improves the plant.

### iADP: native nonlinear F-16

Seed 17, 20 s healthy training, then independent 30 s evaluation episodes.
The fault reduces the command's deviation from trim by 50% at 15 s. This is
an actuator-command fault, not a changed aerodynamic table. The servo and
nonlinear dynamics are unchanged. Actual mean surface motion is fed back.

| Evaluation | Continuous adaptation RMSE | Fixed-parameter diagnostic RMSE |
| --- | ---: | ---: |
| Healthy | 0.2093 deg/s | 0.2568 deg/s |
| Fault | 0.1616 deg/s | 0.1850 deg/s |

All five runs completed without a numerical failure, sign reversal of the
identified pitch-control gain, or violation of the demonstration envelope.
These short runs do not establish long-horizon stability. The fixed-parameter
arms are diagnostic baselines, not the deployed adaptation strategy.

### AA-INDI: 60-second rigid-body trial

Analytic three-axis Euler rotational dynamics, independent navigation, noisy
IMU, GPS 10 Hz and controller 100 Hz. At 20 s the pitch actuator loses 30%
effectiveness and gyro q acquires +0.02 rad/s bias. The controller never receives
those values or that time. Actual input is limited to 25 degrees and 180 deg/s.

| Metric, 30–60 s | Correction enabled | Correction disabled |
| --- | ---: | ---: |
| Pitch-rate tracking RMSE, seed 17 | 0.005620 rad/s | 0.020662 rad/s |
| Gyro q fault-estimate RMSE, seed 17 | 0.000581 rad/s | 0.000609 rad/s |
| Pitch-rate tracking RMSE, seed 23 | 0.005620 rad/s | — |

The ablation still runs the observer but does not apply its output. Pitch
tracking improves by about 73% in this particular comparison. No actuator
amplitude saturation occurred in these 180 deg/s runs; all 6000 identifier
updates remained active. **Parameter convergence is incomplete:** seed 17's
final pitch derivative is 0.01636 against the true 0.021875, with errors in
other derivatives too. Tracking success is not a parameter-identification proof.

## Failures and limits retained in the results

- With the same AA-INDI trial at **60 deg/s**, 10 Hz filtering and rate gain 4,
  the observer/controller diverged; the covariance check stopped it at 28.58 s.
  Other tested 60 deg/s settings stopped at 18.32 s (2 Hz, gain 4) and 8.90 s
  (5 Hz, gain 2), including before the injected fault. This is an unresolved
  closed-loop robustness limit, not a successful fault-recovery result.
- The former scalar F-16 500-second example, now using the published critic,
  left its demonstration flight envelope at step 4461 (**89.24 s**).
  Old softened-critic tuning cannot be assumed valid after removing that update.
  The updated integral-feature runs stopped even earlier: **8.36 s of valid
  trajectory**, with the envelope guard firing at the next step, before either
  fault. Healthy and fault-labelled runs coincide until that point. Both
  long-horizon notebooks save these partial trajectories and explicit failure
  status. These settings are not a usable 500-second tracking controller.
- AA-INDI's physical-coordinate observer interpretation and application-specific
  outer loop prevent a claim of exact reproduction of the authors' software.
  Real aircraft validation needs independent sensor data, correct geometry,
  excitation, actuator timing and gains appropriate for the intended envelope.

## Examples and reproduction

Executed notebooks with embedded plots:

- [AA-INDI: sensor and actuator faults](../example/reinforcement_learning/incremental_adp/example_aaindi_sensor_actuator_faults.ipynb).
- [iADP: published update and continuous learning](../example/reinforcement_learning/incremental_adp/example_iadp_paper.ipynb).
- [F-16 command-fault long-horizon diagnostic](../example/reinforcement_learning/incremental_adp/example_iadp_small_fault_f16.ipynb).
- [F-16 aerodynamic-fault long-horizon diagnostic](../example/reinforcement_learning/incremental_adp/example_iadp_aero_effectiveness_f16.ipynb).

The old generic notebook paths contain migration links. The old rate-only
head-to-head comparison was removed; the new AA-INDI and scalar iADP trials use
different plants and cannot be used to rank algorithms against each other.

```bash
.venv/bin/python scripts/validate_paper_adaptive.py --agent iadp --trace --output /tmp/iadp.json
.venv/bin/python scripts/validate_paper_adaptive.py --agent aaindi --trace --output /tmp/aaindi.json
.venv/bin/python scripts/validate_paper_adaptive.py --agent aaindi --no-sensor-correction --output /tmp/aaindi-ablation.json
.venv/bin/python scripts/validate_paper_adaptive.py --agent aaindi --rate-limit-deg 60 --output /tmp/aaindi-rate60.json
.venv/bin/python scripts/validate_adaptive_f16.py --repo . --agent iadp --seed 17 --train-duration 20 --eval-duration 30 --output /tmp/iadp-f16.json
```
