# iADP and AA-INDI: implementation-to-paper audit

Date: 2026-09-19. Branch: `fix/agent-runtime-bugs`. Reviewed runtime: commit
`3f013b2`; the changes made by this audit affect documentation and tests only.

> Historical audit of the earlier implementation. The subsequent replacement is documented in [the implementation and validation report](adaptive-paper-implementation.md).

## Verdict

- **iADP: its main algebra matches the checked equations, with configurable
  numerical extensions.** Its default configuration is not a reproduction of
  the flight experiment, and finite-window online convergence is not established.
- **AA-INDI: a simplified adaptive INDI implementation, not the complete published
  Active-Adaptive INDI architecture.** Correct inversion and scalar VFF-RLS do
  not supply the missing sensor-fault reconstruction.

The pre-existing AIDI changes in this working tree are a separate task.

## Primary sources

1. Konatala, Milz, Weiser, Looye and van Kampen, *Flight Testing Reinforcement
   Learning based Online Adaptive Flight Control Laws on CS-25 Class Aircraft*,
   AIAA 2024-2402:
   [official full text](https://research.tudelft.nl/files/173498220/konatala_et_al_2024_flight_testing_reinforcement_learning_based_online_adaptive_flight_control_laws_on_cs_25_class.pdf),
   [DOI](https://doi.org/10.2514/6.2024-2402).
2. Atmaca, de Visser and van Kampen, *Active Incremental Nonlinear Dynamic
   Inversion for Sensor and Actuator Fault-Tolerant Control*, AIAA 2026-1743:
   [official full text](https://repository.tudelft.nl/file/File_ee9931f5-cf45-45a5-b5a3-0225b0f35da2),
   [DOI](https://doi.org/10.2514/6.2026-1743).

Equation numbers below refer to these sources. Interpretations of library
behavior come from source inspection and the executable checks listed below.

## iADP equation mapping

| Source location | Implementation | Assessment |
| --- | --- | --- |
| Eqs. (5)–(6): cost and augmented state | `IADPAgent._augment`, `learn` | Restricted API: equal-size observation/reference, cost on their difference. No independent output map or reference-state dimension. |
| Eq. (9), Fig. 2: incremental model and fixed-forgetting RLS | `IncrementalRLS.update`, `IADPAgent.learn` | Correct transition pairing: [dX_t, du_t] predicts dX_(t+1). Applied action is supported. Joseph covariance agrees with an independent weighted batch solve. |
| Eq. (10), Fig. 2: model-based Bellman target | `learn`, `_policy_evaluation` | Uses the model-predicted next state. The first transition uses the measurement because a previous state increment is unavailable. |
| Eq. (11): policy improvement | `_compute_policy_increment` | Finite-difference cost minimization passes for coupled multi-input data. Pseudoinverse and actuator clamps extend the unconstrained formula. |
| Fig. 2: least-squares critic | `_policy_evaluation` | Zero-ridge SVD passes an exact LQR fixed-point oracle. Ridge, eigenvalue projection and blending change the fitted update. |
| Section III.B: learning phases | `model_learning_only_steps`, scheduling | Both published approaches have initial open-loop identification; defaults skip it. The option alone does not implement complete SLA. |

Code: [model](../tensoraerospace/agent/iadp/model.py),
[identifier](../tensoraerospace/agent/iadp/rls.py).

### Material differences and limitations

1. **Unexcited cold start.** With default G=0, previous input zero and no
   excitation, the controller stays at u=0 on a stable scalar plant despite
   reference=1. After 200 transitions, G and the plant state are still zero.
   This is an initialization/information problem, not a wrong sign in Eq. (11).
   Identity P does not fix it. The same zero-control result occurs with exact
   F=diag(0.9, 1), G=[0.1, 0]^T: P=I has no plant/reference coupling,
   and unexcited data cannot learn that coupling. A squared-tracking-error
   P immediately produces a command of the correct sign on this plant.

2. **Regularization can overwhelm the critic.** The existing LAPAN oracle was
   rerun with 300 independently sampled states from a known LQR controller.
   This checks one critic update at the exact solution, not policy training.

   | Ridge | PSD projection | Relative P error | Relative feedback-gain error |
   | --- | --- | --- | --- |
   | 0 | on | 9.51e-16 | 2.41e-14 |
   | 1e-10 | on | 1.78e-5 | 1.16e-4 |
   | 1e-4 (default) | on | 0.9411 | 1.9150 |

   The default penalty therefore changes this particular well-initialized
   critic by approximately 94%. This is a confirmed scaling sensitivity,
   not evidence that all simulations require zero regularization. With ridge
   disabled, the independent Riccati and Lyapunov solutions also agree to
   1.05e-14 relative error.
   [Complete numerical results](adaptive-paper-conformance-lqr.json).

3. **Scheduling differs.** Defaults give model updates at 100 Hz and critic
   updates at 2 Hz, with a 2-second maximum window. The paper's reported
   model/control and critic rates are 1000 Hz and 20 Hz. Changing dt requires
   reconsidering sample-based forgetting, window size and update intervals.

4. **State completeness remains the caller's responsibility.** The class
   concatenates supplied vectors; it does not reconstruct missing plant or
   reference-generator states. A scalar sine value does not encode its phase.
   It also does not provide the flight implementation's sensor preprocessing.

5. **PSD is not a stability certificate.** Projection occurs after fitting;
   P_init is only symmetrised. A user-supplied warm start must satisfy the
   value-function assumptions. The window stores historical model predictions,
   not freshly recomputed transitions under a single current policy.
   The reviewed paper does not provide enough implementation detail to classify
   that buffering choice as an equation error. Stability of the resulting
   changing-policy loop requires separate evidence.

## AA-INDI equation and architecture mapping

| Source location | Implementation | Assessment |
| --- | --- | --- |
| Eqs. (5)–(9): incremental inversion | `AAINDIAgent.predict` | Recognizable inversion core, using a pseudoinverse and actuator limits. |
| Eqs. (54)–(57): scalar VFF-RLS | `VFFRLSEstimator.update` | Gain, forgetting order and covariance agree. eps_sensitivity² represents Sigma_0. |
| Eqs. (50)–(53): surface/moment regression | `AAINDIAgent.learn` | Different model: filtered acceleration increments versus filtered input increments. No aerodynamic-moment reconstruction or moment-coefficient fit. |
| Section III.A, Fig. 1: OTSEKF-HOSM fault correction | `LowPassDerivative`, `BiasEstimator` | Missing architecture: a backward difference, low-pass filter and residual EMA do not implement these estimators. |
| Section III.C: outer guidance loops | `_advance_reference_model`, optional PI | Different interface and controller. A second-order command filter alone supplies no measured-rate error feedback. |

Code: [model](../tensoraerospace/agent/aa_indi/model.py),
[VFF-RLS](../tensoraerospace/agent/aa_indi/vff_rls.py),
[measurement helpers](../tensoraerospace/agent/aa_indi/sensor_filter.py).

### Material differences and limitations

1. **Sensor-fault tolerance is not implemented as published.** The current API
   receives angular rates, without independent ground-speed/attitude
   measurements or a kinematic process model. Its innovation is measured
   rate minus reintegration of a derivative of that same measurement. An
   additive constant cancels. A test with 200 excited transitions and a
   0.2 rad/s gyro offset produces the same bias estimate as the healthy
   trajectory within 1e-12. This is an observability limitation; changing
   the EMA forgetting factor cannot supply the missing information.

2. **The default rate loop does not remove initial tracking error.** On
   rate_dot=u with exact G=1, zero reference and initial rate 0.2 rad/s,
   default zero PI gains leave the rate at 0.2 after 5 seconds. Setting kp=2
   reduces it below 1e-4 in this test with adaptation still active. That gain
   is a counterexample aid, not a recommendation for an aircraft. Calling the
   default behavior “textbook pure INDI” or a universal 10% offset was incorrect.

3. **The identifier is a different modeling choice.** Directly fitting G can
   be useful, but this test suite does not validate moment reconstruction,
   inertia/dynamic-pressure conversion, or simultaneous sensor/actuator faults.
   Acceleration-increment identification also depends on the validity of the
   local incremental approximation.

4. **VFF extensions and tuning matter.** Multiple outputs share a covariance
   and one forgetting factor computed from the residual norm; the scalar
   paper equation does not itself establish this extension. The default
   maximum below one continuously forgets even at zero residual. A scalar
   equation-level comparison should allow maximum=1 and map Sigma_0 through
   eps_sensitivity² rather than using Sigma_0 directly.

5. **There is no protected random-model startup.** A relative pseudoinverse
   cutoff does not bound the inverse of an absolutely tiny matrix. For scalar
   G=1e-9 and a unit reference, the first command reaches the slew limit.
   A useful model initialization and a separately validated startup sequence
   are needed. The old claim that random initialization stays quiet was false.

## Changes made by this audit

- Added eleven cases in
  [adaptive_paper_scope_test.py](../tests/agents/adaptive_paper_scope_test.py):
  four independent weighted batch/RLS comparisons, excited bias invariance,
  two initial-error cases, two unexcited iADP startup cases, reference/value
  coupling and small-G inversion.
- Corrected EN/RU documentation and Python API descriptions: actual sample
  rates, initial learning phases, reference/observation restrictions, critic
  buffering, RLS time indices, outer feedback and sensor-estimator scope.
- Corrected the iADP quick-start model: F and G now include the Euler step dt,
  and the initial value couples the plant and reference. Both EN/RU loops were
  executed for 2000 steps. They are finite, but the default critic tuning still
  loses tracking: final state 6.88e-10 for reference 0.1. This is recorded as a
  limitation, not a successful tracking demonstration.
- Removed the suggestion that OTSEKF-HOSM can be added solely by subclassing a
  filter without extending measurements. Clarified that angle of attack is
  not an angular-rate input for this rate-control API.
- Numerical expressions and defaults were not changed: an AST comparison
  after removing docstrings matches HEAD for all three edited Python modules.

These are conformance checks and documentation corrections. Implementing
OTSEKF-HOSM or selecting aircraft-specific feedback gains remains separate
engineering work.

## Reproduction and results

Targeted suite: **162 passed**, including eleven new cases. The limitation tests
pass when the documented limitation is reproduced; they are not claims of
successful fault tolerance.

Run from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest \
  tests/agents/adaptive_paper_scope_test.py \
  tests/agents/adaptive_paper_regression_test.py \
  tests/agents/adaptive_actuator_regression_test.py \
  tests/agents/iadp_cost_contract_regression_test.py \
  tests/agents/iadp_window_regression_test.py \
  tests/agents/adaptive_validation_protocol_test.py \
  tests/agents/iadp_test.py tests/agents/aa_indi_test.py \
  -p pytest_mock -p pytest_timeout -o addopts= \
  --strict-markers --strict-config --import-mode=importlib -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
.venv/bin/python scripts/validate_iadp_lapan_critic.py \
  --repo . --output /tmp/adaptive-paper-conformance-iadp-lqr.json
```

The audit does not establish aircraft-wide stability, long-duration convergence
or reproduction of either paper's flight/simulation results. Those require the
missing architecture, matched experiment assumptions and separate validation.
