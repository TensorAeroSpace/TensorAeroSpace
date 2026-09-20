# Declarative step-controller tuning

## Public workflow

`ControllerTuner(env=..., env_kwargs=..., reference={state: Step(...)},
controller=..., method=..., metric="cpi").optimize(...)` owns construction,
native learning, simulation and benchmark evaluation. No user rollout, agent
factory or objective callback is required. Multiple named steps have separate
physical references and metrics. The default objective is mean normalized CPI.

Implemented profiles: AA-INDI, AIDI, iADP, IM-GDHP, IHDP, ET-DHP, HDP and MPC.
Search methods: TPE, simulated annealing, genetic (Optuna NSGA-II), random.
Profiles explicitly restrict compatible environments and reference states.
The lower-level optimizer remains available for custom research protocols.

New application boundaries:

- AIDI `predict_rates` reuses its native allocator, filtering, limits and RLS.
  Equivalence to the existing inner loop is regression-tested.
- ET-DHP `learn(applied_action=...)` optionally supplies the actually applied
  input to online model fitting; default behavior and held policy are preserved.
- Linear F16 environment forwards optional `dt` through construction and reset;
  the discrete model semigroup is checked independently.

Healthy nominal priors never read fault effectiveness. Controller updates remain
active during scored episodes. Native aircraft equations are unchanged. The
MPC profile uses nominal linear dynamics; ET-DHP pretrains its native neural
plant model before online event-driven policy updates. These are documented
application choices, not claims of universal convergence or new paper equations.

## Executed nonlinear B737 example

Source: `example/optimization/adaptive_controller_tuning.ipynb` (six executed
code cells, English narrative, stored tables and plots). EN/RU full tutorials:
`docs/{en,ru}/example/optimization/example_optimization.md`.

Protocol: healthy trim at 10,000 ft and 600 ft/s; RK4; dt=0.02 s; total 40 s;
+1 degree theta step at 20 s. AA-INDI moment identification and sensor correction
stay active. Outer attitude gain 0.35/s; desired-rate limit 3 deg/s; surface command
slew limit 20 deg/s. Search rate gain [1, 12]/s and cutoff [2, 15] Hz, both log; seed 42.
Bounds: tail maximum<=2% of step, pre-step maximum<=2%, 5%-band command settling
<=12s. Baseline (3/s,5Hz) is evaluated first.

| Metric | Baseline | Annealing, 12 trials | Genetic, 16 trials, population 4 |
|---|---:|---:|---:|
| Normalized CPI | 3.476377563 | 3.414149532 | 3.305178099 |
| Command 5% settling [s after step] | 7.74 | 7.68 | 6.38 |
| Command overshoot [%] | 0 | 0.148847 | 0 |
| Tail maximum [% of step] | 0.06236 | 0.14885 | 0.39188 |
| Pre-step maximum [% of step] | 0.03789 | 0.03962 | 0.03975 |

Selected annealing parameters: gain 2.8561394539255787/s,
cutoff 9.122522379036573 Hz. Genetic: gain 1.1552680028079323/s,
cutoff 11.454791487656054 Hz. Genetic CPI is about 4.92% below baseline and settling
is 1.36 s faster. Tail error is larger than baseline but remains below the declared
2% constraint. A lower CPI does not mean that every individual metric improves.
The two searches have different budgets; this does not rank search efficiency.

Two additional amplitude checks reuse the selected genetic parameters without
search: -1degree CPI 3.163721, settling 6.34s, tail maximum 0.1623%; +0.5degree
CPI 3.155985, settling 6.34s, tail maximum 0.1257%. Both start from fresh agents.

A simultaneous theta +0.5 degree / phi +1 degree step at 20s uses the same declarative
API. Eight TPE trials, initialized with the genetic settings, select mean
CPI 3.297747492: theta CPI 3.153512634, phi CPI 3.441982350; settling 6.36s /6.52s;
tail maxima0.1228% /0.4047%. Per-channel results and both commanded signals are
visible in the notebook. This short search has no accuracy constraints.

The archived JSON contains the full single-channel search histories. These
results are not comparable to the earlier report's different flight condition
and callback-based TPE experiment. CPI zero/global optimality is not promised.

## Validation

- 1668 tests passed in 56.62 s: optimization, agents, AIDI, affected F16 environment
  and physical-model regressions. New checks include all native learning loops,
  named multichannel timing/units, seed replay, real sampler selection,
  Metropolis acceptance/cooling, physical rejection, environment cleanup,
  actuator dt propagation, healthy fault-independent priors, and a nonfinite
  IHDP update on the final sample.
- All 35 advertised controller/environment pairs additionally completed finite
  0.3-second integration smoke runs. These check interface compatibility only,
  not useful tracking, long-horizon stability or learned-policy quality. Only
  the AA-INDI/B737 study above has full 40-second search and held-out evaluation.
- Ruff and Black checks pass for the new optimizer and affected implementation.
- Mypy: no issues in 13 checked implementation files.
- MkDocs strict EN/RU build passed; full notebook execution passed.

No new dependency is required. Reproduction used one BLAS/OpenMP thread per
process; the SDK does not alter the user's process-wide threading settings.
