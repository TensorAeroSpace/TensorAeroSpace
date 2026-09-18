# ET-DHP versus PID: nonlinear B747

## Tuned result

The main notebook starts with the final `TUNED` profile. For 500-second step-response
measurements on the damaged aircraft:

| Settling time within ±2% | Tuned ET-DHP | Existing PID |
|---|---:|---:|
| Heading +1° | 60.90 s | 209.25 s |
| Roll +0.5° | 2.60 s | 166.10 s |

The integrated normalized error across all four references decreased by 36.55%
for the heading step and 67.26% for the roll step. Tuning covers Q/R, event thresholds
and learning parameters. Rescaling the common cost improves numerical conditioning
without changing the initial ideal LQR policy. Actor and critic learning remains
active throughout the flight.

Validation includes 34 final trajectories, 181 targeted tests, a 25 ms physics step
and an equation-level comparison with the paper. The executed notebook contains
17 figures including references, coupled errors, commands and update counts.
Speed and altitude share the same longitudinal controller; their own step responses
still need improvement.

```bash
python -m example.reinforcement_learning.incremental_adp.example_etdhp_tuned_b747 standard
python -m example.reinforcement_learning.incremental_adp.example_etdhp_tuned_b747 held_out
```

Full history, failed candidates and limitations: `reports/etdhp-b747-tuning-validation.md`.
Paper comparison: `reports/etdhp-b747-paper-comparison.md`.

## Original configuration before tuning

500 seconds; engine 1 loses 50% of its own thrust at 137 s. Only the environment knows the schedule. Both lateral controllers share the same observation-only longitudinal PI/PD hold.

![ET-DHP and PID on B747](../../../../assets/images/etdhp_vs_pid_b747.svg)

| Post-event metric | ET-DHP | PID |
|---|---:|---:|
| Heading RMSE, deg | **0.030445** | 0.222648 |
| Bank RMSE, deg | 0.037076 | **0.026442** |
| Heading RMSE in the last 100 s, deg | **0.002422** | 0.004673 |

ET-DHP tracks heading more accurately; PID holds bank more accurately. PID is more accurate late in the healthy episode, and can also be more accurate in the last 100 s after an early or complete engine failure.

**Initialization and learning.** The ET-DHP networks start from a healthy local LQR initialization. Its nominal plant model is unchanged throughout, while actor and critic keep learning at state-triggered events: 851 updates, 554 after the fault. Nothing switches or disables learning at the fault time. These results do not isolate online learning from initialization and integral feedback.

PID was tuned only on healthy data, with 150+180 objective evaluations. The searches exhausted their budgets; no globally optimal PID is claimed. Both controllers have identical ±8 deg aileron/rudder bounds.

**The separately tuned online-model variant deteriorates:** post-event heading RMSE is 3.452915 deg, reaching 6.577557 deg in the last 100 s. The notebook includes that result. The plant uses nonlinear 6-DoF equations with local aerodynamic derivatives; actuator lag, engine spool dynamics, wind and sensor noise are absent.

```bash
python -m example.reinforcement_learning.incremental_adp.example_etdhp_vs_pid_b747 --online-model-comparison
```

Executed notebook: `example/reinforcement_learning/incremental_adp/example_etdhp_vs_pid_b747.ipynb`. Full protocol and checks: `reports/etdhp-b747-pid-validation.md` and adjacent JSON. 153 targeted tests passed.

The implementation was checked against [Sun et al., CEAS EuroGNC 2022](https://eurognc.ceas.org/archive/EuroGNC2022/pdf/CEAS-GNC-2022-075.pdf). B747, integral augmentation and local LQR initialization are extensions in this example.
