<!-- tensoraerospace/agent/uftc/monitor/README.md -->
# UFTC monitor — Composite Lyapunov runtime monitor

Phase 4 component. Provides:

- 5-component `VState` and `MonitorOutput` dataclasses
- `CompositeLyapunovMonitor` reading `V_HJ + V_INDI + V_iADP + V_DSAC + V_FDD`
- `AlarmStateMachine` with hysteresis and cooldown
- `MacroActionDispatcher` (Variant B advisory) calling `force_rls_reset`,
  `freeze_l4_learning`, `degrade_reference_to_hold`, `request_actuator_hold`
- `run_certificate(cfg, rollouts)` numerical certificate of Lemma 4.1

## Wiring into UFTCController

```python
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController

ctl = UFTCController(
    n_state=4, n_control=2,
    config=UFTCConfig(
        enable_l1_shield=True,
        enable_l4_outer=True,
        enable_monitor=True,
    ),
)
```

The monitor is always passive: it only writes to layer state via
explicit macro-action methods. Failures inside `monitor.step` or
`dispatcher.dispatch` are caught at the controller boundary and
logged; control loop is unaffected.

## Numerical certificate

```bash
python -m tensoraerospace.agent.uftc.monitor.certificate \
    --config artifacts/uftc/cfg.json \
    --rollouts artifacts/uftc/cert_rollouts.npz \
    --report artifacts/uftc/uub_certificate.json
```

`cfg.json` carries `c_weights`, `a_diag`, `eps_matrix`, `d_disturbance`
(same shapes as `MonitorConfig`). `cert_rollouts.npz` is a dict of
preset → 2D array of `V_total` over time per trajectory.
