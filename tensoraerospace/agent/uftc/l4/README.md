<!-- tensoraerospace/agent/uftc/l4/README.md -->
# UFTC L4 — Distributional SAC outer-loop planner

Phase 3 component. Provides:

- `QRDistCritic` (twin) + `qr_huber_loss`
- `GaussianActor` squashed with reparameterisation
- `cvar_alpha_fn` and `risk_gate(z, severity, alarm)`
- `PrioritizedReplay` carrying FDD/monitor metadata
- `DSACOuter` orchestrator with `freeze_learning`/`degrade_reference_to_hold` macro-action sinks
- `LongitudinalTrimFreeWrapper` for adaptive longitudinal references

## Wiring into UFTCController

```python
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController

ctl = UFTCController(
    n_state=4, n_control=2,
    config=UFTCConfig(
        enable_l4_outer=True,
        l4_n_ref_dim=4,
        l4_action_scale=0.05,
        l4_eval_mode=True,
    ),
)
```

## Offline training

See `example/reinforcement_learning/uftc/train_dsac_offline.py` for a
minimal curriculum that reproduces the spec's pre-training pipeline.
The example is reduced to ~5 000 steps so it completes in minutes; the
real workflow uses 200 000 steps and a full damage-preset mix.
