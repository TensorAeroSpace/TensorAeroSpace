<!-- tensoraerospace/agent/uftc/l1/README.md -->
# UFTC L1 — HJ-Reachability safety shield

Phase 2 component of the UFTC cascade. Provides:

- `HJValueFunction` protocol + `DeepReachValueFn` torch backend.
- `power_iteration_lipschitz` upper bound used by `ConformalMargin`.
- `ConformalMargin` translating FDD severity into the runtime margin εₜ.
- `ValueBank` with worst-case fallback for open-world FDD.
- `HJReachabilityShield` — QP post-filter on `u_indi`.

## Pre-training a value function

```bash
python -m tensoraerospace.agent.uftc.l1.deepreach_train \
    --plant f16-nonlinear-angular --mode nominal --epochs 200 \
    --out artifacts/v_hj/nominal/
```

The CLI wraps `train_value_fn` with argparse; the same call is available
programmatically. See `example/uftc/pretrain_hj_value.py` for a runnable
script that produces a `nominal/` artifact directory with `value_fn.pt`
and `value_fn.json`.

## Wiring into UFTCController

```python
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController

ctl = UFTCController(
    n_state=4, n_control=2,
    config=UFTCConfig(
        enable_l1_shield=True,
        enable_glr=True,
        l1_value_fn_path="artifacts/v_hj/nominal/value_fn.pt",
        l1_u_min=[-1.0, -1.0],
        l1_u_max=[+1.0, +1.0],
    ),
)
```

If `l1_value_fn_path` is `None`, the shield uses an `_Identity` value
function and never intervenes — the QP path is exercised only after a
real artifact is loaded.

## GLR detector

The GLR detector is enabled by `UFTCConfig.enable_glr=True`. It reads the
same Kalman innovations as the CUSUM detector but flags slow-drift
faults via a sliding-window log-likelihood-ratio test. See
`docs/superpowers/specs/2026-05-08-uftc-l1-hjshield-and-glr-design.md`
for the math.
