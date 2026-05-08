# example/uftc/pretrain_hj_value.py
"""Pre-train a DeepReach V_θ on a toy double-integrator and save it.

Real F-16 trainings live in dedicated workflows under
``example/reinforcement_learning/uftc/``; this script is a runnable
reference that completes in seconds and exercises the save/load path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tensoraerospace.agent.uftc.l1.deepreach_train import (
    TrainingConfig,
    train_value_fn,
)
from tensoraerospace.agent.uftc.l1.value_fn import DeepReachConfig


def main(out_dir: str = "artifacts/v_hj/double_integrator") -> None:
    cfg_v = DeepReachConfig(
        n_state=2, hidden_sizes=(32, 32),
        state_bounds=[[-2.0, 2.0], [-2.0, 2.0]],
        time_horizon=1.0,
    )
    train_cfg = TrainingConfig(
        epochs=50, batch_size=512, lr=1e-3, n_state=2, n_control=1,
        u_low=np.array([-1.0]), u_high=np.array([1.0]),
        seed=0,
    )
    fn, history = train_value_fn(
        cfg_v, train_cfg,
        dynamics=lambda x, u: np.array([x[1], u[0]]),
        safe_set=lambda x: 1.0 - max(abs(x[0]), abs(x[1])),
    )
    print(f"final loss = {history['loss'][-1]:.4e}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn.save(out / "value_fn.pt")
    print(f"saved to {out / 'value_fn.pt'}")


if __name__ == "__main__":
    main()
