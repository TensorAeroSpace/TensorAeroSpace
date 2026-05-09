"""Smoke offline-training script for L4 DSAC on F-16 angular dynamics.

Uses a small budget so it completes in well under a minute. Real
trainings should bump ``--steps`` to 200 000 and the curriculum mix to
all 7 damage presets.

Note on the env: the design spec mentions a ``LongitudinalF16`` wrapper
that does not currently exist; this script uses
``NonlinearAngularF16`` with the ``ENGINE_FLAMEOUT`` damage preset,
which is the same env the Phase 1+2 F-16 regression tests use. Pre-
training across all damage presets follows the same pattern.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import ENGINE_FLAMEOUT
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController
from tensoraerospace.agent.uftc.fdd.detector import FDDConfig
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

_LINEAR_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../tensoraerospace/aerospacemodel/f16/linear/data",
)


def _f16_nominal_matrices(dt: float = 0.01):
    A_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "A.mat"))["A_lo"]
    B_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "B.mat"))["B_lo"]
    idx = [7, 8, 9]
    A_sub = A_cont[np.ix_(idx, idx)]
    B_sub = B_cont[idx, :]
    Ad, Bd, _, _, _ = cont2discrete((A_sub, B_sub, np.eye(3), np.zeros((3, 4))), dt=dt)
    return Ad, Bd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--out", type=str, default="artifacts/dsac/v1")
    args = parser.parse_args()

    Ad, Bd = _f16_nominal_matrices(dt=0.01)
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=args.steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=ENGINE_FLAMEOUT,
        split_stab=True,
    )
    ctl = UFTCController(
        n_state=3,
        n_control=4,
        nominal_F=Ad - np.eye(3),
        nominal_G=Bd,
        config=UFTCConfig(
            dt=0.01,
            fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            fdd_cfg=FDDConfig(
                process_noise=1e-6,
                measurement_noise=1e-5,
                adapt_Q=False,
                adapt_R=False,
                h_alarm=15.0,
            ),
            enable_l1_shield=False,
            enable_glr=True,
            enable_l4_outer=True,
            l4_n_ref_dim=3,
            l4_eval_mode=False,  # online learning during offline-training
            l4_replay_capacity=10_000,
            l4_batch_size=128,
            l4_seed=0,
        ),
    )
    obs, _ = env.reset()
    target = np.zeros(3)
    for k in range(int(args.steps)):
        u = ctl.predict(obs[:3], target, time_step=k)
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], target, time_step=k)
        if k % 1000 == 0:
            print(f"step {k:6d} — beta_t={ctl._last_beta:.3f}")
        if terminated or truncated:
            print(f"env terminated/truncated at step {k}")
            break
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctl.l4.save(args.out)
    print(f"saved DSAC weights to {args.out}")


if __name__ == "__main__":
    main()
