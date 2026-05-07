"""Demo: UFTC on F-16 nonlinear angular with a wing-tip strike at t=0.

Run:
    poetry run python example/reinforcement_learning/uftc/uftc_f16_damage_demo.py

Outputs CSV-style trace of FDD severity, omega state norm, and RLS
forgetting factor per ~25-step interval to stdout. Does not require
matplotlib — pipe the printed columns to a separate plotting tool if
desired.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.agent.uftc.fdd.detector import FDDConfig
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


_LINEAR_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "tensoraerospace", "aerospacemodel", "f16", "linear", "data",
)


def _f16_nominal_matrices(dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise the [alpha, beta, p] sub-block of the shipped F-16 linear model."""
    A_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "A.mat"))["A_lo"]
    B_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "B.mat"))["B_lo"]
    idx = [7, 8, 9]
    A_sub = A_cont[np.ix_(idx, idx)]
    B_sub = B_cont[idx, :]
    Ad, Bd, _, _, _ = cont2discrete(
        (A_sub, B_sub, np.eye(3), np.zeros((3, 4))), dt=dt
    )
    return Ad, Bd


def main() -> None:
    n_steps = 1500
    Ad, Bd = _f16_nominal_matrices(dt=0.01)
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=WING_STRIKE_LEFT_TIP,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=Ad - np.eye(3),
        nominal_G=Bd,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=400,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            trust_radius_fault=0.7,
            fdd_cfg=FDDConfig(
                process_noise=1e-6,
                measurement_noise=1e-5,
                adapt_Q=False, adapt_R=False,
                h_alarm=15.0,
            ),
        ),
    )
    ref = np.zeros(3)
    print("# t,omega_norm,severity,fault_present,rls_gamma,mode")
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        if k % 25 == 0:
            d = ctl.diagnostics()
            print(
                f"{k * 0.01:.2f},"
                f"{np.linalg.norm(obs[:3]):.4f},"
                f"{d['severity']:.3f},"
                f"{int(d['fault_present'])},"
                f"{d['rls_gamma']:.4f},"
                f"{d['mode']}"
            )
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
