"""F-16 engine flameout — gradual degradation; aircraft must stay bounded.

Engine flameout reduces thrust gradually rather than abruptly. UFTC's
pure-Mahalanobis CUSUM may or may not fire on this signature — we do
not assert detection, only behavioural success (no divergence).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    ENGINE_FLAMEOUT,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.agent.uftc.fdd.detector import FDDConfig
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

_LINEAR_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../tensoraerospace/aerospacemodel/f16/linear/data",
)


def _f16_nominal_matrices(dt: float = 0.01):
    """Return (Ad_3x3, Bd_3x4) discretised at *dt* for the [alpha, beta, p] block."""
    A_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "A.mat"))["A_lo"]
    B_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "B.mat"))["B_lo"]
    idx = [7, 8, 9]
    A_sub = A_cont[np.ix_(idx, idx)]
    B_sub = B_cont[idx, :]
    Ad, Bd, _, _, _ = cont2discrete((A_sub, B_sub, np.eye(3), np.zeros((3, 4))), dt=dt)
    return Ad, Bd


@pytest.mark.slow
def test_engine_flameout_handled_without_divergence() -> None:
    n_steps = 1500
    Ad, Bd = _f16_nominal_matrices(dt=0.01)
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=ENGINE_FLAMEOUT,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3,
        n_control=4,
        nominal_F=Ad - np.eye(3),
        nominal_G=Bd,
        config=UFTCConfig(
            dt=0.01,
            fdd_warmup_steps=400,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            fdd_cfg=FDDConfig(
                process_noise=1e-6,
                measurement_noise=1e-5,
                adapt_Q=False,
                adapt_R=False,
                h_alarm=15.0,
            ),
        ),
    )
    ref = np.zeros(3)
    max_norm = 0.0
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        max_norm = max(max_norm, float(np.linalg.norm(obs[:3])))
        if terminated or truncated:
            break

    # Behavioural success: no divergence. Whether or not CPD fires is
    # not asserted — gradual faults are pure-Mahalanobis-CUSUM-hard.
    assert max_norm < 8.0, f"State diverged: max ‖x‖ = {max_norm:.2f}"
