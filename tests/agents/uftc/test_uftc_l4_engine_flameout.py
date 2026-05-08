"""F-16 ENGINE_FLAMEOUT with L4 eval-mode — tracking-RMS not worse than Phase 1.

The plan calls for ``LongitudinalF16(damage_profile=...)`` but that env API
does not exist; instead we follow the established ``NonlinearAngularF16``
pattern from ``test_uftc_f16_engine_flameout`` and assert that with
``l4_eval_mode=True`` (untrained actor → mean=tanh(0)=0) the L4 reference
perturbation is effectively zero, so tracking RMS matches the L4-off case
within a small tolerance.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import cont2discrete

torch = pytest.importorskip("torch")

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import ENGINE_FLAMEOUT
from tensoraerospace.agent.aa_indi.model import AAINDIConfig
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
    Ad, Bd, _, _, _ = cont2discrete(
        (A_sub, B_sub, np.eye(3), np.zeros((3, 4))), dt=dt
    )
    return Ad, Bd


def _rollout(enable_l4: bool, n_steps: int = 800) -> float:
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
        n_state=3, n_control=4,
        nominal_F=Ad - np.eye(3),
        nominal_G=Bd,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            inner_cfg=AAINDIConfig(seed=0),
            fdd_cfg=FDDConfig(
                process_noise=1e-6, measurement_noise=1e-5,
                adapt_Q=False, adapt_R=False, h_alarm=15.0,
            ),
            enable_glr=True,
            enable_l4_outer=enable_l4,
            l4_n_ref_dim=3,
            l4_eval_mode=True,
            l4_seed=0,
        ),
    )
    ref = np.zeros(3)
    err = 0.0
    n = 0
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        err += float(np.linalg.norm(obs[:3] - ref) ** 2)
        n += 1
        if terminated or truncated:
            break
    return float(np.sqrt(err / max(n, 1)))


@pytest.mark.slow
def test_l4_eval_mode_not_worse_than_phase1() -> None:
    rms_off = _rollout(enable_l4=False)
    rms_on = _rollout(enable_l4=True)
    # With untrained eval-mode actor (mean=tanh(0)=0), L4 does not push
    # reference; numerical state should be effectively identical.
    assert rms_on <= rms_off * 1.05, (
        f"rms_on={rms_on:.6f} should be ≤ 1.05 * rms_off={rms_off:.6f}"
    )
