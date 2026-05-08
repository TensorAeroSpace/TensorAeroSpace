"""F-16 WING_STRIKE: monitor reaches CRITICAL; macro-actions fire; no divergence."""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import cont2discrete

torch = pytest.importorskip("torch")

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
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


@pytest.mark.slow
def test_critical_alarm_triggers_macro_actions() -> None:
    n_steps = 800
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
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=200,
        omega_indices=[0, 1, 2],
        middle_lookahead_dt=0.05,
        trust_radius_fault=0.7,
        inner_cfg=AAINDIConfig(seed=0),
        fdd_cfg=FDDConfig(
            process_noise=1e-6,
            measurement_noise=1e-5,
            adapt_Q=False, adapt_R=False,
            h_alarm=15.0,
        ),
        enable_l1_shield=False, enable_glr=True,
        enable_l4_outer=True, l4_n_ref_dim=3, l4_eval_mode=True,
        enable_monitor=True,
        # tighten alarm thresholds + fault-channel weighting so CRITICAL is reachable
        monitor_c_weights=(0.1, 0.2, 0.2, 0.0, 0.5),
        monitor_alarm_warn_frac=0.2,
        monitor_alarm_critical_frac=0.4,
        monitor_cooldown_steps=50,
    )
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=Ad - np.eye(3),
        nominal_G=Bd,
        config=cfg,
    )
    ref = np.zeros(3)
    saw_critical = False
    saw_force_reset = False
    last_x = obs[:3]
    for k in range(n_steps):
        u = ctl.predict(last_x, ref, time_step=k)
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        last_x = obs[:3]
        info = ctl.learn(last_x, ref, time_step=k)
        if info.get("monitor", {}).get("alarm") == "CRITICAL":
            saw_critical = True
        if info.get("force_rls_reset") is not None:
            saw_force_reset = True
        if terminated or truncated:
            break
    assert saw_critical, "monitor must reach CRITICAL on WING_STRIKE"
    assert saw_force_reset, "CRITICAL alarm must trigger force_rls_reset macro-action"
    assert np.all(np.isfinite(last_x))
