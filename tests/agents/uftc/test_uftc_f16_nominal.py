"""F-16 nonlinear angular flight: UFTC on nominal — regression guard."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import cont2discrete

from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

# ---------------------------------------------------------------------------
# Helper: derive a 3×3 / 3×4 discrete-time nominal model from the F-16 linear
# data shipped with the package.  We extract the [alpha, beta, p] sub-block
# (rows/cols 7, 8, 9 of the 18-state continuous-time A matrix) which covers
# the three channels the UFTC controller tracks via obs[:3].
# ---------------------------------------------------------------------------
_LINEAR_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../tensoraerospace/aerospacemodel/f16/linear/data",
)


def _f16_nominal_matrices(dt: float = 0.01):
    """Return (Ad_3x3, Bd_3x4) discretised at *dt* for the [alpha, beta, p] block."""
    A_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "A.mat"))["A_lo"]
    B_cont = loadmat(os.path.join(_LINEAR_DATA_DIR, "B.mat"))["B_lo"]
    # Linear state order: npos, epos, alt, phi, theta, psi, vel, alpha, beta, p, q, r, …
    idx = [7, 8, 9]  # alpha(7), beta(8), p/wx(9)
    A_sub = A_cont[np.ix_(idx, idx)]
    B_sub = B_cont[idx, :]  # (3, 4)
    Ad, Bd, _, _, _ = cont2discrete((A_sub, B_sub, np.eye(3), np.zeros((3, 4))), dt=dt)
    return Ad, Bd


@pytest.mark.slow
def test_uftc_holds_attitude_under_nominal_f16() -> None:
    rng = np.random.default_rng(0)
    n_steps = 800
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=None,
        split_stab=True,
    )
    obs, _ = env.reset()

    # Use linearised F-16 sub-matrices so the Kalman warm-start matches the
    # plant well enough that FDD stays quiet under nominal conditions.
    nominal_F, nominal_G = _f16_nominal_matrices(dt=0.01)

    ctl = UFTCController(
        n_state=3,
        n_control=4,
        nominal_F=nominal_F,
        nominal_G=nominal_G,
        config=UFTCConfig(
            dt=0.01,
            fdd_warmup_steps=400,
            omega_indices=[0, 1, 2],  # alpha, beta, wx indices in obs[:3]
            # lookahead_dt=0.05 works well with the F-16 plant at dt=0.01
            middle_lookahead_dt=0.05,
        ),
    )
    ref_zero = np.zeros(3)
    rms = []
    for k in range(n_steps):
        x_omega = obs[:3]
        u = ctl.predict(x_omega, ref_zero, time_step=k)
        # Pad to 4-DOF actuator input expected by env.
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, info = env.step(u_env)
        ctl.learn(obs[:3], ref_zero, time_step=k)
        rms.append(float(np.linalg.norm(obs[:3])))
        if terminated or truncated:
            break
    rms_after_warmup = np.mean(rms[300:])
    # Coarse regression bound: nominal hold must beat 0.5 rad/s RMS.
    assert rms_after_warmup < 0.5
    assert not ctl.diagnostics()["fault_present"]
    # Silence the unused-name warning on `rng`.
    _ = rng
