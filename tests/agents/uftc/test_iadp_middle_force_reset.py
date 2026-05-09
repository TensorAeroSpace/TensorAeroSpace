"""IADPMiddle.force_reset inflates RLS regardless of FDD."""

from __future__ import annotations

import numpy as np

# (IADPAgent factory import depends on existing IADP API)
from tensoraerospace.agent.iadp.model import IADPAgent
from tensoraerospace.agent.uftc.middle import IADPMiddle, RLSResetPolicy


def _build_middle() -> IADPMiddle:
    base = IADPAgent(n_state=3, n_control=2)
    return IADPMiddle(base=base, reset_policy=RLSResetPolicy())


def test_force_reset_inflates_phi() -> None:
    m = _build_middle()
    phi_before = float(np.linalg.norm(m.base.rls.Phi))
    m.force_reset(severity_hint=0.5)
    phi_after = float(np.linalg.norm(m.base.rls.Phi))
    assert phi_after > phi_before
