"""Apply control-surface failures to commanded inputs.

Translates the dict of `DamageState.control_failures` into mutations of the
command vector u, before it is fed to the integrator. The mapping from
surface name to vector index depends on the action layout — for the F-16
angular ODE, two layouts exist:

  - Legacy 3-input mode (action = [stab, ail, dir]): surface failures on
    stab_left/stab_right both map to index 0; aileron_left/aileron_right
    both map to index 1; rudder maps to index 2.
  - Split-stab 4-input mode (action = [stab_l, stab_r, ail, dir]):
    stab_left=0, stab_right=1, ailerons=2 (single channel),
    rudder=3.

Failure modes:
  - "jam":              u[idx] = jam_position_rad (regardless of command)
  - "efficiency_loss":  u[idx] *= efficiency
  - "lost"/"free_floating": u[idx] = 0.0
  - "healthy":          no-op
"""

from __future__ import annotations

import numpy as np

from .state import DamageState

# Default mapping for the split-stab AngularF16 (action = [stab_l, stab_r, ail, dir])
ANGULAR_SPLIT_STAB_INDEX = {
    "stab_left": 0,
    "stab_right": 1,
    "aileron_left": 2,
    "aileron_right": 2,
    "rudder": 3,
}

# Mapping for the legacy 3-input AngularF16 (action = [stab, ail, dir]).
# In legacy mode, separate-side failures are not addressable; the closest
# is to apply to the merged 'stab' channel.
ANGULAR_LEGACY_INDEX = {
    "stab_left": 0, "stab_right": 0,
    "aileron_left": 1, "aileron_right": 1,
    "rudder": 2,
}


def apply_control_failures(
    u_command: np.ndarray,
    state: DamageState,
    surface_to_index: dict[str, int],
) -> np.ndarray:
    """Return modified command vector after applying every active failure.

    Multiple failures targeting the same index compose left-to-right by
    insertion order in `state.control_failures`.
    """
    u_eff = np.asarray(u_command, dtype=np.float64).copy()
    for surface, failure in state.control_failures.items():
        if surface not in surface_to_index:
            continue
        idx = surface_to_index[surface]
        if failure.mode == "healthy":
            continue
        if failure.mode == "jam":
            u_eff[idx] = failure.jam_position_rad
        elif failure.mode == "efficiency_loss":
            u_eff[idx] = u_eff[idx] * failure.efficiency
        elif failure.mode in ("lost", "free_floating"):
            u_eff[idx] = 0.0
    return u_eff
