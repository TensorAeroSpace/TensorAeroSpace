"""Convert a completed F-16 episode into a JSON-serializable flight log.

The flight log is the single source of truth that the future three.js
viewer will consume. Schema is versioned (FLIGHT_LOG_VERSION) so future
viewer revisions can refuse incompatible logs explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
    BaseGeometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)

FLIGHT_LOG_VERSION = 1


def build_flight_log(env) -> dict[str, Any]:
    """Build a JSON-serializable flight log from a completed episode.

    Expects ``env`` to be a ``NonlinearAngularF16`` (or compatible) gym env
    that has been ``reset()`` and stepped through at least once. Reads:

      * ``env.position_history`` — (T, 3) inertial position, m
      * ``env.attitude_history`` — (T, 3) (roll, pitch, yaw), rad
      * ``env.time_history``     — (T,) seconds
      * ``env.model.x_history``  — list of T 14-element state column vectors
      * ``env.damage_events_log`` — accumulated DamageEvent records (may be empty)
      * ``env.damage_state_log``  — DamageState snapshots at events (may be empty)

    Returns
    -------
    dict
        A JSON-serializable flight log conforming to schema version
        ``FLIGHT_LOG_VERSION``.
    """
    if env.model is None:
        raise ValueError("env.model is None — call env.reset() first")

    pos = np.asarray(env.position_history, dtype=np.float64)
    att = np.asarray(env.attitude_history, dtype=np.float64)
    t = np.asarray(env.time_history, dtype=np.float64)
    if pos.shape[1] != 3 or att.shape[1] != 3:
        raise ValueError(
            f"position_history / attitude_history must have 3 columns; "
            f"got {pos.shape}, {att.shape}"
        )
    if not (len(pos) == len(att) == len(t)):
        raise ValueError(
            "position_history, attitude_history, time_history must have the "
            "same length"
        )

    # Pull per-step state channels from the underlying angular model (14-state)
    # x_history is a list of (14, 1) ndarrays, one per step (incl. initial)
    x_hist = np.asarray(
        [x.reshape(-1) for x in env.model.x_history], dtype=np.float64,
    )
    if x_hist.shape[0] != len(t):
        # Different lengths can happen if step count != history count; align
        # to the shorter
        n = min(x_hist.shape[0], len(t))
        x_hist = x_hist[:n]
        pos = pos[:n]
        att = att[:n]
        t = t[:n]

    geometry = _serialise_geometry(env._geo_for_obs or load_f16_geometry())

    return {
        "version": FLIGHT_LOG_VERSION,
        "metadata": {
            "model": "F-16",
            "dt": float(env.dt),
            "n_steps": int(len(t)),
            "airspeed": float(getattr(env, "airspeed", 0.0)),
            "split_stab": bool(getattr(env, "split_stab", False)),
        },
        "geometry": geometry,
        "trajectory": {
            "time": t.tolist(),
            "position": pos.tolist(),
            "attitude": att.tolist(),
            "alpha": x_hist[:, 0].tolist(),
            "beta": x_hist[:, 1].tolist(),
            "wx": x_hist[:, 2].tolist(),
            "wy": x_hist[:, 3].tolist(),
            "wz": x_hist[:, 4].tolist(),
            "stab": x_hist[:, 8].tolist(),
            "ail": x_hist[:, 10].tolist(),
            "dir": x_hist[:, 12].tolist(),
        },
        "damage_events": list(getattr(env, "damage_events_log", [])),
        "damage_state_history": list(getattr(env, "damage_state_log", [])),
    }


def _serialise_geometry(geo: BaseGeometry) -> dict[str, Any]:
    return {
        "sections": [
            {
                "name": s.name,
                "type": s.type,
                "side": s.side,
                "area": s.area,
                "span_position": s.span_position,
                "chord": s.chord,
                "sweep": s.sweep,
                "mass": s.mass,
                "cg_local": list(s.cg_local),
                "aero_x_arm": s.aero_x_arm,
                "controls_input": s.controls_input,
            }
            for s in geo.sections
        ],
    }
