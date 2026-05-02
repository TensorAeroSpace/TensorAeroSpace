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
        [x.reshape(-1) for x in env.model.x_history],
        dtype=np.float64,
    )
    if x_hist.shape[0] != len(t):
        # Different lengths can happen if step count != history count; align
        # to the shorter
        n = min(x_hist.shape[0], len(t))
        x_hist = x_hist[:n]
        pos = pos[:n]
        att = att[:n]
        t = t[:n]

    geo_obj = (
        getattr(env, "_geo_for_obs", None)
        or getattr(env, "_geo_for_damage", None)
        or load_f16_geometry()
    )
    geometry = _serialise_geometry(geo_obj)

    # Per-step state channels — pull by model dimension. Angular = 14
    # (alpha, beta, wx, wy, wz, gamma, psi, theta, stab, dstab, ail, dail,
    # dir, ddir). Longitudinal = 4 (alpha, wz, stab, dstab); the channels
    # not present in the longitudinal ODE are filled with zeros so the
    # viewer JSON schema stays uniform.
    n_state = x_hist.shape[1]
    n_steps = x_hist.shape[0]
    zero_channel = [0.0] * n_steps
    if n_state == 14:
        traj_channels = {
            "alpha": x_hist[:, 0].tolist(),
            "beta": x_hist[:, 1].tolist(),
            "wx": x_hist[:, 2].tolist(),
            "wy": x_hist[:, 3].tolist(),
            "wz": x_hist[:, 4].tolist(),
            "stab": x_hist[:, 8].tolist(),
            "ail": x_hist[:, 10].tolist(),
            "dir": x_hist[:, 12].tolist(),
        }
    elif n_state == 16:
        # Altitude-tracking angular: same as 14-state plus h and V.
        traj_channels = {
            "alpha": x_hist[:, 0].tolist(),
            "beta": x_hist[:, 1].tolist(),
            "wx": x_hist[:, 2].tolist(),
            "wy": x_hist[:, 3].tolist(),
            "wz": x_hist[:, 4].tolist(),
            "stab": x_hist[:, 8].tolist(),
            "ail": x_hist[:, 10].tolist(),
            "dir": x_hist[:, 12].tolist(),
            "altitude_m": x_hist[:, 14].tolist(),
            "airspeed_mps": x_hist[:, 15].tolist(),
        }
    elif n_state == 4:
        # Longitudinal: [alpha, wz, stab, dstab]
        traj_channels = {
            "alpha": x_hist[:, 0].tolist(),
            "beta": zero_channel,
            "wx": zero_channel,
            "wy": zero_channel,
            "wz": x_hist[:, 1].tolist(),
            "stab": x_hist[:, 2].tolist(),
            "ail": zero_channel,
            "dir": zero_channel,
        }
    else:
        raise ValueError(
            f"Unsupported model state dimension {n_state}; expected 4, 14, or 16."
        )

    # Reference / commanded signals for the 3D-viewer chart overlays.
    #
    # Two complementary sources are merged:
    #   1. Auto-detected from the env's standard reference_signal + tracking_states
    #      attributes (the canonical interface — set via gym.make(...)).
    #      Values are assumed to be in raw model units (radians for angles,
    #      m for altitude, m/s for airspeed) and are converted to chart
    #      display units (deg / m / m/s) here.
    #   2. Manually populated env.reference_signals dict (already in display
    #      units). Manual entries OVERRIDE auto-detected ones for the same key.
    n_t = len(t)
    references_out = _auto_extract_references(env, n_t)
    references_raw = getattr(env, "reference_signals", None) or {}
    for key, seq in references_raw.items():
        arr = np.asarray(seq, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        if arr.size < n_t:
            arr = np.concatenate([arr, np.full(n_t - arr.size, arr[-1])])
        elif arr.size > n_t:
            arr = arr[:n_t]
        references_out[str(key)] = arr.tolist()

    return {
        "version": FLIGHT_LOG_VERSION,
        "metadata": {
            "model": "F-16",
            "dt": float(env.dt),
            "n_steps": int(len(t)),
            "airspeed": float(getattr(env, "airspeed", 0.0)),
            "split_stab": bool(getattr(env, "split_stab", False)),
            "params": _serialise_params(env),
        },
        "geometry": geometry,
        "trajectory": {
            "time": t.tolist(),
            "position": pos.tolist(),
            "attitude": att.tolist(),
            **traj_channels,
            "references": references_out,
        },
        "damage_events": list(getattr(env, "damage_events_log", [])),
        "damage_state_history": list(getattr(env, "damage_state_log", [])),
    }


# Map env state-vector names to viewer chart keys + the multiplier
# applied to convert raw env units (rad / m / m/s) to chart display
# units (deg / m / m/s). Channels not in this table are ignored.
_RAD2DEG = 180.0 / np.pi
_REFERENCE_CHANNEL_MAP: dict[str, tuple[str, float]] = {
    # angles (rad → deg)
    "alpha": ("alpha", _RAD2DEG),
    "beta": ("beta", _RAD2DEG),
    "wx": ("wx", _RAD2DEG),
    "wy": ("wy", _RAD2DEG),
    "wz": ("wz", _RAD2DEG),
    "theta": ("theta", _RAD2DEG),
    "gamma": ("roll", _RAD2DEG),  # the codebase uses gamma for roll
    "psi": ("yaw", _RAD2DEG),
    "stab": ("stab", _RAD2DEG),
    "dstab": ("dstab", _RAD2DEG),
    "ail": ("ail", _RAD2DEG),
    "dir": ("dir", _RAD2DEG),
    # absolute units (no conversion)
    "h": ("h", 1.0),
    "V": ("V", 1.0),
}


def _auto_extract_references(env, n_t: int) -> dict[str, list[float]]:
    """Pull per-channel reference arrays from env.reference_signal and
    env.tracking_states (the canonical longitudinal-env interface).

    Returns a dict ``{chart_key: [values_in_display_units]}`` truncated /
    padded to ``n_t``. Empty if the env doesn't expose both attributes.
    """
    ref_arr = getattr(env, "reference_signal", None)
    tracking = getattr(env, "tracking_states", None)
    if ref_arr is None or not tracking:
        return {}
    try:
        ref_arr = np.asarray(ref_arr, dtype=np.float64)
    except (TypeError, ValueError):
        return {}
    if ref_arr.ndim != 2:
        return {}
    n_track, n_steps_ref = ref_arr.shape
    if n_track != len(tracking):
        return {}
    out: dict[str, list[float]] = {}
    for i, name in enumerate(tracking):
        mapping = _REFERENCE_CHANNEL_MAP.get(str(name))
        if mapping is None:
            continue
        key, scale = mapping
        series = ref_arr[i] * scale
        if series.size < n_t:
            series = np.concatenate([series, np.full(n_t - series.size, series[-1])])
        elif series.size > n_t:
            series = series[:n_t]
        out[key] = series.tolist()
    return out


def _serialise_params(env) -> dict[str, float]:
    """Pull a small subset of the F-16 model parameters into the flight
    log so the 3D viewer's HUD can render real airspeed / altitude /
    mass / dynamic-pressure values rather than hardcoded constants.

    Reads from ``env.model.param``, which is an ``F16AngularParameters``
    or ``F16LongParameters`` dataclass. Fields known to vary between the
    two are probed via ``getattr`` so the same exporter works for both.
    """
    if env.model is None or not hasattr(env.model, "param"):
        return {}
    p = env.model.param
    fields = (
        "V",
        "Oy",
        "m",
        "g",
        "q",
        "S",
        "bA",
        "Jx",
        "Jy",
        "Jz",
        "Jxy",
        "Jxz",
        "Jyz",
    )
    out: dict[str, float] = {}
    for f in fields:
        if hasattr(p, f):
            try:
                out[f] = float(getattr(p, f))
            except (TypeError, ValueError):
                continue
    return out


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
