"""Shared event timing and configuration for the two nonlinear F-16 environments."""

from functools import partial

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageManager,
    DamageProfile,
    load_f16_geometry,
)


def decode_profile(profile):
    """Accept a profile object or its saved JSON-compatible configuration."""
    return DamageProfile.from_dict(profile) if isinstance(profile, dict) else profile


def _trigger_damage(env, timestamp, labels):
    manager = env.damage_manager
    triggered = manager.update(timestamp, float(np.nextafter(timestamp, -np.inf)))
    for event in triggered:
        if env.damage_event_callback is not None:
            env.damage_event_callback(event, manager.state)
        label = event.label or event.event_type
        labels.append(label)
        env.damage_events_log.append(
            {
                "time": float(event.trigger_time),
                "label": label,
                "event_type": event.event_type,
                "payload": dict(event.payload),
            }
        )
    if triggered:
        env.damage_state_log.append(
            {"time": float(timestamp), "state": manager.state.snapshot()}
        )


def reset_damage(env, options, seed):
    """Initialize damage, including events at t=0 before the reset observation."""
    profile = decode_profile((options or {}).get("damage_profile", env.damage_profile))
    env.damage_events_log = []
    env.damage_state_log = []
    env.damage_manager = None
    if profile is None and not env.damage_observable:
        return
    geometry = load_f16_geometry()
    manager = DamageManager(geometry, env.model.param, profile)
    manager.reset(seed=seed)
    env.damage_manager = manager
    env.model.damage_state = manager.state
    env.model.damage_geometry = geometry
    env.damage_state_log.append({"time": 0.0, "state": manager.state.snapshot()})
    _trigger_damage(env, 0.0, [])


def step_with_damage(env, command):
    """Integrate to every event, apply it, then integrate the remaining interval."""
    if env.damage_manager is None:
        return env.model.run_step(command), []
    model = env.model
    start = model.t0 + (model.time_step - 1) * model.dt
    end = model.t0 + model.time_step * model.dt
    pending = env.damage_manager.pending_events(
        end, float(np.nextafter(start, -np.inf))
    )
    labels: list[str] = []
    events = [
        (
            float(np.clip(timestamp - start, 0, model.dt)),
            partial(_trigger_damage, env, timestamp, labels),
        )
        for timestamp in sorted({event.trigger_time for event in pending})
    ]
    return model.run_step(command, events=events), labels
