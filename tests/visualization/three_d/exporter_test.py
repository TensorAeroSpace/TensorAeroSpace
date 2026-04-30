"""Schema and round-trip tests for the 3D web flight log exporter."""

from __future__ import annotations

import json

import numpy as np
import pytest


def _make_simple_env(damage_profile=None, n_steps_run: int = 5):
    """Build and step a NonlinearAngularF16 env briefly."""
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=20,
        dt=0.01,
        airspeed=200.0,
        split_stab=damage_profile is not None,
        damage_profile=damage_profile,
    )
    env.reset()
    n_inputs = 4 if env.split_stab else 3
    for _ in range(n_steps_run):
        env.step(np.zeros(n_inputs))
    return env


def test_flight_log_version_present():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env()
    log = build_flight_log(env)
    assert log["version"] == 1


def test_flight_log_metadata_fields():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env()
    log = build_flight_log(env)
    md = log["metadata"]
    assert md["model"] == "F-16"
    assert md["dt"] == pytest.approx(0.01)
    assert md["airspeed"] == pytest.approx(200.0)
    assert md["n_steps"] >= 5
    assert md["split_stab"] is False


def test_trajectory_arrays_aligned():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env(n_steps_run=10)
    log = build_flight_log(env)
    traj = log["trajectory"]
    n = len(traj["time"])
    for key in (
        "position",
        "attitude",
        "alpha",
        "beta",
        "wx",
        "wy",
        "wz",
        "stab",
        "ail",
        "dir",
    ):
        assert (
            len(traj[key]) == n
        ), f"trajectory['{key}'] has length {len(traj[key])}, expected {n}"


def test_geometry_section_schema():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env()
    log = build_flight_log(env)
    geo = log["geometry"]
    assert isinstance(geo["sections"], list)
    assert len(geo["sections"]) >= 13
    s0 = geo["sections"][0]
    required_fields = {
        "name",
        "type",
        "side",
        "area",
        "span_position",
        "chord",
        "sweep",
        "mass",
        "cg_local",
        "aero_x_arm",
        "controls_input",
    }
    assert set(s0.keys()) == required_fields
    assert isinstance(s0["cg_local"], list)
    assert len(s0["cg_local"]) == 3


def test_damage_events_logged_when_profile_active():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.visualization.three_d import build_flight_log

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.02, "section_loss", {"section": "left_tip", "loss_fraction": 1.0}
            ),
        ]
    )
    env = _make_simple_env(damage_profile=profile, n_steps_run=10)
    log = build_flight_log(env)
    events = log["damage_events"]
    assert len(events) == 1
    e = events[0]
    assert e["label"] == "section_loss"  # default when no .label
    assert e["event_type"] == "section_loss"
    assert e["payload"]["section"] == "left_tip"
    assert e["payload"]["loss_fraction"] == 1.0
    assert 0.0 < e["time"] <= 0.10


def test_damage_state_history_snapshots_at_transitions():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.visualization.three_d import build_flight_log

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.02, "section_loss", {"section": "right_tip", "loss_fraction": 0.5}
            ),
        ]
    )
    env = _make_simple_env(damage_profile=profile, n_steps_run=10)
    log = build_flight_log(env)
    history = log["damage_state_history"]
    # At least the t=0 baseline + one event snapshot
    assert len(history) >= 2
    assert history[0]["time"] == 0.0
    # The post-event snapshot should reflect the loss
    assert history[-1]["state"]["section_loss"]["right_tip"] == 0.5


def test_no_damage_profile_means_empty_logs():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env()
    log = build_flight_log(env)
    assert log["damage_events"] == []
    assert log["damage_state_history"] == []


def test_flight_log_json_serializable():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_simple_env()
    log = build_flight_log(env)
    s = json.dumps(log)
    assert "trajectory" in s
    # Round-trip
    loaded = json.loads(s)
    assert loaded["version"] == 1


def test_metadata_includes_model_params():
    """The flight log's metadata.params block must include the F-16
    model parameters (V, Oy, m, g, S, ...) so the 3D viewer HUD can
    render them without hardcoded constants."""
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    from tensoraerospace.visualization.three_d import build_flight_log

    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=20,
        dt=0.01,
        airspeed=200.0,
    )
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    log = build_flight_log(env)
    params = log["metadata"]["params"]
    # F-16 angular has all the geometry/inertia fields
    for key in ("V", "Oy", "m", "g", "S", "bA"):
        assert key in params, f"missing param: {key}"
        assert isinstance(params[key], float)


def test_metadata_params_for_longitudinal_env():
    """Same params block must work for the longitudinal env."""
    import gymnasium as gym

    from tensoraerospace.visualization.three_d import build_flight_log

    env = gym.make(
        "NonlinearLongitudinalF16-v0",
        number_time_steps=20,
        initial_state=np.array([0.0, 0.0]),
        reference_signal=np.zeros((1, 30)),
        state_space=["alpha", "wz"],
        control_space=["stab"],
        tracking_states=["alpha"],
        use_reward=False,
        dt=0.01,
        integrator="euler",
    ).unwrapped
    env.reset()
    for _ in range(3):
        env.step(np.zeros(1))
    log = build_flight_log(env)
    params = log["metadata"]["params"]
    # Longitudinal lacks Jx/Jy/Jxy but has the basic V/Oy/m/g/S/bA
    for key in ("V", "Oy", "m", "g", "S", "bA"):
        assert key in params, f"missing param: {key}"


def test_reset_clears_damage_logs():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
        DamageProfile,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    profile = DamageProfile(
        events=[
            DamageEvent(
                0.02, "section_loss", {"section": "left_tip", "loss_fraction": 1.0}
            ),
        ]
    )
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=20,
        dt=0.01,
        airspeed=200.0,
        damage_profile=profile,
        split_stab=True,
    )
    env.reset()
    for _ in range(5):
        env.step(np.zeros(4))
    assert len(env.damage_events_log) == 1
    env.reset()
    assert env.damage_events_log == []
    # damage_state_log keeps the t=0 baseline after reset
    assert len(env.damage_state_log) == 1
    assert env.damage_state_log[0]["time"] == 0.0
