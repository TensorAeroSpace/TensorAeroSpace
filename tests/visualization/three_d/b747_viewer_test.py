from __future__ import annotations

import numpy as np


def _make_b747_env(n_steps: int = 4):
    from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
        DamageProfile,
        EngineFailureEvent,
    )
    from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

    env = NonlinearB747Env(
        trim_at=(20_000.0, 674.0),
        number_time_steps=20,
        dt=0.05,
        damage_profile=DamageProfile(
            events=[
                EngineFailureEvent(
                    trigger_time=0.10,
                    engine_id=1,
                    thrust_fraction=0.0,
                    label="left_outer_engine_flameout",
                )
            ]
        ),
    )
    env.reset()
    for _ in range(n_steps):
        env.step(np.array([0.0, 0.0, 0.0, 0.55]))
    return env


def test_b747_env_registers_3d_web_mode():
    from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env

    assert "3d_web" in NonlinearB747Env.metadata["render_modes"]


def test_b747_flight_log_uses_b747_metadata_and_12_state_channels():
    from tensoraerospace.visualization.three_d import build_flight_log

    log = build_flight_log(_make_b747_env())
    assert log["metadata"]["model"] == "B-747"
    assert log["metadata"]["aircraft_type"] == "b747"
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
        "altitude_m",
        "airspeed_mps",
    ):
        assert len(traj[key]) == n


def test_b747_damage_log_contains_engine_failure_snapshot():
    from tensoraerospace.visualization.three_d import build_flight_log

    log = build_flight_log(_make_b747_env())
    assert log["damage_events"][0]["event_type"] == "EngineFailureEvent"
    assert log["damage_events"][0]["payload"]["engine_id"] == 1
    assert log["damage_state_history"][-1]["state"]["engines_mu"][1] == 0.0


def test_b747_html_contains_b747_mesh_builder():
    from tensoraerospace.visualization.three_d import build_flight_log, build_html

    html = build_html(build_flight_log(_make_b747_env()), title="B747 test")
    # Synchronous builder still present.
    assert "function buildB747Aircraft" in html
    # GLB loader vendored + glue function in the viewer.
    assert "GLTFLoader" in html
    assert "_loadB747GlbInto" in html
    assert "_onB747GlbLoaded" in html
    # GLB binary is embedded as base64.
    assert "B747_GLB_B64" in html
    # The hinge placeholder names that the per-tick animation looks up
    # are still emitted.
    for hinge_name in (
        "aileron_left", "aileron_right",
        "stab_left", "stab_right",
        "rudder",
    ):
        assert f'"{hinge_name}"' in html
    # Engine placeholder groups for each of the four B-747 engines.
    for engine_id in (1, 2, 3, 4):
        assert f'"engine_{engine_id}"' in html
