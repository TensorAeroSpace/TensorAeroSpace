"""F-16 geometry preset loader."""

from __future__ import annotations

from importlib import resources

import yaml

from .geometry import AeroSection, BaseGeometry

_DATA_PACKAGE = "tensoraerospace.aerospacemodel.f16.nonlinear.damage.data"
_F16_FILE = "f16_geometry.yaml"


def load_f16_geometry() -> BaseGeometry:
    """Load the calibrated F-16 baseline geometry.

    Section masses and wing areas are calibrated to match F16AngularParameters
    defaults (m=9295.44 kg, S=27.87 m²) within 1%.
    """
    data_path = resources.files(_DATA_PACKAGE).joinpath(_F16_FILE)
    with data_path.open("r") as f:
        raw = yaml.safe_load(f)
    sections = []
    for s in raw["sections"]:
        cg_values = [float(v) for v in s["cg_local"]]
        inertia_values = [float(v) for v in s["inertia_local"]]
        sections.append(
            AeroSection(
                name=s["name"],
                side=s["side"],
                type=s["type"],
                area=float(s["area"]),
                span_position=float(s["span_position"]),
                chord=float(s["chord"]),
                sweep=float(s["sweep"]),
                mass=float(s["mass"]),
                cg_local=(cg_values[0], cg_values[1], cg_values[2]),
                inertia_local=(
                    inertia_values[0],
                    inertia_values[1],
                    inertia_values[2],
                    inertia_values[3],
                ),
                cl_alpha_contribution=float(s["cl_alpha_contribution"]),
                cd0_contribution=float(s["cd0_contribution"]),
                controls_input=s.get("controls_input"),
                control_effectiveness=float(s.get("control_effectiveness", 1.0)),
                aero_x_arm=float(s.get("aero_x_arm", 0.0)),
            )
        )
    return BaseGeometry(sections=sections)


# === Built-in damage scenarios ===

from .events import DamageEvent, DamageProfile  # noqa: E402

WING_STRIKE_LEFT_TIP = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=10.0,
            event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 1.0},
            label="left_tip_total_loss",
        ),
    ]
)

WING_STRIKE_LEFT_HALF = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=10.0,
            event_type="section_loss",
            payload={"section": "left_tip", "loss_fraction": 1.0},
            label="left_tip_total_loss",
        ),
        DamageEvent(
            trigger_time=10.0,
            event_type="section_loss",
            payload={"section": "left_mid", "loss_fraction": 0.5},
            label="left_mid_partial",
        ),
    ]
)

ELEVATOR_JAM_NEUTRAL = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=5.0,
            event_type="control_failure",
            payload={"surface": "stab_left", "mode": "jam", "jam_position_rad": 0.0},
            label="stab_left_jam_neutral",
        ),
        DamageEvent(
            trigger_time=5.0,
            event_type="control_failure",
            payload={"surface": "stab_right", "mode": "jam", "jam_position_rad": 0.0},
            label="stab_right_jam_neutral",
        ),
    ]
)

ELEVATOR_JAM_PITCH_UP = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=5.0,
            event_type="control_failure",
            payload={"surface": "stab_left", "mode": "jam", "jam_position_rad": 0.1745},
            label="stab_left_jam_up",
        ),
        DamageEvent(
            trigger_time=5.0,
            event_type="control_failure",
            payload={
                "surface": "stab_right",
                "mode": "jam",
                "jam_position_rad": 0.1745,
            },
            label="stab_right_jam_up",
        ),
    ]
)

RUDDER_LOST = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=5.0,
            event_type="control_failure",
            payload={"surface": "rudder", "mode": "lost"},
            label="rudder_lost",
        ),
    ]
)

ENGINE_FLAMEOUT = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=5.0,
            event_type="engine_failure",
            payload={"thrust_factor": 0.0, "hard_failure": True},
            label="engine_flameout",
        ),
    ]
)

ENGINE_THRUST_DRIFT = DamageProfile(
    events=[
        DamageEvent(
            # trigger on the first sub-step (any t_current > 0); the ramp
            # itself is anchored at trigger_time and runs for ``duration``.
            trigger_time=1e-6,
            event_type="engine_failure",
            payload={
                "thrust_scale_start": 1.0,
                "thrust_scale_end": 0.0,
                "ramp": "linear",
            },
            label="engine_thrust_drift",
            # 1 %/s slow drift over 100 s — matches the plan's behavioural
            # target ("at default 1 %/s loss, after 5 s thrust scale ≈ 0.95").
            duration=100.0,
        ),
    ]
)

BIRDSTRIKE_COMPOUND = DamageProfile(
    events=[
        DamageEvent(
            trigger_time=5.0,
            event_type="section_loss",
            payload={"section": "right_mid", "loss_fraction": 0.2},
            label="right_wing_birdstrike",
        ),
        DamageEvent(
            trigger_time=5.0,
            event_type="engine_failure",
            payload={"thrust_factor": 0.3},
            label="engine_partial_loss",
        ),
    ]
)
