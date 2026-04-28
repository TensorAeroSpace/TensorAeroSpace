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
    sections = [
        AeroSection(
            name=s["name"],
            side=s["side"],
            type=s["type"],
            area=float(s["area"]),
            span_position=float(s["span_position"]),
            chord=float(s["chord"]),
            sweep=float(s["sweep"]),
            mass=float(s["mass"]),
            cg_local=tuple(float(v) for v in s["cg_local"]),
            inertia_local=tuple(float(v) for v in s["inertia_local"]),
            cl_alpha_contribution=float(s["cl_alpha_contribution"]),
            cd0_contribution=float(s["cd0_contribution"]),
            controls_input=s.get("controls_input"),
            control_effectiveness=float(s.get("control_effectiveness", 1.0)),
            aero_x_arm=float(s.get("aero_x_arm", 0.0)),
        )
        for s in raw["sections"]
    ]
    return BaseGeometry(sections=sections)
