"""Public import paths use corrected spelling while legacy aliases still work."""

from __future__ import annotations

import importlib
import sys

import pytest


def _fresh_deprecated_import(module_name: str):
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning):
        return importlib.import_module(module_name)


def test_corrected_public_api_paths_import():
    standard = importlib.import_module("tensoraerospace.signals.standard")
    f16_env = importlib.import_module("tensoraerospace.envs.f16.linear_longitudinal")
    geosat_env = importlib.import_module("tensoraerospace.envs.geosat")
    angular_initial = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.initial"
    )
    longitudinal_initial = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.initial"
    )

    assert callable(standard.unit_step)
    assert f16_env.LinearLongitudinalF16.__name__ == "LinearLongitudinalF16"
    assert geosat_env.GeoSatEnv.__name__ == "GeoSatEnv"
    assert callable(angular_initial.set_initial_state)
    assert callable(longitudinal_initial.set_initial_state)


def test_legacy_misspelled_public_api_paths_warn():
    legacy_standard = _fresh_deprecated_import("tensoraerospace.signals.standart")
    legacy_f16_env = _fresh_deprecated_import(
        "tensoraerospace.envs.f16.linear_longitudial"
    )
    legacy_geosat_env = _fresh_deprecated_import("tensoraerospace.envs.geostat")
    legacy_angular_initial = _fresh_deprecated_import(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.inital"
    )
    legacy_longitudinal_initial = _fresh_deprecated_import(
        "tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.inital"
    )

    assert callable(legacy_standard.unit_step)
    assert legacy_f16_env.LinearLongitudinalF16.__name__ == "LinearLongitudinalF16"
    assert legacy_geosat_env.GeoSatEnv.__name__ == "GeoSatEnv"
    assert callable(legacy_angular_initial.set_initial_state)
    assert callable(legacy_longitudinal_initial.set_initial_state)
