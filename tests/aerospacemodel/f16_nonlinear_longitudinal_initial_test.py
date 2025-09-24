import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest


def _install_fake_matlab():
    fake_matlab = types.ModuleType("matlab")
    fake_matlab.__path__ = []
    fake_matlab.double = lambda x: x
    sys.modules["matlab"] = fake_matlab
    fake_engine = types.ModuleType("matlab.engine")
    fake_engine.start_matlab = lambda: None
    sys.modules["matlab.engine"] = fake_engine


def test_set_initial_state_valid_and_invalid():
    _install_fake_matlab()
    init_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/longitudinal/inital.py"
    )
    spec = importlib.util.spec_from_file_location("longitudinal_inital", str(init_path))
    init_mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(init_mod)
    out = init_mod.set_initial_state({"alpha": np.deg2rad(10)})
    assert isinstance(out, list)

    with pytest.raises(Exception):
        init_mod.set_initial_state({"unknown": 1.0})
