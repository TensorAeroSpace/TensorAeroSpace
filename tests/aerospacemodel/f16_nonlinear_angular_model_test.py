import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest


class _FakeMatlabEngine:
    def __init__(self):
        self.paths = []
        self.called = False

    def addpath(self, p):
        self.paths.append(p)

    def airplane_parameters(self):
        return {"dummy": 1}

    def step(self, x_prev, *_args, **_kwargs):
        self.called = True
        # return numpy array converted from input for downstream logic
        return np.array(list(x_prev))


def _install_fake_matlab(_monkeypatch=None):
    # clear any previously installed fake to avoid stale non-package modules
    sys.modules.pop("matlab", None)
    sys.modules.pop("matlab.engine", None)
    fake_matlab = types.ModuleType("matlab")

    class _FakeDouble:
        def __init__(self, data):
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    fake_matlab.double = _FakeDouble
    fake_matlab.__path__ = []
    sys.modules["matlab"] = fake_matlab

    fake_engine = types.ModuleType("matlab.engine")

    def _start():
        return _FakeMatlabEngine()

    fake_engine.start_matlab = _start
    sys.modules["matlab.engine"] = fake_engine
    # also expose as attribute to support `matlab.engine`
    fake_matlab.engine = fake_engine


def test_import_and_construct_smoke():
    _install_fake_matlab()
    model_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py"
    )
    model_spec = importlib.util.spec_from_file_location(
        "angular_model", str(model_path)
    )
    mod = importlib.util.module_from_spec(model_spec)
    assert model_spec and model_spec.loader
    model_spec.loader.exec_module(mod)
    init_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py"
    )
    init_spec = importlib.util.spec_from_file_location("angular_inital", str(init_path))
    init_mod = importlib.util.module_from_spec(init_spec)
    assert init_spec and init_spec.loader
    init_spec.loader.exec_module(init_mod)
    m = mod.AngularF16(init_mod.initial_state)
    assert m.action_space_length == 3
    assert isinstance(m.eng, _FakeMatlabEngine)


def test_run_step_valid_and_invalid_input():
    _install_fake_matlab()
    model_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py"
    )
    model_spec = importlib.util.spec_from_file_location(
        "angular_model", str(model_path)
    )
    mod = importlib.util.module_from_spec(model_spec)
    assert model_spec and model_spec.loader
    model_spec.loader.exec_module(mod)
    init_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py"
    )
    init_spec = importlib.util.spec_from_file_location("angular_inital", str(init_path))
    init_mod = importlib.util.module_from_spec(init_spec)
    assert init_spec and init_spec.loader
    init_spec.loader.exec_module(init_mod)
    m = mod.AngularF16(init_mod.initial_state)

    # valid shape (3x1) and list -> gets wrapped by matlab.double lambda
    out = m.run_step([[0.0], [0.0], [0.0]])
    assert isinstance(out, np.ndarray)

    # invalid: wrong length
    with pytest.raises(Exception):
        m.run_step([[0.0], [0.0]])


def test_selected_state_output():
    _install_fake_matlab()
    model_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/model.py"
    )
    model_spec = importlib.util.spec_from_file_location(
        "angular_model", str(model_path)
    )
    mod = importlib.util.module_from_spec(model_spec)
    assert model_spec and model_spec.loader
    model_spec.loader.exec_module(mod)
    init_path = (
        pathlib.Path(__file__).parents[2]
        / "tensoraerospace/aerospacemodel/f16/nonlinear/angular/inital.py"
    )
    init_spec = importlib.util.spec_from_file_location("angular_inital", str(init_path))
    init_mod = importlib.util.module_from_spec(init_spec)
    assert init_spec and init_spec.loader
    init_spec.loader.exec_module(init_mod)
    selected = ["alpha", "beta"]
    m = mod.AngularF16(init_mod.initial_state, selected_state_output=selected)
    y = m.run_step([[0.0], [0.0], [0.0]])
    assert y.shape[0] == len(selected)
