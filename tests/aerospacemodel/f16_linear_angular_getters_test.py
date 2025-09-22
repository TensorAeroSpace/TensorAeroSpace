import importlib
import sys
import types

import numpy as np


def _install_fake_matlab():
    fake = types.ModuleType("matlab")
    fake.double = lambda x: x
    sys.modules["matlab"] = fake


def _make_light_instance():
    _install_fake_matlab()
    AngularF16 = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.linear.angular.model"
    ).AngularF16
    inst = object.__new__(AngularF16)
    inst.selected_states = [
        "phi",
        "theta",
        "psi",
        "alpha",
        "beta",
        "p",
        "q",
        "r",
        "ele",
        "ail",
        "rud",
    ]
    inst.selected_input = ["ele", "ail", "rud"]
    inst.number_time_steps = 5
    inst.store_states = np.zeros((len(inst.selected_states), inst.number_time_steps))
    inst.store_input = np.zeros((len(inst.selected_input), inst.number_time_steps))
    return inst


def test_get_state_aliases_and_conversions():
    m = _make_light_instance()
    # exercise aliases
    assert m.get_state("wz").shape[0] == m.number_time_steps - 1
    assert m.get_state("wx").shape[0] == m.number_time_steps - 1
    assert m.get_state("wy").shape[0] == m.number_time_steps - 1
    # conversions
    assert np.allclose(m.get_state("alpha", to_deg=True), 0)
    assert np.allclose(m.get_state("alpha", to_rad=True), 0)


def test_get_control_aliases_and_conversions():
    m = _make_light_instance()
    # aliases
    assert m.get_control("stab").shape[0] == m.number_time_steps - 1
    assert m.get_control("dir").shape[0] == m.number_time_steps - 1
    # conversion branch
    assert np.allclose(m.get_control("ele", to_deg=True), 0)
