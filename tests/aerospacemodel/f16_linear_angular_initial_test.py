import importlib
import sys
import types

import numpy as np
import pytest


def _install_fake_matlab():
    fake = types.ModuleType("matlab")
    fake.double = lambda x: x
    sys.modules["matlab"] = fake


def test_set_initial_state_valid_and_invalid():
    _install_fake_matlab()
    # import after installing fake matlab
    init_mod = importlib.import_module(
        "tensoraerospace.aerospacemodel.f16.linear.angular.initial"
    )
    # valid: subset of allowed keys
    out = init_mod.set_initial_state({"alpha": np.deg2rad(10)})
    assert isinstance(out, list)

    # invalid: unknown key triggers exception
    with pytest.raises(Exception):
        init_mod.set_initial_state({"unknown": 1.0})
