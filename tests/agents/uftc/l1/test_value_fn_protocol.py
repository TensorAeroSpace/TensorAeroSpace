"""Lock in the HJValueFunction protocol surface."""

from __future__ import annotations

import numpy as np


def test_hj_value_function_protocol_methods() -> None:
    from tensoraerospace.agent.uftc.l1.value_fn import HJValueFunction

    class Dummy:
        def value(self, x: np.ndarray) -> float:
            return 0.0

        def gradient(self, x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        def lipschitz_const(self) -> float:
            return 1.0

    d = Dummy()
    assert isinstance(d, HJValueFunction)


def test_l1_subpackage_importable() -> None:
    import tensoraerospace.agent.uftc.l1 as l1

    assert hasattr(l1, "__all__")
    assert "HJValueFunction" in l1.__all__
