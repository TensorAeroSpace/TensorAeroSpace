"""Smoke import test that locks in the public surface of the uftc package."""

from __future__ import annotations


def test_uftc_package_importable() -> None:
    import tensoraerospace.agent.uftc as uftc

    assert hasattr(uftc, "__all__")
    # Phase 1 MVP exports — populated incrementally by later tasks.
    assert isinstance(uftc.__all__, list)


def test_uftc_fdd_subpackage_importable() -> None:
    import tensoraerospace.agent.uftc.fdd as fdd

    assert hasattr(fdd, "__all__")
