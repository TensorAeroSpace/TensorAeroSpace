"""Smoke imports for monitor package and dataclass shapes."""
from __future__ import annotations


def test_package_importable() -> None:
    import tensoraerospace.agent.uftc.monitor as m
    assert hasattr(m, "__all__")
    assert "VState" in m.__all__
    assert "MonitorOutput" in m.__all__
    assert "MonitorConfig" in m.__all__


def test_zero_monitor_output_has_safe_defaults() -> None:
    from tensoraerospace.agent.uftc.monitor import MonitorOutput
    z = MonitorOutput.zero()
    assert z.V_total == 0.0
    assert z.alarm == "OK"
    assert z.interventions == []
