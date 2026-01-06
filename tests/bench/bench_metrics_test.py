import numpy as np
from unittest.mock import patch

from tensoraerospace.benchmark.bench import ControlBenchmark


def test_becnchmarking_one_step_basic():
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 50)
    control = np.zeros_like(t)
    control[10:] = 1.0
    system = np.zeros_like(t)
    system[10:] = np.linspace(0.0, 1.0, 40)

    metrics = cb.becnchmarking_one_step(control, system, signal_val=0.5, dt=t[1] - t[0])
    assert "overshoot" in metrics
    assert "settling_time" in metrics
    assert "performance_index" in metrics


def test_becnchmarking_one_step_static_error_fallback():
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 20)
    control = np.ones_like(t)
    system = np.zeros_like(t)  # never reaches steady-state band
    metrics = cb.becnchmarking_one_step(control, system, signal_val=0.5, dt=t[1] - t[0])
    assert metrics["settling_time"] is not None
    # Fallback branch should use last 10% -> static error near 1.0
    assert np.isclose(metrics["static_error"], 1.0, atol=1e-6)


def test_plot_returns_metrics_and_no_show(monkeypatch):
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 20)
    control = np.zeros_like(t)
    control[5:] = 1.0
    system = control * 0.8

    called = {}

    def fake_show(self):
        called["show"] = True

    with patch("plotly.graph_objects.Figure.show", new=fake_show):
        metrics = cb.plot(
            control_signal=control,
            system_signal=system,
            signal_val=0.5,
            dt=t[1] - t[0],
            tps=t,
            figsize=(10, 6),
            title="test",
        )
    assert "overshoot" in metrics  # returned from becnchmarking_one_step
    assert called.get("show") is True


def test_generate_report_contains_metrics():
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 30)
    control = np.zeros_like(t)
    control[8:] = 1.0
    system = np.zeros_like(t)
    system[8:] = np.linspace(0.0, 0.9, len(t) - 8)
    report = cb.generate_report(
        control_signal=control,
        system_signal=system,
        signal_val=0.5,
        dt=t[1] - t[0],
        system_name="DemoSys",
    )
    assert "ОТЧЕТ О КАЧЕСТВЕ" in report
    assert "DemoSys" in report


def test_generate_report_ratings_branches():
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 30)
    control = np.ones_like(t)
    system = control * 0.5  # large static error
    report = cb.generate_report(
        control_signal=control,
        system_signal=system,
        signal_val=0.5,
        dt=t[1] - t[0],
        system_name="BigError",
    )
    assert "BigError" in report
    # Overshoot likely negative -> still should include rating strings
    assert "Перерегулирование" in report
    assert "Точность" in report
