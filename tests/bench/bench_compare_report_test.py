import numpy as np

from tensoraerospace.benchmark.bench import ControlBenchmark


def test_compare_systems_smoke(monkeypatch):
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 50)
    control1 = np.zeros_like(t)
    control1[10:] = 1.0
    system1 = np.zeros_like(t)
    system1[10:] = np.linspace(0.0, 1.0, 40)

    control2 = control1.copy()
    system2 = np.zeros_like(t)
    system2[10:] = np.linspace(0.0, 0.9, 40)

    from tensoraerospace.benchmark import bench as bench_mod

    def _mock_make_subplots(*_args, **_kwargs):
        class _F:
            def add_trace(self, *a, **k):
                pass

            def add_shape(self, *a, **k):
                pass

            def add_annotation(self, *a, **k):
                pass

            def update_layout(self, *a, **k):
                pass

            def update_xaxes(self, *a, **k):
                pass

            def update_yaxes(self, *a, **k):
                pass

            def show(self):
                return None

        return _F()

    monkeypatch.setattr(bench_mod, "make_subplots", _mock_make_subplots)

    systems_data = {
        "sys1": {"control_signal": control1, "system_signal": system1, "time": t},
        "sys2": {"control_signal": control2, "system_signal": system2, "time": t},
    }
    metrics = cb.compare_systems(systems_data, signal_val=0.5, dt=t[1] - t[0])
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"sys1", "sys2"}


def test_generate_report_basic():
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 50)
    control = np.zeros_like(t)
    control[10:] = 1.0
    system = np.zeros_like(t)
    system[10:] = np.linspace(0.0, 1.0, 40)
    report = cb.generate_report(control, system, signal_val=0.5, dt=t[1] - t[0])
    assert isinstance(report, str)
    assert "ОТЧЕТ О КАЧЕСТВЕ" in report
