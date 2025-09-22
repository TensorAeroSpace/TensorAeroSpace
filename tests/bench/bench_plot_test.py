import numpy as np

from tensoraerospace.benchmark.bench import ControlBenchmark


def test_control_benchmark_plot_smoke(monkeypatch):
    cb = ControlBenchmark()
    t = np.linspace(0, 1, 50)
    control = np.zeros_like(t)
    control[10:] = 1.0
    system = np.zeros_like(t)
    system[10:] = np.linspace(0.0, 1.0, 40)

    # Avoid opening a browser/GUI during tests by monkeypatching fig.show
    class _DummyFig:
        def show(self):
            return None

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

    metrics = cb.plot(
        control_signal=control,
        system_signal=system,
        signal_val=0.5,
        dt=t[1] - t[0],
        tps=t,
        figsize=(10, 6),
    )
    assert isinstance(metrics, dict)
