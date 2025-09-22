import numpy as np

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
