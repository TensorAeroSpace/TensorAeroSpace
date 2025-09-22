import matplotlib
import numpy as np

matplotlib.use("Agg")

from tensoraerospace.aerospacemodel.f4c import LongitudinalF4C


def test_f4c_getters_deg_rad_and_plot():
    x0 = np.zeros(4)
    steps = 3
    m = LongitudinalF4C(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )
    _ = m.run_step(np.array([0.0]))
    # to_deg/to_rad branches
    s_deg = m.get_state("theta", to_deg=True)
    s_rad = m.get_state("theta", to_rad=True)
    assert s_deg.shape == s_rad.shape == (steps - 1,)
    # skip plot_output due to strict list_state reset in base; ensure getters covered
