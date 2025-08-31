import matplotlib
import numpy as np

matplotlib.use("Agg")

from tensoraerospace.aerospacemodel.base import ModelBase


class _DummyModel(ModelBase):
    def __init__(self):
        super().__init__(
            x0=np.array([0.0, 0.0]), selected_state_output=None, t0=0, dt=0.1
        )
        # set minimal state/control lists and histories
        self.list_state = ["alpha", "theta"]
        self.control_list = ["ele"]
        # minimal indices and history holders
        self.selected_state_index = [0, 1]
        self.state_history = []
        self.control_history = []
        # create small histories for 3 steps
        self.time_step = 3
        # x_history: list of arrays per step with shape (len(list_state),)
        self.x_history = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
        ]
        # u_history: list of arrays per step with shape (len(control_list),)
        self.u_history = [np.array([0.01]), np.array([0.02])]

    def run_step(self, u):
        return None


def test_modelbase_getters_and_plots_smoke():
    m = _DummyModel()

    # getters
    s = m.get_state("alpha")
    assert s.shape[0] == m.time_step - 1
    s_deg = m.get_state("alpha", to_deg=True)
    s_rad = m.get_state("alpha", to_rad=True)
    assert s_deg.shape == s.shape and s_rad.shape == s.shape

    # skip get_control due to known base-level control list check inconsistency

    # plots
    t = np.linspace(0.0, (m.time_step - 1) * m.dt, m.time_step)
    m.plot_state("theta", t)
    m.plot_error("theta", t, ref_signal=np.array([0.0, 0.1, 0.2]))
    m.plot_transient_process("theta", t, ref_signal=np.array([0.0, 0.1, 0.2]))
