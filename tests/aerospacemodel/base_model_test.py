import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tensoraerospace.aerospacemodel.base import ModelBase


class _DummyModel(ModelBase):
    def __init__(self):
        super().__init__(
            x0=np.array([0.0, 0.0]), selected_state_output=None, t0=0, dt=0.1
        )
        # Keep control name also in list_state because base checks list_state for controls
        self.list_state = ["alpha", "theta", "ele"]
        self.control_list = ["ele"]
        self.selected_state_output = None
        self._init_selection()
        # histories for 3 steps
        self.time_step = 3
        self.x_history = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.01]),
            np.array([0.2, 0.3, 0.02]),
        ]
        self.u_history = [np.array([0.01]), np.array([0.02]), np.array([0.03])]

    def _init_selection(self):
        if self.selected_state_output:
            self.selected_state_index = [
                self.list_state.index(val) for val in self.selected_state_output
            ]
        else:
            self.selected_state_index = list(range(len(self.list_state)))
        self.yt = None
        self.ut = None
        self.state_history = []
        self.control_history = []
        self.store_outputs = []

    def run_step(self, u):
        return None


def test_initialize_selected_state_index_all_and_subset():
    m = _DummyModel()
    assert m.selected_state_index == [0, 1, 2]
    m.selected_state_output = ["theta"]
    m._init_selection()
    assert m.selected_state_index == [1]


def test_get_state_conversions_and_errors():
    m = _DummyModel()
    with pytest.raises(Exception):
        m.get_state("missing")
    with pytest.raises(Exception):
        m.get_state("alpha", to_deg=True, to_rad=True)

    state_rad = m.get_state("alpha")
    assert state_rad.shape[0] == m.time_step - 1
    state_deg = m.get_state("alpha", to_deg=True)
    state_back_rad = m.get_state("alpha", to_rad=True)
    assert state_deg.shape == state_rad.shape
    assert state_back_rad.shape == state_rad.shape


def test_get_control_conversions_and_errors():
    m = _DummyModel()
    with pytest.raises(Exception):
        m.get_control("missing")
    with pytest.raises(Exception):
        m.get_control("ele", to_deg=True, to_rad=True)

    ctrl = m.get_control("ele")
    assert ctrl.shape[0] == m.time_step - 1
    ctrl_deg = m.get_control("ele", to_deg=True)
    assert ctrl_deg.shape == ctrl.shape


def test_restart_resets_histories():
    m = _DummyModel()
    m.restart()
    assert m.time_step == 1
    assert m.u_history == []
    assert m.state_history == []
    assert m.control_history == []
    assert m.list_state == []
    assert m.control_list == []


def test_plot_state_warns_on_non_str_lang():
    m = _DummyModel()
    t = np.linspace(0, 1, 3)
    with pytest.warns(UserWarning):
        m.plot_state("alpha", time=t, lang=123)


def test_plot_error_and_transient_process_run_basic():
    m = _DummyModel()
    t = np.linspace(0, 1, 3)
    ref = np.array([0.0, 0.1, 0.2])
    m.plot_error("theta", time=t, ref_signal=ref)
    m.plot_transient_process("theta", time=t, ref_signal=ref, to_deg=False)
