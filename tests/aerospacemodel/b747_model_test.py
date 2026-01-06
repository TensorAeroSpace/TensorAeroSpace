import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747 import LongitudinalB747


def test_b747_initialization_and_run_step():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 5
    model = LongitudinalB747(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    # One step with over-limit control to test clipping
    u = np.array([100.0])
    x1 = model.run_step(u)
    assert isinstance(x1, np.ndarray)
    assert x1.shape[0] == 4

    # Stored input must be within magnitude limits
    assert np.all(
        np.abs(model.store_input[:, 0]) <= np.array(model.input_magnitude_limits)
    )

    # Subsequent steps with zero control
    for _ in range(steps - 1):
        model.run_step(np.array([0.0]))

    # State and control histories have expected lengths
    theta_hist = model.get_state("theta")
    assert theta_hist.shape[0] == steps - 1

    ele_hist = model.get_control("ele")
    assert ele_hist.shape[0] == steps - 1

    # Output retrieval without unit conversion
    out_q = model.get_output("q")
    assert out_q.shape[0] == model.time_step - 1


def test_b747_get_state_conversions():
    """Cover to_deg/to_rad branches and aliases."""
    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    state_deg = model.get_state("wz", to_deg=True)
    assert state_deg is not None
    state_rad = model.get_state("theta", to_rad=True)
    assert state_rad is not None


def test_b747_get_control_conversions():
    """Cover to_deg/to_rad and alias branches."""
    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    ctrl_deg = model.get_control("stab", to_deg=True)
    assert ctrl_deg is not None
    ctrl_rad = model.get_control("ele", to_rad=True)
    assert ctrl_rad is not None


def test_b747_get_state_aliases_wx_wy():
    """Cover wx->p and wy->r aliases (lines 267, 269)."""
    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    # These aliases only work if p and r are in selected_states; otherwise expect exception
    # B747 longitudinal has: u, alpha, q, theta - no p or r
    with pytest.raises(Exception):
        model.get_state("wx")  # wx->p, but p not in selected_states
    with pytest.raises(Exception):
        model.get_state("wy")  # wy->r, but r not in selected_states


def test_b747_get_state_invalid():
    """Cover exception when state not found (line 271)."""
    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    with pytest.raises(Exception, match="нет в списке состояний"):
        model.get_state("nonexistent_state")


def test_b747_get_control_rud_alias_and_invalid():
    """Cover rud/dir alias (line 297) and invalid control (line 303)."""
    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    # rud/dir alias tries to get rud, but B747 longitudinal only has ele
    with pytest.raises(Exception, match="нет в списке сигналов управления"):
        model.get_control("dir")  # dir->rud, but rud not in list
    with pytest.raises(Exception, match="нет в списке сигналов управления"):
        model.get_control("bad_control")


def test_b747_selected_state_output():
    """Cover line 243: selected_state_output triggers the if branch."""
    # Note: B747 has a bug where it passes self.selected_states instead of
    # self.selected_state_output to _initialize_selected_state_index, so
    # the actual filtering doesn't work. But we still cover line 243.
    model = LongitudinalB747(
        x0=np.zeros(4),
        number_time_steps=5,
        selected_state_output=["q", "theta"],  # triggers the if branch
        dt=0.01,
    )
    # selected_state_output is truthy, so line 243 branch is covered
    assert model.selected_state_output is not None
    x1 = model.run_step(np.array([1.0]))
    # Due to implementation bug, returns all 4 states
    assert x1.shape[0] == 4


def test_b747_plot_output_validation_errors():
    """Cover plot_output validation (lines 363-368)."""
    import matplotlib

    matplotlib.use("Agg")

    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.list_state = model.selected_states

    t = np.linspace(0, 4 * model.discretisation_time, 5)

    # Both to_rad and to_deg raises
    with pytest.raises(Exception, match="to_rad или to_deg"):
        model.plot_output("theta", time=t, to_rad=True, to_deg=True)

    # Invalid output name raises
    with pytest.raises(Exception, match="нет в списке"):
        model.plot_output("nonexistent", time=t)


def test_b747_plot_output_smoke_mocked():
    """Cover plot_output method with mocked plt to avoid numpy issues."""
    from unittest.mock import MagicMock, patch

    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    model.list_state = model.selected_states

    t = np.linspace(
        0,
        (model.number_time_steps - 1) * model.discretisation_time,
        model.number_time_steps,
    )

    mock_fig = MagicMock()
    with patch("tensoraerospace.aerospacemodel.b747.plt") as mock_plt:
        mock_plt.figure.return_value = mock_fig
        # Test rus lang branch
        fig = model.plot_output("theta", time=t, lang="rus")
        assert fig is mock_fig
        mock_plt.figure.assert_called()
        mock_plt.plot.assert_called()

        # Test eng lang branch
        mock_plt.reset_mock()
        fig_eng = model.plot_output("theta", time=t, lang="eng")
        assert fig_eng is mock_fig

        # Test output_name == "u" branch (line 373)
        mock_plt.reset_mock()
        fig_u = model.plot_output("u", time=t, lang="rus")
        assert fig_u is mock_fig


def test_b747_plot_output_no_matplotlib():
    """Cover line 359: ModuleNotFoundError when plt is None."""
    from unittest.mock import patch

    model = LongitudinalB747(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.list_state = model.selected_states
    t = np.linspace(0, 4 * 0.01, 5)

    with patch("tensoraerospace.aerospacemodel.b747.plt", None):
        with pytest.raises(ModuleNotFoundError, match="matplotlib"):
            model.plot_output("theta", time=t)
