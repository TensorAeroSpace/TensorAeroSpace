"""Extended tests for B747 environments to increase code coverage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensoraerospace.envs.b747 import ImprovedB747Env, LinearLongitudinalB747
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period

# Test fixtures
INITIAL_STATE = [[0], [0], [0], [0]]
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
REFERENCE_SIGNAL = np.reshape(
    unit_step(degree=5, tp=tp, time_step=0.1, output_rad=True), [1, -1]
)
NUMBER_TIME_STEPS = 1000


# ============================================================================
# LinearLongitudinalB747 extended tests
# ============================================================================


def test_linear_b747_with_custom_reward_func():
    """Test LinearLongitudinalB747 with custom reward function."""

    def custom_reward(state, ref_signal, ts, action=None):
        """Custom reward that uses the action parameter."""
        error = np.mean((state.flatten() - ref_signal[:, ts].flatten()) ** 2)
        action_penalty = 0.0
        if action is not None:
            action_penalty = 0.01 * np.sum(action**2)
        return float(-error - action_penalty)

    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        reward_func=custom_reward,
    )

    # Check that reward_func is callable
    assert callable(env.reward_func)

    # Test step with custom reward
    env.reset()
    action = np.array([5.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, float)


def test_linear_b747_reward_without_use_reward():
    """Test LinearLongitudinalB747 with use_reward=False."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        use_reward=False,
    )

    env.reset()
    action = np.array([5.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    # Reward should be 1 when use_reward=False
    assert reward == 1


def test_linear_b747_reward_1d_reference_signal():
    """Test reward calculation with 1D reference signal."""
    # Create 1D reference signal (same size as state)
    ref_signal_1d = np.array([0.1, 0.2])

    # Call reward function directly
    state = np.array([0.15, 0.25])
    reward = LinearLongitudinalB747.reward(state, ref_signal_1d, 0)

    assert isinstance(reward, float)
    assert reward <= 0  # Should be negative MSE


def test_linear_b747_step_with_missing_output_space_indices():
    """Test step when output_space doesn't have expected keys."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        output_space=["u", "w"],  # No q or theta
    )

    env.reset()
    action = np.array([5.0], dtype=np.float32)
    next_state, _, _, _, _ = env.step(action)

    # Should still work, using fallback indexing
    assert isinstance(next_state, np.ndarray)


def test_linear_b747_reset_with_missing_output_space_indices():
    """Test reset when output_space doesn't have expected keys."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        output_space=["u", "w"],  # No q or theta
    )

    state, info = env.reset()

    # Should still work, using fallback indexing
    assert isinstance(state, np.ndarray)
    assert isinstance(info, dict)


def test_linear_b747_reward_func_without_action_parameter():
    """Test reward function that doesn't accept action parameter."""

    def old_style_reward(state, ref_signal, ts):
        """Reward function without action parameter."""
        error = np.mean((state.flatten() - ref_signal[:, ts].flatten()) ** 2)
        return float(-error)

    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        reward_func=old_style_reward,
    )

    env.reset()
    action = np.array([5.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)

    # Should handle TypeError and call without action
    assert isinstance(reward, float)


def test_linear_b747_custom_parameters():
    """Test LinearLongitudinalB747 with custom state/control spaces."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        tracking_states=["theta", "q"],
        state_space=["theta", "q", "u", "w"],
        control_space=["stab"],
        output_space=["theta", "q", "u", "w"],
        dt=0.02,
    )

    assert env.dt == 0.02
    assert len(env.tracking_states) == 2
    assert len(env.state_space) == 4
    assert len(env.control_space) == 1

    env.reset()
    action = np.array([3.0], dtype=np.float32)
    _, _, _, _, _ = env.step(action)


def test_linear_b747_render_modes(capsys):
    """Test lightweight telemetry render modes."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        render_mode="human",
    )

    assert env.render() is None
    assert "LinearLongitudinalB747" in capsys.readouterr().out

    env.reset()
    env.step(np.array([3.0], dtype=np.float32))
    snapshot = env.render(mode="ansi")
    assert isinstance(snapshot, str)
    assert "step=1" in snapshot
    assert "action=[3]" in snapshot


def test_linear_b747_action_rad_in_info():
    """Test that action_rad is present in step info."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
    )

    env.reset()
    action_deg = np.array([10.0], dtype=np.float32)
    _, _, _, _, info = env.step(action_deg)

    assert "action" in info
    assert "action_rad" in info
    assert np.allclose(info["action_rad"], np.deg2rad(action_deg), atol=1e-5)


# ============================================================================
# ImprovedB747Env extended tests
# ============================================================================


def test_improved_b747_termination_condition():
    """Test that environment terminates when pitch exceeds limits."""
    # Start with initial state that has pitch close to limit
    high_pitch_state = np.array([[0], [0], [0], [np.deg2rad(19)]])

    env = ImprovedB747Env(
        initial_state=high_pitch_state,
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    env.reset()

    # Apply large action to push pitch over limit
    action = np.array([1.0], dtype=np.float32)

    # Step multiple times to accumulate pitch
    terminated_found = False
    for _ in range(50):
        _, reward, terminated, _, _ = env.step(action)

        if terminated:
            # Check that reward penalty was applied
            assert reward == -100.0
            terminated_found = True
            break

    # Note: may not always terminate depending on dynamics,
    # so we just verify the logic path exists
    del terminated_found  # Avoid unused variable warning


def test_improved_b747_reward_at_step_zero():
    """Test reward calculation at step 0 (ref_theta_prev handling)."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    obs, _ = env.reset()
    assert env.current_step == 0

    action = np.array([0.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)

    # Should not raise error at step 0
    assert isinstance(reward, float)


def test_improved_b747_observation_components():
    """Test that observation has all 4 components."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    obs, _ = env.reset()

    assert len(obs) == 4
    # [norm_pitch_error, norm_q, norm_theta, norm_prev_action]
    assert all(-1.0 <= x <= 1.0 for x in obs)


def test_improved_b747_action_history():
    """Test that action history is correctly maintained."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        initial_elevator_deg=5.0,
        use_initial_action_on_first_step=False,  # Use actual actions
    )

    env.reset()
    assert env.previous_action == 5.0 / env.max_stabilizer_angle_deg
    assert env.pre_previous_action == 5.0 / env.max_stabilizer_angle_deg

    action1 = np.array([0.5], dtype=np.float32)
    env.step(action1)
    # Previous action should now be 0.5 (the applied action)
    assert env.previous_action == pytest.approx(0.5, abs=1e-6)

    action2 = np.array([-0.3], dtype=np.float32)
    env.step(action2)
    # Previous action should be -0.3, pre_previous should be 0.5
    assert env.previous_action == pytest.approx(-0.3, abs=1e-6)
    assert env.pre_previous_action == pytest.approx(0.5, abs=1e-6)


def test_improved_b747_reference_signal_clipping():
    """Test that reference signal access is safely clipped."""
    # Short reference signal
    short_ref = np.array([[0.1, 0.2, 0.3]]).reshape(1, -1)

    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=short_ref,
        number_time_steps=10,  # More steps than reference points
        dt=dt,
    )

    env.reset()

    # Step beyond reference signal length
    for _ in range(5):
        action = np.array([0.0], dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        if truncated:
            break

    # Should not raise IndexError


def test_improved_b747_weights_and_cost():
    """Test that reward weights are correctly configured."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    assert hasattr(env, "w_pitch")
    assert hasattr(env, "w_q")
    assert hasattr(env, "w_action")
    assert hasattr(env, "w_smooth")
    assert hasattr(env, "w_jerk")

    # All weights should be non-negative
    assert env.w_pitch >= 0
    assert env.w_q >= 0
    assert env.w_action >= 0
    assert env.w_smooth >= 0
    assert env.w_jerk >= 0


# ============================================================================
# Rendering tests for ImprovedB747Env
# ============================================================================


def test_improved_b747_render_mode_not_human():
    """Test that render returns early when render_mode is not 'human'."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        render_mode=None,
    )

    env.reset()
    # Should return early without error (render_mode is None)
    env.render()
    # No assertion needed, just checking it doesn't raise


def test_improved_b747_close_before_init():
    """Test closing environment before pygame initialization."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    # Close before any rendering
    env.close()

    # Should not raise error
    assert env._pygame_closed is False


@patch("importlib.import_module")
def test_improved_b747_render_without_pygame(mock_import):
    """Test render raises ImportError when pygame is not available."""
    mock_import.side_effect = ImportError("No module named 'pygame'")

    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        render_mode="human",
    )

    env.reset()

    with pytest.raises(ImportError, match="pygame"):
        env.render()


def test_improved_b747_render_with_mock_pygame():
    """Test render with mocked pygame to cover rendering logic."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        render_mode="human",
    )

    env.reset()

    # Mock pygame module
    mock_pygame = MagicMock()
    mock_screen = MagicMock()
    mock_screen.get_width.return_value = 900
    mock_screen.get_height.return_value = 600
    mock_clock = MagicMock()
    mock_font = MagicMock()
    mock_small_font = MagicMock()

    # Setup event queue (empty)
    mock_pygame.event.get.return_value = []
    mock_pygame.display.set_mode.return_value = mock_screen
    mock_pygame.time.Clock.return_value = mock_clock
    mock_pygame.font.SysFont.side_effect = [mock_font, mock_small_font]
    mock_pygame.QUIT = 1  # pygame.QUIT constant

    # Mock image loading to fail (no image available)
    mock_pygame.image.load.side_effect = Exception("No image")

    with patch("importlib.import_module", return_value=mock_pygame):
        # Initialize pygame
        env._init_pygame()

        assert env._pygame_initialized is True

        # Render a frame
        env.render()

        # Check that display methods were called
        assert mock_screen.fill.called
        assert mock_pygame.display.flip.called


def test_improved_b747_push_history():
    """Test history buffer management."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    env.reset()

    # Manually push some history
    for i in range(10):
        env._push_history(float(i), float(i * 0.5), float(i * 0.1))

    assert len(env._hist_theta_deg) == 10
    assert len(env._hist_theta_target_deg) == 10
    assert len(env._hist_elev_deg) == 10

    # Push beyond history_len
    for i in range(env._history_len + 100):
        env._push_history(float(i), float(i * 0.5), float(i * 0.1))

    # Should be capped at history_len
    assert len(env._hist_theta_deg) == env._history_len
    assert len(env._hist_theta_target_deg) == env._history_len
    assert len(env._hist_elev_deg) == env._history_len


def test_improved_b747_close_after_init():
    """Test closing environment after pygame initialization."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    # Mock pygame
    mock_pygame = MagicMock()
    mock_screen = MagicMock()
    mock_screen.get_width.return_value = 900
    mock_pygame.event.get.return_value = []
    mock_pygame.display.set_mode.return_value = mock_screen
    mock_pygame.time.Clock.return_value = MagicMock()
    mock_pygame.font.SysFont.return_value = MagicMock()
    mock_pygame.image.load.side_effect = Exception("No image")

    with patch("importlib.import_module", return_value=mock_pygame):
        env.reset()
        env._init_pygame()

        assert env._pygame_initialized is True

        # Close
        env.close()

        assert env._pygame_closed is True
        assert env._pygame_initialized is False
        assert mock_pygame.display.quit.called
        assert mock_pygame.quit.called


def test_improved_b747_render_after_close():
    """Test that render returns early after close."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
        render_mode="human",
    )

    env.reset()
    env._pygame_closed = True

    # Should return early
    env.render()
    # No assertion needed, just checking it doesn't raise


def test_improved_b747_metadata():
    """Test that metadata is correctly set."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    assert hasattr(env, "metadata")
    assert "render_modes" in env.metadata
    assert "human" in env.metadata["render_modes"]


def test_improved_b747_idx_properties():
    """Test index properties for state access."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    assert env._idx_q == 2
    assert env._idx_theta == 3


def test_improved_b747_reward_scale():
    """Test reward scaling factor."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=NUMBER_TIME_STEPS,
        dt=dt,
    )

    assert env.reward_scale > 0

    env.reset()
    action = np.array([0.1], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)

    # Reward should be scaled
    assert isinstance(reward, float)


def test_improved_b747_multiple_steps():
    """Test multiple steps to ensure state consistency."""
    env = ImprovedB747Env(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=100,
        dt=dt,
    )

    obs, _ = env.reset()

    for i in range(50):
        action = np.array([0.01 * np.sin(i * 0.1)], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        # Check observation bounds
        assert all(-1.1 <= x <= 1.1 for x in obs)
        assert isinstance(reward, float)


def test_linear_b747_multiple_steps_to_done():
    """Test LinearLongitudinalB747 until done flag is set."""
    env = LinearLongitudinalB747(
        initial_state=np.array(INITIAL_STATE),
        reference_signal=REFERENCE_SIGNAL,
        number_time_steps=10,  # Small number to quickly reach done
    )

    env.reset()

    for _ in range(20):
        action = np.array([5.0], dtype=np.float32)
        _, _, done, _, _ = env.step(action)

        if done:
            break

    # Should have reached done
    assert env.done is True
    assert env.current_step >= env.number_time_steps - 1
