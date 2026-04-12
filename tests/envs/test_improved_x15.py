"""Extended tests for ImprovedX15Env and LinearLongitudinalX15 to increase coverage.

Targets uncovered lines in tensoraerospace/envs/x15.py:
- Line 61: custom reward_func branch in LinearLongitudinalX15
- Lines 495-505: _push_history method
- Lines 858+: render() early-return paths
- Lines 922-931: close() method
- Full episode rollouts, observation normalization, double resets, etc.
"""

import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.x15 import ImprovedX15Env, LinearLongitudinalX15
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import generate_time_period

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_STEPS = 50
DT = 0.01


def _make_reference_signal(n_steps, degree=2.0):
    """Create a unit-step reference signal of shape (1, n_steps)."""
    tp = generate_time_period(tn=n_steps * DT, dt=DT)
    sig = unit_step(degree=degree, tp=tp, time_step=int(len(tp) * 0.3), output_rad=True)
    return np.reshape(sig, [1, -1])[:, :n_steps]


def _make_improved_env(**overrides):
    """Factory for ImprovedX15Env with sensible defaults."""
    defaults = dict(
        initial_state=np.array([500.0, 0.0, 0.0, 0.0], dtype=np.float64),
        reference_signal=_make_reference_signal(N_STEPS),
        number_time_steps=N_STEPS,
        dt=DT,
        initial_elevator_deg=0.0,
        use_initial_action_on_first_step=False,
    )
    defaults.update(overrides)
    return ImprovedX15Env(**defaults)


# ===========================================================================
# LinearLongitudinalX15 - custom reward_func branch (line 61)
# ===========================================================================


class TestLinearLongitudinalX15CustomReward:
    """Cover the `if reward_func:` true branch in __init__."""

    def test_custom_reward_func_is_used(self):
        tp = generate_time_period(tn=20, dt=DT)
        ref = np.reshape(
            unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]
        )
        calls = []

        def my_reward(state, ref_signal, ts):
            calls.append(ts)
            return -42.0

        env = LinearLongitudinalX15(
            initial_state=[[0], [0], [0], [0]],
            reference_signal=ref,
            number_time_steps=1000,
            reward_func=my_reward,
        )
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert reward == -42.0
        assert len(calls) == 1

    def test_render_raises(self):
        tp = generate_time_period(tn=20, dt=DT)
        ref = np.reshape(
            unit_step(degree=5, tp=tp, time_step=10, output_rad=True), [1, -1]
        )
        env = LinearLongitudinalX15(
            initial_state=[[0], [0], [0], [0]],
            reference_signal=ref,
            number_time_steps=1000,
        )
        with pytest.raises(NotImplementedError):
            env.render()


# ===========================================================================
# ImprovedX15Env - _push_history (lines 495-505)
# ===========================================================================


class TestPushHistory:
    """Cover _push_history including the trimming branches."""

    def test_push_history_basic(self):
        env = _make_improved_env()
        env._push_history(1.0, 2.0, 3.0)
        assert env._hist_theta_deg == [1.0]
        assert env._hist_theta_target_deg == [2.0]
        assert env._hist_elev_deg == [3.0]

    def test_push_history_trimming(self):
        env = _make_improved_env()
        # Set a small history length to trigger trimming quickly
        env._history_len = 3
        for i in range(5):
            env._push_history(float(i), float(i * 10), float(i * 100))
        assert len(env._hist_theta_deg) == 3
        assert len(env._hist_theta_target_deg) == 3
        assert len(env._hist_elev_deg) == 3
        # Should keep last 3 values
        assert env._hist_theta_deg == [2.0, 3.0, 4.0]
        assert env._hist_theta_target_deg == [20.0, 30.0, 40.0]
        assert env._hist_elev_deg == [200.0, 300.0, 400.0]


# ===========================================================================
# ImprovedX15Env - close() (lines 922-931)
# ===========================================================================


class TestClose:
    """Cover close() when pygame is not initialized."""

    def test_close_not_initialized(self):
        env = _make_improved_env()
        # pygame not initialized -> close should be a no-op, no error
        assert not env._pygame_initialized
        env.close()
        # Still not initialized after close
        assert not env._pygame_initialized

    def test_close_already_closed(self):
        env = _make_improved_env()
        env._pygame_closed = True
        env.close()
        assert env._pygame_closed


# ===========================================================================
# ImprovedX15Env - render() early returns (lines 858+)
# ===========================================================================


class TestRenderEarlyReturns:
    """Cover render() early-return branches without needing pygame."""

    def test_render_non_human_mode(self):
        env = _make_improved_env()
        env.reset()
        # mode != "human" -> early return
        result = env.render(mode="rgb_array")
        assert result is None

    def test_render_pygame_closed(self):
        env = _make_improved_env()
        env.reset()
        env._pygame_closed = True
        result = env.render(mode="human")
        assert result is None

    def test_render_state_is_none(self):
        env = _make_improved_env()
        env.state = None
        result = env.render(mode="human")
        assert result is None


# ===========================================================================
# ImprovedX15Env - observation normalization
# ===========================================================================


class TestObservationNormalization:
    """Verify observations are always within [-1, 1]."""

    def test_obs_bounds_after_reset(self):
        env = _make_improved_env()
        obs, _ = env.reset()
        assert obs.shape == (4,)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_obs_bounds_during_rollout(self):
        env = _make_improved_env()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert obs.shape == (4,)
            assert np.all(obs >= -1.0), f"Obs below -1: {obs}"
            assert np.all(obs <= 1.0), f"Obs above 1: {obs}"
            if terminated or truncated:
                break

    def test_obs_with_nonzero_initial_state(self):
        env = _make_improved_env(
            initial_state=np.array([500.0, 0.01, 0.01, 0.01], dtype=np.float64)
        )
        obs, _ = env.reset()
        assert obs.shape == (4,)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)


# ===========================================================================
# ImprovedX15Env - full episode rollout
# ===========================================================================


class TestFullEpisode:
    """Run a complete episode to exercise the step/truncation logic."""

    def test_full_episode_reaches_truncation(self):
        env = _make_improved_env()
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = np.array([0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        # Should run until truncation at number_time_steps - 2
        assert steps == N_STEPS - 2
        assert isinstance(total_reward, float)

    def test_full_episode_with_random_actions(self):
        env = _make_improved_env()
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        assert steps > 0
        assert steps <= N_STEPS


# ===========================================================================
# ImprovedX15Env - reward computation details
# ===========================================================================


class TestRewardComputation:
    """Verify reward properties."""

    def test_reward_is_negative(self):
        """LQR-style cost should produce non-positive rewards."""
        env = _make_improved_env()
        env.reset()
        _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        if not terminated:
            assert reward <= 0.0

    def test_reward_penalty_on_termination(self):
        """Termination from pitch limit should give -100 reward."""
        env = _make_improved_env()
        env.reset()

        def _forced_run_step(action):
            return np.array(
                [500.0, 0.0, 0.0, env.max_pitch_rad * 2.0], dtype=np.float64
            )

        env.model.run_step = _forced_run_step
        _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert terminated is True
        assert reward == -100.0

    def test_reward_with_nonzero_reference(self):
        """Reward should vary with different reference signals."""
        ref = np.ones((1, N_STEPS), dtype=np.float64) * np.deg2rad(1.0)
        env = _make_improved_env(reference_signal=ref)
        env.reset()
        _, reward1, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        # Zero reference should give different reward
        ref_zero = np.zeros((1, N_STEPS), dtype=np.float64)
        env2 = _make_improved_env(reference_signal=ref_zero)
        env2.reset()
        _, reward2, _, _, _ = env2.step(np.array([0.0], dtype=np.float32))

        # With initial theta=0, a nonzero reference should produce larger error -> more negative
        assert reward1 != reward2


# ===========================================================================
# ImprovedX15Env - double reset
# ===========================================================================


class TestDoubleReset:
    """Verify environment can be reset multiple times."""

    def test_reset_twice_gives_same_obs(self):
        env = _make_improved_env()
        obs1, _ = env.reset()
        # Step a few times
        for _ in range(5):
            env.step(np.array([0.5], dtype=np.float32))
        obs2, _ = env.reset()
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_reset_clears_state(self):
        env = _make_improved_env()
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        assert env.current_step == 5
        env.reset()
        assert env.current_step == 0
        assert env._last_reward == 0.0


# ===========================================================================
# ImprovedX15Env - construction with various parameters
# ===========================================================================


class TestConstruction:
    """Test different constructor parameter combinations."""

    def test_custom_dt(self):
        env = _make_improved_env(dt=0.05)
        assert env.dt == 0.05
        obs, _ = env.reset()
        assert obs.shape == (4,)

    def test_nonzero_initial_elevator(self):
        env = _make_improved_env(
            initial_elevator_deg=5.0,
            use_initial_action_on_first_step=True,
        )
        assert env.initial_elevator_deg == 5.0
        assert env.initial_action_norm == pytest.approx(
            5.0 / env.max_elevator_angle_deg, abs=1e-6
        )

    def test_initial_action_applied_on_first_step(self):
        env = _make_improved_env(
            initial_elevator_deg=10.0,
            use_initial_action_on_first_step=True,
        )
        env.reset()
        # Agent sends -1, but first step should use initial_elevator_deg
        obs, _, _, _, _ = env.step(np.array([-1.0], dtype=np.float32))
        expected_norm = 10.0 / env.max_elevator_angle_deg
        assert env.previous_action == pytest.approx(expected_norm, abs=1e-6)

    def test_second_step_uses_agent_action(self):
        env = _make_improved_env(
            initial_elevator_deg=10.0,
            use_initial_action_on_first_step=True,
        )
        env.reset()
        env.step(np.array([0.0], dtype=np.float32))  # first step -> initial elevator
        env.step(np.array([0.5], dtype=np.float32))  # second step -> agent action
        assert env.previous_action == pytest.approx(0.5, abs=1e-6)

    def test_negative_action_clamped(self):
        env = _make_improved_env()
        env.reset()
        env.step(np.array([-5.0], dtype=np.float32))  # clamp to -1.0
        assert env.previous_action == pytest.approx(-1.0, abs=1e-6)


# ===========================================================================
# ImprovedX15Env - action history tracking (pre_previous_action)
# ===========================================================================


class TestActionHistory:
    """Verify action history chain for jerk/smoothness terms."""

    def test_pre_previous_action_tracks(self):
        env = _make_improved_env()
        env.reset()
        env.step(np.array([0.3], dtype=np.float32))
        first_prev = env.previous_action
        env.step(np.array([0.7], dtype=np.float32))
        assert env.pre_previous_action == pytest.approx(first_prev, abs=1e-6)
        assert env.previous_action == pytest.approx(0.7, abs=1e-6)

    def test_reset_restores_action_history(self):
        env = _make_improved_env(initial_elevator_deg=3.0)
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        env.reset()
        expected = 3.0 / env.max_elevator_angle_deg
        assert env.previous_action == pytest.approx(expected, abs=1e-6)
        assert env.pre_previous_action == pytest.approx(expected, abs=1e-6)


# ===========================================================================
# ImprovedX15Env - get_init_args value verification
# ===========================================================================


class TestGetInitArgsValues:
    """Verify get_init_args returns correct values, not just keys."""

    def test_values_match_construction(self):
        env = _make_improved_env(
            number_time_steps=N_STEPS,
            dt=DT,
            initial_elevator_deg=2.5,
            use_initial_action_on_first_step=True,
        )
        args = env.get_init_args()
        assert args["number_time_steps"] == N_STEPS
        assert args["dt"] == DT
        assert args["initial_elevator_deg"] == 2.5
        assert args["use_initial_action_on_first_step"] is True


# ===========================================================================
# ImprovedX15Env - idx properties
# ===========================================================================


class TestIdxProperties:
    """Cover the _idx_q and _idx_theta properties."""

    def test_idx_q(self):
        env = _make_improved_env()
        assert env._idx_q == 2

    def test_idx_theta(self):
        env = _make_improved_env()
        assert env._idx_theta == 3


# ===========================================================================
# ImprovedX15Env - _get_obs edge case with large state
# ===========================================================================


class TestGetObsEdgeCases:
    """Cover _get_obs normalization clipping paths."""

    def test_obs_clips_large_pitch_error(self):
        env = _make_improved_env()
        env.reset()
        # Force a very large theta to check clipping
        env.state = np.array([500.0, 0.0, 0.0, 1.0], dtype=np.float64)
        obs = env._get_obs()
        assert obs.shape == (4,)
        # norm_theta should be clipped to 1.0 or -1.0
        assert abs(obs[2]) <= 1.0

    def test_obs_clips_large_pitch_rate(self):
        env = _make_improved_env()
        env.reset()
        # Force a very large q
        env.state = np.array([500.0, 0.0, 10.0, 0.0], dtype=np.float64)
        obs = env._get_obs()
        assert abs(obs[1]) <= 1.0


# ===========================================================================
# ImprovedX15Env - close() with mocked pygame (lines 922-931)
# ===========================================================================


class TestCloseWithPygame:
    """Cover close() internal logic by simulating initialized pygame state."""

    def test_close_when_pygame_initialized(self):
        env = _make_improved_env()
        # Simulate that pygame was initialized
        env._pygame_initialized = True
        env._pygame_closed = False

        class FakeDisplay:
            quit_called = False

            def quit(self):
                FakeDisplay.quit_called = True

        class FakePg:
            display = FakeDisplay()
            quit_called = False

            def quit(self):
                FakePg.quit_called = True

        env._pg = FakePg()
        env.close()
        assert env._pygame_closed is True
        assert env._pygame_initialized is False

    def test_close_when_pg_is_none(self):
        env = _make_improved_env()
        env._pygame_initialized = True
        env._pygame_closed = False
        env._pg = None
        env.close()
        assert env._pygame_closed is True
        assert env._pygame_initialized is False

    def test_close_when_pg_raises(self):
        env = _make_improved_env()
        env._pygame_initialized = True
        env._pygame_closed = False

        class FakeDisplay:
            def quit(self):
                raise RuntimeError("fake error")

        class FakePg:
            display = FakeDisplay()

            def quit(self):
                raise RuntimeError("fake error")

        env._pg = FakePg()
        # Should not raise despite internal exception
        env.close()
        assert env._pygame_closed is True
        assert env._pygame_initialized is False


# ===========================================================================
# ImprovedX15Env - _push_history edge cases
# ===========================================================================


class TestPushHistoryEdgeCases:
    """Test _push_history with exact boundary conditions."""

    def test_push_exactly_at_limit(self):
        env = _make_improved_env()
        env._history_len = 3
        # Push exactly 3 items - should NOT trigger trimming
        for i in range(3):
            env._push_history(float(i), float(i), float(i))
        assert len(env._hist_theta_deg) == 3
        assert env._hist_theta_deg == [0.0, 1.0, 2.0]

    def test_push_one_over_limit(self):
        env = _make_improved_env()
        env._history_len = 3
        # Push 4 items - should trigger trimming to 3
        for i in range(4):
            env._push_history(float(i), float(i), float(i))
        assert len(env._hist_theta_deg) == 3
        assert env._hist_theta_deg == [1.0, 2.0, 3.0]


# ===========================================================================
# ImprovedX15Env - render with mocked pygame (lines 861-918)
# ===========================================================================


class TestRenderWithMockedPygame:
    """Cover render() body by mocking pygame objects."""

    def _make_mock_surface(self):
        """Create a mock pygame surface."""

        class FakeRect:
            def __init__(self, **kwargs):
                pass

        class FakeSurface:
            def __init__(self, w=900, h=600):
                self._w = w
                self._h = h

            def get_width(self):
                return self._w

            def get_height(self):
                return self._h

            def fill(self, color):
                pass

            def blit(self, surface, pos):
                pass

            def get_rect(self, **kwargs):
                return FakeRect()

        return FakeSurface()

    def _make_mock_pg(self):
        """Create a mock pygame module."""

        class FakeFont:
            def render(self, text, antialias, color):
                return self._make_mock_surface()

            def _make_mock_surface(self):
                class S:
                    def get_width(self):
                        return 100

                    def get_height(self):
                        return 20

                    def get_rect(self, **kw):
                        return type("R", (), {})()

                return S()

        class FakeDraw:
            @staticmethod
            def rect(surface, color, rect_tuple, width=0, border_radius=0):
                pass

            @staticmethod
            def line(surface, color, start, end, width=1):
                pass

            @staticmethod
            def lines(surface, color, closed, points, width=1):
                pass

            @staticmethod
            def polygon(surface, color, points):
                pass

        class FakeDisplay:
            @staticmethod
            def flip():
                pass

            @staticmethod
            def quit():
                pass

        class FakeClock:
            def tick(self, fps):
                pass

        class FakeEvent:
            pass

        class FakePg:
            draw = FakeDraw()
            display = FakeDisplay()
            QUIT = 256

            @staticmethod
            def quit():
                pass

            class event:
                @staticmethod
                def get():
                    return []

        return FakePg, FakeFont(), FakeFont(), FakeClock()

    def test_render_full_body(self):
        """Exercise the full render path with mocked pygame."""
        env = _make_improved_env()
        env.reset()
        # Step once to have non-zero state
        env.step(np.array([0.1], dtype=np.float32))

        pg, font, small_font, clock = self._make_mock_pg()
        screen = self._make_mock_surface()

        env._pygame_initialized = True
        env._pygame_closed = False
        env._pg = pg
        env._screen = screen
        env._clock = clock
        env._font = font
        env._small_font = small_font
        env._plane_img_scaled = None  # No sprite, skip draw_aircraft sprite path

        env.render(mode="human")
        # If we get here without error, the render path was exercised
        assert env.current_step == 1

    def test_render_with_sprite(self):
        """Exercise draw_aircraft sprite branch."""
        env = _make_improved_env()
        env.reset()
        env.step(np.array([0.1], dtype=np.float32))

        pg, font, small_font, clock = self._make_mock_pg()
        screen = self._make_mock_surface()

        class FakeImage:
            def get_rect(self, **kw):
                class R:
                    pass

                return R()

        class FakeTransform:
            @staticmethod
            def rotate(img, angle):
                return FakeImage()

        pg.transform = FakeTransform()

        env._pygame_initialized = True
        env._pygame_closed = False
        env._pg = pg
        env._screen = screen
        env._clock = clock
        env._font = font
        env._small_font = small_font
        env._plane_img_scaled = FakeImage()

        env.render(mode="human")

    def test_render_quit_event(self):
        """Exercise render when a QUIT event is received."""
        env = _make_improved_env()
        env.reset()

        pg, font, small_font, clock = self._make_mock_pg()
        screen = self._make_mock_surface()

        class QuitEvent:
            type = 256  # pg.QUIT

        class FakeEventModule:
            @staticmethod
            def get():
                return [QuitEvent()]

        pg.event = FakeEventModule()

        env._pygame_initialized = True
        env._pygame_closed = False
        env._pg = pg
        env._screen = screen
        env._clock = clock
        env._font = font
        env._small_font = small_font

        env.render(mode="human")
        # close() should have been called
        assert env._pygame_closed is True


# ===========================================================================
# ImprovedX15Env - _draw_timeseries, _draw_aircraft, _draw_elevator_gauge,
# _draw_hud with mocked pygame
# ===========================================================================


class TestDrawMethods:
    """Directly test drawing helper methods with mocked pygame."""

    def _setup_env_with_mocks(self):
        env = _make_improved_env()
        env.reset()

        class FakeFont:
            def render(self, text, antialias, color):
                class S:
                    pass

                return S()

        class FakeDraw:
            @staticmethod
            def rect(surface, color, rect_tuple, width=0, border_radius=0):
                pass

            @staticmethod
            def line(surface, color, start, end, width=1):
                pass

            @staticmethod
            def lines(surface, color, closed, points, width=1):
                pass

            @staticmethod
            def polygon(surface, color, points):
                pass

        class FakeSurface:
            def get_width(self):
                return 900

            def get_height(self):
                return 600

            def fill(self, color):
                pass

            def blit(self, surface, pos):
                pass

        class FakePg:
            draw = FakeDraw()

        env._pg = FakePg()
        env._screen = FakeSurface()
        env._font = FakeFont()
        env._small_font = FakeFont()
        return env

    def test_draw_hud(self):
        env = self._setup_env_with_mocks()
        env._draw_hud(5.0, 3.0)

    def test_draw_elevator_gauge(self):
        env = self._setup_env_with_mocks()
        env._draw_elevator_gauge(10.0)

    def test_draw_aircraft_no_sprite(self):
        env = self._setup_env_with_mocks()
        env._plane_img_scaled = None
        env._draw_aircraft(0.1, 5.0, (460, 240))

    def test_draw_aircraft_with_sprite(self):
        env = self._setup_env_with_mocks()

        class FakeImage:
            def get_rect(self, **kw):
                class R:
                    pass

                return R()

        class FakeTransform:
            @staticmethod
            def rotate(img, angle):
                return FakeImage()

        env._pg.transform = FakeTransform()
        env._plane_img_scaled = FakeImage()
        env._draw_aircraft(0.1, 5.0, (460, 240))

    def test_draw_timeseries_with_data(self):
        env = self._setup_env_with_mocks()
        # Add enough history data to trigger drawing
        for i in range(10):
            env._push_history(float(i), float(i * 0.5), float(i * 2))
        env._draw_timeseries()

    def test_draw_timeseries_empty(self):
        env = self._setup_env_with_mocks()
        # No history data - should handle gracefully (draw_series returns early)
        env._draw_timeseries()


# ===========================================================================
# ImprovedX15Env - _init_pygame (lines 731-771)
# ===========================================================================


class TestInitPygame:
    """Cover _init_pygame error path."""

    def test_init_pygame_already_initialized(self):
        env = _make_improved_env()
        env._pygame_initialized = True
        # Should return immediately
        env._init_pygame()
        assert env._pygame_initialized is True
