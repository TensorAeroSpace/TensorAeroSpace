import numpy as np
import pytest
from gymnasium import spaces

from tensoraerospace.envs.ultrastick import (
    ImprovedUltrastickEnv,
    LinearLongitudinalUltrastick,
)


class _StubUltrastickModel:
    """Stub for aerospacemodel.Ultrastick used by envs.ultrastick.

    The ImprovedUltrastickEnv assumes model output order:
    [Va, alpha, theta, q, h]
    """

    selected_states = ["Va", "alpha", "theta", "q", "h"]

    def __init__(
        self, initial_state, number_time_steps, selected_state_output, t0, dt=0.01
    ):  # noqa: ARG002
        self._state = np.array(initial_state, dtype=float).reshape(-1)
        self.last_action = None

    def initialise_system(self, x0, number_time_steps):  # noqa: ARG002
        self._state = np.array(x0, dtype=float).reshape(-1)

    def run_step(self, action):
        # action = [elev_rad, throttle]
        a = np.array(action, dtype=float).reshape(-1)
        self.last_action = a.copy()
        elev_rad = float(a[0])
        # simple deterministic evolution
        y = self._state.copy()
        # theta, q respond slightly to elevator
        if y.size >= 4:
            y[2] = float(y[2] + 0.01 * elev_rad)
            y[3] = float(y[3] + 0.02 * elev_rad)
        self._state = y
        return y


@pytest.fixture
def ultra_env_default(monkeypatch):
    import tensoraerospace.envs.ultrastick as ultramod

    monkeypatch.setattr(ultramod, "Ultrastick", _StubUltrastickModel)

    initial_state = np.zeros(5, dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    env = ImprovedUltrastickEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=0.0,
        initial_throttle=0.2,
        use_initial_action_on_first_step=False,
    )
    return env


def test_improved_ultrastick_init(ultra_env_default):
    env = ultra_env_default
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (5,)
    assert env.current_step == 0


def test_improved_ultrastick_reset(ultra_env_default):
    env = ultra_env_default
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_improved_ultrastick_to_norm_action_edge_cases(ultra_env_default):
    env = ultra_env_default
    env.reset()

    # empty action -> uses [0.0, prev_thr_norm]
    a0 = env._to_norm_action([])  # noqa: SLF001
    assert a0.shape == (2,)
    assert a0[0] == pytest.approx(0.0)
    assert a0[1] == pytest.approx(env.prev_thr_norm)

    # scalar -> elevator only, throttle defaults to prev_thr_norm
    a1 = env._to_norm_action(0.7)  # noqa: SLF001
    assert a1.shape == (2,)
    assert a1[0] == pytest.approx(0.7)
    assert a1[1] == pytest.approx(env.prev_thr_norm)

    # nested list -> flattened
    a2 = env._to_norm_action([[2.0], [-2.0]])  # noqa: SLF001
    assert a2.shape == (2,)
    # clipped to [-1, 1]
    assert a2[0] == pytest.approx(1.0)
    assert a2[1] == pytest.approx(-1.0)


def test_improved_ultrastick_invalid_action_raises(ultra_env_default):
    env = ultra_env_default
    env.reset()

    with pytest.raises(ValueError, match="numeric"):
        env._to_norm_action(["bad-action"])  # noqa: SLF001

    with pytest.raises(ValueError, match="finite"):
        env._to_norm_action([np.nan, 0.0])  # noqa: SLF001


def test_improved_ultrastick_step_rate_limit_and_throttle_mapping(ultra_env_default):
    env = ultra_env_default
    env.reset()

    # command max elevator and max throttle
    obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 1.0], dtype=np.float32)
    )
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # With dt=0.01, step limit = 300*dt=3 deg, prev elev=0 deg.
    # cmd elev=1.0 -> 15 deg, low-pass 0.5 => 7.5 deg, then rate limit => 3 deg.
    assert info["elevator_deg"] == pytest.approx(3.0, abs=1e-6)
    # throttle mapping from cmd_thr_norm=1 -> thr=1.0
    assert info["throttle"] == pytest.approx(1.0, abs=1e-6)


def test_improved_ultrastick_initial_action_override(monkeypatch):
    import tensoraerospace.envs.ultrastick as ultramod

    monkeypatch.setattr(ultramod, "Ultrastick", _StubUltrastickModel)
    initial_state = np.zeros(5, dtype=np.float32)
    reference_signal = np.zeros((1, 20), dtype=np.float32)
    env = ImprovedUltrastickEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=20,
        dt=0.01,
        initial_elevator_deg=10.0,
        initial_throttle=0.7,
        use_initial_action_on_first_step=True,
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert info["elevator_deg"] == pytest.approx(10.0, abs=1e-6)
    assert info["throttle"] == pytest.approx(0.7, abs=1e-6)


def test_improved_ultrastick_truncation_flag(monkeypatch):
    import tensoraerospace.envs.ultrastick as ultramod

    monkeypatch.setattr(ultramod, "Ultrastick", _StubUltrastickModel)
    initial_state = np.zeros(5, dtype=np.float32)
    reference_signal = np.zeros((1, 2), dtype=np.float32)
    env = ImprovedUltrastickEnv(
        initial_state=initial_state,
        reference_signal=reference_signal,
        number_time_steps=2,
        dt=0.01,
        use_initial_action_on_first_step=False,
    )
    env.reset()
    _, _, _, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert truncated is True


def test_improved_ultrastick_termination_on_pitch_limit(monkeypatch, ultra_env_default):
    env = ultra_env_default
    env.reset()

    def _forced_run_step(action):
        y = np.zeros(5, dtype=np.float32)
        y[2] = env.max_pitch_rad * 1.1  # theta
        y[3] = 0.0  # q
        return y

    monkeypatch.setattr(env.model, "run_step", _forced_run_step)
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True


def test_linear_ultrastick_render_raises(monkeypatch):
    # Cover LinearLongitudinalUltrastick.render NotImplementedError without real model.
    import tensoraerospace.envs.ultrastick as ultramod

    monkeypatch.setattr(ultramod, "Ultrastick", _StubUltrastickModel)
    initial_state = np.zeros(5, dtype=np.float32)
    ref = np.zeros((1, 5), dtype=np.float32)
    env = LinearLongitudinalUltrastick(
        initial_state=initial_state,
        reference_signal=ref,
        number_time_steps=5,
        tracking_states=["theta", "q"],
        state_space=["theta", "q"],
        control_space=["stab"],
        output_space=["theta", "q"],
    )
    with pytest.raises(NotImplementedError):
        env.render()
