import gymnasium as gym
import numpy as np

from tensoraerospace.envs.ultrastick import LinearLongitudinalUltrastick


class _StubModel:
    _FULL_STATE_ORDER = ["u", "w", "q", "theta", "h"]
    _TWO_STATE_ORDER = ["theta", "q"]

    def __init__(self, initial_state, number_time_steps, selected_state_output, t0):
        # Simulate ModelBase behavior: when selected_state_output is None,
        # use all states. Ultrastick has 5 states: ["u", "w", "q", "theta", "h"]
        # But in the test we use 2 states: ["theta", "q"]
        # So we need to match the actual state space length
        self.selected_states = self._infer_state_names(
            initial_state, selected_state_output
        )
        self.list_state = self.selected_states
        self.selected_state_index = list(range(len(self.selected_states)))

    def initialise_system(self, x0, number_time_steps):
        self._state = np.array(x0, dtype=np.float32)
        # Ensure selected_state_index is set after initialization
        if (
            not hasattr(self, "selected_state_index")
            or self.selected_state_index is None
        ):
            self.selected_state_index = list(range(len(x0)))

    def run_step(self, action):
        # simple deterministic next state
        self._state = self._state + 0.1 * np.ones_like(self._state)
        return self._state

    @classmethod
    def _infer_state_names(cls, initial_state, selected_state_output):
        if (
            selected_state_output is not None
            and isinstance(selected_state_output, (list, tuple))
            and all(isinstance(name, str) for name in selected_state_output)
        ):
            return list(selected_state_output)

        state_len = len(initial_state)
        if state_len == 2:
            return cls._TWO_STATE_ORDER.copy()
        if state_len == len(cls._FULL_STATE_ORDER):
            return cls._FULL_STATE_ORDER.copy()
        if state_len < len(cls._FULL_STATE_ORDER):
            return cls._FULL_STATE_ORDER[:state_len]
        return [f"x{i}" for i in range(state_len)]


def test_ultrastick_env_reset_and_step(monkeypatch):
    # Monkeypatch the Ultrastick model used inside the env
    import tensoraerospace.envs.ultrastick as ultramod

    monkeypatch.setattr(ultramod, "Ultrastick", _StubModel)

    initial_state = [0.0, 0.0]
    ref = np.zeros((2, 20), dtype=np.float32)

    env = LinearLongitudinalUltrastick(
        initial_state=initial_state,
        reference_signal=ref,
        number_time_steps=20,
        tracking_states=["theta", "q"],
        state_space=["theta", "q"],
        control_space=["stab"],
        output_space=["theta", "q"],
    )

    obs, info = env.reset()
    assert obs.shape == (2, 1)

    # Step with an out-of-range action to trigger clipping
    next_obs, reward, done, truncated, info = env.step(
        np.array([100.0], dtype=np.float32)
    )
    assert next_obs.shape == (2, 1)
    # Reward is returned as float from reward() method, not as numpy array
    assert isinstance(
        reward, (float, np.floating, np.ndarray)
    ), f"Reward should be float or array, got {type(reward)}"
    # If it's an array, check it's not empty; if it's a scalar, that's fine too
    if isinstance(reward, np.ndarray):
        assert reward.shape != (), "If reward is array, it should not be scalar"
    assert isinstance(done, (np.bool_, bool)), f"Done should be bool, got {type(done)}"
    assert isinstance(
        truncated, (np.bool_, bool)
    ), f"Truncated should be bool, got {type(truncated)}"
