import gymnasium as gym
import numpy as np

from tensoraerospace.envs.ultrastick import LinearLongitudinalUltrastick


class _StubModel:
    def __init__(self, initial_state, number_time_steps, selected_state_output, t0):
        self.selected_state_index = list(range(len(selected_state_output)))

    def initialise_system(self, x0, number_time_steps):
        self._state = np.array(x0, dtype=np.float32)

    def run_step(self, action):
        # simple deterministic next state
        self._state = self._state + 0.1 * np.ones_like(self._state)
        return self._state


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
    assert isinstance(reward, np.ndarray)
    assert reward.shape != ()
    assert isinstance(done, np.bool_) or isinstance(done, bool)
    assert isinstance(truncated, np.bool_) or isinstance(truncated, bool)
