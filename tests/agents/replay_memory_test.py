import os
from pathlib import Path

import numpy as np

from tensoraerospace.agent.sac.replay_memory import ReplayMemory


def _make_transition(i: int):
    state = np.array([i, i + 1], dtype=np.float32)
    action = np.array([i], dtype=np.float32)
    reward = float(i) * 0.1
    next_state = state + 1
    done = bool(i % 2)
    return state, action, reward, next_state, done


def test_replay_memory_push_len_and_sample():
    mem = ReplayMemory(capacity=3, seed=123)

    for i in range(3):
        mem.push(*_make_transition(i))

    assert len(mem) == 3

    # Wrap-around behavior (capacity reached)
    for i in range(3, 5):
        mem.push(*_make_transition(i))

    assert len(mem) == 3

    batch = mem.sample(batch_size=3)
    state, action, reward, next_state, done = batch

    assert state.shape == (3, 2)
    assert action.shape == (3, 1)
    assert reward.shape == (3,)
    assert next_state.shape == (3, 2)
    assert done.shape == (3,)

    # Rewards must come from the inserted set (tolerant to float rounding)
    possible = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    for r in reward:
        assert np.any(np.isclose(r, possible))


def test_replay_memory_save_and_load(tmp_path: Path):
    mem = ReplayMemory(capacity=3, seed=1)
    for i in range(3):
        mem.push(*_make_transition(i))

    save_path = tmp_path / "buf.pkl"
    mem.save_buffer(env_name="test", save_path=str(save_path))
    assert os.path.exists(save_path)

    mem2 = ReplayMemory(capacity=3, seed=0)
    mem2.load_buffer(str(save_path))
    assert len(mem2) == 3
