import json
from pathlib import Path

import numpy as np

from tensoraerospace.agent.ppo.model import PPO


class _DummyEnv:
    def __init__(self):
        self.observation_space = type("S", (), {"shape": (3,)})
        self.action_space = type("A", (), {"shape": (2,)})
        self.unwrapped = self

    def reset(self):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def test_ppo_save_and_load(tmp_path: Path):
    env = _DummyEnv()
    agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
    agent.save(path=str(tmp_path))

    # Find saved folder
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs, "No saved directory found"
    model_dir = subdirs[0]

    # Ensure config exists and is valid json
    config_path = model_dir / "config.json"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        json.load(f)

    loaded = PPO.from_pretrained(str(model_dir))
    assert isinstance(loaded, PPO)
