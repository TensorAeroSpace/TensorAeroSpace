"""save/load round-trip preserves UFTCController behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _drive(ctl: UFTCController, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(ctl.n_state)
    out = []
    for k in range(n):
        ref = rng.normal(scale=0.05, size=ctl.n_state)
        u = ctl.predict(x, ref, time_step=k)
        out.append(u.copy())
        x = x + 0.05 * rng.normal(size=ctl.n_state) + 0.1 * u
        ctl.learn(x, ref, time_step=k)
    return np.stack(out)


def test_save_creates_expected_files(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=3, n_control=3,
        nominal_F=np.zeros((3, 3)), nominal_G=np.eye(3) * 0.1,
        config=UFTCConfig(fdd_warmup_steps=10),
    )
    _drive(ctl, 30)
    folder = ctl.save(tmp_path)
    p = Path(folder)
    assert (p / "config.json").is_file()
    assert (p / "controller_state.npz").is_file()
    assert (p / "fdd" / "kalman.npz").is_file()
    assert (p / "fdd" / "cpd.npz").is_file()
    inner_dirs = list((p / "inner").glob("*"))
    middle_dirs = list((p / "middle").glob("*"))
    assert len(inner_dirs) == 1
    assert len(middle_dirs) == 1


def test_round_trip_predict_matches(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=3, n_control=3,
        nominal_F=np.zeros((3, 3)), nominal_G=np.eye(3) * 0.1,
        config=UFTCConfig(fdd_warmup_steps=10),
    )
    _drive(ctl, 50, seed=42)
    folder = ctl.save(tmp_path)
    restored = UFTCController.from_pretrained(folder)

    # Both should produce identical predictions on the same input.
    x = np.array([0.1, -0.05, 0.02])
    ref = np.array([0.05, 0.0, -0.01])
    u_orig = ctl.predict(x, ref, time_step=100)
    u_restored = restored.predict(x, ref, time_step=100)
    assert np.allclose(u_orig, u_restored, atol=1e-9)


def test_config_json_human_readable(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=2, n_control=2,
        nominal_F=np.zeros((2, 2)), nominal_G=np.eye(2),
        config=UFTCConfig(fdd_warmup_steps=5),
    )
    folder = ctl.save(tmp_path)
    payload = json.loads((Path(folder) / "config.json").read_text())
    assert payload["policy"]["params"]["n_state"] == 2
    assert payload["policy"]["params"]["n_control"] == 2
