"""Exercise concurrent experiment logging with the real W&B SDK, offline."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.integration
def test_two_wandb_writers_remain_usable_until_individually_closed(tmp_path, repo_root):
    # A subprocess keeps W&B's global service and settings out of other tests.
    script = textwrap.dedent("""
        from tensoraerospace.agent.metrics import MetricWriter, schema

        writers = []
        try:
            first = MetricWriter(wandb_project="metrics-regression", wandb_run_name="first")
            writers.append(first)
            second = MetricWriter(wandb_project="metrics-regression", wandb_run_name="second")
            writers.append(second)
            first.add_scalar(schema.LOSS_ACTOR, 1.0, env_step=1)
            second.add_scalar(schema.LOSS_ACTOR, 2.0, env_step=1)
            first.close()
            second.add_scalar(schema.LOSS_ACTOR, 3.0, env_step=2)
        finally:
            for writer in writers:
                writer.close()
        """)
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "WANDB_MODE": "offline",
        "WANDB_API_KEY": "0" * 40,
        "WANDB_DIR": str(tmp_path),
        "WANDB_CONFIG_DIR": str(tmp_path / "config"),
        "WANDB_CACHE_DIR": str(tmp_path / "cache"),
        "WANDB_DATA_DIR": str(tmp_path / "data"),
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
    }
    # pytest-cov 5 can leave config discovery to the child, whose cwd is a
    # temporary directory. Reuse the project config so its existing omissions
    # also apply to subprocess coverage; preserve explicitly supplied configs.
    if environment.get("COV_CORE_CONFIG") == os.pathsep:
        environment["COV_CORE_CONFIG"] = str(repo_root / "pyproject.toml")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(tmp_path.glob("wandb/offline-run-*/*.wandb"))) == 2
