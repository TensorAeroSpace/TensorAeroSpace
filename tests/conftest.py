"""Global pytest configuration for filesystem isolation and optional deps.

Goal: make sure tests do not create files/folders in the repository root.

We achieve this by:
  - Switching the current working directory (cwd) to a unique directory under
    /tmp for every test (most code writes relative paths like "checkpoints/",
    "runs/", etc.).
  - Redirecting Python bytecode cache so __pycache__ isn't written into the repo.

If a test needs to access repository files, use the provided `repo_root` fixture.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Auto-skip tests that require optional dependencies (TensorFlow, Ray, etc.)
# ---------------------------------------------------------------------------
collect_ignore_glob: list[str] = []

# IHDP is now PyTorch-based — no TensorFlow needed

try:
    import ray  # noqa: F401
except ImportError:
    collect_ignore_glob += ["optimization/ray_test.py"]

try:
    import tensorboard  # noqa: F401
except ImportError:
    collect_ignore_glob += ["agents/a2c_narx_learner_test.py"]

# Ensure the repository root is importable even when tests change cwd to /tmp
# and/or when pytest uses importlib import mode.
# NOTE: We always move _REPO_ROOT to position 0 so that root packages (e.g.
# scripts/) take precedence over same-named packages inside tests/ subdirs
# (e.g. tests/scripts/ would shadow scripts/ if tests/ comes first).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config) -> None:  # noqa: ANN001
    """Re-anchor root-level packages after all conftest files have loaded.

    pytest loads tests/scripts/conftest.py before collection starts, which
    registers tests/scripts/ as the 'scripts' package in sys.modules.  We
    clear that stale entry here (pytest_configure runs after all conftest
    imports but before collection) so the first real import resolves to the
    root scripts/ package via the sys.path we set above.
    """
    _stale = [k for k in list(sys.modules) if k == "scripts" or k.startswith("scripts.")]
    for _key in _stale:
        # Only remove entries that point inside tests/ — leave the real one.
        mod = sys.modules[_key]
        mod_file = getattr(mod, "__file__", None) or ""
        if "tests" + os.sep + "scripts" in mod_file or "tests/scripts" in mod_file:
            del sys.modules[_key]


def _tmp_root_prefer_tmp() -> Path:
    """Prefer /tmp when available (Linux CI/local), otherwise fallback."""
    tmp = Path("/tmp")
    if tmp.is_dir() and os.access(tmp, os.W_OK):
        return tmp
    # Fallback for non-Linux platforms / restricted environments.
    import tempfile

    return Path(tempfile.gettempdir())


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (TensorAeroSpace/)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def _tests_tmp_base() -> Path:
    """Base directory for all test workdirs (under /tmp)."""
    base = _tmp_root_prefer_tmp() / f"tensoraerospace-tests-{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session", autouse=True)
def _redirect_bytecode_cache(_tests_tmp_base: Path) -> None:
    """Avoid creating __pycache__ inside the repository."""
    # Prevent writing .pyc next to source files.
    sys.dont_write_bytecode = True

    # Also redirect pycache prefix for any code that still writes bytecode.
    # (e.g. if sys.dont_write_bytecode is overridden somewhere)
    pycache = _tests_tmp_base / "pycache"
    pycache.mkdir(parents=True, exist_ok=True)
    try:
        sys.pycache_prefix = str(pycache)  # type: ignore[attr-defined]
    except Exception:
        # Older Python / restricted environments: best-effort only.
        pass


@pytest.fixture(autouse=True)
def _isolate_cwd(
    _tests_tmp_base: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run each test from its own working directory under /tmp."""
    # request.node.nodeid example: "tests/agents/test_x.py::TestY::test_z[param]"
    nodeid = request.node.nodeid
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid).strip("_")
    # Keep paths reasonably short for filesystem limits.
    if len(safe) > 160:
        safe = safe[:160]

    workdir = _tests_tmp_base / safe
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)

    # A few common tools default to relative dirs; make the intent explicit.
    monkeypatch.setenv("TENSORBOARD_LOGDIR", str(workdir / "runs"))
    monkeypatch.setenv("RAY_TMPDIR", str(workdir / "ray_tmp"))
    return workdir
