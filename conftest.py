"""Root-level conftest: ensure repo root is first on sys.path.

This conftest is loaded before any other conftest, before tests/ is inserted
into sys.path by pytest. By putting the repo root at position 0 here and
eagerly importing the ``scripts`` package, we guarantee that root-level
packages (e.g. ``scripts/``) are found before same-named packages that may
exist inside ``tests/`` subdirectories (e.g. ``tests/scripts/`` would shadow
``scripts/`` if ``tests/`` came first in sys.path).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# Eagerly anchor the root scripts/ package in sys.modules so that pytest
# cannot later shadow it with tests/scripts/ when it inserts tests/ into
# sys.path during collection.
importlib.import_module("scripts")
