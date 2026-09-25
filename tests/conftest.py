"""Shared isolation for baseline tests.

The real Suite2p import initializes a settings directory in the user's home
directory.  Unit tests replace that external boundary with minimal modules so
imports remain deterministic and side-effect free.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / ".pytest_cache" / "matplotlib"))


def _noop(*args, **kwargs):
    return None


def _install_suite2p_stubs() -> None:
    suite2p = types.ModuleType("suite2p")
    suite2p.__path__ = []

    io = types.ModuleType("suite2p.io")
    for name in (
        "tiff_to_binary",
        "h5py_to_binary",
        "nwb_to_binary",
        "sbx_to_binary",
        "nd2_to_binary",
        "ome_to_binary",
        "movie_to_binary",
        "dcimg_to_binary",
    ):
        setattr(io, name, _noop)

    class UnconfiguredBinaryFile:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("BinaryFile must be replaced by a test double")

    io.BinaryFile = UnconfiguredBinaryFile

    parameters = types.ModuleType("suite2p.parameters")
    parameters.default_db = lambda: {}
    parameters.default_settings = lambda: {}
    parameters.set_db = lambda db: db
    parameters.set_settings = lambda settings: settings
    parameters.convert_settings_orig = lambda ops, db, settings: (db, settings, {})

    registration = types.ModuleType("suite2p.registration")
    registration.__path__ = []
    for name in ("register", "bidiphase", "rigid", "nonrigid", "zalign"):
        child = types.ModuleType(f"suite2p.registration.{name}")
        setattr(registration, name, child)
        sys.modules[child.__name__] = child

    extraction = types.ModuleType("suite2p.extraction")
    extraction.__path__ = []
    for name in ("extract", "masks"):
        child = types.ModuleType(f"suite2p.extraction.{name}")
        setattr(extraction, name, child)
        sys.modules[child.__name__] = child

    run_s2p = types.ModuleType("suite2p.run_s2p")

    suite2p.io = io
    suite2p.parameters = parameters
    suite2p.registration = registration
    suite2p.extraction = extraction

    sys.modules.update(
        {
            "suite2p": suite2p,
            "suite2p.io": io,
            "suite2p.parameters": parameters,
            "suite2p.registration": registration,
            "suite2p.extraction": extraction,
            "suite2p.run_s2p": run_s2p,
        }
    )


_install_suite2p_stubs()


@pytest.fixture
def import_zregister(monkeypatch):
    """Import zregister without selecting its module-level Qt backend."""
    import matplotlib

    monkeypatch.setattr(matplotlib, "use", _noop)
    sys.modules.pop("suite2p_in_depth.zregister", None)
    module = importlib.import_module("suite2p_in_depth.zregister")
    return module


@pytest.fixture
def import_correct_traces():
    sys.modules.pop("suite2p_in_depth.correct_traces", None)
    return importlib.import_module("suite2p_in_depth.correct_traces")
