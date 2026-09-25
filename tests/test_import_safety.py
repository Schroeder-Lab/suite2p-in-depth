from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ImportSideEffectDetectedError(RuntimeError):
    pass


SAFE_UTILITY_MODULES = [
    "suite2p_in_depth.utilities.determine_planes_channels_rate",
    "suite2p_in_depth.utilities.extract_lowest_pixel_values",
    "suite2p_in_depth.utilities.plot_neuropil_masks",
    "suite2p_in_depth.utilities.plot_piezo_traces",
    "suite2p_in_depth.utilities.recreate_registered_bins",
]


@pytest.mark.parametrize("module_name", SAFE_UTILITY_MODULES)
def test_utility_import_does_not_create_directories_or_load_data(module_name, monkeypatch) -> None:
    import matplotlib

    def block_mkdir(*args, **kwargs):
        raise ImportSideEffectDetectedError("module attempted to create a directory during import")

    def block_load(*args, **kwargs):
        raise ImportSideEffectDetectedError("module attempted to load data during import")

    monkeypatch.setattr(Path, "mkdir", block_mkdir)
    monkeypatch.setattr(np, "load", block_load)
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: None)

    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


@pytest.mark.parametrize(
    "script_name",
    ["fig01_algorithms.py", "fig02_zstack.py", "fig04_zregistration.py"],
)
def test_paper_generation_scripts_are_import_safe(script_name, monkeypatch) -> None:
    def block_mkdir(*args, **kwargs):
        raise ImportSideEffectDetectedError("paper script attempted to create output during import")

    monkeypatch.setattr(Path, "mkdir", block_mkdir)
    runpy.run_path(
        str(REPOSITORY_ROOT / "paper" / script_name),
        run_name="characterization_import",
    )
