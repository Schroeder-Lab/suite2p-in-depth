"""Shared input and output locations for the paper figure scripts."""

import os
from pathlib import Path

DATA_ROOT = Path(
    os.environ.get("SUITE2P_IN_DEPTH_DATA_ROOT", Path.home() / "suite2p-in-depth-paper-data")
)
PLOTS_ROOT = DATA_ROOT / "plots"
