"""Run suite2p-in-depth directly from a source checkout."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))
main = import_module("suite2p_in_depth.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
