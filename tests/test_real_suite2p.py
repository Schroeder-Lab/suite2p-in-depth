from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_installed_official_suite2p_and_local_extensions(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "tests" / "real_suite2p_smoke.py"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    source_path = str(repository / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + existing_pythonpath if existing_pythonpath else source_path
    )

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "suite2p-home")],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
