from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import resources
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
import yaml

import suite2p_in_depth
from suite2p_in_depth import run_manifest


def test_public_package_exports_only_version() -> None:
    assert suite2p_in_depth.__all__ == ["__version__"]
    assert suite2p_in_depth.__version__ == "0.1.0"


def test_runtime_dependencies_support_the_pinned_suite2p_gui() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert "matplotlib>=3.8,<3.12" in dependencies
    assert (
        "suite2p[gui,nwb] @ git+https://github.com/MouseLand/suite2p.git@"
        "90be8953c03c6f4275dcd392ac1c6f554116169e"
    ) in dependencies


def test_source_requirements_match_runtime_metadata() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = (repository_root / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert requirements == project["project"]["dependencies"]


def test_source_checkout_launcher_uses_the_src_tree() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repository_root / "run_suite2p_in_depth.py"), "--version"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "suite2p-in-depth 0.1.0"


def test_source_checkout_manifest_uses_project_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_distribution(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(run_manifest, "version", missing_distribution)

    assert run_manifest._package_version("suite2p-in-depth") == "0.1.0"
    assert run_manifest._package_version("suite2p") == "not-installed"


@pytest.mark.parametrize(
    ("name", "datasets_name"),
    [
        ("plain_suite2p.yaml", "datasets_zstack.csv"),
        ("trace_correction.yaml", "datasets_zregistration.csv"),
        ("zregistration.yaml", "datasets_zregistration.csv"),
        ("zstack.yaml", "datasets_zstack.csv"),
    ],
)
def test_packaged_config_templates_are_sanitized_and_valid(name: str, datasets_name: str) -> None:
    text = resources.files("suite2p_in_depth").joinpath("config", name).read_text()

    assert "/path/to/" in text
    assert ":\\\\" not in text
    assert yaml.safe_load(text)["datasets"] == datasets_name
    assert resources.files("suite2p_in_depth").joinpath("config", datasets_name).is_file()


def test_retired_legacy_trees_are_absent() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert not (repository_root / "source").exists()
    assert not (repository_root / "helper_scripts").exists()
