from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load_yaml(path: Path) -> dict:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def test_ci_workflow_has_required_platforms_gates_and_immutable_actions() -> None:
    workflow = _load_yaml(REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    jobs = workflow["jobs"]
    assert set(jobs) == {"quality", "tests", "integration", "source-checkout", "clean-wheel"}
    assert jobs["tests"]["strategy"]["matrix"]["os"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    assert jobs["clean-wheel"]["needs"] == "quality"
    assert "requirements.txt" in str(jobs["source-checkout"])
    assert "run_suite2p_in_depth.py" in str(jobs["source-checkout"])

    uses = [step["uses"] for job in jobs.values() for step in job["steps"] if "uses" in step]
    assert any(item.startswith("actions/checkout@") for item in uses)
    assert any(item.startswith("actions/setup-python@") for item in uses)
    for item in uses:
        _action, revision = item.split("@", maxsplit=1)
        assert FULL_SHA.fullmatch(revision)

    serialized = str(workflow)
    assert "--cov-fail-under=75" in serialized
    assert "MPLBACKEND" in workflow["env"]
    assert "scratch_files" not in serialized
    assert "https://github.com/MouseLand/suite2p.git" in serialized
    assert "90be8953c03c6f4275dcd392ac1c6f554116169e" in serialized


@pytest.mark.parametrize(
    "job_name", ["quality", "tests", "integration", "source-checkout", "clean-wheel"]
)
def test_ci_installs_cpu_dependencies_before_project_without_caching(job_name: str) -> None:
    workflow = _load_yaml(REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml")
    assert workflow["env"]["PIP_NO_CACHE_DIR"] == "1"
    steps = workflow["jobs"][job_name]["steps"]
    install_commands = [
        shlex.split(step["run"])
        for step in steps
        if step.get("run", "").startswith("python -m pip install ")
    ]
    assert len(install_commands) == 2
    cpu_install = install_commands[0]
    assert {"torch", "torchvision"} <= set(cpu_install)
    assert cpu_install[cpu_install.index("--index-url") + 1] == (
        "https://download.pytorch.org/whl/cpu"
    )
    assert all("cache" not in step.get("with", {}) for step in steps)


def test_dependabot_updates_python_and_action_dependencies_weekly() -> None:
    config = _load_yaml(REPOSITORY_ROOT / ".github" / "dependabot.yml")
    updates = config["updates"]
    assert {item["package-ecosystem"] for item in updates} == {"pip", "github-actions"}
    assert all(item["schedule"]["interval"] == "weekly" for item in updates)


def test_coverage_measures_the_installable_package_only() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = project["tool"]["pytest"]["ini_options"]

    assert "--cov=suite2p_in_depth" in pytest_options["addopts"]
    assert "--cov=paper" not in pytest_options["addopts"]
    assert project["tool"]["coverage"]["run"]["source"] == ["suite2p_in_depth"]
