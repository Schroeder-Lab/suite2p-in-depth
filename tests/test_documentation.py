from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

from suite2p_in_depth import cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SHELL_BLOCK = re.compile(r"```shell\n(.*?)```", re.DOTALL)


def _documentation_files() -> list[Path]:
    return [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "CONTRIBUTING.md",
        REPOSITORY_ROOT / "SECURITY.md",
        *sorted((REPOSITORY_ROOT / "docs").glob("*.md")),
    ]


@pytest.mark.parametrize("document", _documentation_files(), ids=lambda path: path.name)
def test_local_documentation_links_resolve(document: Path) -> None:
    assert document.is_file()
    for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
        target = raw_target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = target.split("#", maxsplit=1)[0]
        assert (document.parent / relative).resolve().exists(), (
            f"Broken local link {target!r} in {document.relative_to(REPOSITORY_ROOT)}"
        )


def test_documented_suite2p_in_depth_commands_match_the_parser() -> None:
    parser = cli.build_parser()
    commands: list[list[str]] = []
    source_commands = 0
    for document in _documentation_files():
        text = document.read_text(encoding="utf-8")
        for block in SHELL_BLOCK.findall(text):
            for line in block.splitlines():
                if line.startswith("suite2p-in-depth "):
                    commands.append(shlex.split(line)[1:])
                elif line.startswith("python run_suite2p_in_depth.py "):
                    commands.append(shlex.split(line)[2:])
                    source_commands += 1

    assert commands
    assert source_commands
    documented_subcommands = {args[0] for args in commands if not args[0].startswith("--")}
    expected_subcommands = {
        "init",
        "validate",
        "run-suite2p",
        "zregister",
        "correct-traces",
        "inspect-imaging",
        "estimate-dark-level",
        "plot-piezo",
        "plot-neuropil-masks",
        "recreate-bin-files",
    }
    assert documented_subcommands == expected_subcommands

    for args in commands:
        if args in (["--help"], ["--version"]):
            with pytest.raises(SystemExit) as exit_info:
                parser.parse_args(args)
            assert exit_info.value.code == 0
        else:
            parser.parse_args(args)


def test_initialized_examples_validate_all_documented_workflows(tmp_path: Path) -> None:
    target = tmp_path / "analysis-config"
    assert cli.main(["init", str(target)]) == 0

    examples = (
        ("run-suite2p", "plain_suite2p.yaml"),
        ("zregister", "zregistration.yaml"),
        ("correct-traces", "trace_correction.yaml"),
        ("correct-traces", "zstack.yaml"),
    )
    for workflow, filename in examples:
        assert (
            cli.main(
                [
                    "validate",
                    "--workflow",
                    workflow,
                    "--config",
                    str(target / filename),
                ]
            )
            == 0
        )
        assert cli.main([workflow, "--config", str(target / filename), "--dry-run"]) == 0


def test_maintainer_metadata_and_private_root_guards() -> None:
    citation = yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["authors"] == [{"name": "Schroeder Lab"}]
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "GPL-3.0-only"
    assert "preferred-citation" not in citation
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "GPL-3.0-only"' in pyproject

    acknowledgements = (REPOSITORY_ROOT / "ACKNOWLEDGEMENTS.md").read_text(encoding="utf-8")
    for expected in (
        "Copyright 2026 University of Sussex",
        "Sylvia Schröder",
        "Liad J Baruchin",
        "Andrew Mikhniak",
        "Wellcome Trust",
    ):
        assert expected in acknowledgements

    ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    protected = {
        "/plain_suite2p.yaml",
        "/zregistration.yaml",
        "/zstack.yaml",
        "/trace_correction.yaml",
        "/preprocess_boutons.csv",
        "/preprocess_neurons.csv",
        "/planes_to_flyback.csv",
        "/zoom_to_microns.csv",
        "/Data/",
        "/scratch_files/",
    }
    assert protected <= set(ignore)
    for ignored_root in protected:
        assert not (REPOSITORY_ROOT / ignored_root.strip("/")).exists()


def test_issue_templates_are_valid_yaml() -> None:
    template_directory = REPOSITORY_ROOT / ".github" / "ISSUE_TEMPLATE"
    names = {path.name for path in template_directory.glob("*.yml")}
    assert names == {"bug_report.yml", "feature_request.yml", "config.yml"}
    for path in template_directory.glob("*.yml"):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
