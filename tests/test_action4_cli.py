from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from suite2p_in_depth import cli
from suite2p_in_depth.run_manifest import manifest_path
from suite2p_in_depth.validation import (
    _read_datasets,
    _validate_dataset_row,
    is_windows_path,
    resolve_path,
    validate_workflow,
)


def _write_zregistration_project(tmp_path: Path, *, process: bool = True) -> Path:
    raw = tmp_path / "raw" / "mouse-a" / "2026-01-02" / "1"
    raw.mkdir(parents=True)
    (raw / "movie.tif").write_bytes(b"synthetic")
    (tmp_path / "datasets.csv").write_text(
        "Name,Date,Experiments,Planes,Channels,Frame_rate,Depth,Process\n"
        f'mouse-a,2026-01-02,"[1]",4,1,30.0,0,{process}\n',
        encoding="utf-8",
    )
    (tmp_path / "flybacks.csv").write_text(
        'Planes,Frame_rate_range,Flyback_planes\n4,"[20, 40]",[0]\n',
        encoding="utf-8",
    )
    config = {
        "datasets": "datasets.csv",
        "planes_to_flyback": "flybacks.csv",
        "directories": {
            "tiffs": "raw/{Name}/{Date}/{Experiments}",
            "output": "outputs/{Name}/{Date}",
        },
        "suite2p_settings": {
            "torch_device": "cpu",
            "diameter": 3,
            "run": {
                "do_registration": True,
                "do_regmetrics": True,
                "do_detection": True,
                "do_deconvolution": False,
                "multi_registration": False,
            },
        },
        "suite2p_db": {"functional_chan": 1},
    }
    path = tmp_path / "zregistration.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "command",
    [
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
    ],
)
def test_all_unified_cli_help_commands(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args([command, "--help"])
    assert raised.value.code == 0


def test_cli_version(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["--version"])
    assert raised.value.code == 0
    assert "suite2p-in-depth 0.1.0" in capsys.readouterr().out


def test_init_copies_only_sanitized_templates_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0
    assert sorted(path.name for path in target.iterdir()) == sorted(cli.TEMPLATE_NAMES)
    assert "5 / 400" not in (target / "trace_correction.yaml").read_text()

    assert cli.main(["init", str(target)]) == 2
    assert sorted(path.name for path in target.iterdir()) == sorted(cli.TEMPLATE_NAMES)


@pytest.mark.parametrize(
    ("workflow", "filename"),
    [
        ("run-suite2p", "plain_suite2p.yaml"),
        ("zregister", "zregistration.yaml"),
        ("correct-traces", "trace_correction.yaml"),
        ("correct-traces", "zstack.yaml"),
    ],
)
def test_all_sanitized_examples_validate_and_dry_run(
    tmp_path: Path, workflow: str, filename: str
) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0
    config = target / filename

    assert cli.main(["validate", "--workflow", workflow, "--config", str(config)]) == 0
    assert cli.main([workflow, "--config", str(config), "--dry-run"]) == 0
    assert not (target / "outputs").exists()
    assert not list(target.rglob("run_manifest.json"))


def test_supported_utility_dry_runs_use_sanitized_configs(tmp_path: Path) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0

    assert (
        cli.main(
            [
                "inspect-imaging",
                "--config",
                str(target / "zregistration.yaml"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "estimate-dark-level",
                "--config",
                str(target / "trace_correction.yaml"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "recreate-bin-files",
                "--config",
                str(target / "zstack.yaml"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not list(target.rglob("run_manifest.json"))


def test_relative_paths_selected_inputs_and_dry_run_are_resolved_without_writes(
    tmp_path: Path,
) -> None:
    config = _write_zregistration_project(tmp_path)
    validated = validate_workflow("zregister", config)

    assert validated.config["datasets"] == str(tmp_path / "datasets.csv")
    assert validated.plans[0].inputs == (str(tmp_path / "raw" / "mouse-a" / "2026-01-02" / "1"),)
    assert validated.plans[0].output == str(tmp_path / "outputs" / "mouse-a" / "2026-01-02")
    assert cli.main(["zregister", "--config", str(config), "--dry-run"]) == 0
    assert not (tmp_path / "outputs").exists()


def test_dataset_numeric_value_is_accepted_when_blank_cells_force_string_dtype(
    tmp_path: Path,
) -> None:
    datasets_path = tmp_path / "datasets.csv"
    datasets_path.write_text(
        "Name,Date,Zstack_folder,Ignore_planes,Depth,Process\n"
        'unused,2026-01-01,zstack,"[]",,False\n'
        'selected,2026-01-02,zstack,"[]",67,True\n',
        encoding="utf-8",
    )

    datasets = _read_datasets(str(datasets_path), "correct-traces")
    values = _validate_dataset_row(datasets.iloc[1], 3, "correct-traces", True)

    assert values is not None
    assert values["Depth"] == 67.0


@pytest.mark.parametrize(
    "path",
    [r"C:\RawData\{Name}\{Date}", r"\\server\share\{Name}\{Date}"],
)
def test_windows_paths_are_not_reinterpreted_as_posix(tmp_path: Path, path: str) -> None:
    assert is_windows_path(path)
    resolved = resolve_path(path, tmp_path)
    assert resolved == path
    assert not resolved.startswith(str(tmp_path))


def test_expression_like_numeric_yaml_fails_before_output_creation(
    tmp_path: Path,
) -> None:
    (tmp_path / "datasets.csv").write_text(
        "Name,Date,Process\nexample,2026-01-01,False\n", encoding="utf-8"
    )
    (tmp_path / "zoom.csv").write_text("Zoom,FOV_size\n1,700\n", encoding="utf-8")
    config = {
        "datasets": "datasets.csv",
        "zoom_to_size": "zoom.csv",
        "pixels_to_microns": True,
        "use_zstack": False,
        "directories": {
            "tiffs": "raw/{Name}/{Date}",
            "suite2p": "processed/{Name}/{Date}",
            "output": "output/{Name}/{Date}",
        },
        "delta_F": {
            "absolute_zero": -100,
            "F0_percentile": 8,
            "F0_window": 60,
            "max_roi_plots": "all",
            "plot_zoomed_traces": True,
        },
        "daq": {
            "sampling_rate": 1000,
            "channel_names": "channels.csv",
            "data": "input.bin",
            "piezo_channel_name": "piezo",
            "clock_channel_name": "clock",
            "piezo_volt_per_micron": "5 / 400",
        },
    }
    path = tmp_path / "trace.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert (
        cli.main(
            [
                "correct-traces",
                "--config",
                str(path),
                "--dry-run",
            ]
        )
        == 2
    )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("value", [-1, 1.5, 10.0, True, "10", "ALL"])
def test_invalid_max_roi_plots_is_rejected(tmp_path: Path, value, capsys) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0
    config_path = target / "trace_correction.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["delta_F"]["max_roi_plots"] = value
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert cli.main(["validate", "--workflow", "correct-traces", "--config", str(config_path)]) == 2
    assert "max_roi_plots" in capsys.readouterr().out


def test_all_roi_plots_and_boolean_zoom_setting_validate(tmp_path: Path) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0
    config_path = target / "trace_correction.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["delta_F"]["max_roi_plots"] = "all"
    config["delta_F"]["plot_zoomed_traces"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert cli.main(["validate", "--workflow", "correct-traces", "--config", str(config_path)]) == 0


def test_invalid_plot_zoomed_traces_is_rejected(tmp_path: Path, capsys) -> None:
    target = tmp_path / "project"
    assert cli.main(["init", str(target)]) == 0
    config_path = target / "zstack.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["delta_F"]["plot_zoomed_traces"] = "yes"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert cli.main(["validate", "--workflow", "correct-traces", "--config", str(config_path)]) == 2
    assert "plot_zoomed_traces" in capsys.readouterr().out


def test_missing_selected_input_fails_before_output_creation(tmp_path: Path) -> None:
    config = _write_zregistration_project(tmp_path)
    movie = tmp_path / "raw" / "mouse-a" / "2026-01-02" / "1" / "movie.tif"
    movie.unlink()

    assert cli.main(["zregister", "--config", str(config)]) == 2
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config["directories"].__setitem__("tiffs", "raw/{Name}/{Date}"),
            "placeholders",
        ),
        (lambda config: config["suite2p_settings"].__setitem__("diameter", 0), "at least"),
        (
            lambda config: config["suite2p_settings"].__setitem__(
                "registration", {"align_by_chan2": "yes"}
            ),
            "align_by_chan2",
        ),
    ],
)
def test_invalid_schema_fails_with_exit_two_before_output(
    tmp_path: Path, mutation, message: str, capsys
) -> None:
    config_path = _write_zregistration_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mutation(config)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert cli.main(["zregister", "--config", str(config_path)]) == 2
    assert message in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()


def test_out_of_range_flyback_index_is_rejected(tmp_path: Path, capsys) -> None:
    config = _write_zregistration_project(tmp_path)
    (tmp_path / "flybacks.csv").write_text(
        'Planes,Frame_rate_range,Flyback_planes\n4,"[20, 40]",[4]\n',
        encoding="utf-8",
    )

    assert cli.main(["validate", "--workflow", "zregister", "--config", str(config)]) == 2
    assert "contains plane 4" in capsys.readouterr().out


def test_selected_dataset_requires_exactly_one_flyback_rule(tmp_path: Path, capsys) -> None:
    config = _write_zregistration_project(tmp_path)
    (tmp_path / "flybacks.csv").write_text(
        'Planes,Frame_rate_range,Flyback_planes\n4,"[40, 60]",[0]\n',
        encoding="utf-8",
    )

    assert cli.main(["validate", "--workflow", "zregister", "--config", str(config)]) == 2
    assert "must match exactly one flyback rule" in capsys.readouterr().out


def test_recreate_bin_files_is_dry_run_by_default_and_lists_exact_target(
    tmp_path: Path, capsys
) -> None:
    tiffs = tmp_path / "raw" / "mouse-a" / "2026-01-02"
    plane = tmp_path / "processed" / "mouse-a" / "2026-01-02" / "suite2p" / "plane0"
    tiffs.mkdir(parents=True)
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {"nchannels": 1})
    (tmp_path / "datasets.csv").write_text(
        'Name,Date,Ignore_planes,Process\nmouse-a,2026-01-02,"[]",True\n',
        encoding="utf-8",
    )
    config = {
        "datasets": "datasets.csv",
        "use_zstack": False,
        "pixels_to_microns": False,
        "directories": {
            "tiffs": "raw/{Name}/{Date}",
            "suite2p": "processed/{Name}/{Date}",
            "output": "output/{Name}/{Date}",
        },
        "delta_F": {
            "absolute_zero": -100,
            "F0_percentile": 8,
            "F0_window": 60,
            "max_roi_plots": "all",
            "plot_zoomed_traces": True,
        },
    }
    config_path = tmp_path / "trace.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert cli.main(["recreate-bin-files", "--config", str(config_path)]) == 0
    assert str(plane / "data.bin") in capsys.readouterr().out
    assert not (plane / "data.bin").exists()


def test_completed_output_requires_overwrite_and_manifest_is_atomic(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_zregistration_project(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli,
        "_run_implementation",
        lambda workflow, loaded_config, datasets: calls.append(workflow),
    )

    assert cli.main(["zregister", "--config", str(config)]) == 0
    validated = validate_workflow("zregister", config)
    path = manifest_path(validated.plans[0], "zregister")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["completion_state"] == "complete"
    assert manifest["suite2p_in_depth_version"] == "0.1.0"
    assert manifest["dataset"]["Name"] == "mouse-a"
    assert manifest["device"] == "cpu"
    assert not list(path.parent.glob("*.tmp"))

    assert cli.main(["zregister", "--config", str(config)]) == 2
    assert calls == ["zregister"]
    assert cli.main(["zregister", "--config", str(config), "--overwrite"]) == 0
    assert calls == ["zregister", "zregister"]


def test_prior_tool_manifest_requires_new_output_root(tmp_path: Path) -> None:
    config = _write_zregistration_project(tmp_path)
    plan = validate_workflow("zregister", config).plans[0]
    prior = Path(plan.output) / ".prior-tool" / "zregister" / "run_manifest.json"
    prior.parent.mkdir(parents=True)
    prior.write_text("{}", encoding="utf-8")

    assert cli.main(["zregister", "--config", str(config), "--overwrite"]) == 2
    assert not manifest_path(plan, "zregister").exists()


def test_processing_failure_returns_one_and_records_failed_summary(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_zregistration_project(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic processing failure")

    monkeypatch.setattr(cli, "_run_implementation", fail)
    assert cli.main(["zregister", "--config", str(config)]) == 1
    validated = validate_workflow("zregister", config)
    manifest = json.loads(
        manifest_path(validated.plans[0], "zregister").read_text(encoding="utf-8")
    )
    assert manifest["completion_state"] == "failed"
    assert manifest["error"] == "synthetic processing failure"


@pytest.mark.parametrize(
    "script", ["fig01_algorithms.py", "fig02_zstack.py", "fig04_zregistration.py"]
)
def test_paper_cli_requires_explicit_roots_and_supports_style_seed(
    script: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "paper" / script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--input-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--style" in result.stdout
    assert "--seed" in result.stdout

    module = importlib.import_module(f"paper.{Path(script).stem}")
    calls = []
    monkeypatch.setattr(
        module,
        "generate_figures",
        lambda input_root, output_dir, *, style, seed: calls.append(
            (input_root, output_dir, style, seed)
        ),
    )
    for missing_option in (
        ["--input-root", str(tmp_path)],
        ["--output-dir", str(tmp_path)],
    ):
        monkeypatch.setattr(sys, "argv", [script, *missing_option])
        with pytest.raises(SystemExit) as error:
            module.cli()
        assert error.value.code == 2

    input_root = tmp_path / "input"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script,
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--style",
            "classic",
            "--seed",
            "7",
        ],
    )
    module.cli()
    assert calls == [(input_root, output_dir, "classic", 7)]
