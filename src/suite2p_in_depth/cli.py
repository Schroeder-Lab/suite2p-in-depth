"""Unified command-line interface for suite2p-in-depth."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .run_manifest import check_output_policy, write_manifest
from .validation import (
    ConfigurationError,
    ValidatedWorkflow,
    WorkflowName,
    validate_daq_config,
    validate_workflow,
)

LOGGER = logging.getLogger("suite2p_in_depth")
WORKFLOWS: tuple[WorkflowName, ...] = (
    "run-suite2p",
    "zregister",
    "correct-traces",
)
TEMPLATE_NAMES = (
    "datasets_zregistration.csv",
    "datasets_zstack.csv",
    "plain_suite2p.yaml",
    "planes_to_flyback.csv",
    "trace_correction.yaml",
    "zoom_to_microns.csv",
    "zregistration.yaml",
    "zstack.yaml",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for workflows and utility commands.

    Returns
    -------
    argparse.ArgumentParser
        Parser with configuration, dry-run, and output-control options.
    """
    parser = argparse.ArgumentParser(
        prog="suite2p-in-depth",
        description="Two-photon depth-motion preprocessing workflows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Copy sanitized configuration templates.")
    init_parser.add_argument("directory", type=Path)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration and selected inputs without writing."
    )
    validate_parser.add_argument("--workflow", choices=WORKFLOWS, required=True)
    validate_parser.add_argument("--config", type=Path, required=True)

    for workflow in WORKFLOWS:
        workflow_parser = subparsers.add_parser(
            workflow, help=f"Validate and run the {workflow} workflow."
        )
        workflow_parser.add_argument("--config", type=Path, required=True)
        workflow_parser.add_argument("--dry-run", action="store_true")
        workflow_parser.add_argument("--overwrite", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect-imaging", help="Inspect plane, channel, and frame-rate metadata."
    )
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--dry-run", action="store_true")

    dark_parser = subparsers.add_parser(
        "estimate-dark-level", help="Estimate registered-movie dark levels."
    )
    dark_parser.add_argument("--config", type=Path, required=True)
    dark_parser.add_argument("--dry-run", action="store_true")

    piezo_parser = subparsers.add_parser("plot-piezo", help="Plot synchronized piezo traces.")
    piezo_parser.add_argument("--config", type=Path, required=True)
    piezo_parser.add_argument("--output-dir", type=Path, required=True)

    masks_parser = subparsers.add_parser(
        "plot-neuropil-masks", help="Plot accepted Suite2p neuropil masks."
    )
    masks_parser.add_argument("--plane-dir", type=Path, required=True)
    masks_parser.add_argument("--output", type=Path, required=True)

    recreate_parser = subparsers.add_parser(
        "recreate-bin-files",
        help="Plan or explicitly recreate missing registered binary files.",
    )
    recreate_parser.add_argument("--config", type=Path, required=True)
    recreate_parser.add_argument("--dry-run", action="store_true")
    recreate_parser.add_argument("--overwrite", action="store_true")
    return parser


def _configure_logging() -> None:
    """Send package log messages at INFO level to standard output.

    Existing root logger handlers are replaced so CLI output has a consistent
    format on repeated invocations.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _init_directory(target: Path) -> None:
    """Copy bundled configuration templates into an empty directory.

    Parameters
    ----------
    target : Path
        Destination directory, created if necessary.

    Raises
    ------
    ConfigurationError
        If the target is a file, is nonempty, or would overwrite a template.
    """
    target = target.expanduser().resolve(strict=False)
    if target.exists() and not target.is_dir():
        raise ConfigurationError(f"Initialization target is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        raise ConfigurationError(
            f"Initialization target must be empty; refusing to overwrite: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    package_config = resources.files("suite2p_in_depth").joinpath("config")
    for name in TEMPLATE_NAMES:
        destination = target / name
        if destination.exists():
            raise ConfigurationError(f"Refusing to overwrite template: {destination}")
        destination.write_bytes(package_config.joinpath(name).read_bytes())
    LOGGER.info("Created %d sanitized templates in %s", len(TEMPLATE_NAMES), target)


def _device_for(validated: ValidatedWorkflow) -> str:
    """Resolve the execution device for a validated workflow.

    Parameters
    ----------
    validated : ValidatedWorkflow
        Workflow and requested device from configuration validation.

    Returns
    -------
    str
        ``cpu`` for trace correction, otherwise the resolved PyTorch device.
    """
    if validated.workflow == "correct-traces":
        return "cpu"
    from .general import assign_torch_device

    return str(assign_torch_device(validated.requested_device))


def _log_plan(validated: ValidatedWorkflow, device: str) -> None:
    """Log selected datasets, their inputs and outputs, and the device.

    Parameters
    ----------
    validated : ValidatedWorkflow
        Validated dataset plans to report.
    device : str
        Resolved execution device.
    """
    LOGGER.info(
        "Validated %s with %d selected dataset(s); device=%s",
        validated.workflow,
        len(validated.plans),
        device,
    )
    for plan in validated.plans:
        LOGGER.info(
            "PLAN dataset=%s inputs=%s output=%s",
            plan.identifier,
            list(plan.inputs),
            plan.output,
        )
    if not validated.plans:
        LOGGER.info("No datasets are selected (Process=true).")


def _run_implementation(
    workflow: WorkflowName,
    config: dict[str, Any],
    datasets: pd.DataFrame,
) -> None:
    """Call the implementation for one validated processing workflow.

    Parameters
    ----------
    workflow : WorkflowName
        Processing command to run.
    config : dict[str, Any]
        Resolved workflow configuration.
    datasets : pandas.DataFrame
        Dataset table with the rows to process selected by ``Process``.
    """
    if workflow == "run-suite2p":
        from .run_suite2p import process_all_datasets
    elif workflow == "zregister":
        from .zregister import process_all_datasets
    else:
        from .correct_traces import process_all_datasets
    process_all_datasets(config, datasets)


def _run_workflow(
    workflow: WorkflowName,
    config_file: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Validate, plan, and execute a workflow one dataset at a time.

    A manifest records the running and final state of each dataset. Individual
    dataset failures are logged while remaining selected datasets continue.

    Parameters
    ----------
    workflow : WorkflowName
        Processing workflow to run.
    config_file : Path
        YAML configuration to validate and resolve.
    dry_run : bool
        If true, report the plan without running datasets or writing files.
    overwrite : bool
        Allow replacement of existing completed or incompatible outputs.

    Returns
    -------
    int
        Zero when all selected datasets succeed, one if any dataset fails.
    """
    validated = validate_workflow(workflow, config_file)
    device = _device_for(validated)
    _log_plan(validated, device)
    manifest_paths = {
        plan.index: check_output_policy(plan, workflow, validated.config, overwrite=overwrite)
        for plan in validated.plans
    }
    if dry_run:
        LOGGER.info("Dry run complete; no files or directories were written.")
        return 0

    successes: list[str] = []
    failures: list[str] = []
    for plan in validated.plans:
        manifest = manifest_paths[plan.index]
        selected = validated.datasets.copy(deep=True)
        selected["Process"] = False
        for key, value in plan.values.items():
            if key in selected.columns:
                selected.at[plan.index, key] = value
        selected.at[plan.index, "Process"] = True
        write_manifest(
            manifest,
            workflow=workflow,
            config_file=validated.config_path,
            config=validated.config,
            plan=plan,
            device=device,
            completion_state="running",
        )
        try:
            _run_implementation(workflow, validated.config, selected)
        except Exception as error:
            LOGGER.exception("Dataset %s failed", plan.identifier)
            write_manifest(
                manifest,
                workflow=workflow,
                config_file=validated.config_path,
                config=validated.config,
                plan=plan,
                device=device,
                completion_state="failed",
                error=str(error),
            )
            failures.append(plan.identifier)
        else:
            write_manifest(
                manifest,
                workflow=workflow,
                config_file=validated.config_path,
                config=validated.config,
                plan=plan,
                device=device,
                completion_state="complete",
            )
            successes.append(plan.identifier)
    LOGGER.info(
        "SUMMARY workflow=%s success=%d failure=%d",
        workflow,
        len(successes),
        len(failures),
    )
    if successes:
        LOGGER.info("Successful datasets: %s", ", ".join(successes))
    if failures:
        LOGGER.error("Failed datasets: %s", ", ".join(failures))
        return 1
    return 0


def _inspect_imaging(config_file: Path, dry_run: bool) -> None:
    """Print imaging metadata for selected z-registration datasets as CSV.

    Parameters
    ----------
    config_file : Path
        Z-registration configuration and dataset table.
    dry_run : bool
        If true, only validate and log the dataset plan.
    """
    validated = validate_workflow("zregister", config_file)
    validate_daq_config(validated.config)
    _log_plan(validated, _device_for(validated))
    if dry_run:
        LOGGER.info("Dry run complete; no files or directories were written.")
        return
    from .utilities.determine_planes_channels_rate import determine_imaging_config

    result = determine_imaging_config(  # type: ignore[no-untyped-call]
        validated.config, validated.datasets.copy()
    )
    print(
        result.loc[result["Process"], ["Name", "Date", "Planes", "Channels", "Frame_rate"]].to_csv(
            index=False
        ),
        end="",
    )


def _estimate_dark_level(config_file: Path, dry_run: bool) -> None:
    """Print registered-movie dark-level estimates as CSV.

    Parameters
    ----------
    config_file : Path
        Trace-correction configuration and dataset table.
    dry_run : bool
        If true, only validate and log the dataset plan.
    """
    validated = validate_workflow("correct-traces", config_file)
    _log_plan(validated, "cpu")
    if dry_run:
        LOGGER.info("Dry run complete; no files or directories were written.")
        return
    from .utilities.extract_lowest_pixel_values import plane_percentiles_all_datasets

    result = plane_percentiles_all_datasets(validated.config, validated.datasets.copy())
    print(result.to_csv(index=False), end="")


def _plot_piezo(config_file: Path, output_directory: Path) -> None:
    """Save piezo traces for selected datasets without replacing plots.

    Parameters
    ----------
    config_file : Path
        Z-registration configuration containing DAQ inputs.
    output_directory : Path
        Destination for per-dataset PNG plots.

    Raises
    ------
    ConfigurationError
        If a requested plot already exists.
    """
    validated = validate_workflow("zregister", config_file)
    validate_daq_config(validated.config)
    output_directory = output_directory.expanduser().resolve(strict=False)
    conflicts = [
        output_directory / f"{plan.name}_{plan.date}_piezo_per_plane.png"
        for plan in validated.plans
        if (output_directory / f"{plan.name}_{plan.date}_piezo_per_plane.png").exists()
    ]
    if conflicts:
        raise ConfigurationError(f"Refusing to overwrite existing piezo plot: {conflicts[0]}")
    from .utilities.plot_piezo_traces import load_piezo_data

    output_directory.mkdir(parents=True, exist_ok=True)
    load_piezo_data(  # type: ignore[no-untyped-call]
        validated.config, validated.datasets, output_directory
    )


def _plot_neuropil_masks(plane_directory: Path, output: Path) -> None:
    """Plot accepted neuropil masks from one Suite2p plane.

    Parameters
    ----------
    plane_directory : Path
        Directory containing the plane's Suite2p outputs.
    output : Path
        PNG path, which must not already exist.

    Raises
    ------
    ConfigurationError
        If the plane directory is absent or the output exists.
    """
    plane_directory = plane_directory.expanduser().resolve(strict=False)
    output = output.expanduser().resolve(strict=False)
    if not plane_directory.is_dir():
        raise ConfigurationError(f"Plane directory does not exist: {plane_directory}")
    if output.exists():
        raise ConfigurationError(f"Refusing to overwrite existing plot: {output}")
    from .utilities.plot_neuropil_masks import plot_neuropil_masks

    plot_neuropil_masks(  # type: ignore[no-untyped-call]
        plane_directory, output_path=output, show=False
    )


def _recreate_bin_files(config_file: Path, *, dry_run: bool, overwrite: bool) -> None:
    """List missing registered binaries and optionally recreate them.

    Parameters
    ----------
    config_file : Path
        Trace-correction configuration and dataset table.
    dry_run : bool
        If true, report targets without writing binaries.
    overwrite : bool
        Required to perform recreation when ``dry_run`` is false.
    """
    validated = validate_workflow("correct-traces", config_file)
    from .utilities.recreate_registered_bins import (
        find_missing_bins,
        plan_missing_bin_targets,
    )

    targets = plan_missing_bin_targets(  # type: ignore[no-untyped-call]
        validated.config, validated.datasets
    )
    for target in targets:
        LOGGER.info("PLAN recreate registered binary: %s", target)
    if not targets:
        LOGGER.info("No registered binary files need recreation.")
        return
    if dry_run or not overwrite:
        LOGGER.info(
            "Dry run complete; pass --overwrite (without --dry-run) to recreate "
            "the exact files listed above."
        )
        return
    find_missing_bins(  # type: ignore[no-untyped-call]
        validated.config, validated.datasets
    )


def dispatch(args: argparse.Namespace) -> int:
    """Route parsed command-line arguments to the requested operation.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments returned by :func:`build_parser`.

    Returns
    -------
    int
        Process exit status from the chosen command.

    Raises
    ------
    RuntimeError
        If ``args.command`` is not a supported command.
    """
    if args.command == "init":
        _init_directory(args.directory)
        return 0
    if args.command == "validate":
        validated = validate_workflow(args.workflow, args.config)
        _log_plan(validated, _device_for(validated))
        return 0
    if args.command in WORKFLOWS:
        return _run_workflow(
            args.command,
            args.config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    if args.command == "inspect-imaging":
        _inspect_imaging(args.config, args.dry_run)
        return 0
    if args.command == "estimate-dark-level":
        _estimate_dark_level(args.config, args.dry_run)
        return 0
    if args.command == "plot-piezo":
        _plot_piezo(args.config, args.output_dir)
        return 0
    if args.command == "plot-neuropil-masks":
        _plot_neuropil_masks(args.plane_dir, args.output)
        return 0
    if args.command == "recreate-bin-files":
        _recreate_bin_files(args.config, dry_run=args.dry_run, overwrite=args.overwrite)
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and run the CLI, translating failures into exit statuses.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Arguments to parse; ``None`` uses the process command line.

    Returns
    -------
    int
        Zero on success, two for invalid input, or one for processing errors.
    """
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except ConfigurationError as error:
        LOGGER.error("%s", error)
        return 2
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    except Exception:
        LOGGER.exception("Processing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
