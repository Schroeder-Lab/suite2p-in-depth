"""Atomic run manifests and conservative output-conflict handling."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import __version__
from .validation import ConfigurationError, DatasetPlan


def _package_version(distribution: str) -> str:
    """Look up an installed distribution version for a run manifest.

    Parameters
    ----------
    distribution : str
        Distribution name to query.

    Returns
    -------
    str
        Installed version, the local package version, or ``not-installed``.
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        if distribution == "suite2p-in-depth":
            return str(__version__)
        return "not-installed"


def _jsonable(value: Any) -> Any:
    """Convert nested configuration values to JSON-compatible values.

    Parameters
    ----------
    value : Any
        Scalar, mapping, or sequence to normalize.

    Returns
    -------
    Any
        JSON-compatible copy; unsupported objects become strings.
    """
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def configuration_digest(config: dict[str, Any]) -> str:
    """Hash a configuration using deterministic JSON serialization.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved configuration to identify.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_path(plan: DatasetPlan, workflow: str) -> Path:
    """Locate the tool-owned manifest for a dataset and workflow.

    Parameters
    ----------
    plan : DatasetPlan
        Dataset whose output directory contains the manifest.
    workflow : str
        Workflow name used as a manifest subdirectory.

    Returns
    -------
    Path
        Path to ``run_manifest.json`` under the dataset output.
    """
    return Path(plan.output) / ".suite2p-in-depth" / workflow / "run_manifest.json"


def check_output_policy(
    plan: DatasetPlan,
    workflow: str,
    config: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    """Check whether an existing output manifest permits a run.

    Parameters
    ----------
    plan : DatasetPlan
        Dataset and output being checked.
    workflow : str
        Workflow that will produce the output.
    config : dict[str, Any]
        Resolved configuration compared with a previous manifest.
    overwrite : bool
        Permit completed, incompatible, or unreadable prior manifests.

    Returns
    -------
    Path
        Manifest path to use for this run.

    Raises
    ------
    ConfigurationError
        If a previous manifest conflicts and overwrite is false.
    """
    path = manifest_path(plan, workflow)
    if not path.exists():
        prior_manifests = [
            candidate
            for candidate in Path(plan.output).glob(f".*/{workflow}/run_manifest.json")
            if candidate != path
        ]
        if prior_manifests:
            raise ConfigurationError(
                f"Existing run manifest uses another tool directory: {prior_manifests[0]}. "
                "Use a new output directory for this workflow."
            )
        return path
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if not overwrite:
            raise ConfigurationError(
                f"Existing run manifest is unreadable: {path}. "
                "Use --overwrite only after checking the output directory."
            ) from error
        return path
    compatible = (
        previous.get("workflow") == workflow
        and previous.get("dataset", {}).get("Name") == plan.name
        and previous.get("dataset", {}).get("Date") == plan.date
        and previous.get("configuration_sha256") == configuration_digest(config)
    )
    if previous.get("completion_state") == "complete" and not overwrite:
        raise ConfigurationError(
            f"Completed output already exists for {plan.identifier}: {path}. "
            "Pass --overwrite to run it again."
        )
    if not compatible and not overwrite:
        raise ConfigurationError(
            f"Existing output manifest is incompatible for {plan.identifier}: {path}. "
            "Pass --overwrite only if replacement is intended."
        )
    return path


def write_manifest(
    path: Path,
    *,
    workflow: str,
    config_file: Path,
    config: dict[str, Any],
    plan: DatasetPlan,
    device: str,
    completion_state: str,
    error: str | None = None,
) -> None:
    """Atomically record a dataset run and its completion state.

    Parameters
    ----------
    path : Path
        Destination manifest; its tool-owned parent is created if needed.
    workflow : str
        Workflow being recorded.
    config_file : Path
        Source configuration file.
    config : dict[str, Any]
        Resolved configuration, stored with its digest.
    plan : DatasetPlan
        Dataset values, inputs, and output directory.
    device : str
        Execution device used for the run.
    completion_state : str
        Current state, such as ``running``, ``failed``, or ``complete``.
    error : str or None, optional
        Error detail for a failed run.
    """
    now = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": 1,
        "workflow": workflow,
        "completion_state": completion_state,
        "updated_at": now,
        "suite2p_in_depth_version": _package_version("suite2p-in-depth"),
        "suite2p_version": _package_version("suite2p"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "configuration_file": str(config_file),
        "configuration_sha256": configuration_digest(config),
        "configuration": _jsonable(config),
        "dataset": _jsonable(plan.values),
        "inputs": list(plan.inputs),
        "output": plan.output,
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
