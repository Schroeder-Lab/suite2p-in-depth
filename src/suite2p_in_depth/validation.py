"""Configuration, dataset, and cross-platform path validation."""

from __future__ import annotations

import ast
import ntpath
import os
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from string import Formatter
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

WorkflowName = Literal["run-suite2p", "zregister", "correct-traces"]
_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_EXPRESSION = re.compile(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*[/+*]\s*")


class ConfigurationError(ValueError):
    """An actionable configuration or input validation error."""


@dataclass(frozen=True)
class DatasetPlan:
    """Resolved, write-free plan for one selected dataset."""

    index: int
    name: str
    date: str
    values: dict[str, Any]
    inputs: tuple[str, ...]
    output: str

    @property
    def identifier(self) -> str:
        """Combine the dataset name and date for logs and error messages.

        Returns
        -------
        str
            Dataset identifier in ``Name/Date`` form.
        """
        return f"{self.name}/{self.date}"


@dataclass(frozen=True)
class ValidatedWorkflow:
    """Validated workflow inputs and selected dataset plans."""

    workflow: WorkflowName
    config_path: Path
    config: dict[str, Any]
    datasets: pd.DataFrame
    plans: tuple[DatasetPlan, ...]
    requested_device: str


def is_windows_path(value: str) -> bool:
    """Check whether a path begins with Windows drive or UNC syntax.

    Parameters
    ----------
    value : str
        Path string to inspect.

    Returns
    -------
    bool
        True for Windows absolute paths.
    """
    return bool(_WINDOWS_PATH.match(value))


def resolve_path(value: str, base_directory: Path) -> str:
    """Expand and normalize a configured path or path template.

    Relative paths are based on the configuration directory. Windows drive
    and UNC paths retain Windows syntax on other operating systems.

    Parameters
    ----------
    value : str
        Nonempty path, possibly with environment or user variables.
    base_directory : Path
        Base for relative paths.

    Returns
    -------
    str
        Expanded absolute or Windows-normalized path.

    Raises
    ------
    ConfigurationError
        If ``value`` is empty or is not a string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("Paths must be non-empty strings.")
    value = os.path.expandvars(os.path.expanduser(value.strip()))
    if is_windows_path(value):
        return ntpath.normpath(value)
    path = Path(value)
    if not path.is_absolute():
        path = base_directory / path
    return os.path.normpath(str(path.resolve(strict=False)))


def _mapping(value: Any, location: str) -> dict[str, Any]:
    """Require a YAML mapping at a named configuration location.

    Parameters
    ----------
    value : Any
        Parsed YAML value.
    location : str
        Location included in validation errors.

    Returns
    -------
    dict[str, Any]
        The original mapping.

    Raises
    ------
    ConfigurationError
        If the value is not a dictionary.
    """
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a YAML mapping.")
    return value


def _required(mapping: dict[str, Any], key: str, location: str) -> Any:
    """Get a required key from a configuration mapping.

    Parameters
    ----------
    mapping : dict[str, Any]
        Mapping to inspect.
    key : str
        Required key.
    location : str
        Parent location used in error messages.

    Returns
    -------
    Any
        Value stored under ``key``.

    Raises
    ------
    ConfigurationError
        If the key is absent.
    """
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration key: {location}.{key}")
    return mapping[key]


def _boolean(value: Any, location: str) -> bool:
    """Parse a boolean or case-insensitive true/false string.

    Parameters
    ----------
    value : Any
        YAML or CSV value to validate.
    location : str
        Field name used in validation errors.

    Returns
    -------
    bool
        Parsed boolean value.

    Raises
    ------
    ConfigurationError
        If the value cannot be interpreted as true or false.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ConfigurationError(f"{location} must be true or false, not {value!r}.")


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> int | float:
    """Validate a numeric YAML value and optional bounds.

    Parameters
    ----------
    value : Any
        Numeric value; strings and booleans are rejected.
    location : str
        Field name used in validation errors.
    minimum, maximum : float or None, optional
        Inclusive numeric bounds.
    integer : bool, optional
        Require an integral value and return an ``int``.

    Returns
    -------
    int or float
        Validated number.

    Raises
    ------
    ConfigurationError
        If type, integrality, or bounds are invalid.
    """
    if isinstance(value, str):
        if _EXPRESSION.match(value):
            raise ConfigurationError(
                f"{location} must be a numeric YAML value, not the expression {value!r}; "
                "calculate it first (for example, use 0.0125 instead of '5 / 400')."
            )
        raise ConfigurationError(f"{location} must be numeric, not {value!r}.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ConfigurationError(f"{location} must be numeric, not {value!r}.")
    if integer and (not float(value).is_integer()):
        raise ConfigurationError(f"{location} must be an integer, not {value!r}.")
    numeric = int(float(value)) if integer else float(value)
    if minimum is not None and numeric < minimum:
        raise ConfigurationError(f"{location} must be at least {minimum}, not {value!r}.")
    if maximum is not None and numeric > maximum:
        raise ConfigurationError(f"{location} must be at most {maximum}, not {value!r}.")
    return numeric


def _dataset_number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> int | float:
    """Validate a CSV number that may have been read as a string.

    Parameters
    ----------
    value : Any
        CSV cell value to parse.
    location : str
        Field name used in validation errors.
    minimum, maximum : float or None, optional
        Inclusive numeric bounds.
    integer : bool, optional
        Require an integral value.

    Returns
    -------
    int or float
        Validated numeric value.
    """
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except (OverflowError, ValueError):
            numeric = None
        if numeric is not None and np.isfinite(numeric):
            value = numeric
    return _number(
        value,
        location,
        minimum=minimum,
        maximum=maximum,
        integer=integer,
    )


def _list(value: Any, location: str) -> list[Any]:
    """Parse a YAML or CSV list into a Python list.

    Parameters
    ----------
    value : Any
        List, tuple, or string containing a literal list.
    location : str
        Field name used in validation errors.

    Returns
    -------
    list[Any]
        Parsed list values.

    Raises
    ------
    ConfigurationError
        If the value does not represent a list.
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ConfigurationError(
                f"{location} must be a YAML/CSV list such as [1, 2]."
            ) from error
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{location} must be a list, not {value!r}.")
    return list(value)


def _validate_template(
    template: Any,
    location: str,
    *,
    required: set[str],
    allowed: set[str],
) -> str:
    """Validate placeholders in a configured path template.

    Parameters
    ----------
    template : Any
        Nonempty format string to inspect.
    location : str
        Field name used in validation errors.
    required : set[str]
        Placeholders that must be present.
    allowed : set[str]
        Complete set of supported placeholders.

    Returns
    -------
    str
        Original validated template.

    Raises
    ------
    ConfigurationError
        If braces or placeholders are invalid.
    """
    if not isinstance(template, str) or not template.strip():
        raise ConfigurationError(f"{location} must be a non-empty path template.")
    try:
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as error:
        raise ConfigurationError(f"{location} has invalid braces: {error}") from error
    unknown = fields - allowed
    if unknown:
        raise ConfigurationError(
            f"{location} contains unsupported placeholders: {sorted(unknown)}."
        )
    missing = required - fields
    if missing:
        raise ConfigurationError(f"{location} must contain placeholders: {sorted(missing)}.")
    return template


def _format_template(template: str, values: dict[str, Any]) -> str:
    """Fill dataset placeholders in a validated path template.

    Parameters
    ----------
    template : str
        Path template with dataset fields.
    values : dict[str, Any]
        Dataset values including ``Name`` and ``Date``.

    Returns
    -------
    str
        Expanded path.

    Raises
    ------
    ConfigurationError
        If template expansion fails.
    """
    replacements = {
        "Name": values["Name"],
        "Date": values["Date"],
        "Experiments": values.get("Experiments", ""),
        "Zstack": values.get("Zstack_folder", values.get("Zstack", "")),
    }
    try:
        return template.format(**replacements)
    except (KeyError, ValueError) as error:
        raise ConfigurationError(f"Could not expand path template {template!r}: {error}") from error


def load_and_resolve_config(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    """Load YAML and resolve paths relative to the configuration file.

    Parameters
    ----------
    path : str or os.PathLike
        YAML configuration file.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        Absolute configuration path and resolved configuration mapping.

    Raises
    ------
    ConfigurationError
        If the file is missing, invalid YAML, or not a mapping.
    """
    config_path = Path(path).expanduser().resolve(strict=False)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error
    config = _mapping(loaded, "configuration")
    base = config_path.parent
    for key in ("datasets", "planes_to_flyback", "zoom_to_size"):
        if key in config:
            config[key] = resolve_path(config[key], base)
    if "directories" in config:
        directories = _mapping(config["directories"], "directories")
        config["directories"] = {
            key: resolve_path(value, base) for key, value in directories.items()
        }
    return config_path, config


def _validate_flyback_table(path: str) -> list[tuple[int, float, float]]:
    """Validate flyback plane rules from a CSV table.

    Parameters
    ----------
    path : str
        CSV with ``Planes``, ``Frame_rate_range``, and ``Flyback_planes``.

    Returns
    -------
    list[tuple[int, float, float]]
        Plane count and lower/upper frame-rate bounds per rule.

    Raises
    ------
    ConfigurationError
        If the file or any rule is invalid.
    """
    if not Path(path).is_file():
        raise ConfigurationError(f"Flyback-plane table does not exist: {path}")
    try:
        table = pd.read_csv(path)
    except Exception as error:
        raise ConfigurationError(f"Could not read flyback-plane table {path}: {error}") from error
    required = {"Planes", "Frame_rate_range", "Flyback_planes"}
    missing = required - set(table.columns)
    if missing:
        raise ConfigurationError(f"Flyback-plane table is missing columns: {sorted(missing)}.")
    rules = []
    for row_number, (_index, row) in enumerate(table.iterrows(), start=2):
        planes = int(
            _number(
                row["Planes"],
                f"flyback row {row_number}.Planes",
                minimum=1,
                integer=True,
            )
        )
        frame_range = _list(row["Frame_rate_range"], f"flyback row {row_number}.Frame_rate_range")
        if len(frame_range) != 2:
            raise ConfigurationError(
                f"flyback row {row_number}.Frame_rate_range must contain two values."
            )
        low = float(
            _number(
                frame_range[0],
                f"flyback row {row_number}.Frame_rate_range[0]",
                minimum=0,
            )
        )
        high = float(
            _number(
                frame_range[1],
                f"flyback row {row_number}.Frame_rate_range[1]",
                minimum=0,
            )
        )
        if low >= high:
            raise ConfigurationError(f"flyback row {row_number}.Frame_rate_range must increase.")
        rules.append((planes, low, high))
        flybacks = _list(row["Flyback_planes"], f"flyback row {row_number}.Flyback_planes")
        for flyback in flybacks:
            plane = _number(
                flyback,
                f"flyback row {row_number}.Flyback_planes",
                minimum=0,
                integer=True,
            )
            if plane >= planes:
                raise ConfigurationError(
                    f"flyback row {row_number} contains plane {plane}, but Planes is {planes}."
                )
    return rules


def _validate_suite2p_config(config: dict[str, Any]) -> str:
    """Validate required Suite2p settings and database fields.

    Parameters
    ----------
    config : dict[str, Any]
        Workflow configuration with Suite2p sections.

    Returns
    -------
    str
        Requested torch device name.

    Raises
    ------
    ConfigurationError
        If a required setting or field value is invalid.
    """
    settings = _mapping(_required(config, "suite2p_settings", "configuration"), "suite2p_settings")
    device = _required(settings, "torch_device", "suite2p_settings")
    if not isinstance(device, str) or device not in {"cpu", "cuda", "mps"}:
        raise ConfigurationError("suite2p_settings.torch_device must be one of: cpu, cuda, mps.")
    _number(
        _required(settings, "diameter", "suite2p_settings"),
        "suite2p_settings.diameter",
        minimum=1,
    )
    run = _mapping(_required(settings, "run", "suite2p_settings"), "suite2p_settings.run")
    for key in (
        "do_registration",
        "do_regmetrics",
        "do_detection",
        "do_deconvolution",
        "multi_registration",
    ):
        if key in run:
            _boolean(run[key], f"suite2p_settings.run.{key}")
    if "registration" in settings:
        registration = _mapping(settings["registration"], "suite2p_settings.registration")
        if "align_by_chan2" in registration:
            _boolean(
                registration["align_by_chan2"],
                "suite2p_settings.registration.align_by_chan2",
            )
    database = _mapping(_required(config, "suite2p_db", "configuration"), "suite2p_db")
    if "functional_chan" in database:
        _number(
            database["functional_chan"],
            "suite2p_db.functional_chan",
            minimum=1,
            integer=True,
        )
    return device


def _validate_trace_config(config: dict[str, Any]) -> None:
    """Validate trace-correction settings and optional DAQ inputs.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration with ``delta_F`` and optional z-correction settings.

    Raises
    ------
    ConfigurationError
        If a required setting, range, or referenced table is invalid.
    """
    delta_f = _mapping(_required(config, "delta_F", "configuration"), "delta_F")
    _number(_required(delta_f, "absolute_zero", "delta_F"), "delta_F.absolute_zero")
    _number(
        _required(delta_f, "F0_percentile", "delta_F"),
        "delta_F.F0_percentile",
        minimum=0,
        maximum=100,
    )
    _number(
        _required(delta_f, "F0_window", "delta_F"),
        "delta_F.F0_window",
        minimum=0,
    )
    for key in ("save_F_neuropilcorrected", "plot"):
        if key in delta_f:
            _boolean(delta_f[key], f"delta_F.{key}")
    max_roi_plots = _required(delta_f, "max_roi_plots", "delta_F")
    if isinstance(max_roi_plots, str):
        if max_roi_plots != "all":
            raise ConfigurationError(
                "delta_F.max_roi_plots must be a non-negative integer or 'all', "
                f"not {max_roi_plots!r}."
            )
    else:
        if not isinstance(max_roi_plots, Integral) or isinstance(max_roi_plots, (bool, np.bool_)):
            raise ConfigurationError(
                "delta_F.max_roi_plots must be a non-negative integer or 'all', "
                f"not {max_roi_plots!r}."
            )
        _number(max_roi_plots, "delta_F.max_roi_plots", minimum=0, integer=True)
    _boolean(
        _required(delta_f, "plot_zoomed_traces", "delta_F"),
        "delta_F.plot_zoomed_traces",
    )
    if "pixels_to_microns" in config:
        pixels_to_microns = _boolean(config["pixels_to_microns"], "pixels_to_microns")
        if pixels_to_microns:
            zoom_path = _required(config, "zoom_to_size", "configuration")
            if not Path(zoom_path).is_file():
                raise ConfigurationError(f"Zoom-to-size table does not exist: {zoom_path}")
    if "daq" in config:
        validate_daq_config(config)
    use_zstack = _boolean(config.get("use_zstack", False), "use_zstack")
    if use_zstack:
        correction = _mapping(_required(config, "z_correction", "configuration"), "z_correction")
        _number(
            _required(correction, "ref_frames", "z_correction"),
            "z_correction.ref_frames",
            minimum=0.000001,
            maximum=1,
        )
        _number(
            _required(correction, "spacing", "z_correction"),
            "z_correction.spacing",
            minimum=0.000001,
        )
        _number(
            _required(correction, "sigma_trace", "z_correction"),
            "z_correction.sigma_trace",
            minimum=0,
        )


def validate_daq_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate DAQ sampling, conversion, and channel settings.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration containing the ``daq`` section.

    Returns
    -------
    dict[str, Any]
        Original validated DAQ mapping.

    Raises
    ------
    ConfigurationError
        If required rates or channel names are invalid.
    """
    daq = _mapping(_required(config, "daq", "configuration"), "daq")
    _number(
        _required(daq, "sampling_rate", "daq"),
        "daq.sampling_rate",
        minimum=0.000001,
    )
    _number(
        _required(daq, "piezo_volt_per_micron", "daq"),
        "daq.piezo_volt_per_micron",
        minimum=0.000001,
    )
    for key in ("channel_names", "data", "piezo_channel_name", "clock_channel_name"):
        value = _required(daq, key, "daq")
        if not isinstance(value, str) or not value:
            raise ConfigurationError(f"daq.{key} must be a non-empty string.")
    return daq


def _read_datasets(path: str, workflow: WorkflowName) -> pd.DataFrame:
    """Read a dataset CSV and check required workflow columns.

    Parameters
    ----------
    path : str
        Dataset CSV path.
    workflow : WorkflowName
        Workflow determining required columns.

    Returns
    -------
    pandas.DataFrame
        Dataset table with text-preserved names and dates.

    Raises
    ------
    ConfigurationError
        If the file cannot be read or columns are missing.
    """
    if not Path(path).is_file():
        raise ConfigurationError(f"Dataset CSV does not exist: {path}")
    try:
        datasets = pd.read_csv(path, dtype={"Name": str, "Date": str}, keep_default_na=False)
    except Exception as error:
        raise ConfigurationError(f"Could not read dataset CSV {path}: {error}") from error
    base = {"Name", "Date", "Process"}
    workflow_columns = (
        {"Experiments", "Planes", "Channels", "Frame_rate"}
        if workflow in {"run-suite2p", "zregister"}
        else set()
    )
    if workflow == "correct-traces":
        workflow_columns = set()
    missing = (base | workflow_columns) - set(datasets.columns)
    if missing:
        raise ConfigurationError(f"Dataset CSV is missing columns: {sorted(missing)}.")
    return datasets


def _validate_dataset_row(
    row: pd.Series, row_number: int, workflow: WorkflowName, use_zstack: bool
) -> dict[str, Any] | None:
    """Validate and normalize a selected dataset CSV row.

    Parameters
    ----------
    row : pandas.Series
        Dataset values from one CSV row.
    row_number : int
        One-based CSV line number for errors.
    workflow : WorkflowName
        Workflow selecting required imaging metadata.
    use_zstack : bool
        Require z-stack fields for trace correction when true.

    Returns
    -------
    dict[str, Any] or None
        Normalized values, or ``None`` when ``Process`` is false.

    Raises
    ------
    ConfigurationError
        If selected dataset fields are missing or invalid.
    """
    location = f"dataset row {row_number}"
    process = _boolean(row["Process"], f"{location}.Process")
    if not process:
        return None
    name = str(row["Name"]).strip()
    date = str(row["Date"]).strip()
    if not name or not date:
        raise ConfigurationError(f"{location} requires non-empty Name and Date.")
    values: dict[str, Any] = {str(key): value for key, value in row.items()}
    values["Name"] = name
    values["Date"] = date
    values["Process"] = True
    if workflow in {"run-suite2p", "zregister"}:
        experiments = _list(row["Experiments"], f"{location}.Experiments")
        if not experiments:
            raise ConfigurationError(f"{location}.Experiments must not be empty.")
        values["Experiments"] = experiments
        values["Planes"] = _dataset_number(
            row["Planes"], f"{location}.Planes", minimum=1, integer=True
        )
        values["Channels"] = _dataset_number(
            row["Channels"], f"{location}.Channels", minimum=1, integer=True
        )
        values["Frame_rate"] = _dataset_number(
            row["Frame_rate"], f"{location}.Frame_rate", minimum=0.000001
        )
    for key in ("Flyback", "Ignore_planes"):
        if key in row.index and row[key] != "":
            values[key] = [
                _number(item, f"{location}.{key}", minimum=0, integer=True)
                for item in _list(row[key], f"{location}.{key}")
            ]
            if key == "Flyback" and workflow in {"run-suite2p", "zregister"}:
                invalid = [item for item in values[key] if item >= values["Planes"]]
                if invalid:
                    raise ConfigurationError(
                        f"{location}.Flyback contains {invalid[0]}, but Planes is "
                        f"{values['Planes']}."
                    )
    if use_zstack:
        for key in ("Zstack_folder", "Ignore_planes", "Depth"):
            if key not in row.index:
                raise ConfigurationError(
                    f"Dataset CSV requires column {key!r} when use_zstack is true."
                )
        if not str(row["Zstack_folder"]).strip():
            raise ConfigurationError(f"{location}.Zstack_folder must not be empty.")
        values["Zstack_folder"] = str(row["Zstack_folder"]).strip()
        values["Depth"] = _dataset_number(row["Depth"], f"{location}.Depth")
    elif "Depth" in row.index and row["Depth"] != "":
        values["Depth"] = _dataset_number(row["Depth"], f"{location}.Depth")
    return values


def _existing_directory(path: str, location: str) -> None:
    """Require an input directory to exist.

    Parameters
    ----------
    path : str
        Directory path to inspect.
    location : str
        Description used in validation errors.

    Raises
    ------
    ConfigurationError
        If ``path`` is not a directory.
    """
    if not os.path.isdir(path):
        raise ConfigurationError(f"{location} directory does not exist: {path}")


def _dataset_plan(
    index: int,
    values: dict[str, Any],
    workflow: WorkflowName,
    directories: dict[str, str],
    use_zstack: bool,
) -> DatasetPlan:
    """Resolve input and output paths for one selected dataset.

    Parameters
    ----------
    index : int
        Dataset table row index.
    values : dict[str, Any]
        Normalized dataset values.
    workflow : WorkflowName
        Workflow determining required input directories.
    directories : dict[str, str]
        Validated path templates.
    use_zstack : bool
        Include the z-stack directory when true.

    Returns
    -------
    DatasetPlan
        Resolved, write-free dataset execution plan.

    Raises
    ------
    ConfigurationError
        If required directories or TIFF files are missing.
    """
    inputs: list[str] = []
    if workflow in {"run-suite2p", "zregister"}:
        for experiment in values["Experiments"]:
            expanded_values = dict(values, Experiments=experiment)
            path = _format_template(directories["tiffs"], expanded_values)
            _existing_directory(path, f"{values['Name']}/{values['Date']} TIFF input")
            if not any(entry.lower().endswith((".tif", ".tiff")) for entry in os.listdir(path)):
                raise ConfigurationError(f"No TIFF files found in input directory: {path}")
            inputs.append(path)
    else:
        for key in ("tiffs", "suite2p"):
            path = _format_template(directories[key], values)
            if key == "suite2p":
                path = os.path.join(path, "suite2p")
            _existing_directory(path, f"{values['Name']}/{values['Date']} {key}")
            inputs.append(path)
        if use_zstack:
            path = _format_template(directories["zstack"], values)
            _existing_directory(path, f"{values['Name']}/{values['Date']} z-stack")
            tiffs = [
                entry for entry in os.listdir(path) if entry.lower().endswith((".tif", ".tiff"))
            ]
            if len(tiffs) != 1:
                raise ConfigurationError(
                    f"Expected exactly one z-stack TIFF in {path}; found {len(tiffs)}."
                )
            inputs.append(path)
    output = _format_template(directories["output"], values)
    return DatasetPlan(
        index=index,
        name=values["Name"],
        date=values["Date"],
        values=values,
        inputs=tuple(inputs),
        output=output,
    )


def validate_workflow(
    workflow: WorkflowName, config_file: str | os.PathLike[str]
) -> ValidatedWorkflow:
    """Validate a workflow and plan selected datasets without writing files.

    Parameters
    ----------
    workflow : WorkflowName
        Workflow to validate.
    config_file : str or os.PathLike
        YAML configuration path.

    Returns
    -------
    ValidatedWorkflow
        Resolved configuration, dataset table, and selected dataset plans.

    Raises
    ------
    ConfigurationError
        If configuration, dataset values, or required inputs are invalid.
    """
    config_path, config = load_and_resolve_config(config_file)
    directories = _mapping(_required(config, "directories", "configuration"), "directories")
    allowed = {"Name", "Date", "Experiments", "Zstack"}
    if workflow in {"run-suite2p", "zregister"}:
        requested_device = _validate_suite2p_config(config)
        flybacks = _required(config, "planes_to_flyback", "configuration")
        flyback_rules = _validate_flyback_table(flybacks)
        required_directories = {"tiffs", "output"}
        for key in required_directories:
            _required(directories, key, "directories")
        _validate_template(
            directories["tiffs"],
            "directories.tiffs",
            required={"Name", "Date", "Experiments"},
            allowed=allowed,
        )
        _validate_template(
            directories["output"],
            "directories.output",
            required={"Name", "Date"},
            allowed=allowed,
        )
        use_zstack = False
    else:
        requested_device = "cpu"
        flyback_rules = []
        _validate_trace_config(config)
        use_zstack = _boolean(config.get("use_zstack", False), "use_zstack")
        required_directories = {"tiffs", "suite2p", "output"}
        if use_zstack:
            required_directories.add("zstack")
        for key in required_directories:
            _required(directories, key, "directories")
        for key in ("tiffs", "suite2p", "output"):
            _validate_template(
                directories[key],
                f"directories.{key}",
                required={"Name", "Date"},
                allowed=allowed,
            )
        if use_zstack:
            _validate_template(
                directories["zstack"],
                "directories.zstack",
                required={"Name", "Date", "Zstack"},
                allowed=allowed,
            )
    datasets_path = _required(config, "datasets", "configuration")
    datasets = _read_datasets(datasets_path, workflow).astype(object)
    selected: list[tuple[int, dict[str, Any]]] = []
    for row_index, (_index, row) in enumerate(datasets.iterrows()):
        values = _validate_dataset_row(row, row_index + 2, workflow, use_zstack)
        if values is not None:
            if workflow in {"run-suite2p", "zregister"}:
                matches = [
                    rule
                    for rule in flyback_rules
                    if rule[0] == values["Planes"] and rule[1] <= values["Frame_rate"] <= rule[2]
                ]
                if len(matches) != 1:
                    raise ConfigurationError(
                        f"dataset row {row_index + 2} must match exactly one flyback "
                        f"rule for Planes={values['Planes']} and "
                        f"Frame_rate={values['Frame_rate']}; found {len(matches)}."
                    )
            selected.append((row_index, values))
            for key, value in values.items():
                if key in datasets.columns:
                    datasets.at[row_index, key] = value
    plans = tuple(
        _dataset_plan(index, values, workflow, directories, use_zstack)
        for index, values in selected
    )
    return ValidatedWorkflow(
        workflow=workflow,
        config_path=config_path,
        config=config,
        datasets=datasets,
        plans=plans,
        requested_device=requested_device,
    )
