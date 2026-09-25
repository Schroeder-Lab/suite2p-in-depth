"""General configuration, path-template, device, and logging helpers."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_TQDM_PROGRESS_PATTERN = re.compile(r"^\s*\d{1,3}%\|.*\|\s*\d+/\d+\s*\[.*\]\s*$")


class _SuppressTqdmProgress(logging.Filter):
    """Drop Suite2p's logged tqdm refreshes while retaining useful messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep a log record unless it is a tqdm progress refresh.

        Parameters
        ----------
        record : logging.LogRecord
            Candidate Suite2p log record.

        Returns
        -------
        bool
            True for messages that should be emitted.
        """
        return _TQDM_PROGRESS_PATTERN.fullmatch(record.getMessage()) is None


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML configuration mapping.

    Parameters
    ----------
    path
        YAML file to read. This low-level helper preserves values exactly and
        does not resolve relative paths; production CLI callers use the
        validated configuration boundary instead.

    Returns
    -------
    dict
        Parsed YAML mapping.
    """
    with Path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return loaded


def expand_path_template(
    path_template: str,
    subject: str,
    date: str,
    exp: list[str],
) -> str | list[str] | None:
    """Expand a dataset path template into existing experiment paths.

    Parameters
    ----------
    path_template
        Template supporting ``{Name}``, ``{Date}``, and optionally
        ``{Experiments}``. Native string operations are intentional here so a
        Windows template is not reinterpreted as POSIX syntax on Linux.
    subject, date
        Values substituted for the subject and acquisition date.
    exp
        Requested experiment identifiers. When empty, experiments are
        discovered beneath the template prefix if the template contains the
        experiment placeholder.

    Returns
    -------
    str or list of str or None
        A formatted non-experiment path, existing experiment paths in input
        order, or ``None`` when expansion cannot resolve an existing input.
    """
    if not isinstance(path_template, str) or not path_template:
        return None
    experiments = exp
    if not experiments:
        if "{Experiments}" in path_template:
            parent = path_template.split("{Experiments}", 1)[0].format(Name=subject, Date=date)
            if not os.path.exists(parent):
                return None
            experiments = [
                entry for entry in os.listdir(parent) if os.path.isdir(os.path.join(parent, entry))
            ]
        else:
            return path_template.format(Name=subject, Date=date)

    paths = []
    for experiment in experiments:
        path = path_template.format(Name=subject, Date=date, Experiments=str(experiment))
        if os.path.exists(path):
            paths.append(path)
    return paths or None


def assign_torch_device(requested_device: str) -> torch.device:
    """Return an available torch device, falling back to CPU.

    Parameters
    ----------
    requested_device
        Torch device string such as ``"cuda"``, ``"cpu"``, or ``"mps"``.

    Returns
    -------
    torch.device
        Requested device when a small allocation succeeds, otherwise CPU.
    """
    if requested_device == "cpu":
        return torch.device("cpu")
    try:
        device = torch.device(requested_device)
        torch.zeros([1, 2, 3]).to(device)
    except (AssertionError, RuntimeError):
        return torch.device("cpu")
    return device


def setup_suite2p_logging(suite2p_output_dir: str | os.PathLike[str]) -> None:
    """Configure Suite2p logging without truncating Suite2p's ``run.log``.

    The function creates ``suite2p_output_dir`` and replaces handlers only on
    the ``suite2p`` logger. Messages are written to stdout and to the
    tool-owned ``run_correct.log`` file, which is replaced on each run.

    Parameters
    ----------
    suite2p_output_dir : str or os.PathLike
        Directory receiving the correction log.
    """
    output_directory = Path(suite2p_output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    suite2p_logger = logging.getLogger("suite2p")
    suite2p_logger.setLevel(logging.INFO)
    suite2p_logger.propagate = False

    for handler in list(suite2p_logger.handlers):
        suite2p_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_SuppressTqdmProgress())
    suite2p_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        output_directory / "run_correct.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_SuppressTqdmProgress())
    suite2p_logger.addHandler(file_handler)
