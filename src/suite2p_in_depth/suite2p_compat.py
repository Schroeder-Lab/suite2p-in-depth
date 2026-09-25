"""Compatibility adapters for supported Suite2p output formats."""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
from suite2p import parameters

REG_OUTPUTS = {
    "refImg": None,
    "meanImg": None,
    "meanImgE": None,
    "meanImg_chan2": None,
    "yoff": None,
    "xoff": None,
    "corrXY": None,
    "yoff1": None,
    "xoff1": None,
    "corrXY1": None,
    "badframes": None,
    "badframes0": None,
    "yrange": None,
    "xrange": None,
    "rmin": None,
    "rmax": None,
    "bidiphase": None,
    "zpos_registration": None,
    "cmax_registration": None,
    "tPC": None,
    "regPC": None,
    "regDX": None,
}

DETECT_OUTPUTS = {
    "max_proj": None,
    "meanImg_crop": None,
    "Vcorr": None,
    "diameter": None,
    "Vmax": None,
    "Vmap": None,
    "spatscale_pix": None,
    "chan2_masks": None,
}


Suite2pMapping = dict[str, Any]


def load_parameters(
    folder_path: str | os.PathLike[str],
) -> tuple[Suite2pMapping, Suite2pMapping]:
    """Load current or legacy Suite2p database and settings mappings.

    Parameters
    ----------
    folder_path : str or os.PathLike
        Plane directory with ``db.npy`` and ``settings.npy``, or legacy
        ``ops.npy`` metadata.

    Returns
    -------
    tuple[dict, dict]
        Database and settings mappings in the current format.

    Raises
    ------
    FileNotFoundError
        If neither supported metadata layout exists.
    """
    ops_file = os.path.join(folder_path, "ops.npy")
    db_file = os.path.join(folder_path, "db.npy")
    settings_file = os.path.join(folder_path, "settings.npy")

    if os.path.isfile(db_file) and os.path.isfile(settings_file):
        db = cast(Suite2pMapping, np.load(db_file, allow_pickle=True).item())
        settings = cast(Suite2pMapping, np.load(settings_file, allow_pickle=True).item())
    elif os.path.isfile(ops_file):
        ops = np.load(ops_file, allow_pickle=True).item()
        converted = parameters.convert_settings_orig(
            ops, db=parameters.default_db(), settings=parameters.default_settings()
        )
        db = cast(Suite2pMapping, converted[0])
        settings = cast(Suite2pMapping, converted[1])
        plane = ops["save_path"].split("plane")[-1]
        try:
            plane = int(plane)
        except ValueError:
            pass
        db["save_path"] = ops["save_path"]
        db["fast_disk"] = os.path.join(db["fast_disk"], "suite2p", f"plane{plane}")
        db["settings_path"] = os.path.join(db["save_path"], "settings.npy")
        db["db_path"] = os.path.join(db["save_path"], "db.npy")
        db["reg_file"] = ops["reg_file"]
        db["iplane"] = plane
        db["file_list"] = ops["filelist"]
        db["first_files"] = ops["first_tiffs"]
        db["Ly"] = ops["Ly"]
        db["Lx"] = ops["Lx"]
        db["nframes"] = ops["nframes"]
        db["frames_per_file"] = ops["frames_per_file"]
        db["frames_per_folder"] = ops["frames_per_folder"]
        db["meanImg"] = ops["meanImg"]

        if db["keep_movie_raw"]:
            db["raw_file"] = ops["raw_file"]

        if db["nchannels"] > 1:
            db["reg_file_chan2"] = ops["reg_file_chan2"]
            db["meanImg_chan2"] = ops["meanImg_chan2"]
            if db["keep_movie_raw"]:
                db["raw_file_chan2"] = ops["raw_file_chan2"]

        settings["registration"]["align_by_chan2"] = True if ops["align_by_chan"] == 2 else False

    else:
        raise FileNotFoundError(
            f"Neither ops.npy nor db.npy and settings.npy found in {folder_path}"
        )

    return db, settings


def load_parameters_recording(
    folder_path: str | os.PathLike[str],
) -> tuple[Suite2pMapping, Suite2pMapping]:
    """Load parameters from the first plane directory in a recording.

    Parameters
    ----------
    folder_path : str or os.PathLike
        Recording directory containing at least one ``plane*`` folder.

    Returns
    -------
    tuple[dict, dict]
        Database and settings returned by :func:`load_parameters`.

    Raises
    ------
    FileNotFoundError
        If no plane directory is found.
    """
    entries = os.listdir(folder_path)
    plane_name = None
    for name in entries:
        if name.startswith("plane"):
            full = os.path.join(folder_path, name)
            if os.path.isdir(full):
                plane_name = name
                break

    if plane_name is None:
        raise FileNotFoundError(f"No plane* folder found in {folder_path}")

    return load_parameters(os.path.join(folder_path, plane_name))


def convert_outputs_orig(
    ops: Suite2pMapping,
) -> tuple[Suite2pMapping, Suite2pMapping]:
    """Split a legacy ``ops.npy`` mapping into output categories.

    Parameters
    ----------
    ops : dict
        Legacy Suite2p operations mapping.

    Returns
    -------
    tuple[dict, dict]
        Registration and detection output mappings.
    """
    reg_outputs = REG_OUTPUTS.copy()
    for key in reg_outputs.keys():
        if key in ops:
            reg_outputs[key] = ops[key]

    detect_outputs = DETECT_OUTPUTS.copy()
    for key in detect_outputs.keys():
        if key in ops:
            detect_outputs[key] = ops[key]

    return reg_outputs, detect_outputs


def load_outputs(
    file_path: str | os.PathLike[str],
) -> tuple[Suite2pMapping | None, Suite2pMapping | None]:
    """Load current outputs, falling back to legacy ``ops.npy``.

    Parameters
    ----------
    file_path : str or os.PathLike
        Plane directory containing current or legacy output files.

    Returns
    -------
    tuple[dict or None, dict or None]
        Registration and detection outputs, or two ``None`` values if no
        supported output files are found.
    """
    ops_file = os.path.join(file_path, "ops.npy")
    reg_file = os.path.join(file_path, "reg_outputs.npy")
    detect_file = os.path.join(file_path, "detect_outputs.npy")

    reg_outputs = (
        cast(Suite2pMapping, np.load(reg_file, allow_pickle=True).item())
        if os.path.isfile(reg_file)
        else None
    )
    detect_outputs = (
        cast(Suite2pMapping, np.load(detect_file, allow_pickle=True).item())
        if os.path.isfile(detect_file)
        else None
    )

    if reg_outputs or detect_outputs:
        return reg_outputs, detect_outputs

    if os.path.isfile(ops_file):
        ops = cast(Suite2pMapping, np.load(ops_file, allow_pickle=True).item())
        return convert_outputs_orig(ops)

    return None, None
