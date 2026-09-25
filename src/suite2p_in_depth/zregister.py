import argparse
import ast
import contextlib
import copy
import importlib
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from suite2p import io, parameters, registration
from suite2p.registration import bidiphase, register, rigid

from . import general, process_tiff, registration_plotting, suite2p_compat
from .data_types import RegistrationPaths
from .registration_numerics import (
    average_neighbor_frames,
    determine_best_ref,
    equalize_frame_numbers,
)

# Import the module to avoid the name collision with ``suite2p.run_s2p``.
run_s2p_mod = importlib.import_module("suite2p.run_s2p")

logger = logging.getLogger("suite2p.zregister")

CORR_THRESH = 4  # threshold in std dev below mean correlation to mark frame as bad

# copy file format to a binary file
files_to_binary = {
    "tif": io.tiff_to_binary,
    "h5": io.h5py_to_binary,
    "nwb": io.nwb_to_binary,
    "sbx": io.sbx_to_binary,
    "nd2": io.nd2_to_binary,
    "bruker": io.ome_to_binary,
    "movie": io.movie_to_binary,
    "dcimg": io.dcimg_to_binary,
}


def make_paths(dataset: pd.Series, directories: dict[str, str]) -> RegistrationPaths:
    """Resolve TIFF input directories and create a dataset output directory.

    Parameters
    ----------
    dataset : pandas.Series
        Row containing ``Name``, ``Date``, and ``Experiments``.
    directories : dict
        ``tiffs`` and ``output`` path templates.

    Returns
    -------
    dict
        Resolved ``tiffs`` and ``output`` paths. ``tiffs`` is ``None`` when
        any selected experiment has no TIFF file.

    Raises
    ------
    ValueError
        If the output template cannot be resolved.
    """
    subject = str(dataset.Name)
    date = str(dataset.Date)
    exp = dataset["Experiments"]

    paths = {
        "tiffs": general.expand_path_template(directories["tiffs"], subject, date, exp),
        "output": general.expand_path_template(directories["output"], subject, date, []),
    }

    if paths["tiffs"] is not None:
        tiff_directories = paths["tiffs"] if isinstance(paths["tiffs"], list) else [paths["tiffs"]]
        missing_tiffs = [
            path
            for path in tiff_directories
            if not os.path.isdir(path)
            or not any(
                os.path.isfile(os.path.join(path, name))
                and os.path.splitext(name)[1].lower() in (".tif", ".tiff")
                for name in os.listdir(path)
            )
        ]
        if missing_tiffs:
            paths["tiffs"] = None
    if paths["output"] is None:
        raise ValueError("Output path template could not be resolved.")
    os.makedirs(paths["output"], exist_ok=True)

    return paths


def get_settings(data_entry: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Build per-dataset Suite2p settings from user configuration.

    Parameters
    ----------
    data_entry : pandas.Series
        Dataset row with frame rate, plane count, and channel count.
    config : dict
        Base Suite2p settings; copied before modification.

    Returns
    -------
    dict
        Settings with per-plane frame rate and registration channel defaults.
    """
    settings = copy.deepcopy(config)
    # Hard-coded settings for this specific pipeline
    settings["fs"] = data_entry.Frame_rate / data_entry.Planes
    settings.setdefault("io", {})
    settings["io"]["combined"] = False
    settings["io"]["delete_bin"] = False
    settings["io"]["move_bin"] = False
    settings.setdefault("registration", {})
    # Channel 2 is the structural registration channel whenever it is present.
    settings["registration"].setdefault("align_by_chan2", data_entry.Channels != 1)
    return settings


def get_db(
    paths: RegistrationPaths, data_entry: pd.Series, config: dict[str, Any]
) -> dict[str, Any]:
    """Build a Suite2p database with flyback planes for one dataset.

    Parameters
    ----------
    paths : dict
        Resolved TIFF and output paths.
    data_entry : pandas.Series
        Dataset row with plane count, channel count, and frame rate.
    config : dict
        Base database and flyback-rule CSV path.

    Returns
    -------
    dict
        Suite2p database including ``ignore_flyback`` plane IDs.

    Raises
    ------
    FileNotFoundError
        If the flyback-rule table is missing.
    ValueError
        If no rule matches the dataset.
    """
    db0 = config["suite2p_db"].copy()
    db1 = {
        "data_path": paths["tiffs"],
        "input_format": "tif",
        "nplanes": data_entry.Planes,
        "nchannels": data_entry.Channels,
        "save_path0": paths["output"],
    }

    # Loop up flyback planes from table based on number of planes and frame rate
    filename = config["planes_to_flyback"]
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    planes_flybacks = pd.read_csv(
        filename, dtype={"Planes": "Int64", "Frame_rate_range": str, "Flyback_planes": str}
    )
    planes_flybacks["Frame_rate_range"] = planes_flybacks["Frame_rate_range"].apply(
        ast.literal_eval
    )
    planes_flybacks["Flyback_planes"] = planes_flybacks["Flyback_planes"].apply(ast.literal_eval)
    row = next(
        (
            row
            for _, row in planes_flybacks.iterrows()
            if row["Planes"] == db1["nplanes"]
            and row["Frame_rate_range"][0] <= data_entry.Frame_rate <= row["Frame_rate_range"][1]
        ),
        None,
    )
    if row is not None:
        db1["ignore_flyback"] = row["Flyback_planes"]
    else:
        raise (
            ValueError(
                f"No matching entry found in '{filename}' for {db1['nplanes']} planes at {data_entry.Frame_rate} Hz."
            )
        )

    db = {**db0, **db1}
    return db


def check_registration_status(save_folder: str | os.PathLike[str]) -> tuple[bool, bool]:
    """Check whether plane binaries and z-registration outputs exist.

    Backup plane folders are restored to plane folders when complete binaries
    are found there.

    Parameters
    ----------
    save_folder : str or os.PathLike
        Suite2p directory containing plane or backup folders.

    Returns
    -------
    tuple[bool, bool]
        Whether TIFFs have been converted and whether every retained plane
        has a saved z-registration result.
    """
    plane_folders = natsorted(
        [f.path for f in os.scandir(save_folder) if f.is_dir() and re.match(r"^plane\d+$", f.name)]
    )
    backup_folders = natsorted(
        [f.path for f in os.scandir(save_folder) if f.is_dir() and re.match(r"^backup\d+$", f.name)]
    )
    tiffs_copied = False
    planes_registered = False
    do_rename = False
    if len(plane_folders) > 0:
        dbs_exist = all(os.path.exists(os.path.join(folder, "db.npy")) for folder in plane_folders)
        settings_exist = all(
            os.path.exists(os.path.join(folder, "settings.npy")) for folder in plane_folders
        )
        ops_exist = all(os.path.exists(os.path.join(folder, "ops.npy")) for folder in plane_folders)
        tiffs_copied = all(
            os.path.exists(os.path.join(folder, "data.bin")) for folder in plane_folders
        ) & (ops_exist | (dbs_exist & settings_exist))
    elif len(backup_folders) > 0:
        tiffs_copied = all(
            os.path.exists(os.path.join(folder, "data.bin")) for folder in backup_folders
        )
        plane_folders = backup_folders
        do_rename = True

    if tiffs_copied:
        db, _ = suite2p_compat.load_parameters(plane_folders[0])
        planes = np.delete(np.arange(db["nplanes"]), db["ignore_flyback"])
        if do_rename:
            for plane in planes:
                folder = os.path.join(save_folder, f"backup{plane}")
                new_folder_path = os.path.join(save_folder, f"plane{plane}")
                os.rename(folder, new_folder_path)
        reg_output_list = [
            suite2p_compat.load_outputs(os.path.join(save_folder, f"plane{plane}"))[0]
            for plane in planes
        ]
        planes_registered = all(
            (reg_outputs is not None)
            and ("zpos_registration" in reg_outputs)
            and (reg_outputs["zpos_registration"] is not None)
            for reg_outputs in reg_output_list
        )

    return tiffs_copied, planes_registered


def save_plane_settings(
    plane_paths: Sequence[str], settings_per_plane: Sequence[dict[str, Any]]
) -> None:
    """Save one Suite2p settings mapping per imaging plane.

    Parameters
    ----------
    plane_paths : sequence[str]
        Plane directories receiving ``settings.npy``.
    settings_per_plane : sequence[dict]
        Settings corresponding to those directories.

    Raises
    ------
    ValueError
        If the two sequences have different lengths.
    """
    if len(plane_paths) != len(settings_per_plane):
        raise ValueError("plane_paths and settings_per_plane must have the same length.")
    for plane_path, plane_settings in zip(plane_paths, settings_per_plane, strict=True):
        np.save(os.path.join(plane_path, "settings.npy"), plane_settings)


def register_dataset(
    db0: dict[str, Any], user_settings: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Register individual planes and assemble a depth-corrected movie.

    Existing converted binaries or plane registration results are reused when
    available. The combined movie, database, settings, and registration
    outputs are saved under ``plane_z``.

    Parameters
    ----------
    db0 : dict
        Dataset-level Suite2p database and input paths.
    user_settings : dict
        Suite2p registration and run settings.

    Returns
    -------
    tuple[dict, dict]
        Combined plane database and registration outputs.

    Raises
    ------
    ValueError
        If all imaging planes are excluded as flyback planes.
    """
    t0 = time.time()

    settings0 = parameters.default_settings()
    parameters.set_settings(settings0, copy.deepcopy(user_settings))

    save_folder = run_s2p_mod.get_save_folder(db0)
    os.makedirs(save_folder, exist_ok=True)

    # Check if .bin files already created and if z-registration already done.
    tiffs_copied, planes_registered = check_registration_status(save_folder)

    if tiffs_copied:
        logger.info("Skipping tiff conversion")

    # Convert tiffs to one .bin file per imaging plane.
    else:
        logger.info("Creating bin files out of tiff")
        # Find files
        fs, first_files = io.get_file_list(db0)
        db0["file_list"] = fs
        db0["first_files"] = first_files

        # Copy dbs to list per plane + create folders.
        dbs = io.init_dbs(db0)
        save_folder = os.path.join(db0["save_path0"], db0["save_folder"])
        np.save(os.path.join(save_folder, "db.npy"), db0)
        np.save(os.path.join(save_folder, "settings.npy"), settings0)

        # Keep all per-plane binaries open while the TIFF data are distributed.
        with contextlib.ExitStack() as stack:
            raw_str = "raw" if db0.get("keep_movie_raw", False) else "reg"
            fnames = [db[f"{raw_str}_file"] for db in dbs]
            files = [stack.enter_context(open(f, "wb")) for f in fnames]
            if db0["nchannels"] > 1:
                fnames_chan2 = [db[f"{raw_str}_file_chan2"] for db in dbs]
                files_chan2 = [stack.enter_context(open(f, "wb")) for f in fnames_chan2]
            else:
                files_chan2 = None

            dbs = files_to_binary[db0["input_format"]](dbs, settings0, files, files_chan2)

        logger.info(
            "Wrote {} frames per binary, {} folders + {} channels, {:0.2f}sec".format(
                dbs[0]["nframes"], len(dbs), dbs[0]["nchannels"], time.time() - t0
            )
        )

    # Create array of all imaging plane indices and those without flyback planes.
    planes = np.arange(db0["nplanes"])
    planes_wo_fb = np.setdiff1d(planes, db0["ignore_flyback"])
    if planes_wo_fb.size == 0:
        raise ValueError("No imaging planes remain after excluding flyback planes.")

    # Create list with dbs and settings for planes (excluding flyback planes).
    plane_paths = [os.path.join(save_folder, f"plane{plane}") for plane in planes_wo_fb]
    db_planes_wo_fb = []
    settings_planes_wo_fb = []
    for plane in range(len(planes_wo_fb)):
        db, settings = suite2p_compat.load_parameters(plane_paths[plane])
        db_planes_wo_fb.append(db)
        parameters.set_settings(settings, copy.deepcopy(user_settings))
        settings_planes_wo_fb.append(settings)

    device = general.assign_torch_device(settings0["torch_device"])

    if planes_registered:  # If individual planes already z-registered.
        logger.info(
            "Skipping single-plane z-registration. Removing previous detection and extraction files, if present."
        )
        corrs_time_refs_planes = []
        zposition_each_plane_across_time = []
        save_plane_settings(plane_paths, settings_planes_wo_fb)

        for plane in plane_paths:
            # Load correlations between planes and reference images and z-positions across time.
            reg_outputs = np.load(os.path.join(plane, "reg_outputs.npy"), allow_pickle=True).item()
            corrs_time_refs_planes.append(
                reg_outputs["cmax_registration"]
            )  # shape: n_frames x n_refs
            zposition_each_plane_across_time.append(
                reg_outputs["zpos_registration"]
            )  # shape: n_frames

        ref_imgs = reg_outputs["refImg"]

        # Remove plane_z folder if present
        plane_z_path = os.path.join(save_folder, "plane_z")
        if os.path.exists(plane_z_path):
            shutil.rmtree(plane_z_path)

    else:  # If individual planes not yet z-registered.
        # Get frames to exclude from registration and detection (e.g. photostim frames)
        bad_frames_wo_fb = []
        for plane in range(len(planes_wo_fb)):
            badframes_path = os.path.join(db_planes_wo_fb[plane]["data_path"][0], "bad_frames.npy")
            if not os.path.exists(badframes_path):
                badframes_path = os.path.join(
                    db_planes_wo_fb[plane]["save_path0"], "bad_frames.npy"
                )
            badframes_path = badframes_path if os.path.exists(badframes_path) else None
            badframes0 = np.zeros(db["nframes"], "bool")
            if badframes_path is not None and os.path.exists(badframes_path):
                bf_indices = np.load(badframes_path).flatten().astype("int")
                badframes0[bf_indices] = True
                logger.info(
                    f"badframes file - plane {planes_wo_fb[plane]}: "
                    f"{badframes_path};\n # of badframes: {badframes0.sum()}"
                )

            bad_frames_wo_fb.append(badframes0)

        # Create reference image for each plane and determine bidirectional phase if necessary.
        logger.info("Computing reference images")
        ref_imgs = []
        bidi_phases = []
        for plane in range(len(planes_wo_fb)):
            logger.info(f"  PLANE {planes_wo_fb[plane]}")

            # Check that there are sufficient numbers of frames
            if db_planes_wo_fb[plane]["nframes"] < 10:
                raise ValueError("Number of frames should be at least 50")
            elif db["nframes"] < 200:
                logger.warning("WARNING: number of frames < 200, unpredictable behaviors may occur")

            ref, bidi = create_ref_per_plane(
                db_planes_wo_fb[plane],
                settings_planes_wo_fb[plane]["registration"],
                bad_frames_wo_fb[plane],
                device,
            )
            ref_imgs.append(ref)
            bidi_phases.append(bidi)

        # Align reference images of all planes to each other.
        ref_imgs = align_ref_images(settings0["registration"], ref_imgs, device=device)

        logger.info("Registering each plane in all three directions")
        corrs_time_refs_planes = []
        zposition_each_plane_across_time = []
        for plane in range(len(planes_wo_fb)):
            logger.info(f"  PLANE {planes_wo_fb[plane]}")
            settings_planes_wo_fb[plane]["registration"]["do_bidiphase"] = False
            settings_planes_wo_fb[plane]["registration"]["bidiphase"] = bidi_phases[plane]
            reg_outputs = register_plane(
                db_planes_wo_fb[plane],
                settings_planes_wo_fb[plane]["registration"],
                ref_imgs,
                bad_frames_wo_fb[plane],
                device,
            )

            np.save(
                os.path.join(db_planes_wo_fb[plane]["save_path"], "reg_outputs.npy"), reg_outputs
            )

            corrs_time_refs_planes.append(
                reg_outputs["cmax_registration"]
            )  # shape: n_frames x n_refs
            zposition_each_plane_across_time.append(
                reg_outputs["zpos_registration"]
            )  # shape: n_frames

    # Disregard extra frame in planes if necessary.
    equalize_frame_numbers(
        corrs_time_refs_planes, zposition_each_plane_across_time, db_planes_wo_fb
    )

    # Determine best reference image.
    corrs_time_refs_planes = np.dstack(
        corrs_time_refs_planes
    )  # shape: n_frames x n_refs x n_planes
    best_reference = determine_best_ref(corrs_time_refs_planes)

    logger.info("Creating new file")
    # At this point the files are registered properly according to where they are
    # now we need to go over each zposition and replace the frame on the channel with a weighted
    # frame on the plane it actually is
    correlations_refs_neighbours = determine_ref_similarities(
        ref_imgs, best_reference, settings0["registration"], device
    )
    db_new, reg_outputs_new = create_registered_file(
        db_planes_wo_fb,
        settings_planes_wo_fb,
        best_reference,
        corrs_time_refs_planes[:, best_reference, :],
        correlations_refs_neighbours,
        device,
        user_settings["registration"]["delete_singleplanes"],
    )
    reg_outputs_new["corrs_time_refs_planes"] = corrs_time_refs_planes

    np.save(os.path.join(db_new["db_path"]), db_new)
    np.save(os.path.join(db_new["settings_path"]), settings_planes_wo_fb[best_reference])
    np.save(os.path.join(db_new["save_path"], "reg_outputs.npy"), reg_outputs_new)

    t1 = time.time() - t0
    logger.info("----------- Total time for dataset z-registration: %.2f sec", t1)
    return db_new, reg_outputs_new


def register_plane(
    db: dict[str, Any],
    settings_reg: dict[str, Any],
    ref_imgs: Sequence[np.ndarray],
    bad_frames: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Register one plane's binary movie against all plane references.

    Parameters
    ----------
    db : dict
        Plane database with binary paths, dimensions, and frame count.
    settings_reg : dict
        Suite2p registration settings.
    ref_imgs : sequence[numpy.ndarray]
        Aligned reference image for each retained plane.
    bad_frames : numpy.ndarray
        Boolean mask of frames excluded from registration.
    device : torch.device
        Device for registration calculations.

    Returns
    -------
    dict
        Suite2p registration results, including z-position estimates.
    """
    t0 = time.time()

    reg_file = db["reg_file"]
    raw_file = db.get("raw_file", None)
    logger.info(f"binary output path: {reg_file}")
    if raw_file is not None:
        logger.info(f"raw binary path: {raw_file}")

    if db["nchannels"] > 1:
        reg_file_chan2 = db["reg_file_chan2"]
        raw_file_chan2 = db.get("raw_file_chan2", None)

    with contextlib.ExitStack() as stack:
        f_reg = stack.enter_context(
            io.BinaryFile(
                Ly=db["Ly"], Lx=db["Lx"], filename=reg_file, n_frames=db["nframes"], write=True
            )
        )
        f_raw_chan2 = None
        if db["keep_movie_raw"]:
            f_raw = stack.enter_context(
                io.BinaryFile(
                    Ly=db["Ly"], Lx=db["Lx"], filename=raw_file, n_frames=db["nframes"], write=False
                )
            )
            if db["nchannels"] > 1:
                f_raw_chan2 = stack.enter_context(
                    io.BinaryFile(
                        Ly=db["Ly"],
                        Lx=db["Lx"],
                        filename=raw_file_chan2,
                        n_frames=db["nframes"],
                        write=False,
                    )
                )
        else:
            f_raw = None

        if db["nchannels"] > 1:
            f_reg_chan2 = stack.enter_context(
                io.BinaryFile(
                    Ly=db["Ly"],
                    Lx=db["Lx"],
                    filename=reg_file_chan2,
                    n_frames=db["nframes"],
                    write=True,
                )
            )
        else:
            f_reg_chan2 = None

        registration_outputs = register.registration_wrapper(
            f_reg,
            f_raw=f_raw,
            f_reg_chan2=f_reg_chan2,
            f_raw_chan2=f_raw_chan2,
            save_path=db["save_path"],
            refImg=ref_imgs,
            align_by_chan2=settings_reg["align_by_chan2"],
            badframes=bad_frames,
            settings=settings_reg,
            device=device,
        )

    t1 = time.time() - t0
    logger.info("----------- Total %.2f sec", t1)
    return registration_outputs


def align_ref_images(
    settings_reg: dict[str, Any], ref_imgs: Sequence[np.ndarray], device: torch.device
) -> list[np.ndarray]:
    """Clip and align plane reference images to one another.

    Parameters
    ----------
    settings_reg : dict
        Registration settings; nonrigid alignment is disabled in a copy.
    ref_imgs : sequence[numpy.ndarray]
        Reference images, clipped in place to their 1st and 99th percentiles.
    device : torch.device
        Device used for cross-plane alignment.

    Returns
    -------
    list[numpy.ndarray]
        Aligned reference images in plane order.
    """
    for frame in ref_imgs:
        rmin = np.int16(np.percentile(frame, 1))
        rmax = np.int16(np.percentile(frame, 99))
        frame[:] = np.clip(frame, rmin, rmax)

    settings_refs = settings_reg.copy()
    settings_refs["nonrigid"] = False

    ref_imgs, y_off, x_off = process_tiff.register_zstack_across_planes(
        zstack=np.array(ref_imgs), settings_reg=settings_refs, device=device
    )[:3]
    ref_imgs = list(ref_imgs)

    logger.info("  Shifts of reference images: (y,x) = %s, %s", y_off, x_off)
    return ref_imgs


def create_ref_per_plane(
    db: dict[str, Any], settings_reg: dict[str, Any], bad_frames: np.ndarray, device: torch.device
) -> tuple[np.ndarray, int]:
    """Build one plane reference from central, usable movie frames.

    Parameters
    ----------
    db : dict
        Plane binary paths, dimensions, and frame count.
    settings_reg : dict
        Registration channel, initial-image count, and bidiphase settings.
    bad_frames : numpy.ndarray
        Boolean mask of frames to exclude.
    device : torch.device
        Device for reference computation.

    Returns
    -------
    tuple[numpy.ndarray, int]
        Reference image and applied bidirectional phase shift in pixels.
    """
    # get binary file paths
    if settings_reg["align_by_chan2"]:
        bin_file = db["reg_file_chan2"]
    else:
        bin_file = db["reg_file"]

    n_frames, ly, lx = db["nframes"], db["Ly"], db["Lx"]

    # Grab frames from middle of recordings for reference image
    frame_ind = np.arange(0, n_frames)[~bad_frames]
    frames_centre = frame_ind[
        range(
            round(len(frame_ind) / 2) - round(settings_reg["nimg_init"] / 2),
            round(len(frame_ind) / 2) + round(settings_reg["nimg_init"] / 2),
        )
    ]
    with io.BinaryFile(Ly=ly, Lx=lx, filename=bin_file, n_frames=n_frames) as f_align_in:
        ref_frames = f_align_in[frames_centre]

    # Compute bidi_phase shift
    if settings_reg["do_bidiphase"] and settings_reg["bidiphase"] == 0:
        bidi_phase = bidiphase.compute(ref_frames)
        logger.info("    Estimated bidiphase offset from data: %d pixels", bidi_phase)
    else:
        bidi_phase = settings_reg["bidiphase"]

    # shift frames
    if bidi_phase != 0:
        bidiphase.shift(ref_frames, bidi_phase)

    # Compute reference image
    ref = register.compute_reference(ref_frames, settings=settings_reg, device=device)
    return ref, bidi_phase


def create_registered_file(
    db_planes: list[dict[str, Any]],
    settings_planes: list[dict[str, Any]],
    best_reference: int,
    corrs_time_planes: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    delete_extra: bool = False,
    corr_threshold: float = CORR_THRESH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create and re-register a combined movie from neighboring planes.

    The best-correlated plane is chosen at each time point. Its neighboring
    frames are combined with correlation weights, then the result is aligned
    to a weighted reference image.

    Parameters
    ----------
    db_planes : list[dict]
        Database mappings for retained imaging planes.
    settings_planes : list[dict]
        Matching per-plane Suite2p settings.
    best_reference : int
        Index of the selected reference among retained planes.
    corrs_time_planes : numpy.ndarray
        Correlations in ``(frames, planes)`` order.
    weights : list or numpy.ndarray
        Weights for previous, selected, and next plane.
    device : torch.device
        Device for final registration.
    delete_extra : bool, optional
        Delete original plane directories rather than renaming them as backup.
    corr_threshold : float, optional
        Standard deviations below mean correlation used to flag bad frames.

    Returns
    -------
    tuple[dict, dict]
        Combined-plane database and registration outputs.
    """
    t = time.time()

    # Planes matching best reference at each time point.
    planes_across_time = np.nanargmax(corrs_time_planes, axis=1)  # index into plane_ids

    # Prepare db, paths, and plane IDs
    db_old = db_planes[best_reference]
    new_save_path = os.path.join(db_old["save_path0"], "suite2p", "plane_z")
    os.makedirs(new_save_path, exist_ok=True)
    new_bin_file_path = os.path.join(new_save_path, "data.bin")
    plane_ids = np.setdiff1d(
        np.arange(db_old["nplanes"]), db_old["ignore_flyback"]
    )  # IDs of non-flyback planes

    db_new, reg_outputs0 = prepare_registration_ops(
        best_reference, new_bin_file_path, new_save_path, db_old, plane_ids, planes_across_time
    )

    # For all used planes at each time point, determine bad frames and shifts in x and y.
    badframes, n_frames, xoff, yoff = gather_single_plane_results(db_planes, planes_across_time)

    # Create new .bin file with weighted average of neighboring planes at each time point.
    write_new_file(
        badframes,
        n_frames,
        new_bin_file_path,
        new_save_path,
        db_new,
        db_planes,
        planes_across_time,
        weights,
    )
    t1 = time.time() - t
    logger.info("  %.2f sec to write new file", t1)

    # Now realign new movie to best reference image (weighted average of neighbor ref images)
    reg_outputs1 = realign_frames(
        badframes[:, 1],
        best_reference,
        n_frames,
        db_new,
        copy.deepcopy(settings_planes[best_reference]),
        reg_outputs0,
        weights,
        xoff,
        yoff,
        device,
    )
    t2 = time.time() - t - t1
    logger.info("  %.2f sec to re-register new movie", t2)

    reg_outputs = {**reg_outputs0, **reg_outputs1}
    # Integrate all badframe information: (1) bad registration of individual planes, (2) low correlation during
    # z-registration, (3) final registration
    corrs_time = np.nanmax(corrs_time_planes, axis=1)
    badframes_corr = corrs_time <= np.nanmean(corrs_time) - corr_threshold * np.nanstd(corrs_time)
    reg_outputs["badframes"] = badframes[:, 1] | badframes_corr | reg_outputs["badframes"]

    # Delete old plane folders (independently registered planes)
    handle_single_plane_data(delete_extra, db_new)

    return db_new, reg_outputs


def handle_single_plane_data(delete_extra: bool, db: dict[str, Any]) -> None:
    """Remove or back up individual plane directories after combination.

    Parameters
    ----------
    delete_extra : bool
        Delete all ``planeN`` folders when true; otherwise rename retained
        planes to ``backupN`` and delete flyback planes.
    db : dict
        Combined-plane database with save path and flyback IDs.
    """
    plane_folders = [
        f.path
        for f in os.scandir(os.path.join(db["save_path0"], db["save_folder"]))
        if f.is_dir() and re.match(r"^plane\d+$", f.name)
    ]
    if delete_extra:
        for f in plane_folders:
            shutil.rmtree(f)
    else:
        for folder in plane_folders:
            base_dir, folder_name = os.path.split(folder)
            plane_re = re.compile(r"^plane(\d+)$")
            m = plane_re.match(folder_name)
            if int(m.group(1)) in db["ignore_flyback_singleplanes"]:
                shutil.rmtree(folder)
            else:
                new_folder_name = re.sub(r"^plane(\d+)$", r"backup\1", folder_name)
                new_folder_path = os.path.join(base_dir, new_folder_name)
                os.rename(folder, new_folder_path)


def realign_frames(
    badframes: np.ndarray,
    best_reference: int,
    n_frames: int,
    db: dict[str, Any],
    settings: dict[str, Any],
    reg_outputs_old: dict[str, Any],
    weights: np.ndarray,
    xoff: np.ndarray,
    yoff: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Align the combined movie to a weighted neighboring-plane reference.

    Parameters
    ----------
    badframes : numpy.ndarray
        Boolean flags for frames selected from the central plane.
    best_reference : int
        Index of the selected reference plane.
    n_frames : int
        Number of frames in the combined movie.
    db : dict
        Combined-plane database and binary paths.
    settings : dict
        Suite2p settings; registration options are adjusted in place.
    reg_outputs_old : dict
        Single-plane registration outputs containing reference images.
    weights : numpy.ndarray
        Neighbor weights for the reference image.
    xoff, yoff : numpy.ndarray
        Original plane offsets, updated with final shifts.
    device : torch.device
        Device for Suite2p registration.

    Returns
    -------
    dict
        Final registration outputs and combined shift ranges.
    """
    # Build a reference from the same neighboring-plane weights as the movie.
    ref_img = np.zeros((db["Ly"], db["Lx"]), dtype=np.float32)
    # Check whether neighboring planes exist
    neigh_idxs = np.arange(best_reference - 1, best_reference + 2)
    w = weights.copy()
    valid = (neigh_idxs >= 0) & (neigh_idxs < len(reg_outputs_old["refImg"]))
    w[~valid] = 0
    w /= np.sum(w)
    for idx, neighbor in enumerate(neigh_idxs):
        ref_img += w[idx] * np.asarray(reg_outputs_old["refImg"][neighbor], dtype=np.float32)

    # Align new plane to new reference image
    settings["registration"]["nonrigid"] = False
    settings["registration"]["do_bidiphase"] = False
    settings["registration"]["bidiphase"] = 0
    with contextlib.ExitStack() as stack:
        f_reg = stack.enter_context(
            io.BinaryFile(
                Ly=db["Ly"], Lx=db["Lx"], filename=db["reg_file"], n_frames=n_frames, write=True
            )
        )
        if db["nchannels"] == 2:
            f_reg_chan2 = stack.enter_context(
                io.BinaryFile(
                    Ly=db["Ly"],
                    Lx=db["Lx"],
                    filename=db["reg_file_chan2"],
                    n_frames=n_frames,
                    write=True,
                )
            )
        else:
            f_reg_chan2 = None

        reg_outputs = register.registration_wrapper(
            f_reg,
            f_raw=None,
            f_reg_chan2=f_reg_chan2,
            f_raw_chan2=None,
            refImg=ref_img,
            align_by_chan2=settings["registration"]["align_by_chan2"],
            save_path=db["save_path"],
            badframes=badframes,
            settings=settings["registration"],
            device=device,
        )

        # Compute metrics for registration
        if settings["run"]["do_regmetrics"] and n_frames >= 1500:
            yrange, xrange = reg_outputs["yrange"], reg_outputs["xrange"]
            out = registration.get_pc_metrics(
                f_reg, yrange=yrange, xrange=xrange, settings=settings["registration"]
            )
            reg_outputs["tPC"], reg_outputs["regPC"], reg_outputs["regDX"] = out

    # References images used to register individual planes.
    reg_outputs["refImg_singleplanes"] = reg_outputs_old["refImg"]

    # Shifts in x and y from final registration.
    reg_outputs["xoff_singleplanes"] = xoff
    reg_outputs["yoff_singleplanes"] = yoff

    # Shifts and ranges from z-registration of used planes.
    xoff += reg_outputs["xoff"][:, np.newaxis]
    yoff += reg_outputs["yoff"][:, np.newaxis]
    reg_outputs["xrange"] = [np.nanmax(xoff), db["Lx"] + np.nanmin(xoff)]
    reg_outputs["yrange"] = [np.nanmax(yoff), db["Ly"] + np.nanmin(yoff)]
    return reg_outputs


def write_new_file(
    badframes: np.ndarray,
    n_frames: int,
    new_bin_file_path: str,
    new_save_path: str,
    db: dict[str, Any],
    db_planes: Sequence[dict[str, Any]],
    planes_across_time: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Stream weighted plane frames into a combined registered binary.

    Parameters
    ----------
    badframes : numpy.ndarray
        Boolean flags in ``(frames, three neighbors)`` order.
    n_frames : int
        Number of time points to write.
    new_bin_file_path : str
        Destination for first-channel ``data.bin``.
    new_save_path : str
        Directory for an optional second-channel binary.
    db : dict
        Combined-plane image dimensions and channel count. Its second-channel
        binary path is filled when applicable.
    db_planes : sequence[dict]
        Source plane databases and registered binary paths.
    planes_across_time : numpy.ndarray
        Selected plane index for each time point.
    weights : numpy.ndarray
        Weights for previous, selected, and next plane.
    """
    # Keep source and target binaries open while frames are streamed once.
    with contextlib.ExitStack() as stack:
        src_files = [
            stack.enter_context(
                io.BinaryFile(
                    Ly=db_planes[i]["Ly"],
                    Lx=db_planes[i]["Lx"],
                    filename=db_planes[i]["reg_file"],
                    n_frames=n_frames,
                    write=False,
                )
            )
            for i in range(len(db_planes))
        ]

        # optionally open chan2 source files
        has_chan2 = db["nchannels"] == 2
        if has_chan2:
            src_files_chan2 = [
                stack.enter_context(
                    io.BinaryFile(
                        Ly=db_planes[i]["Ly"],
                        Lx=db_planes[i]["Lx"],
                        filename=db_planes[i].get("reg_file_chan2"),
                        n_frames=n_frames,
                        write=False,
                    )
                )
                for i in range(len(db_planes))
            ]
        else:
            src_files_chan2 = None

        # open target files for writing
        new_file = stack.enter_context(
            io.BinaryFile(
                Ly=db["Ly"], Lx=db["Lx"], filename=new_bin_file_path, n_frames=n_frames, write=True
            )
        )
        if has_chan2:
            new_bin_file_path_chan2 = os.path.join(new_save_path, "data_chan2.bin")
            db["reg_file_chan2"] = new_bin_file_path_chan2
            new_file_chan2 = stack.enter_context(
                io.BinaryFile(
                    Ly=db["Ly"],
                    Lx=db["Lx"],
                    filename=new_bin_file_path_chan2,
                    n_frames=n_frames,
                    write=True,
                )
            )
        else:
            new_file_chan2 = None

        # Iterate frames and build weighted average from neighbor planes
        for t in range(n_frames):
            sel_idx = int(planes_across_time[t])  # index into ops_paths / src_files
            neigh_idxs = np.arange(sel_idx - 1, sel_idx + 2)

            # Check whether neighboring planes had bad frames; set weights to zero if yes
            w = weights.copy()

            # 1) Zero weights for out-of-range neighbor indices
            valid = (neigh_idxs >= 0) & (neigh_idxs < len(db_planes))
            w[~valid] = 0

            # 2) Zero weights for badframes in neighboring planes
            if np.any(badframes[t, [0, 2]]):
                bf_idx = np.full(3, False)
                bf_idx[[0, 2]] = badframes[t, [0, 2]]
                w[bf_idx] = 0

            if any(w == 0):
                w /= np.sum(w)

            # Create weighted average of neighboring plane frames
            new_file[t : t + 1] = average_neighbor_frames(
                neigh_idxs, db["Ly"], db["Lx"], src_files, t, w
            )

            # Do the same for channel 2 if present
            if has_chan2:
                new_file_chan2[t : t + 1] = average_neighbor_frames(
                    neigh_idxs, db["Ly"], db["Lx"], src_files_chan2, t, w
                )


def gather_single_plane_results(
    db_list: Sequence[dict[str, Any]], planes_across_time: np.ndarray
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Gather bad-frame flags and shifts for each selected plane's neighbors.

    Parameters
    ----------
    db_list : sequence[dict]
        Per-plane databases pointing to saved registration outputs.
    planes_across_time : numpy.ndarray
        Selected plane index for each frame.

    Returns
    -------
    tuple[numpy.ndarray, int, numpy.ndarray, numpy.ndarray]
        Neighbor bad-frame flags, frame count, x offsets, and y offsets.
        Array outputs have shape ``(frames, three neighbors)``.
    """
    n_frames = len(planes_across_time)
    neighbors_across_time = planes_across_time[:, np.newaxis] + np.array([-1, 0, 1])
    used_planes = np.unique(neighbors_across_time)
    badframes_all_neighbors = np.zeros_like(neighbors_across_time, dtype=bool)
    xoff = np.zeros_like(neighbors_across_time, dtype=np.int32)
    yoff = np.zeros_like(neighbors_across_time, dtype=np.int32)

    for plane in used_planes:
        if plane < 0 or plane >= len(db_list):
            continue

        # Find where this plane is used in neighbors_across_time
        plane_indices = np.where(neighbors_across_time == plane)

        # Load registration outputs for this plane
        reg_outputs = np.load(
            os.path.join(db_list[plane]["save_path"], "reg_outputs.npy"), allow_pickle=True
        ).item()

        # Replace entries in badframes_all_neighbors with True if the current plane has a badframe at that time point
        badframes_all_neighbors[plane_indices] = reg_outputs["badframes"][plane_indices[0]]

        # Copy shifts
        goodframes = ~reg_outputs["badframes"][plane_indices[0]]
        xoff[plane_indices[0][goodframes], plane_indices[1][goodframes]] = reg_outputs["xoff"][
            plane_indices[0][goodframes]
        ]
        yoff[plane_indices[0][goodframes], plane_indices[1][goodframes]] = reg_outputs["yoff"][
            plane_indices[0][goodframes]
        ]

    return badframes_all_neighbors, n_frames, xoff, yoff


def prepare_registration_ops(
    best_reference: int,
    new_bin_file_path: str,
    new_save_path: str,
    db_old: dict[str, Any],
    plane_ids: np.ndarray,
    planes_across_time: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create combined-plane metadata from a selected single-plane database.

    Parameters
    ----------
    best_reference : int
        Index of the reference among retained planes.
    new_bin_file_path : str
        Combined movie binary path.
    new_save_path : str
        Combined plane output directory.
    db_old : dict
        Source plane database.
    plane_ids : numpy.ndarray
        Physical IDs of retained planes.
    planes_across_time : numpy.ndarray
        Selected retained-plane index at each time point.

    Returns
    -------
    tuple[dict, dict]
        Updated database and registration outputs with physical plane IDs.
    """
    db_new = copy.deepcopy(db_old)
    db_new["save_path"] = new_save_path
    db_new["fast_disk"] = new_save_path
    db_new["db_path"] = os.path.join(new_save_path, "db.npy")
    db_new["settings_path"] = os.path.join(new_save_path, "settings.npy")
    db_new["reg_file"] = new_bin_file_path
    db_new["ignore_flyback_singleplanes"] = db_new["ignore_flyback"]
    db_new["ignore_flyback"] = []

    reg_outputs_new = np.load(
        os.path.join(db_old["save_path"], "reg_outputs.npy"), allow_pickle=True
    ).item()
    reg_outputs_new["reference_plane"] = plane_ids[best_reference]
    reg_outputs_new["planes_across_time"] = plane_ids[planes_across_time]
    return db_new, reg_outputs_new


def determine_ref_similarities(
    ref_imgs: list[np.ndarray],
    best_reference: int,
    settings_reg: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Calculate normalized correlations for adjacent reference images.

    Parameters
    ----------
    ref_imgs : list[numpy.ndarray]
        Reference image for each retained plane.
    best_reference : int
        Index of the chosen reference image.
    settings_reg : dict
        Phase-correlation filter and shift settings.
    device : torch.device
        Device for correlation calculations.

    Returns
    -------
    numpy.ndarray
        Three weights for previous, selected, and next plane; absent neighbors
        have zero weight.
    """
    if device.type == "cuda":
        ref_imgs_torch = torch.from_numpy(np.stack(ref_imgs, axis=0)).pin_memory().to(device)
    else:
        ref_imgs_torch = torch.from_numpy(np.stack(ref_imgs, axis=0)).to(device)
    ref_img = ref_imgs[best_reference]

    mask_mul, mask_offset, cf_ref_img = register.compute_filters_and_norm(
        ref_img,
        False,
        settings_reg["smooth_sigma"],
        settings_reg["spatial_taper"],
        block_size=None,
        device=device,
    )[:3]
    cmax = rigid.phasecorr(
        ref_imgs_torch,
        cf_ref_img,
        mask_mul,
        mask_offset,
        maxregshift=settings_reg["maxregshift"],
        smooth_sigma_time=settings_reg["smooth_sigma_time"],
    )[2]

    corrs = np.zeros((3,))
    idx = np.arange(best_reference - 1, best_reference + 2)
    valid = (idx >= 0) & (idx < len(ref_imgs))
    corrs[valid] = cmax[idx[valid]].detach().cpu().numpy()
    corrs /= np.sum(corrs)

    return corrs


def load_dataset_list(path: str) -> pd.DataFrame:
    """Read and type the z-registration dataset table.

    Parameters
    ----------
    path : str
        Dataset CSV path.

    Returns
    -------
    pandas.DataFrame
        Dataset rows with parsed experiment lists and typed imaging fields.
    """
    df = pd.read_csv(
        path,
        dtype={
            "Name": str,
            "Date": str,
            "Depth": "float64",
            "Experiments": str,
            "Planes": "Int64",
            "Frame_rate": "float64",
            "Channels": "Int64",
            "Process": bool,
        },
    )
    df["Experiments"] = df["Experiments"].apply(ast.literal_eval)
    return df


def parse_args() -> argparse.Namespace:
    """Parse the standalone z-registration configuration path.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` path.
    """
    parser = argparse.ArgumentParser(description="Z-register source data")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="zregistration.yaml",
        help="Path to configuration file (yaml)",
    )
    return parser.parse_args()


def process_all_datasets(config: dict, datasets: pd.DataFrame) -> None:
    """
    Run Suite2p z-registration for each selected dataset.

    Datasets whose TIFF inputs are unavailable are skipped. Optional ROI detection
    runs after the combined z-registered plane has been created.

    Parameters
    ----------
    config : dict
        Directory templates, Suite2p settings, and flyback-plane rules.
    datasets : pandas.DataFrame
        Dataset rows with ``Name``, ``Date``, ``Experiments``, ``Planes``,
        ``Channels``, ``Frame_rate``, and ``Process`` fields.

    """
    for i in range(len(datasets)):
        if not datasets.loc[i]["Process"]:
            continue
        logger.info(
            "Starting dataset %s/%s",
            datasets.loc[i]["Name"],
            datasets.loc[i]["Date"],
        )
        paths = make_paths(datasets.loc[i], config["directories"])

        general.setup_suite2p_logging(os.path.join(paths["output"], "suite2p"))

        logger.info(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
        if paths["tiffs"] is None:
            logger.info(
                f"NOTE: Imaging directory for {datasets.loc[i]['Name']} "
                f"{datasets.loc[i]['Date']} not found. Skipping.\n\n"
            )
            continue

        # Create settings and db (as used in suite2p) for z-registration.
        user_settings = get_settings(datasets.loc[i], config["suite2p_settings"])
        user_settings["date_proc"] = datetime.now().astimezone()
        db = parameters.default_db()
        parameters.set_db(db, get_db(paths, datasets.loc[i], copy.deepcopy(config)))

        # Run z-registration.
        # NOTE: if z-registration was already performed, previous results will be overwritten.
        if user_settings["run"]["do_registration"]:
            db_new, reg_outputs = register_dataset(db, user_settings)
            registration_plotting.make_plots(
                db_new,
                reg_outputs,
                os.path.join(paths["output"], db_new["save_folder"], "plots"),
            )

        # Run detection of ROIs after z-registration.
        if user_settings["run"]["do_detection"]:
            plane_z = os.path.join(db["save_path0"], db["save_folder"], "plane_z")
            db_old, settings_old = suite2p_compat.load_parameters(plane_z)
            if not os.path.exists(os.path.join(plane_z, "db.npy")):
                np.save(os.path.join(plane_z, "db.npy"), db_old)
            if not os.path.exists(os.path.join(plane_z, "settings.npy")):
                np.save(os.path.join(plane_z, "settings.npy"), settings_old)
            # Overwrite parameters from registration (ops1) with user defined parameters (ops0) about detection.
            keys_to_keep = [
                "tau",
                "fs",
                "torch_device",
                "diameter",
                "run",
                "io",
                "detection",
                "classification",
                "extraction",
                "dcnv_preprocess",
            ]
            settings = {key: user_settings[key] for key in keys_to_keep if key in user_settings}
            parameters.set_settings(settings_old, settings)
            run_s2p_mod.run_s2p(db=db_old, settings=settings_old)


def main() -> None:
    """Run z-registration from a standalone configuration file.

    The command reads its dataset CSV and processes rows selected by the
    ``Process`` column. Missing input paths terminate the command.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    conf = general.load_config(args.config)
    if not os.path.exists(conf["datasets"]):
        print(f"Dataset file {conf['datasets']} does not exist.")
        exit(1)
    ds = load_dataset_list(conf["datasets"])
    if ds is None:
        exit(1)

    process_all_datasets(conf, ds)


if __name__ == "__main__":
    main()
