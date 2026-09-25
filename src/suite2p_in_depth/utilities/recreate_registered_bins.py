import argparse
import glob
import logging
import os
import re
import shutil
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from suite2p import io, registration

from .. import correct_traces, general
from ..data_types import TracePaths

logger = logging.getLogger("suite2p_in_depth.recreate_bin_files")


def _plane_id(path: str | os.PathLike[str]) -> int:
    """Extract the numeric Suite2p plane ID from a directory path.

    Parameters
    ----------
    path : str
        Path ending in ``planeN``.

    Returns
    -------
    int
        Plane number ``N``.

    Raises
    ------
    ValueError
        If the path does not identify a plane directory.
    """
    match = re.search(r"plane(\d+)$", os.path.normpath(path))
    if match is None:
        raise ValueError(f"Not a plane directory: {path}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    """Parse the configuration path for binary recreation.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` path.
    """
    parser = argparse.ArgumentParser(description="Find datasets without .bin files")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="zstack.yaml",
        help="Path to configuration file (yaml)",
    )
    return parser.parse_args()


def plan_missing_bin_targets(config: dict[str, Any], datasets: pd.DataFrame) -> list[str]:
    """List registered binary files absent from selected datasets.

    Parameters
    ----------
    config : dict
        Directory templates for Suite2p outputs.
    datasets : pandas.DataFrame
        Dataset rows with ``Name``, ``Date``, ``Process``, and optional
        ``Ignore_planes`` values.

    Returns
    -------
    list[str]
        Missing first- and second-channel binary paths.

    Raises
    ------
    FileNotFoundError
        If a selected Suite2p directory or required ``ops.npy`` is missing.
    """
    targets = []
    directories = config["directories"]
    for _, dataset in datasets.iterrows():
        if not bool(dataset["Process"]):
            continue
        suite2p_root = directories["suite2p"].format(
            Name=str(dataset["Name"]), Date=str(dataset["Date"])
        )
        suite2p_dir = os.path.join(suite2p_root, "suite2p")
        if not os.path.isdir(suite2p_dir):
            raise FileNotFoundError(
                f"Suite2p directory for {dataset['Name']} {dataset['Date']} "
                f"does not exist: {suite2p_dir}"
            )
        ignore_planes = np.asarray(dataset.get("Ignore_planes", []), dtype=int)
        plane_directories = sorted(glob.glob(os.path.join(suite2p_dir, "plane*")), key=_plane_id)
        for plane_directory in plane_directories:
            if _plane_id(plane_directory) in ignore_planes:
                continue
            ops_path = os.path.join(plane_directory, "ops.npy")
            if not os.path.isfile(ops_path):
                raise FileNotFoundError(
                    f"Cannot recreate binaries without Suite2p metadata: {ops_path}"
                )
            ops = np.load(ops_path, allow_pickle=True).item()
            data_path = os.path.join(plane_directory, "data.bin")
            if not os.path.isfile(data_path):
                targets.append(data_path)
            if int(ops.get("nchannels", 1)) > 1:
                channel_path = os.path.join(plane_directory, "data_chan2.bin")
                if not os.path.isfile(channel_path):
                    targets.append(channel_path)
    return targets


def find_missing_bins(config: dict[str, Any], datasets: pd.DataFrame) -> None:
    """Recreate missing registered binaries for selected datasets.

    Reuses saved Suite2p shifts and removes temporary Suite2p outputs after
    moving completed binaries into each original plane directory.

    Parameters
    ----------
    config : dict
        Directory templates for source TIFFs and Suite2p outputs.
    datasets : pandas.DataFrame
        Dataset rows with ``Process`` and ``Ignore_planes`` fields.
    """
    for _, dataset in datasets.iterrows():
        if not dataset["Process"]:
            continue

        logger.info(
            "Processing dataset %s/%s",
            dataset["Name"],
            dataset["Date"],
        )
        paths = correct_traces.make_paths(dataset, config["directories"])
        if not paths["suite2p"]:
            logger.warning(
                "Suite2p directory for %s/%s was not found; skipping",
                dataset["Name"],
                dataset["Date"],
            )
            continue

        ignore_planes = np.array(dataset["Ignore_planes"]).astype(int)
        plane_directories = glob.glob(os.path.join(paths["suite2p"], "plane*"))
        plane_directories = sorted(plane_directories, key=_plane_id)
        planes = [_plane_id(d) for d in plane_directories]
        ind_ignore = np.where(np.isin(planes, ignore_planes))[0]
        plane_directories = np.delete(plane_directories, ind_ignore)

        planes_missing = False
        for f_plane in plane_directories:
            # Load the ops file for the current plane.
            ops = np.load(os.path.join(f_plane, "ops.npy"), allow_pickle=True).item()
            # Check if the .bin file exists for the current plane.
            if not os.path.exists(os.path.join(f_plane, "data.bin")) or (
                ops["nchannels"] > 1 and not os.path.exists(os.path.join(f_plane, "data_chan2.bin"))
            ):
                planes_missing = True
                break

        # If any .bin files are missing, recreate them using the shifts in x and y found previously (saved in ops).
        if planes_missing:
            recreate_registered_binary(ops, paths, plane_directories)
            # Delete temporary files and folders
            temporary_suite2p = os.path.realpath(os.path.join(paths["output"], "suite2p"))
            source_suite2p = os.path.realpath(paths["suite2p"])
            if temporary_suite2p == source_suite2p:
                raise ValueError(
                    "Refusing to remove temporary files because the configured "
                    "output/suite2p path is the source Suite2p directory."
                )
            shutil.rmtree(temporary_suite2p, ignore_errors=True)


def recreate_registered_binary(
    ops: dict[str, Any], paths: TracePaths, plane_directories: Sequence[str]
) -> None:
    """Recreate plane binaries from TIFFs using saved registration shifts.

    Parameters
    ----------
    ops : dict
        Suite2p operations with original data paths and registration shifts.
        Output and data paths are updated in this dictionary.
    paths : dict
        Source TIFF, temporary output, and piezo directory paths.
    plane_directories : sequence[str]
        Existing plane directories that receive recreated binaries.
    """
    # Change critical values in ops
    ops["save_path0"] = paths["output"]
    ops["fast_disk"] = paths["output"]
    # Replace drive letters in data_paths with Z:
    # Find all last folder names in data_paths and add them to paths["piezo"]
    piezo_path = paths.get("piezo")
    if piezo_path is None:
        raise ValueError("A piezo directory is required to recreate registered binaries.")
    data_paths = []
    for path in ops["data_path"]:
        last_folder = os.path.basename(os.path.normpath(path))
        data_paths.append(os.path.join(piezo_path, last_folder))
    # Update ops with modified data_paths
    ops["data_path"] = data_paths

    # Convert TIFFs to one binary file per plane
    io.tiff_to_binary(ops)

    # Loop across all planes to register movies
    ops_paths = [
        os.path.join(f, "ops.npy") for f in plane_directories
    ]  # As in suite2p\run_s2p.run_s2p l. 512.
    planes = [_plane_id(d) for d in plane_directories]
    for ipl, ops_path in enumerate(ops_paths):  # As in suite2p\run_s2p.run_s2p l. 534.
        ops_i = np.load(ops_path, allow_pickle=True).item()
        reg_file = os.path.join(paths["output"], "suite2p", f"plane{planes[ipl]}", "data.bin")
        # Create binary file objects, as in suite2p\run_s2p.run_plane l. 334
        if ops_i["nonrigid"]:
            yoff1 = ops_i["yoff1"]
            xoff1 = ops_i["xoff1"]
        else:
            yoff1 = None
            xoff1 = None
        with io.BinaryFile(
            filename=reg_file, n_frames=ops_i["nframes"], Ly=ops_i["Ly"], Lx=ops_i["Lx"]
        ) as f_reg:
            # Shift frames and write to binary file, as in suite2p\registration\register.registration_wrapper l. 653
            registration.register.shift_frames_and_write(
                f_reg, yoff=ops_i["yoff"], xoff=ops_i["xoff"], yoff1=yoff1, xoff1=xoff1, ops=ops_i
            )
        # Move registered bin file to long-term storage (where suite2p data is)
        shutil.move(reg_file, os.path.join(plane_directories[ipl], "data.bin"))

        # Do the same for channel 2, if it exists
        if ops_i["nchannels"] > 1:
            reg_file_chan2 = os.path.join(
                paths["output"], "suite2p", f"plane{planes[ipl]}", "data_chan2.bin"
            )
            with io.BinaryFile(
                filename=reg_file_chan2, n_frames=ops_i["nframes"], Ly=ops_i["Ly"], Lx=ops_i["Lx"]
            ) as f_reg_chan2:
                registration.register.shift_frames_and_write(
                    f_reg_chan2,
                    yoff=ops_i["yoff"],
                    xoff=ops_i["xoff"],
                    yoff1=yoff1,
                    xoff1=xoff1,
                    ops=ops_i,
                )
            shutil.move(reg_file_chan2, os.path.join(plane_directories[ipl], "data_chan2.bin"))

    return


def main() -> None:
    """Run binary recreation from a standalone configuration file.

    Loads the configured dataset table and restores absent registered
    binaries for selected datasets using saved Suite2p registration shifts.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    config = general.load_config(args.config)
    if not os.path.exists(config["datasets"]):
        print(f"Dataset file {config['datasets']} does not exist.")
        exit(1)
    datasets = correct_traces.load_dataset_list(config["datasets"])
    find_missing_bins(config, datasets)


if __name__ == "__main__":
    main()
