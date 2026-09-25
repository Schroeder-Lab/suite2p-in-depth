import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from natsort import natsorted
from suite2p import io

from .. import correct_traces, general, suite2p_compat

logger = logging.getLogger("suite2p_in_depth.estimate_dark_level")


def _find_plane_folders(suite2p_dir: str) -> list[str]:
    """List existing ``planeN`` directories in natural numeric order.

    Parameters
    ----------
    suite2p_dir : str
        Suite2p output directory to scan.

    Returns
    -------
    list[str]
        Plane paths, or an empty list if the directory is absent.
    """
    if not os.path.isdir(suite2p_dir):
        return []
    plane_folders = natsorted(
        [
            f.path
            for f in os.scandir(suite2p_dir)
            if f.is_dir() and re.fullmatch(r"plane\d+", f.name)
        ]
    )
    return cast(list[str], plane_folders)


def _load_ops(plane_folder: str) -> dict[str, Any] | None:
    """Load Suite2p operations metadata when present.

    Parameters
    ----------
    plane_folder : str
        Directory containing a possible ``ops.npy`` file.

    Returns
    -------
    dict or None
        Loaded operations, or ``None`` if the file is absent.
    """
    ops_path = os.path.join(plane_folder, "ops.npy")
    if not os.path.exists(ops_path):
        return None
    return cast(dict[str, Any], np.load(ops_path, allow_pickle=True).item())


def _plane_bin_path(plane_folder: str) -> str | None:
    """Locate a plane's registered binary if it exists.

    Parameters
    ----------
    plane_folder : str
        Suite2p plane directory.

    Returns
    -------
    str or None
        Path to ``data.bin``, or ``None`` when missing.
    """
    bin_path = os.path.join(plane_folder, "data.bin")
    if os.path.exists(bin_path):
        return bin_path
    return None


def percentile_first_1000_frames(
    bin_path: str, db: dict[str, Any], percentile: float = 1.0, nframes: int = 1000
) -> float:
    """Return a movie-pixel percentile from a bounded initial frame sample.

    The registered binary is interpreted using Suite2p ``Ly``, ``Lx``, and
    ``nframes`` metadata. Frames use ``(time, y, x)`` order and native pixel
    intensity units. NaNs are ignored by the percentile calculation.

    Parameters
    ----------
    bin_path : str
        Registered Suite2p binary file.
    db : dict
        Metadata with positive ``Ly``, ``Lx``, and ``nframes`` values.
    percentile : float, optional
        Pixel percentile between zero and 100.
    nframes : int, optional
        Maximum number of initial frames to sample.

    Returns
    -------
    float
        Percentile of sampled pixel intensities.

    Raises
    ------
    FileNotFoundError
        If the registered binary does not exist.
    ValueError
        If arguments or binary metadata are invalid.
    """
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    if nframes < 1:
        raise ValueError("nframes must be at least 1.")
    ly = int(db["Ly"])
    lx = int(db["Lx"])
    total_frames = int(db.get("nframes", 0))
    if ly < 1 or lx < 1 or total_frames < 1:
        raise ValueError("Binary metadata must contain positive Ly, Lx, and nframes values.")
    n = min(nframes, total_frames)

    with io.BinaryFile(Ly=ly, Lx=lx, filename=bin_path, n_frames=total_frames, write=False) as bf:
        frames = bf[:n, :, :].copy()  # shape: (n, Ly, Lx)
    if frames.shape != (n, ly, lx):
        raise ValueError(f"Binary reader returned shape {frames.shape}; expected {(n, ly, lx)}.")

    out = float(np.nanpercentile(frames, percentile))
    return out


def plane_percentiles_all_datasets(config: dict[str, Any], datasets: pd.DataFrame) -> pd.DataFrame:
    """Estimate a dark pixel level for each selected dataset.

    For each available plane, the first percentile of up to 200 initial
    frames is measured. The smallest plane estimate becomes ``Dark value``.

    Parameters
    ----------
    config : dict
        Configuration containing Suite2p directory templates.
    datasets : pandas.DataFrame
        Table with ``Name`` and ``Date``; ``Process`` is optional.

    Returns
    -------
    pandas.DataFrame
        Copy of the identifying columns with a ``Dark value`` column.
        Datasets without usable planes receive NaN.
    """
    columns = ["Name", "Date"] + (["Process"] if "Process" in datasets.columns else [])
    datasets = datasets.loc[:, columns].copy()

    for i in range(len(datasets)):
        if not bool(datasets.loc[i].get("Process", True)):
            continue

        logger.info(
            "Processing dataset %s/%s",
            datasets.loc[i, "Name"],
            datasets.loc[i, "Date"],
        )
        paths = correct_traces.make_paths(cast(pd.Series, datasets.loc[i]), config["directories"])

        suite2p_dir = paths["suite2p"]
        plane_folders = _find_plane_folders(suite2p_dir) if suite2p_dir is not None else []

        per_plane: list[float] = []
        for plane_folder in plane_folders:
            db = suite2p_compat.load_parameters(plane_folder)[0]
            bin_path = os.path.join(plane_folder, "data.bin")
            if db is None or not os.path.exists(bin_path):
                continue
            per_plane.append(
                percentile_first_1000_frames(bin_path, db, percentile=1.0, nframes=200)
            )

        vals = np.asarray(per_plane, dtype=float)
        datasets.loc[i, "Dark value"] = (
            np.nanmin(vals) if vals.size and not np.isnan(vals).all() else np.nan
        )

    return datasets


def parse_args() -> argparse.Namespace:
    """Parse configuration and destination CSV paths.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` and ``output`` values.
    """
    p = argparse.ArgumentParser(
        description="Compute 1st percentile from first 1000 registered frames per plane."
    )
    p.add_argument(
        "--config",
        type=str,
        required=False,
        default="zregistration.yaml",
        help="Path to configuration file (yaml).",
    )
    p.add_argument("--output", type=Path, required=True, help="Output CSV file.")
    return p.parse_args()


def main() -> None:
    """Estimate dark levels from a standalone configuration.

    Reads the configured dataset CSV and writes a copy with ``Dark value``
    estimates to the requested output path.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    conf = general.load_config(args.config)
    if not os.path.exists(conf["datasets"]):
        raise FileNotFoundError(f"Dataset file not found: {conf['datasets']}")

    ds = pd.read_csv(conf["datasets"])
    results = plane_percentiles_all_datasets(conf, ds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
