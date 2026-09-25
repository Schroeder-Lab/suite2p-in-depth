import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile

from .. import correct_traces, general
from ..extract_data import get_frame_times, get_piezo_data

logger = logging.getLogger("suite2p_in_depth.inspect_imaging")


def calculate_imaging_metadata(
    piezo: npt.ArrayLike, piezo_time: npt.ArrayLike, frame_times: npt.ArrayLike
) -> tuple[int, float]:
    """Calculate plane count and pooled frame rate from DAQ signals.

    Parameters
    ----------
    piezo : array_like
        Sampled piezo position signal.
    piezo_time : array_like
        Times corresponding to piezo samples.
    frame_times : array_like
        Strictly increasing times of detected imaging frames.

    Returns
    -------
    tuple[int, float]
        Number of planes and pooled frames per second.

    Raises
    ------
    ValueError
        If frame times are invalid or plane count cannot be determined.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    if frame_times.ndim != 1 or frame_times.size < 3 or np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame_times must contain at least three strictly increasing values.")
    num_planes = correct_traces.determine_number_of_planes(piezo, piezo_time, frame_times)
    if num_planes is None or num_planes < 1:
        raise ValueError("Could not determine the number of imaging planes from the piezo trace.")
    return int(num_planes), float(1 / np.median(np.diff(frame_times)))


def _imaging_path(config: dict[str, Any], dataset: pd.Series, experiment: str | int) -> str:
    """Resolve the TIFF directory for one dataset experiment.

    Parameters
    ----------
    config : dict
        Configuration containing imaging directory templates.
    dataset : pandas.Series
        Row with ``Name`` and ``Date`` fields.
    experiment : str or int
        Experiment identifier to insert into the path.

    Returns
    -------
    str
        Formatted TIFF directory path.
    """
    directories = config["directories"]
    if "tiffs" in directories:
        return str(
            directories["tiffs"].format(
                Name=dataset.Name,
                Date=dataset.Date,
                Experiments=experiment,
            )
        )
    return os.path.join(directories["imaging"], dataset.Name, dataset.Date, str(experiment))


def determine_imaging_config(config: dict[str, Any], datasets: pd.DataFrame) -> pd.DataFrame:
    """Determine plane count, channel count, and pooled frame rate.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    datasets : pd.Series
        A DataFrame listing all datasets.

    Returns
    -------
    pandas.DataFrame
        Copy-compatible input table with ``Planes``, ``Channels``, and
        ``Frame_rate`` populated for selected rows. Frame rate is measured in
        frames per second pooled across planes. The input table is mutated and
        returned for compatibility with the original helper.
    """
    # initialize columns in the DataFrame
    datasets["Planes"] = np.nan
    datasets["Frame_rate"] = np.nan
    datasets["Channels"] = np.nan

    for i, dataset in datasets.iterrows():
        if not dataset["Process"]:
            continue
        logger.info("Processing dataset %s/%s", dataset.Name, dataset.Date)
        data_path = _imaging_path(config, dataset, dataset.Experiments[0])

        daq = config["daq"]
        frame_times = get_frame_times(
            data_path,
            data_file=daq["data"],
            channel_name_file=daq["channel_names"],
            clock_channel=daq["clock_channel_name"],
            sampling_rate=daq["sampling_rate"],
        )
        piezo, piezo_time = get_piezo_data(
            data_path,
            data_file=daq["data"],
            channel_name_file=daq["channel_names"],
            piezo_channel=daq["piezo_channel_name"],
            sampling_rate=daq["sampling_rate"],
        )
        num_planes, frame_rate = calculate_imaging_metadata(piezo, piezo_time, frame_times)
        datasets.at[i, "Planes"] = num_planes
        datasets.at[i, "Frame_rate"] = float(frame_rate)

        exp = 0
        while True:
            data_path = _imaging_path(config, dataset, dataset.Experiments[exp])
            all_files = [
                f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
            ]
            tiff_files = sorted(
                f for f in all_files if os.path.splitext(f)[1].lower() in (".tif", ".tiff")
            )
            if tiff_files:
                tiff_file = os.path.join(data_path, tiff_files[0])
                with tifffile.TiffFile(os.path.join(tiff_file)) as tif:
                    if len(tif.series[0].shape) > 3:
                        datasets.at[i, "Channels"] = 2
                    else:
                        datasets.at[i, "Channels"] = 1
                break
            else:
                exp += 1
                if exp >= len(dataset.Experiments):
                    break

    return datasets


def parse_args() -> argparse.Namespace:
    """Parse configuration and output CSV paths from the command line.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` and ``output`` values.
    """
    parser = argparse.ArgumentParser(
        description="Determine number of imaging planes and frame rate for datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="zregistration.yaml",
        help="Path to configuration file (yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV file to receive the calculated imaging metadata.",
    )
    return parser.parse_args()


def main() -> None:
    """Calculate imaging metadata from a standalone configuration.

    Selected datasets are written to the requested output CSV with plane
    counts, channel counts, and pooled frame rates.
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
    datasets = determine_imaging_config(config, datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    datasets[["Name", "Date", "Planes", "Channels", "Frame_rate"]].to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
