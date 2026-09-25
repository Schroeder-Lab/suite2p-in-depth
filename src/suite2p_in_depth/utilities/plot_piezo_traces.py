import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .. import correct_traces, extract_data, general, zregister


def load_piezo_data(
    config: dict[str, Any], datasets: pd.DataFrame, output_directory: str | Path
) -> None:
    """Write one synchronized piezo-depth panel per selected dataset.

    DAQ samples are loaded from the configured TIFF experiment directory.
    Piezo values are converted from volts to micrometres using
    ``daq.piezo_volt_per_micron``; output PNG names contain subject and date.
    Existing files are replaced only after the CLI has checked conflicts.

    Parameters
    ----------
    config : dict
        Directory templates and DAQ channel settings.
    datasets : pandas.DataFrame
        Dataset rows with ``Name``, ``Date``, ``Experiments``, ``Planes``,
        and ``Process`` fields.
    output_directory : str or Path
        Directory receiving per-dataset PNG plots.
    """
    plot_path = Path(output_directory)
    if not str(plot_path):
        raise ValueError("An output directory is required.")
    for _index, dataset in datasets.iterrows():
        if not dataset["Process"]:
            continue

        daq = config["daq"]
        piezo_path = config["directories"]["tiffs"].format(
            Name=dataset.Name, Date=dataset.Date, Experiments=dataset.Experiments[0]
        )
        frame_times = extract_data.get_frame_times(
            piezo_path,
            data_file=daq["data"],
            channel_name_file=daq["channel_names"],
            clock_channel=daq["clock_channel_name"],
            sampling_rate=daq["sampling_rate"],
        )
        piezo_trace, piezo_time = extract_data.get_piezo_data(
            piezo_path,
            data_file=daq["data"],
            channel_name_file=daq["channel_names"],
            piezo_channel=daq["piezo_channel_name"],
            sampling_rate=daq["sampling_rate"],
        )
        piezo_per_plane = correct_traces.make_piezo_trace_per_plane(
            piezo_trace,
            piezo_time,
            frame_times,
            num_planes=dataset.Planes,
            volt_per_microns=daq["piezo_volt_per_micron"],
        )
        correct_traces.plot_piezo_per_plane(str(plot_path), piezo_per_plane, daq["sampling_rate"])

        src = plot_path / "piezo_per_plane.png"
        dst = plot_path / f"{dataset.Name}_{dataset.Date}_piezo_per_plane.png"
        src.replace(dst)


def parse_args() -> argparse.Namespace:
    """Parse the configuration and plot destination directory.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` and ``output_dir`` values.
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
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which piezo plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone piezo plotting command.

    Loads the configured dataset table and saves one synchronized piezo plot
    per selected dataset in the requested output directory.
    """
    args = parse_args()
    conf_path = Path(args.config)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file {conf_path} does not exist.")
    conf = general.load_config(args.config)
    data_path = Path(conf["datasets"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")
    datasets = zregister.load_dataset_list(conf["datasets"])
    load_piezo_data(conf, datasets, args.output_dir)


if __name__ == "__main__":
    main()
