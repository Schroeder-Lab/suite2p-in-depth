"""DAQ and TIFF metadata extraction helpers."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from tifffile import TiffFile


def get_nidaq_data(
    data_path: str | os.PathLike[str],
    data_file: str,
    channel_name_file: str,
    sampling_rate: float = 1000,
    num_channels: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.str_] | npt.NDArray[np.int_],
    npt.NDArray[np.float64],
]:
    """Load an interleaved float64 DAQ binary and its channel labels.

    Partial samples at the end of a crashed recording are discarded. The
    returned data have shape ``(samples, channels)`` and time is measured in
    seconds. The function reads files but does not write any output.

    Parameters
    ----------
    data_path : str or os.PathLike
        Directory containing DAQ data and channel names.
    data_file : str
        Prefix of the interleaved binary data file.
    channel_name_file : str
        Channel label filename; used when ``num_channels`` is omitted.
    sampling_rate : float, optional
        DAQ samples per second.
    num_channels : int or None, optional
        Explicit channel count for data without labels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Sample-by-channel values, channel labels or indices, and sample times.

    Raises
    ------
    FileNotFoundError
        If the required data or channel label file is absent.
    ValueError
        If sampling metadata are invalid or the file is too short.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero.")
    # Prefer recorded channel labels; callers may supply a count for unlabeled data.
    if num_channels is None:
        dirs = glob.glob(os.path.join(data_path, channel_name_file))
        if not dirs:
            raise FileNotFoundError(
                f"Missing channel file `{channel_name_file}` in `{data_path}` and "
                "num_channels is None."
            )
        channels = np.loadtxt(dirs[0], delimiter=",", dtype=str)
        if channels.ndim == 0:
            channels = np.array([str(channels)], dtype=str)
        num_channels = len(channels)
    else:
        if num_channels < 1:
            raise ValueError("num_channels must be at least 1.")
        channels = np.arange(num_channels)

    # Load data.
    data_paths = glob.glob(os.path.join(data_path, data_file + "*"), recursive=True)
    if data_paths:
        data_path = data_paths[0]
    else:
        raise FileNotFoundError(f"No data file matching `{data_file}` found in `{data_path}`.")
    nidaq = np.fromfile(data_path, dtype=np.float64)
    if nidaq.size < num_channels:
        raise ValueError(
            f"NIDAQ file contains {nidaq.size} values, fewer than the "
            f"{num_channels} configured channels."
        )

    if len(nidaq) % num_channels == 0:
        nidaq = nidaq.reshape(-1, num_channels)
    else:
        # A partial final sample can remain when acquisition ends unexpectedly.
        complete_samples = len(nidaq) // num_channels
        nidaq = nidaq[: complete_samples * num_channels].reshape(complete_samples, num_channels)

    nidaq_time = np.arange(nidaq.shape[0]) / sampling_rate

    return nidaq, channels, nidaq_time


def assign_frame_time(
    frame_clock: npt.ArrayLike,
    th: float = 0.5,
    sampling_rate: float = 1000,
    plot_path: str | os.PathLike[str] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Assign a time in seconds to each frame-clock rising edge.

    Parameters
    ----------
    frame_clock : array_like
        One-dimensional sampled frame-clock signal.
    th : float, optional
        Voltage threshold used to detect rising edges.
    sampling_rate : float, optional
        DAQ sampling rate in samples per second.
    plot_path : str or os.PathLike or None, optional
        Directory in which to save a frame-clock diagnostic JPG.

    Returns
    -------
    numpy.ndarray
        Frame start times in seconds.

    Raises
    ------
    ValueError
        If the signal shape or sampling rate is invalid, or a requested plot
        has no detected frame-clock edges.

    """
    frame_clock = np.asarray(frame_clock)
    if frame_clock.ndim == 2 and frame_clock.shape[1] == 1:
        frame_clock = frame_clock[:, 0]
    if frame_clock.ndim != 1:
        raise ValueError("frame_clock must be one-dimensional or have shape (samples, 1).")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero.")

    # A rising-edge detector yields one timestamp even when the TTL pulse spans
    # several DAQ samples.
    peak_times = np.flatnonzero(np.diff((frame_clock > th).astype(int), prepend=False, axis=0) > 0)
    if plot_path is not None:
        if peak_times.size == 0:
            raise ValueError("No frame-clock rising edges were found.")
        plt.plot(np.arange(len(frame_clock)) / sampling_rate, frame_clock)
        plt.plot(
            peak_times / sampling_rate,
            np.full(len(peak_times), np.min(frame_clock)),
            "r*",
        )
        plt.xlim(
            (
                (peak_times[0] / sampling_rate) - 1,
                (peak_times[min(10, len(peak_times) - 1)] / sampling_rate) + 1,
            )
        )
        plt.xlabel("time (ms)")
        plt.ylabel("Amplitude (V)")
        plt.savefig(Path(plot_path) / "frame_clock.jpg", format="jpg", dpi=300)
    return peak_times / sampling_rate


def get_piezo_data(
    data_path: str | os.PathLike[str],
    data_file: str,
    channel_name_file: str,
    piezo_channel: str,
    sampling_rate: float = 1000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load the named piezo channel and corresponding DAQ sample times.

    Parameters
    ----------
    data_path : str or os.PathLike
        Directory containing DAQ files.
    data_file : str
        DAQ binary file prefix.
    channel_name_file : str
        File listing the channel names.
    piezo_channel : str
        Name of the piezo voltage channel.
    sampling_rate : float, optional
        DAQ samples per second.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Piezo voltage samples and their times in seconds.

    Raises
    ------
    ValueError
        If the named channel is missing or ambiguous.
    """
    data, channels, time = get_nidaq_data(
        data_path, data_file, channel_name_file, sampling_rate=sampling_rate
    )
    channel_indices = np.flatnonzero(channels == piezo_channel)
    if channel_indices.size != 1:
        raise ValueError(
            f"Expected exactly one piezo channel named {piezo_channel!r}; found {channel_indices.size}."
        )
    piezo = data[:, channel_indices[0]]
    return piezo, time


def get_frame_times(
    data_path: str | os.PathLike[str],
    data_file: str,
    channel_name_file: str,
    clock_channel: str,
    sampling_rate: float = 1000,
) -> npt.NDArray[np.float64]:
    """Return frame-clock rising-edge times from a named DAQ channel.

    Parameters
    ----------
    data_path : str or os.PathLike
        Directory containing DAQ files.
    data_file : str
        DAQ binary file prefix.
    channel_name_file : str
        File listing channel names.
    clock_channel : str
        Name of the frame-clock channel.
    sampling_rate : float, optional
        DAQ samples per second.

    Returns
    -------
    numpy.ndarray
        Frame start times in seconds.

    Raises
    ------
    ValueError
        If the named channel is missing or ambiguous.
    """
    data_sampled, channels, _ = get_nidaq_data(
        data_path, data_file, channel_name_file, sampling_rate=sampling_rate
    )
    channel_indices = np.flatnonzero(channels == clock_channel)
    if channel_indices.size != 1:
        raise ValueError(
            f"Expected exactly one frame-clock channel named {clock_channel!r}; found {channel_indices.size}."
        )
    two_p_clock = data_sampled[:, channel_indices[0]]
    frame_times = assign_frame_time(two_p_clock, sampling_rate=sampling_rate)
    return frame_times


def extract_zoom_factor(tif: TiffFile) -> float:
    """Extract the dimensionless ScanImage zoom factor from TIFF metadata.

    Parameters
    ----------
    tif : tifffile.TiffFile
        Open TIFF whose first page contains an ``Artist`` metadata tag.

    Returns
    -------
    float
        ScanImage ``scanZoomFactor`` value.
    """
    tif_header = tif.pages[0].tags["Artist"].value
    zoom_factor = float(re.findall(r'"scanZoomFactor":\s*(\d*\.?\d+)', tif_header)[0])
    return zoom_factor
