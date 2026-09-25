import argparse
import glob
import logging
import os
import re
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import skimage
import tifffile
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure

from . import extract_data, general, preprocess_traces, process_tiff, suite2p_compat, zstack
from .data_types import PlaneTraceResults, TracePaths

logger = logging.getLogger("suite2p")

zoom_window = (0, 5000)


def _determine_reference_depth(ztrace: npt.ArrayLike, reference_frames: int | None = None) -> int:
    """Find the median finite z position in a reference interval.

    Parameters
    ----------
    ztrace : array_like
        Frame-wise axial positions.
    reference_frames : int or None, optional
        Number of initial frames used as the reference interval.

    Returns
    -------
    int
        Median reference position converted to an integer slice index.

    Raises
    ------
    ValueError
        If the selected interval has no finite positions.
    """
    reference_trace = np.asarray(ztrace)
    if reference_frames is not None:
        reference_trace = reference_trace[:reference_frames]
    if not np.isfinite(reference_trace).any():
        raise ValueError("No finite z-trace positions remain in the reference interval.")
    return int(np.nanmedian(reference_trace))


def select_roi_plot_indices(
    n_rois: int, max_roi_plots: int | Literal["all"]
) -> npt.NDArray[np.int_]:
    """Select evenly distributed accepted ROI indices for plotting.

    Parameters
    ----------
    n_rois : int
        Number of accepted ROIs.
    max_roi_plots : int or str
        Maximum number to plot, or ``all``.

    Returns
    -------
    numpy.ndarray
        Sorted zero-based ROI indices; empty when the requested count is zero.

    Raises
    ------
    ValueError
        If either count is invalid.
    """
    if not isinstance(n_rois, Integral) or isinstance(n_rois, (bool, np.bool_)) or n_rois < 0:
        raise ValueError("n_rois must be a non-negative integer.")
    if max_roi_plots == "all":
        count = int(n_rois)
    elif (
        not isinstance(max_roi_plots, Integral)
        or isinstance(max_roi_plots, (bool, np.bool_))
        or max_roi_plots < 0
    ):
        raise ValueError("max_roi_plots must be a non-negative integer or 'all'.")
    else:
        count = min(int(n_rois), int(max_roi_plots))
    if count == 0:
        return np.array([], dtype=int)
    return np.floor((np.arange(count) + 0.5) * int(n_rois) / count).astype(int)


def scale_roi_locations_to_microns(
    roi_locations: npt.ArrayLike, ly: int, lx: int, fov_size: float
) -> npt.NDArray[np.float64]:
    """Convert ROI y and x locations from pixels to micrometres.

    Parameters
    ----------
    roi_locations : array_like
        Locations with shape ``(ROIs, 3)`` in ``[y, x, z]`` order.
    ly, lx : int
        Image height and width in pixels.
    fov_size : float
        Field-of-view extent in micrometres.

    Returns
    -------
    numpy.ndarray
        Float copy with y and x converted; z values are unchanged.

    Raises
    ------
    ValueError
        If the shape or dimensions are invalid.
    """
    locations = np.asarray(roi_locations, dtype=float).copy()
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("roi_locations must have shape (n_rois, 3) in [y, x, z] order.")
    if ly <= 0 or lx <= 0 or fov_size <= 0:
        raise ValueError("Ly, Lx, and fov_size must be positive.")
    locations[:, 0] = locations[:, 0] / ly * fov_size
    locations[:, 1] = locations[:, 1] / lx * fov_size
    return locations


def make_paths(dataset: pd.Series, directories: dict[str, str]) -> TracePaths:
    """
    Build paths for a single experiment row.

    Parameters
    ----------
    dataset : pandas Series
        A single row (experiment) from the preprocess DataFrame.
    directories : dict
        Base directories dictionary.

    Returns
    -------
    dict
        Paths for tiffs, suite2p, piezo, and output.
    """
    subject = str(dataset.Name)
    date = str(dataset.Date)
    paths = {
        "tiffs": directories["tiffs"].format(Name=subject, Date=date),
        "suite2p": os.path.join(directories["suite2p"].format(Name=subject, Date=date), "suite2p"),
        "output": directories["output"].format(Name=subject, Date=date),
    }

    if not os.path.exists(paths["suite2p"]):
        paths["suite2p"] = None
        logger.warning(
            f"NOTE: Suite2p directory for {dataset['Name']} {dataset['Date']} not found. Skipping.\n\n"
        )
        return paths

    if not os.path.exists(paths["tiffs"]):
        paths["tiffs"] = None
        logger.warning(
            f"NOTE: Tiff directory for {dataset['Name']} {dataset['Date']} not found. Skipping.\n\n"
        )
        return paths

    if "piezo" in directories:
        paths["piezo"] = directories["piezo"].format(Name=subject, Date=date)
        if not os.path.exists(paths["piezo"]):
            paths["piezo"] = None
            logger.warning(
                f"NOTE: Piezo data for {dataset['Name']} {dataset['Date']} not found."
                f" No z-correction can be performed.\n"
                "Frame rate of each imaging plane will be taken from ops variable.\n\n"
            )

    else:
        paths["piezo"] = None

    os.makedirs(paths["output"], exist_ok=True)
    return paths


def determine_pixels_to_microns(db: dict[str, Any], tiff_path: str, zoom_sizes_path: str) -> float:
    """Estimate field-of-view size from TIFF zoom and a calibration table.

    Parameters
    ----------
    db : dict
        Suite2p database with the recorded TIFF ``file_list``.
    tiff_path : str
        Current directory containing the recorded TIFF.
    zoom_sizes_path : str
        CSV mapping ``Zoom`` values to ``FOV_size`` in micrometres.

    Returns
    -------
    float
        Interpolated field-of-view size in micrometres.

    Raises
    ------
    FileNotFoundError
        If the zoom calibration table is absent.
    """
    # Determine zoom factor from tiff header.
    # Normalize the base directory and the "full" file path coming from suite2p
    new_tiff_path = os.path.normpath(tiff_path)
    old_tiff_path = os.path.normpath(db["file_list"][-1])

    # Split into components
    new_parts = new_tiff_path.split(os.sep)
    old_parts = old_tiff_path.split(os.sep)

    # Find where final part of new_parts occurs inside old_parts
    index = old_parts.index(new_parts[-1])
    tiff_file = os.path.join(new_tiff_path, *old_parts[index + 1 :])
    tif = tifffile.TiffFile(tiff_file)
    zoom_factor = extract_data.extract_zoom_factor(tif)

    # Read csv file mapping zoom factors to FOV sizes.
    if not os.path.exists(zoom_sizes_path):
        raise FileNotFoundError(f"The file '{zoom_sizes_path}' does not exist.")
    zooms_sizes = pd.read_csv(zoom_sizes_path)

    # Interpolate to get FOV size for the current zoom factor.
    zoom_to_size = sp.interpolate.interp1d(zooms_sizes["Zoom"], zooms_sizes["FOV_size"])
    fov_size = zoom_to_size(zoom_factor)

    return fov_size


def load_dataset_list(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read the trace-correction dataset table with stable column types.

    Parameters
    ----------
    path : str or os.PathLike
        Dataset CSV path.

    Returns
    -------
    pandas.DataFrame
        Dataset rows with names and dates preserved as strings.
    """
    return pd.read_csv(
        path,
        dtype={
            "Name": str,
            "Date": str,
            "Depth": "float64",
            "Zstack_folder": str,
            "Ignore_planes": "Int64",
            "Process": bool,
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse the standalone trace-correction configuration path.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` path.
    """
    parser = argparse.ArgumentParser(description="Correct fluorescence traces")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="zstack.yaml",
        help="Path to configuration file (yaml)",
    )
    return parser.parse_args()


def plot_signal_per_roi(
    plots_path: str,
    stat: np.ndarray,
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    fluorescence_zcorrected: np.ndarray | None,
    neuropil_zcorrected: np.ndarray | None,
    fluorescence_ncorrected: np.ndarray,
    baseline: np.ndarray,
    delta_f: np.ndarray,
    ztrace: np.ndarray | None,
    fluorescence_profiles: np.ndarray | None,
    peaks: Sequence[np.ndarray] | None,
    plane_stack: np.ndarray | None,
    reference_depth: int | None,
    ind_best_slice: int | np.integer[Any] | None,
    plane_rate: float,
    n_frames: npt.ArrayLike,
    prctile: float,
    roi_indices: Iterable[int] | None = None,
    zoom_in: bool = False,
) -> None:
    """Save raw, corrected, baseline, and dF/F plots for selected ROIs.

    When a z-stack is available, each figure also shows the axial profile and
    z trace. Figures are saved under ``plots_path/ROIs`` and then closed.

    Parameters
    ----------
    plots_path : str
        Parent plot directory.
    stat : sequence[dict]
        Accepted ROI metadata in plotted index order.
    fluorescence, neuropil : numpy.ndarray
        Raw traces in ``(frames, ROIs)`` order.
    fluorescence_zcorrected, neuropil_zcorrected : numpy.ndarray or None
        Axial-motion corrected traces for stack-enabled processing.
    fluorescence_ncorrected, baseline, delta_f : numpy.ndarray
        Neuropil-corrected traces, baseline, and dF/F values.
    ztrace : numpy.ndarray or None
        Estimated axial slice per frame.
    fluorescence_profiles, peaks : array_like or None
        Axial ROI profiles and their peak slice indices.
    plane_stack : numpy.ndarray or None
        Registered stack in ``(z, y, x)`` order.
    reference_depth, ind_best_slice : float or None
        Depth markers shown when a stack is available.
    plane_rate : float
        Frames per second for this plane.
    n_frames : array_like
        Experiment frame boundaries.
    prctile : float
        Fluorescence percentile used for axial profiles.
    roi_indices : iterable[int] or None, optional
        ROI indices to plot; ``None`` plots all accepted ROIs.
    zoom_in : bool, optional
        Use the compact zoomed plot layout and filename suffix.
    """
    colors_paired = plt.get_cmap("Paired")

    if roi_indices is None:
        roi_indices = range(len(stat))
    for i in roi_indices:
        i = int(i)
        roi = stat[i]

        fig = plt.figure(figsize=(19.2, 9.76))

        if plane_stack is not None:
            if zoom_in:
                gs = gridspec.GridSpec(4, 2, width_ratios=(1, 9))
                profile_axis = fig.add_subplot(gs[:, 0])
                stack_axis = None
                trace_column = 1
            else:
                gs = gridspec.GridSpec(4, 3, width_ratios=(1, 1, 8))
                profile_axis = fig.add_subplot(gs[:, 0])
                stack_axis = fig.add_subplot(gs[:, 1])
                trace_column = 2
            ztrace_axis = fig.add_subplot(gs[0, trace_column])
            zstack.plot_zprofile(
                fluorescence[:, i],
                neuropil[:, i],
                ztrace,
                fluorescence_profiles[:, i],
                peaks[i],
                roi,
                plane_stack,
                reference_depth,
                ind_best_slice,
                plane_rate,
                n_frames,
                prctile,
                colors_paired,
                profile_axis=profile_axis,
                stack_axis=stack_axis,
                ztrace_axis=ztrace_axis,
                zoom_in=zoom_in,
            )
            trace_axes = [fig.add_subplot(gs[row, trace_column]) for row in range(1, 4)]
            bouton_layout = False
        else:
            gs = gridspec.GridSpec(3, 1)
            trace_axes = [fig.add_subplot(gs[row, 0]) for row in range(3)]
            bouton_layout = True

        compact_legend = {
            "fontsize": 8,
            "framealpha": 0.8,
            "borderpad": 0.25,
            "labelspacing": 0.25,
            "handlelength": 1.4,
            "handletextpad": 0.4,
            "columnspacing": 0.8,
        }

        # Raw and z-motion corrected ROI and neuropil traces
        plt.sca(trace_axes[0])
        time = np.arange(fluorescence.shape[0]) / plane_rate
        raw_handles = [
            plt.plot(time, fluorescence[:, i], color=colors_paired(1))[0],
            plt.plot(time, neuropil[:, i], color=colors_paired(7))[0],
        ]
        if plane_stack is None:
            raw_labels = ["F(raw)", "N(raw)"]
        else:
            raw_handles.extend(
                (
                    plt.plot(time, fluorescence_zcorrected[:, i], color=colors_paired(0))[0],
                    plt.plot(time, neuropil_zcorrected[:, i], color=colors_paired(6))[0],
                )
            )
            raw_labels = ["F(raw)", "N(raw)", "F(z-corr.)", "N(z-corr.)"]
        if not bouton_layout:
            trace_axes[0].legend(
                raw_handles,
                raw_labels,
                loc="upper right",
                ncols=len(raw_labels),
                **compact_legend,
            )
        else:
            trace_axes[0].legend(
                raw_handles,
                raw_labels,
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                borderaxespad=0,
                **compact_legend,
            )
        [plt.axvline(x / plane_rate, color="k", linewidth=1) for x in n_frames]
        plt.xlim(0, fluorescence.shape[0] / plane_rate)
        plt.gca().set_xticklabels([])
        plt.tick_params(axis="y", right=True, labelleft=False, labelright=True)

        # Neuropil-corrected ROI traces and F0
        plt.sca(trace_axes[1])
        neuropil_handles = [
            plt.plot(time, fluorescence_ncorrected[:, i], color=colors_paired(1))[0],
            plt.plot(time, baseline[:, i], color=colors_paired(2), linewidth=4)[0],
        ]
        neuropil_labels = ["F(Npil corr.)", "F0"]
        if not bouton_layout:
            trace_axes[1].legend(
                neuropil_handles,
                neuropil_labels,
                loc="upper right",
                ncols=len(neuropil_labels),
                **compact_legend,
            )
        else:
            trace_axes[1].legend(
                neuropil_handles,
                neuropil_labels,
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                borderaxespad=0,
                **compact_legend,
            )
        [plt.axvline(x / plane_rate, color="k", linewidth=1) for x in n_frames]
        plt.xlim(0, fluorescence.shape[0] / plane_rate)
        plt.gca().set_xticklabels([])
        plt.tick_params(axis="y", right=True, labelleft=False, labelright=True)

        # dF/F
        plt.sca(trace_axes[2])
        delta_handle = plt.plot(time, delta_f[:, i], color=colors_paired(3))[0]
        if not bouton_layout:
            trace_axes[2].legend([delta_handle], ["dF/F"], loc="upper right", **compact_legend)
        else:
            trace_axes[2].legend(
                [delta_handle],
                ["dF/F"],
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                borderaxespad=0,
                **compact_legend,
            )
        [plt.axvline(x / plane_rate, color="k", linewidth=1) for x in n_frames]
        plt.xlim(0, fluorescence.shape[0] / plane_rate)
        plt.xlabel("Time (s)")
        plt.tick_params(axis="y", right=True, labelleft=False, labelright=True)

        title = f"ROI {i} (zoom-in)" if zoom_in else f"ROI {i}"
        fig.suptitle(title, fontsize=20, fontweight="bold")
        if plane_stack is not None:
            fig.subplots_adjust(
                left=0.06, right=0.95, bottom=0.08, top=0.9, wspace=0.02, hspace=0.25
            )
        else:
            fig.subplots_adjust(left=0.06, right=0.89, bottom=0.08, top=0.9, hspace=0.25)

        file_name = f"ROI{str(i).zfill(4)}_zoomed.png" if zoom_in else f"ROI{str(i).zfill(4)}.png"
        fig.savefig(os.path.join(plots_path, "ROIs", file_name), dpi=300)
        plt.close(fig)


def plot_rois_and_reference(
    plots_path: str, db: dict[str, Any], stat: np.ndarray, mean_img: np.ndarray
) -> None:
    """Save accepted ROI masks beside their registered mean image.

    Parameters
    ----------
    plots_path : str
        Directory receiving ``ROIs_and_meanImg.png``.
    db : dict
        Suite2p image dimensions ``Ly`` and ``Lx``.
    stat : sequence[dict]
        ROI pixel masks, overlaps, and center coordinates.
    mean_img : numpy.ndarray
        Registered mean image in ``(y, x)`` order.
    """
    # Create a mask with ROIs numbered from 1 to n_rois.
    mask = np.zeros((db["Ly"], db["Lx"]), dtype=np.uint16)
    for n, roi in enumerate(stat):
        ypix = roi["ypix"][~roi["overlap"]]
        xpix = roi["xpix"][~roi["overlap"]]
        mask[ypix, xpix] = n + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # (1) ROI masks (on black background)
    # Create a colormap: black for 0, then rainbow for ROIs
    n_rois = len(stat)
    rainbow = plt.get_cmap("turbo", n_rois)
    colors = np.vstack(([0, 0, 0, 1], rainbow(np.arange(n_rois))))
    cmap = ListedColormap(colors)
    axes[0].set_adjustable("box")
    axes[0].imshow(mask, cmap=cmap, vmin=0, vmax=n_rois)

    # Add ROI IDs as text labels in the center of each ROI
    for n, roi in enumerate(stat):
        y, x = roi["med"]
        axes[0].text(
            x, y, str(n), color="white", fontsize=12, ha="center", va="center", fontweight="bold"
        )

    axes[0].set_title("ROI Masks")
    axes[0].axis("off")

    # (2) ROI outlines on mean image
    mean_img = np.clip(mean_img, np.percentile(mean_img, 1), np.percentile(mean_img, 99))
    axes[1].imshow(mean_img, cmap="gray")
    for n in range(0, n_rois):
        mask_roi = (mask == n + 1).astype(np.uint8)
        contours = measure.find_contours(mask_roi, 0.5)
        for contour in contours:
            axes[1].plot(contour[:, 1], contour[:, 0], color="green", linewidth=1.0)

    axes[1].set_title("Mean Image with ROI Masks")
    axes[1].axis("off")

    fig.savefig(os.path.join(plots_path, "ROIs_and_meanImg.png"), dpi=300)
    plt.close(fig)


def plot_piezo_per_plane(plot_path: str, piezo_per_plane: np.ndarray, sampling_rate: float) -> None:
    """Save within-frame piezo depth trajectories for all imaging planes.

    Parameters
    ----------
    plot_path : str
        Directory receiving ``piezo_per_plane.png``.
    piezo_per_plane : numpy.ndarray
        Depth in micrometres in ``(samples, planes)`` order.
    sampling_rate : float
        Sampling rate used for the plotted time axis.
    """
    os.makedirs(plot_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    t = np.arange(piezo_per_plane.shape[0]) / sampling_rate
    for j in range(piezo_per_plane.shape[1]):
        ax.plot(t, piezo_per_plane[:, j], color=zstack.plane_color(j), label=f"Plane {j}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (µm)")
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(os.path.join(plot_path, "piezo_per_plane.png"), dpi=300)
    plt.close(fig)


def make_piezo_trace_per_plane(
    piezo: npt.ArrayLike,
    piezo_time: npt.ArrayLike,
    frame_times: npt.ArrayLike,
    num_planes: int | None = None,
    volt_per_microns: float = 5 / 400,
    win_size: float = 10,
    max_avg: int = 1000,
) -> npt.NDArray[np.float64]:
    """Calculate the mean within-frame piezo trajectory for each plane.

    Parameters
    ----------
    piezo : array_like
        Sampled piezo voltage trace.
    piezo_time : array_like
        Sample times in seconds for ``piezo``.
    frame_times : array_like
        Imaging frame start times in seconds.
    num_planes : int or None, optional
        Number of planes; inferred from the trace when omitted.
    volt_per_microns : float, optional
        Piezo volts per micrometre.
    win_size : int, optional
        Smoothing window width in milliseconds.
    max_avg : int, optional
        Approximate maximum number of frames averaged per plane.

    Returns
    -------
    numpy.ndarray
        Depth in micrometres with shape ``(samples per frame, planes)``,
        relative to the minimum sampled depth.

    Raises
    ------
    ValueError
        If signal shapes, times, conversion factor, or frame coverage are
        insufficient for a complete trajectory.
    """
    piezo = np.asarray(piezo, dtype=float)
    piezo_time = np.asarray(piezo_time, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    if piezo.ndim != 1 or piezo_time.ndim != 1 or piezo.size != piezo_time.size:
        raise ValueError("piezo and piezo_time must be one-dimensional arrays of equal length.")
    if piezo.size < 2 or frame_times.ndim != 1 or frame_times.size < 3:
        raise ValueError("At least two piezo samples and three frame times are required.")
    if np.any(np.diff(piezo_time) <= 0) or np.any(np.diff(frame_times) <= 0):
        raise ValueError("piezo_time and frame_times must be strictly increasing.")
    if volt_per_microns <= 0:
        raise ValueError("volt_per_microns must be greater than zero.")
    if max_avg < 1:
        raise ValueError("max_avg must be at least 1.")

    sample_interval = np.median(np.diff(piezo_time))
    window_samples = max(1, int(win_size / 1000 / sample_interval))
    # Returns a Hanning window of size winSize.
    w = np.hanning(window_samples) if window_samples > 2 else np.ones(window_samples)

    # Divides the values in the window by the sum of the values.
    # This averages the window so that the area under the curve is 1.
    w /= np.sum(w)

    # Smooth high-frequency acquisition noise before converting voltage to depth.
    piezo = np.convolve(piezo, w, "same")

    # Divides the piezo trace by the voltage ratio to convert the voltage values into distance in microns.
    piezo /= volt_per_microns

    # Determines the number of imaging planes if not provided.
    if num_planes is None:
        num_planes = determine_number_of_planes(piezo, piezo_time, frame_times)
    if num_planes is None or num_planes < 1:
        raise ValueError("Could not determine a positive number of imaging planes.")

    # Determines how often we sample the piezo trace throughout recording a single frame.
    piezo_samples_per_frame = int(round(np.median(np.diff(frame_times)) / sample_interval))
    if piezo_samples_per_frame < 1:
        raise ValueError("Frame intervals are shorter than the piezo sampling interval.")
    # Store one within-frame depth trajectory per imaging plane.
    piezo_per_plane = np.zeros((piezo_samples_per_frame, num_planes))
    n = range(piezo_samples_per_frame)
    num_reps = max(1, round(len(frame_times) / num_planes / max_avg))

    # Runs over the imaging planes and calculates the average piezo trace per frame.
    for plane in range(num_planes):
        # Determines the time at which the piezo starts and ends for each plane but ignoring the first frame
        # because the location of the first frame is when the piezo starts moving so it is inaccurate.
        plane_frames = np.arange(num_planes + plane, len(frame_times) - 1, num_planes)

        # Determines the range over which to sample over the piezo trace given the batchFactor specified.
        rep_indices = plane_frames[range(0, len(plane_frames), num_reps)]
        if rep_indices.size == 0:
            raise ValueError(f"No complete frames are available for imaging plane {plane}.")

        # Align sampled trajectories by their within-frame offset.
        aligned_piezo = np.zeros((piezo_samples_per_frame, len(rep_indices)))
        for rep, frame_index in enumerate(rep_indices):
            # Determines the section of the piezo trace to take into account given the piezo start and end times
            # specified above.
            time_inds = np.flatnonzero(piezo_time >= frame_times[frame_index])[:1]
            if time_inds.size == 0 or time_inds[0] + piezo_samples_per_frame > piezo.size:
                raise ValueError(
                    f"Piezo samples do not cover the complete frame at index {frame_index}."
                )
            aligned_piezo[:, rep] = piezo[time_inds + n]

        piezo_per_plane[:, plane] = np.nanmean(aligned_piezo, 1)

    # Subtracts the minimum value so that top-most position is 0.
    piezo_per_plane -= np.min(piezo_per_plane)

    return piezo_per_plane


def determine_number_of_planes(
    piezo: npt.ArrayLike, piezo_time: npt.ArrayLike, frame_times: npt.ArrayLike
) -> int | None:
    """Infer the imaging plane cycle length from piezo autocorrelation.

    Parameters
    ----------
    piezo : array_like
        Sampled piezo position signal.
    piezo_time : array_like
        Time in seconds for each piezo sample.
    frame_times : array_like
        Imaging frame start times in seconds.

    Returns
    -------
    int or None
        Number of planes, or ``None`` if no autocorrelation peak is found.

    """
    # Each interval between frame starts corresponds to one acquired frame.
    bin_ids = np.digitize(piezo_time, frame_times)

    # One thousand frames are sufficient to reveal the periodic plane cycle.
    piezo_positions = np.full(min(1000, len(frame_times) - 1), np.nan)
    for b in range(1, piezo_positions.shape[0] + 1):
        inds = np.where(bin_ids == b)[0]
        if inds.size > 0:
            piezo_positions[b - 1] = np.nanmean(piezo[inds])

    x = piezo_positions - np.nanmean(piezo_positions)
    ac_full = np.correlate(x, x, mode="same")
    ac = ac_full[ac_full.size // 2 :]  # lags >= 0

    # The first non-zero-lag autocorrelation peak is the plane-cycle length.
    peaks, _ = sp.signal.find_peaks(ac[1:])
    if peaks.size == 0:
        return None

    return int(peaks[0] + 1)


def process_plane(
    config: dict[str, Any],
    plane_directory: str,
    tiff_path: str,
    zstack_raw_path: str | None,
    output_directory: str,
    piezo: np.ndarray | None,
    frame_rate: float,
    depth: float,
) -> PlaneTraceResults | None:
    """Process accepted ROIs in one imaging plane.

    Applies neuropil correction and optional z-stack motion correction,
    calculates dF/F, and saves selected per-plane plots and intermediate data.

    Parameters
    ----------
    config : dict
        Trace-correction and plot settings.
    plane_directory : str
        Suite2p plane containing ROI traces and metadata.
    tiff_path : str
        Original TIFF directory used to recover field-of-view calibration.
    zstack_raw_path : str or None
        Original z-stack TIFF; ``None`` disables stack-based correction.
    output_directory : str
        Directory receiving processed arrays and plots.
    piezo : numpy.ndarray or None
        Within-frame depth trajectory in ``(samples, planes)`` order.
    frame_rate : float
        Pooled imaging frames per second.
    depth : float
        Baseline depth in micrometres added to ROI locations.

    Returns
    -------
    dict or None
        Traces, axial profiles and positions, ROI IDs, and locations. Returns
        ``None`` if traces are absent or no ROIs are accepted.

    """
    if not (os.path.exists(os.path.join(plane_directory, "F.npy"))):
        return None

    # Load suite2p parameters and ROI data.
    is_cell = np.load(os.path.join(plane_directory, "iscell.npy"))  # (nROIs, 2)
    stat = np.load(os.path.join(plane_directory, "stat.npy"), allow_pickle=True)
    stat = stat[is_cell[:, 0].astype(bool)]
    db, settings = suite2p_compat.load_parameters(plane_directory)
    reg_outputs = suite2p_compat.load_outputs(plane_directory)[0]
    if reg_outputs is None or "badframes" not in reg_outputs:
        raise ValueError(f"Missing registration outputs or badframes for {plane_directory}.")

    if len(stat) == 0:
        return None

    # Normalize Windows/Unix separators and strip trailing slashes
    leaf = os.path.basename(os.path.normpath(plane_directory))
    plane_match = re.search(r"plane(\d+)$", leaf)

    # Determine location of each ROI in 3D, in units of microns. Depth is relative to the top-most position of the
    # z-actuator (piezo).
    roi_locations = np.zeros((len(stat), 3))

    # First, determine depth.
    if piezo is not None and plane_match is not None:
        plane = int(plane_match.group(1))
        for i, roi in enumerate(stat):
            # The piezo depth depends on when the ROI's row is scanned.
            y_position = roi["med"][0] / db["Ly"]
            # Determine position of actuator (piezo), which depends on relative Y position. (Scanning along X dimension is
            # so fast that we can ignore it.)
            piezo_sample = int(np.round((piezo.shape[0] - 1) * y_position))
            z_position = piezo[piezo_sample, plane]
            roi_locations[i, :] = np.append(roi["med"], z_position + depth)

    else:
        for i, roi in enumerate(stat):
            roi_locations[i, :] = np.append(roi["med"], depth)

        plane = leaf.split("plane", 1)[-1] if "plane" in leaf else ""

    logger.info(f"  PLANE {plane}")

    # Convert horizontal locations from pixels to microns.
    if config["pixels_to_microns"]:
        fov_size = determine_pixels_to_microns(db, tiff_path, config["zoom_to_size"])
        roi_locations = scale_roi_locations_to_microns(roi_locations, db["Ly"], db["Lx"], fov_size)
    else:
        fov_size = db["Ly"]

    # Load neural data.
    fluorescence = np.load(os.path.join(plane_directory, "F.npy"), allow_pickle=True).T
    neuropil = np.load(os.path.join(plane_directory, "Fneu.npy")).T
    # Only include "good" ROIs.
    fluorescence = fluorescence[:, is_cell[:, 0].astype(bool)]
    neuropil = neuropil[:, is_cell[:, 0].astype(bool)]
    if fluorescence.shape != neuropil.shape:
        raise ValueError(
            "F and Fneu shapes differ after ROI selection: "
            f"{fluorescence.shape} versus {neuropil.shape}."
        )
    badframes = np.asarray(reg_outputs["badframes"], dtype=bool)
    if badframes.ndim != 1 or badframes.size != fluorescence.shape[0]:
        raise ValueError(
            f"badframes length {badframes.size} does not match trace length "
            f"{fluorescence.shape[0]}."
        )
    # Remove bad frames.
    fluorescence[badframes, :] = np.nan
    neuropil[badframes, :] = np.nan

    # Add value at absolute zero (dark signal) to the traces.
    fluorescence -= config["delta_F"]["absolute_zero"]
    neuropil -= config["delta_F"]["absolute_zero"]

    processed_path = os.path.join(output_directory, "2P_processed", f"plane{plane}")
    os.makedirs(processed_path, exist_ok=True)
    device = general.assign_torch_device(settings["torch_device"])

    # If Z stack was generated (and path to file provided), perform Z correction.
    if zstack_raw_path is not None and plane_match:
        (
            data_path,
            db_zcorr,
            settings_reg_within,
            settings_reg_across,
            settings_reg_ref,
            settings_reg_frames,
            sigma,
            spacing,
        ) = zstack.prep_zstack_parameters(plane_directory, config, db, settings, fov_size)

        ind_best_slice, plane_stack = zstack.extract_reference_slice(
            zstack_raw_path,
            processed_path,
            plane,
            db_zcorr,
            settings_reg_within,
            settings_reg_across,
            settings_reg_ref,
            config["z_correction"]["ref_frames"],
            piezo,
            sigma,
            spacing,
            reg_outputs,
            device,
        )
        best_slice = plane_stack[ind_best_slice]

        zcorr_path = os.path.join(processed_path, f"zcorr_plane{plane}.npy")
        zcorr, ztrace = zstack.compute_correlations_reference(
            data_path,
            zcorr_path,
            db_zcorr,
            settings_reg_frames,
            ind_best_slice,
            plane_stack,
            config["z_correction"]["stack_step"],
            config["z_correction"]["sigma_trace"],
            device,
        )

        # If we used the non-functional channel (2) for alignment, we now need to use the registered Z stack of the
        # functional channel (1) to correct the recorded calcium traces.
        if db["functional_chan"] == 1 and settings["registration"]["align_by_chan2"]:
            plane_stack_functional_path = os.path.join(
                processed_path, f"stack_plane{plane}_chan{db['functional_chan']}.tif"
            )
            plane_stack = skimage.io.imread(plane_stack_functional_path)

        # Determine Z profile for each ROI.
        logger.info("  Determine z-profiles and reference depth")
        fluorescence_profiles = process_tiff.extract_zprofiles(
            plane_directory, plane_stack, abs_zero=config["delta_F"]["absolute_zero"], device=device
        )

        # Determine best reference depth.
        if config["z_correction"]["reference"] == "first":  # across first experiment
            reference_depth = _determine_reference_depth(ztrace, db["frames_per_folder"][0])
        else:  # across all experiments
            reference_depth = _determine_reference_depth(ztrace)

        # Correct ROI and neuropil traces for z motion.
        logger.info("  Correct traces for z-motion")
        fluorescence_zcorrected, neuropil_zcorrected, peaks = preprocess_traces.correct_zmotion(
            fluorescence,
            neuropil,
            fluorescence_profiles,
            ztrace,
            reference_depth,
            ignore_faults=config["z_correction"]["remove_outliers"],
            frames_per_experiment=db["frames_per_folder"],
        )

        settings["z_correction"] = config["z_correction"]
        settings["suite2p_stack_within"] = config["suite2p_stack_within"]
        settings["suite2p_stack_across"] = config["suite2p_stack_across"]
        settings["suite2p_stack_to_ref"] = config["suite2p_stack_to_ref"]
        settings["suite2p_stack_to_frames"] = config["suite2p_stack_to_frames"]
    else:
        # If no Z correction is performed (for example if no Z stack was given)
        # only the uncorrected delta F over F is considered.
        fluorescence_zcorrected = fluorescence
        neuropil_zcorrected = neuropil
        ind_best_slice = None
        best_slice = None
        plane_stack = None
        zcorr = None
        ztrace = None
        fluorescence_profiles = None
        reference_depth = None
        peaks = None

    logger.info("  Perform neuropil correction and calculate dF/F")

    plane_rate = frame_rate / db["nplanes"]

    # Calculate baseline fluorescence F0.
    baseline = preprocess_traces.get_f0(
        fluorescence_zcorrected,
        plane_rate,
        f0_percentile=config["delta_F"]["F0_percentile"],
        window_size=config["delta_F"]["F0_window"],
        frames_per_folder=db["frames_per_folder"],
    )

    # Perform neuropil correction.
    fluorescence_ncorrected, regression_pars, fluorescence_values, neuropil_values = (
        preprocess_traces.correct_neuropil(
            fluorescence_zcorrected,
            neuropil_zcorrected,
            plane_rate,
            baseline,
            baseline_percentile=config["delta_F"]["F0_percentile"],
            baseline_window=config["delta_F"]["F0_window"],
            frames_per_folder=db["frames_per_folder"],
        )
    )

    # Calculate delta F over F.
    delta_f = preprocess_traces.get_delta_f_over_f(fluorescence_ncorrected, baseline)

    # Places all the results in a dictionary (dF/F, Z corrected dF/F,
    # z profiles, z traces and the cell locations in X, Y and Z).
    results = {
        "zCorr_stack": zcorr,
        "zTrace": ztrace,
        "zProfiles": fluorescence_profiles,
        "F_zcorrected": fluorescence_zcorrected,
        "N_zcorrected": neuropil_zcorrected,
        "F_ncorrected": fluorescence_ncorrected,
        "N_regression": regression_pars,
        "F_bin_values": fluorescence_values,
        "N_bin_values": neuropil_values,
        "dff": delta_f,
        "locs": roi_locations,
        "cellId": np.where(is_cell[:, 0].astype(bool))[0].reshape(-1, 1),
    }

    settings["fs"] = frame_rate
    settings["delta_F"] = config["delta_F"]
    settings["daq"] = config["daq"]

    detect_file = os.path.join(plane_directory, "detect_outputs.npy")
    detect_outputs = {}
    if os.path.isfile(detect_file):
        detect_outputs = np.load(
            os.path.join(plane_directory, "detect_outputs.npy"), allow_pickle=True
        ).item()

    detect_outputs["fov_size"] = fov_size if config["pixels_to_microns"] else None

    np.save(os.path.join(plane_directory, "db_correcting.npy"), db)
    np.save(os.path.join(plane_directory, "settings_correcting.npy"), settings)
    np.save(os.path.join(plane_directory, "detect_outputs_correcting.npy"), detect_outputs)

    if ind_best_slice is not None:
        zcorrect_outputs = {
            "slice_matching_refImg": ind_best_slice,
            "reference_depth": reference_depth,
        }
        np.save(os.path.join(plane_directory, "zcorrect_outputs.npy"), zcorrect_outputs)

    if config["delta_F"]["plot"]:
        plots_path = os.path.join(output_directory, "2P_processed", f"plane{plane}", "plots")
        os.makedirs(plots_path, exist_ok=True)
        plot_rois_and_reference(plots_path, db, stat, reg_outputs["meanImg"])
        if best_slice is not None:
            zstack.plot_reference_slice(
                plots_path, reg_outputs["refImg"], np.asarray(best_slice, dtype=np.float32)
            )

        roi_plot_indices = select_roi_plot_indices(len(stat), config["delta_F"]["max_roi_plots"])
        logger.info(
            "  Plot ROI traces for %d of %d accepted ROIs",
            len(roi_plot_indices),
            len(stat),
        )
        if roi_plot_indices.size:
            os.makedirs(os.path.join(plots_path, "ROIs"), exist_ok=True)
            plot_signal_per_roi(
                plots_path,
                stat,
                fluorescence,
                neuropil,
                fluorescence_zcorrected,
                neuropil_zcorrected,
                fluorescence_ncorrected,
                baseline,
                delta_f,
                ztrace,
                fluorescence_profiles,
                peaks,
                plane_stack,
                reference_depth,
                ind_best_slice,
                plane_rate,
                np.cumsum(db["frames_per_folder"]),
                config["delta_F"]["F0_percentile"],
                roi_indices=roi_plot_indices,
            )

            if config["delta_F"]["plot_zoomed_traces"] and fluorescence.shape[0] > zoom_window[1]:
                t = range(zoom_window[0], zoom_window[1])
                zoomed_ztrace = None if ztrace is None else ztrace[t]
                plot_signal_per_roi(
                    plots_path,
                    stat,
                    fluorescence[t],
                    neuropil[t],
                    fluorescence_zcorrected[t],
                    neuropil_zcorrected[t],
                    fluorescence_ncorrected[t],
                    baseline[t],
                    delta_f[t],
                    zoomed_ztrace,
                    fluorescence_profiles,
                    peaks,
                    plane_stack,
                    reference_depth,
                    ind_best_slice,
                    plane_rate,
                    np.cumsum(db["frames_per_folder"]),
                    config["delta_F"]["F0_percentile"],
                    roi_indices=roi_plot_indices,
                    zoom_in=True,
                )

    return results


def process_dataset(
    suite2p_directory: str,
    config: dict[str, Any],
    tiff_path: str,
    frame_rate: float,
    piezo: np.ndarray | None = None,
    zstack_path: str | None = None,
    output_directory: str | None = None,
    ignore_planes: npt.ArrayLike | None = None,
    depth: float = 0,
) -> None:
    """Process selected planes and save combined ROI traces and metadata.

    Per-plane signals are trimmed to the shortest valid recording before
    concatenation. Optional z-stack outputs are saved alongside dF/F, ROI
    identities, and ROI locations.

    Parameters
    ----------
    suite2p_directory : str
        Directory containing numeric plane folders or ``plane_z``.
    config : dict
        Trace-correction, z-stack, and output settings.
    tiff_path : str
        Original TIFF directory for field-of-view calibration.
    frame_rate : float
        Pooled imaging frames per second.
    piezo : numpy.ndarray or None, optional
        Within-frame piezo depth trajectory by plane.
    zstack_path : str or None, optional
        Original z-stack TIFF for motion correction.
    output_directory : str or None, optional
        Directory receiving combined processed arrays.
    ignore_planes : array_like or None, optional
        Physical plane IDs excluded from processing.
    depth : float, optional
        Baseline depth added to ROI locations, in micrometres.

    Raises
    ------
    ValueError
        If no valid plane directories or accepted ROI results remain.

    """
    # Algorithm 2 post-processing reads the combined z-registered movie. Z-stack
    # correction instead processes each conventionally registered numeric plane.
    if config["use_zstack"]:
        plane_directories = [
            path
            for path in glob.glob(os.path.join(suite2p_directory, "plane*"))
            if re.search(r"plane\d+$", path)
        ]
        plane_directories = sorted(
            plane_directories, key=lambda path: int(re.search(r"plane(\d+)$", path).group(1))
        )
    else:
        plane_z = os.path.join(suite2p_directory, "plane_z")
        plane_directories = [plane_z] if os.path.isdir(plane_z) else []
    if not plane_directories:
        raise ValueError(f"No plane directories found in {suite2p_directory}.")

    # Determine IDs of roi_planes previously processed with suite2p.
    planes = [
        int(m.group(1)) if (m := re.search(r"plane(\d+)$", s)) else -1 for s in plane_directories
    ]

    # Ignore roi_planes if specified.
    ind_ignore = np.where(np.isin(planes, ignore_planes))[0]
    planes = np.delete(planes, ind_ignore)
    plane_directories = np.delete(plane_directories, ind_ignore)

    # Process each plane and return list with results for each plane. If no data for plane at planes[i]
    # exists (output from suite2p), results[i] will be None.
    results = [
        process_plane(
            config,
            plane_directory,
            tiff_path,
            zstack_path,
            output_directory,
            piezo,
            frame_rate,
            depth,
        )
        for plane_directory in plane_directories
    ]

    # Identify roi_planes for which no data was found.
    ind_valid_planes = [i for i, res in enumerate(results) if res is not None]
    if not ind_valid_planes:
        raise ValueError("No valid planes with accepted ROIs remain after filtering.")
    # Clip all signals the same length (to the shortest signal), and create vector with plane indices for each ROI.
    min_length = np.inf
    for i in ind_valid_planes:
        min_length = np.min((results[i]["dff"].shape[0], min_length)).astype(int)
    roi_planes = np.array([])
    for i in ind_valid_planes:
        results[i]["dff"] = results[i]["dff"][:min_length, :]
        results[i]["F_ncorrected"] = results[i]["F_ncorrected"][:min_length, :]
        if results[i]["zTrace"] is not None:
            results[i]["zTrace"] = results[i]["zTrace"][:min_length]
            results[i]["zCorr_stack"] = results[i]["zCorr_stack"][:, :min_length]
            results[i]["F_zcorrected"] = results[i]["F_zcorrected"][:min_length, :]
            results[i]["N_zcorrected"] = results[i]["N_zcorrected"][:min_length, :]

        roi_planes = np.append(roi_planes, np.ones((len(results[i]["cellId"]), 1)) * planes[i])

    if zstack_path is not None:
        if config["z_correction"]["plot"]:
            plot_path = os.path.join(output_directory, "2P_processed", "plots")
            os.makedirs(plot_path, exist_ok=True)
            # Load suite2p db dictionary (used parameters are the same across roi_planes).
            db = suite2p_compat.load_parameters(plane_directories[-1])[0]
            zstack.plot_ztraces(
                plot_path,
                results,
                ind_valid_planes,
                plane_ids=planes[ind_valid_planes],
                plane_rate=frame_rate / db["nplanes"],
            )

        np.save(
            os.path.join(output_directory, "2pPlanes.zCorrelations"),
            np.stack([results[i]["zCorr_stack"] for i in ind_valid_planes], axis=0),
        )
        np.save(
            os.path.join(output_directory, "2pPlanes.zTraces"),
            np.stack([results[i]["zTrace"] for i in ind_valid_planes], axis=0),
        )
        np.save(
            os.path.join(output_directory, "2pRois.zProfiles.npy"),
            np.vstack([results[i]["zProfiles"].T for i in ind_valid_planes]),
        )
        if config["z_correction"]["save_traces_zcorrected"]:
            np.save(
                os.path.join(output_directory, "2pCalcium.F_zcorrected.npy"),
                np.hstack([results[i]["F_zcorrected"] for i in ind_valid_planes]),
            )
            np.save(
                os.path.join(output_directory, "2pCalcium.N_zcorrected.npy"),
                np.hstack([results[i]["N_zcorrected"] for i in ind_valid_planes]),
            )

    np.save(
        os.path.join(output_directory, "2pCalcium.dff.npy"),
        np.hstack([results[i]["dff"] for i in ind_valid_planes]),
    )
    np.save(
        os.path.join(output_directory, "2pRois.xyz.npy"),
        np.vstack([results[i]["locs"] for i in ind_valid_planes]),
    )
    np.save(
        os.path.join(output_directory, "2pRois.ids.npy"),
        np.vstack([results[i]["cellId"] for i in ind_valid_planes]),
    )
    np.save(os.path.join(output_directory, "2pRois.2pPlanes.npy"), roi_planes)
    if config["delta_F"]["save_F_neuropilcorrected"]:
        np.save(
            os.path.join(output_directory, "2pCalcium.F_ncorrected.npy"),
            np.hstack([results[i]["F_ncorrected"] for i in ind_valid_planes]),
        )
        np.save(
            os.path.join(output_directory, "2pCalcium.N_regression.npy"),
            np.vstack([results[i]["N_regression"] for i in ind_valid_planes]),
        )
        np.save(
            os.path.join(output_directory, "2pCalcium.F_bin_values.npy"),
            np.vstack([results[i]["F_bin_values"] for i in ind_valid_planes]),
        )
        np.save(
            os.path.join(output_directory, "2pCalcium.N_bin_values.npy"),
            np.vstack([results[i]["N_bin_values"] for i in ind_valid_planes]),
        )


def process_all_datasets(config: dict, datasets: pd.DataFrame) -> None:
    """Run trace correction for all selected dataset rows.

    Missing Suite2p or TIFF directories are skipped. Optional DAQ piezo data
    determine plane trajectories and frame rate; optional z-stacks enable
    depth-motion correction.

    Parameters
    ----------
    config : dict
        Directory templates and trace-correction settings.
    datasets : pandas.DataFrame
        Dataset rows with ``Name``, ``Date``, and ``Process`` fields, plus
        optional depth, frame rate, ignored-plane, and z-stack fields.
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
        if not paths["suite2p"] or not paths["tiffs"]:
            continue

        general.setup_suite2p_logging(paths["suite2p"])

        logger.info(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")

        # Load parameters that were used to process data with Suite2p (from first plane in folder).
        db, settings = suite2p_compat.load_parameters_recording(paths["suite2p"])

        # Disregard imaging planes that user wants to ignore ( e.g., fly-back plane when using piezo as z-actuator).
        if "Ignore_planes" in datasets.columns and not pd.isna(datasets.iloc[i]["Ignore_planes"]):
            ignore_planes = np.array(datasets.loc[i]["Ignore_planes"]).astype(int)
            if ignore_planes.size == 0 & ("ignore_flyback" in db):
                ignore_planes = db["ignore_flyback"]
        else:
            ignore_planes = None

        # Returns the movement of the piezo (in microns along depth, relative to top-most position of piezo
        # trace) aligned to onset of each frame.
        if paths["piezo"] is None:
            piezo_per_plane = None
            if "Frame_rate" in datasets.columns and not np.isnan(datasets.loc[i]["Frame_rate"]):
                frame_rate = datasets.loc[i]["Frame_rate"]
            else:
                frame_rate = settings["fs"]
        else:
            logger.info("  Extracting piezo trace and frame times.")
            first_experiment_path = os.path.normpath(db["data_path"][0])
            first_experiment = first_experiment_path.split(os.sep)[-1]
            paths["piezo"] = os.path.join(paths["piezo"], first_experiment)
            frame_times = extract_data.get_frame_times(
                paths["piezo"],
                data_file=config["daq"]["data"],
                channel_name_file=config["daq"]["channel_names"],
                clock_channel=config["daq"]["clock_channel_name"],
                sampling_rate=config["daq"]["sampling_rate"],
            )
            frame_rate = np.nanmedian(np.diff(frame_times)) ** -1
            piezo_trace, piezo_time = extract_data.get_piezo_data(
                paths["piezo"],
                data_file=config["daq"]["data"],
                channel_name_file=config["daq"]["channel_names"],
                piezo_channel=config["daq"]["piezo_channel_name"],
                sampling_rate=config["daq"]["sampling_rate"],
            )
            piezo_per_plane = make_piezo_trace_per_plane(
                piezo_trace,
                piezo_time,
                frame_times,
                num_planes=db["nplanes"],
                volt_per_microns=config["daq"]["piezo_volt_per_micron"],
            )
            if config["daq"]["plot_piezo"]:
                plot_piezo_per_plane(
                    os.path.join(paths["output"], "2P_processed", "plots"),
                    piezo_per_plane,
                    sampling_rate=config["daq"]["sampling_rate"],
                )

        # If zstack path is not specified, no z-correction will be performed.
        if config["use_zstack"]:
            paths = zstack.append_paths(datasets.loc[i], config["directories"], paths)
            if paths["zstack"] is None:
                logger.info("  Path to z-stack does not exist or was not provided!")
        else:
            paths["zstack"] = None

        # Call main processing function.
        if "Depth" in datasets.columns and not np.isnan(datasets.loc[i]["Depth"]):
            depth = datasets.loc[i]["Depth"]
        else:
            depth = 0
        process_dataset(
            paths["suite2p"],
            config,
            tiff_path=paths["tiffs"],
            frame_rate=frame_rate,
            piezo=piezo_per_plane,
            zstack_path=paths["zstack"],
            output_directory=paths["output"],
            ignore_planes=ignore_planes,
            depth=depth,
        )


def main() -> None:
    """Run trace correction from a standalone configuration file.

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
