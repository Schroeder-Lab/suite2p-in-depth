import glob
import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_hex
from skimage.io import imread, imsave
from suite2p import parameters

from . import process_tiff, suite2p_extensions
from .data_types import PlaneTraceResults, TracePaths

ZStackParameters = tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple[float, float, float],
    float,
]

logger = logging.getLogger("suite2p")


def plane_color(plane_id: int) -> str:
    """Choose the active Matplotlib cycle color for a physical plane ID.

    Parameters
    ----------
    plane_id : int
        Nonnegative physical imaging plane number.

    Returns
    -------
    str
        Cycle color, wrapping after the final configured color.

    Raises
    ------
    ValueError
        If ``plane_id`` is negative or nonintegral.
    """
    plane_index = int(plane_id)
    if plane_index < 0 or plane_index != plane_id:
        raise ValueError("plane_id must be a non-negative integer.")
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        return f"C{plane_index % 10}"
    return to_hex(colors[plane_index % len(colors)])


def _invalidate_boundary_positions(
    ztrace: npt.ArrayLike, range_total: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Mark z estimates on the evaluated stack boundaries as unresolved.

    Parameters
    ----------
    ztrace : array_like
        Estimated z-stack slice index for each frame.
    range_total : array_like
        Evaluated slice indices in increasing order.

    Returns
    -------
    numpy.ndarray
        Float z trace with boundary positions replaced by NaN.
    """
    trace = np.asarray(ztrace, dtype=float)
    boundary = (trace == range_total[0]) | (trace == range_total[-1])
    trace[boundary] = np.nan
    return trace


def append_paths(dataset: pd.Series, directories: dict[str, str], paths: TracePaths) -> TracePaths:
    """Add a selected dataset's z-stack TIFF path to its path mapping.

    Parameters
    ----------
    dataset : pandas.Series
        Dataset with ``Name``, ``Date``, and ``Zstack_folder``.
    directories : dict
        Directory templates including ``zstack``.
    paths : dict
        Path mapping updated in place.

    Returns
    -------
    dict
        The input mapping with ``zstack`` set to its TIFF file, or ``None``
        if no z-stack is available.

    Raises
    ------
    ValueError
        If more than one z-stack TIFF is found.
    """
    zstack_directory = directories["zstack"].format(
        Name=dataset["Name"], Date=dataset["Date"], Zstack=str(dataset["Zstack_folder"])
    )

    if not os.path.exists(zstack_directory):
        paths["zstack"] = None

    else:
        zstack_files = sorted(
            path
            for path in glob.glob(os.path.join(zstack_directory, "*"))
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in (".tif", ".tiff")
        )
        if len(zstack_files) == 0:
            paths["zstack"] = None
            logger.warning(
                f"NOTE: Z-stack for {dataset['Name']} "
                f"{dataset['Date']} not found. No z-correction will be performed.\n\n"
            )
        elif len(zstack_files) > 1:
            raise ValueError(
                f"Expected one z-stack TIFF for {dataset['Name']} {dataset['Date']}, "
                f"but found {len(zstack_files)}: {zstack_files}"
            )
        else:
            paths["zstack"] = zstack_files[0]

    return paths


def compute_correlations_reference(
    data_path: str,
    zcorr_path: str,
    db: dict[str, Any],
    settings_reg: dict[str, Any],
    best_slice: int,
    plane_stack: np.ndarray,
    stack_step: int,
    sigma: float,
    device: torch.device,
) -> tuple[np.ndarray, npt.NDArray[np.float64]]:
    """Estimate each movie frame's axial position against a z-stack.

    Correlations are expanded outward from ``best_slice`` until no frame's
    best match is on the edge of the evaluated range. Existing correlations
    are loaded from ``zcorr_path`` when available.

    Parameters
    ----------
    data_path : str
        Registered movie binary to correlate.
    zcorr_path : str
        Cache path for the plane-by-frame correlation array.
    db : dict
        Suite2p movie dimensions, frame count, and experiment boundaries.
    settings_reg : dict
        Correlation and batching settings.
    best_slice : int
        Initial z-stack slice matching the reference image.
    plane_stack : numpy.ndarray
        Registered z-stack in ``(z, y, x)`` order.
    stack_step : int
        Number of neighboring slices added at each search expansion.
    sigma : float
        Gaussian smoothing scale across stack slices.
    device : torch.device
        Device used for Suite2p correlations.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Correlations in ``(z, frames)`` order and frame-wise slice indices;
        estimates on the evaluated boundaries are NaN.

    Raises
    ------
    ValueError
        If stack, settings, or cached correlation shape are invalid.
    """
    plane_stack = np.asarray(plane_stack)
    if plane_stack.ndim != 3 or plane_stack.shape[0] == 0:
        raise ValueError("plane_stack must have shape (z, y, x) with at least one slice.")
    if not 0 <= best_slice < plane_stack.shape[0]:
        raise ValueError("best_slice is outside the z-stack.")
    if int(db.get("nframes", 0)) < 1:
        raise ValueError("db['nframes'] must be at least 1.")
    if stack_step < 1:
        raise ValueError("stack_step must be at least 1.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if not os.path.exists(zcorr_path):
        logger.info("  Determine z-trace")
        range_stack = np.arange(
            max(best_slice - stack_step, 0), min(best_slice + stack_step + 1, plane_stack.shape[0])
        )
        range_total = range_stack.copy()
        zcorr = np.ones((plane_stack.shape[0], db["nframes"])) * np.nan
        while True:
            stack_in_range = plane_stack[range_stack, :, :]

            # For each frame, determine its correlation with each plane in the considered range of the z-stack.
            zcorr_in_range = suite2p_extensions.compute_zpos(
                stack_in_range, db, settings_reg, reg_file=data_path, device=device
            )
            if zcorr_in_range.shape != (len(range_stack), db["nframes"]):
                raise ValueError(
                    "Suite2p returned z correlations with shape "
                    f"{zcorr_in_range.shape}; expected {(len(range_stack), db['nframes'])}."
                )
            zcorr[range_stack, :] = zcorr_in_range

            # Smooth the correlations across planes in the z-stack, then find the best-matching plane for each frame;
            # Here, ztrace indexes into the total range of the zstack considered so far.
            ztrace = np.nanargmax(
                sp.ndimage.gaussian_filter1d(zcorr[range_total, :], sigma, axis=0, mode="nearest"),
                axis=0,
            ).astype(int)
            # Now, ztrace indexes into the complete z-stack.
            ztrace = range_total[ztrace]

            # Check whether any best-matching planes of the z-stack are at the edge of the currently considered range.
            # If so, increase the considered range and repeat whole loop.
            # Do not consider first frames of experiments, because piezo trajectory is always different to all other
            # frames.
            range_stack = np.array([], dtype=int)
            first_frames = np.cumsum(db["frames_per_folder"])[:-1]
            ztrace_excluded = np.delete(ztrace, np.insert(first_frames, 0, 0))
            if ztrace_excluded.size == 0:
                ztrace_excluded = ztrace
            if np.min(ztrace_excluded) == np.min(range_total):
                range_stack = np.arange(
                    max(0, np.min(range_total) - stack_step), np.min(range_total)
                )

            if np.max(ztrace_excluded) == np.max(range_total):
                range_stack = np.concatenate(
                    (
                        range_stack,
                        np.arange(
                            np.max(range_total) + 1,
                            min(plane_stack.shape[0], np.max(range_total) + stack_step + 1),
                        ),
                    )
                )

            if range_stack.size > 0:
                range_total = np.union1d(range_total, range_stack)
            else:
                break
        np.save(zcorr_path, zcorr)

    else:
        logger.info("  Load z-trace")
        zcorr = np.load(zcorr_path)
        expected_shape = (plane_stack.shape[0], db["nframes"])
        if zcorr.shape != expected_shape:
            raise ValueError(
                f"Cached z correlations have shape {zcorr.shape}; expected {expected_shape}."
            )
        range_total = np.where(~np.isnan(zcorr).any(axis=1))[0]
        if range_total.size == 0:
            raise ValueError("Cached z correlations contain no complete z-stack slices.")
        ztrace = np.nanargmax(
            sp.ndimage.gaussian_filter1d(zcorr[range_total, :], sigma, axis=0, mode="nearest"),
            axis=0,
        ).astype(int)
        ztrace = range_total[ztrace]

    return zcorr, _invalidate_boundary_positions(ztrace, range_total)


def prep_zstack_parameters(
    plane_directory: str,
    config: dict[str, Any],
    db: dict[str, Any],
    settings: dict[str, Any],
    fov_size: float,
) -> ZStackParameters:
    """Prepare movie paths, registration settings, and z-stack scales.

    Parameters
    ----------
    plane_directory : str
        Suite2p plane containing registered binaries.
    config : dict
        Z-correction and z-stack registration settings.
    db : dict
        Plane database, including dimensions and registered movie path.
    settings : dict
        Base Suite2p settings and registration channel selection.
    fov_size : float
        Field-of-view size in micrometres when pixel conversion is enabled.

    Returns
    -------
    tuple
        Alignment-channel movie path, correlation database, four registration
        settings mappings, Gaussian sigma in stack/pixel units, and z spacing.
    """
    file_name = "data_chan2.bin" if settings["registration"]["align_by_chan2"] else "data.bin"
    data_path = os.path.join(plane_directory, file_name)

    db_zcorr = db.copy()
    db_zcorr["reg_file"] = os.path.join(plane_directory, "data.bin")

    settings_reg_within = settings["registration"].copy()
    parameters.set_settings(settings_reg_within, config["suite2p_stack_within"].copy())
    settings_reg_across = settings["registration"].copy()
    parameters.set_settings(settings_reg_across, config["suite2p_stack_across"].copy())
    settings_reg_ref = settings["registration"].copy()
    parameters.set_settings(settings_reg_ref, config["suite2p_stack_to_ref"].copy())
    settings_reg_frames = settings["registration"].copy()
    parameters.set_settings(settings_reg_frames, config["suite2p_stack_to_frames"].copy())

    # Initiate parameters for Z stack alignment and reslicing. Specifies path to the registered movie (.bin) and
    # non-rigid registration method.
    if config["pixels_to_microns"]:
        pix_per_micron = db_zcorr["Ly"] / fov_size
    else:
        pix_per_micron = 1

    sig = config["z_correction"]["sigma_stack"]
    sigma = (sig[0], sig[1] * pix_per_micron, sig[2] * pix_per_micron)
    spacing = config["z_correction"]["spacing"]
    return (
        data_path,
        db_zcorr,
        settings_reg_within,
        settings_reg_across,
        settings_reg_ref,
        settings_reg_frames,
        sigma,
        spacing,
    )


def extract_reference_slice(
    zstack_raw_path: str,
    processed_path: str,
    plane: int,
    db: dict[str, Any],
    settings_reg_within: dict[str, Any],
    settings_reg_across: dict[str, Any],
    settings_reg_ref: dict[str, Any],
    ref_frames: float,
    piezo: np.ndarray | None,
    sigma: tuple[float, float, float],
    spacing: float,
    reg_outputs: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray | np.integer[Any], np.ndarray]:
    """Load or create a registered z-stack and its best reference slice.

    Parameters
    ----------
    zstack_raw_path : str
        Original z-stack TIFF.
    processed_path : str
        Directory for cached registered stacks and best-slice index.
    plane : int
        Physical imaging plane ID.
    db : dict
        Plane channel and image metadata.
    settings_reg_within, settings_reg_across, settings_reg_ref : dict
        Registration settings within stack planes, across stack planes, and
        against the movie reference.
    ref_frames : numpy.ndarray
        Movie frames used to compare the z-stack with the reference.
    piezo : numpy.ndarray or None
        Sampled piezo positions by imaging plane.
    sigma : tuple[float, float, float]
        Gaussian smoothing scale for the z-stack.
    spacing : float
        Axial spacing of resliced stack planes.
    reg_outputs : dict
        Registration outputs containing ``refImg``.
    device : torch.device
        Device for image registration.

    Returns
    -------
    tuple[int, numpy.ndarray]
        Best-matching stack slice index and registered stack in ``(z, y, x)``
        order.
    """
    channel = 2 if settings_reg_within["align_by_chan2"] else 1
    channel_functional = db["functional_chan"]

    plane_stack_path = os.path.join(processed_path, f"stack_plane{plane}_chan{channel}.tif")

    # If Z stack not saved yet, register and reslice raw Z Stack and determine slice matching reference image best.
    if not (os.path.exists(plane_stack_path)):
        logger.info("  Register and reslice z-stack")
        if piezo is None:
            piezo_plane = None
        else:
            if piezo.ndim != 2 or plane >= piezo.shape[1]:
                raise ValueError(
                    f"Piezo array shape {piezo.shape} does not contain imaging plane {plane}."
                )
            piezo_plane = np.vstack(
                (
                    piezo[:, plane : plane + 1],
                    np.reshape(piezo[0:1, (plane + 1) % piezo.shape[1]], (1, 1)),
                )
            )

        plane_stack, best_slice, plane_stack_functional = process_tiff.register_zstack(
            zstack_raw_path,
            settings_reg_within,
            settings_reg_across,
            settings_reg_ref,
            spacing=spacing,
            piezo=piezo_plane,
            target_image=reg_outputs["refImg"],
            channel_align=channel,
            channel_functional=channel_functional,
            n_channels=db["nchannels"],
            sigma=sigma,
            ref_frames=ref_frames,
            device=device,
        )
        imsave(plane_stack_path, plane_stack)
        np.save(os.path.join(processed_path, f"bestSlice_plane{plane}.npy"), best_slice)
        if channel_functional != channel:
            plane_stack_functional_path = os.path.join(
                processed_path, f"stack_plane{plane}_chan{channel_functional}.tif"
            )
            imsave(plane_stack_functional_path, plane_stack_functional)
    else:
        logger.info("  Load registered and resliced z-stack")
        plane_stack = imread(plane_stack_path)
        best_slice = np.load(os.path.join(processed_path, f"bestSlice_plane{plane}.npy"))
    return best_slice, plane_stack


def plot_reference_slice(plots_path: str, ref_img: np.ndarray, reference_slice: np.ndarray) -> None:
    """Save a comparison of a movie reference and matching z-stack slice.

    Parameters
    ----------
    plots_path : str
        Directory receiving ``Reference_and_matchingSlice.png``.
    ref_img : numpy.ndarray
        Two-dimensional movie reference image.
    reference_slice : numpy.ndarray
        Two-dimensional best-matching stack slice.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    # (1) Reference image
    ref_img = np.clip(ref_img, np.percentile(ref_img, 1), np.percentile(ref_img, 99))
    axes[0].imshow(ref_img, cmap="gray")
    axes[0].set_title("Reference Image")
    axes[0].axis("off")

    # (2) Best matching slice in stack
    reference_slice = np.clip(
        reference_slice, np.percentile(reference_slice, 1), np.percentile(reference_slice, 99)
    )
    axes[1].imshow(reference_slice, cmap="gray")
    axes[1].set_title("Best Matching Slice in Stack")
    axes[1].axis("off")

    # (3) Comparison between reference image and best matching slice in stack
    reference_normalized = (ref_img - np.min(ref_img)) / np.ptp(ref_img)
    zstack_norm = (reference_slice - np.min(reference_slice)) / np.ptp(reference_slice)
    # Stack into RGB: Red = zstack_plane, Green = reference_image, Blue = 0
    rgb = np.zeros((*reference_slice.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.round(zstack_norm * 255).astype(np.uint8)  # Red
    rgb[..., 1] = np.round(reference_normalized * 255).astype(np.uint8)  # Green
    axes[2].imshow(rgb)
    axes[2].set_title("Reference Image (green) vs Best Matching Slice in Stack (red)")
    axes[2].axis("off")

    fig.savefig(os.path.join(plots_path, "Reference_and_matchingSlice.png"), dpi=300)
    plt.close(fig)


def plot_zprofile(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    ztrace: np.ndarray,
    fluorescence_profiles: np.ndarray | None,
    peaks: npt.ArrayLike,
    roi: dict[str, Any],
    plane_stack: np.ndarray,
    reference_depth: int,
    ind_best_slice: int | np.integer[Any],
    plane_rate: float,
    n_frames: npt.ArrayLike,
    prctile: float,
    colors_paired: Colormap,
    profile_axis: Axes,
    stack_axis: Axes | None,
    ztrace_axis: Axes,
    zoom_in: bool = False,
) -> None:
    """Draw one ROI's axial fluorescence profile and z trace on given axes.

    Parameters
    ----------
    fluorescence, neuropil : numpy.ndarray
        ROI and neuropil fluorescence traces; ``neuropil`` is retained for the
        caller interface and is not plotted here.
    ztrace : numpy.ndarray
        Estimated stack slice index per movie frame.
    fluorescence_profiles : numpy.ndarray or None
        ROI fluorescence measured across stack slices.
    peaks : array_like
        Slice indices of profile peaks.
    roi : dict
        ROI metadata with center pixel in ``med``.
    plane_stack : numpy.ndarray
        Registered stack in ``(z, y, x)`` order.
    reference_depth, ind_best_slice : float
        Reference depth and best-matching slice used as plot markers.
    plane_rate : float
        Frames per second for this imaging plane.
    n_frames : array_like
        Experiment frame boundaries marked on the time axis.
    prctile : float
        Recording fluorescence percentile per occupied depth.
    colors_paired : callable
        Color-map function for recorded and stack fluorescence.
    profile_axis, stack_axis, ztrace_axis : matplotlib.axes.Axes or None
        Existing axes; ``stack_axis`` may be omitted.
    zoom_in : bool, optional
        Restrict the profile to depths sampled by the z trace.
    """
    # Compare the stack profile with low fluorescence values recorded at each depth.
    if fluorescence_profiles is not None:
        plt.sca(profile_axis)
        depth_fluorescence_percentiles = np.full((plane_stack.shape[0], 1), np.nan)
        valid_ztrace = np.asarray(ztrace)[np.isfinite(ztrace)].astype(int)
        for p in np.unique(valid_ztrace):
            depth_fluorescence_percentiles[p] = np.percentile(
                fluorescence[ztrace == p], prctile, axis=0
            )

        if zoom_in:
            valid = np.where(np.isfinite(depth_fluorescence_percentiles))[0]
        else:
            valid = np.arange(depth_fluorescence_percentiles.shape[0])
            plt.plot(
                fluorescence_profiles[peaks],
                peaks,
                "o",
                color=colors_paired(1),
                markersize=3,
            )

        plt.plot(
            depth_fluorescence_percentiles[valid],
            valid,
            color=colors_paired(0),
            linewidth=3,
            label="F(recording)",
        )
        plt.plot(
            fluorescence_profiles[valid],
            valid,
            color=colors_paired(1),
            linewidth=1,
            label="F(stack)",
        )
        plt.axhline(reference_depth, color="k", linewidth=1, label="Ref. depth")
        plt.axhline(
            ind_best_slice, color="red", linestyle="--", linewidth=1, label="Ref. im. match"
        )

        handles, labels = profile_axis.get_legend_handles_labels()
        profile_axis.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            borderaxespad=0,
            frameon=False,
            fontsize=7.5,
            ncols=2,
            labelspacing=0.25,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=0.7,
        )
        plt.ylim(0, fluorescence_profiles.shape[0])
        plt.gca().invert_yaxis()
        plt.xlabel("Fluorescence")
        plt.ylabel("Depth (µm)")

    # Z-stack around ROI center, omitted from the denser zoomed layout.
    if stack_axis is not None:
        plt.sca(stack_axis)
        x_ind = np.clip(roi["med"][1] + np.arange(-10, 11), 0, plane_stack.shape[2] - 1)
        plt.imshow(plane_stack[:, roi["med"][0], x_ind], cmap="gray", aspect="auto")
        plt.axvline(10, color="red", linewidth=1)  # Center index is always at position 10
        plt.yticks([])
        plt.xlabel("Pixels")

    # z trace of plane
    plt.sca(ztrace_axis)
    plt.plot(np.arange(len(ztrace)) / plane_rate, ztrace, color=(0.5, 0.5, 0.5))
    plt.gca().invert_yaxis()
    plt.axhline(reference_depth, color="k", linewidth=3)
    [plt.axvline(x / plane_rate, color="k", linewidth=1) for x in n_frames]
    plt.xlim(0, ztrace.shape[0] / plane_rate)
    plt.gca().set_xticklabels([])
    plt.tick_params(axis="y", right=True, labelleft=False, labelright=True)
    plt.title("Z-trace")


def plot_ztraces(
    plot_path: str,
    results: Sequence[PlaneTraceResults | None],
    ind_valid_planes: Sequence[int],
    plane_ids: Sequence[int] | np.ndarray,
    plane_rate: float,
) -> None:
    """Save the z traces of valid imaging planes on a shared time axis.

    Parameters
    ----------
    plot_path : str
        Directory receiving ``zTraces_allPlanes.png``.
    results : sequence[dict]
        Per-plane results with ``zTrace`` arrays.
    ind_valid_planes : sequence[int]
        Indices into ``results`` to plot.
    plane_ids : sequence[int]
        Physical plane IDs corresponding to those indices.
    plane_rate : float
        Frames per second for the plotted planes.

    Raises
    ------
    ValueError
        If no planes are selected or plane IDs do not align with indices.
    """
    if not ind_valid_planes:
        raise ValueError("At least one valid plane is required for z-trace plotting.")
    if len(ind_valid_planes) != len(plane_ids):
        raise ValueError("plane_ids must correspond one-to-one with ind_valid_planes.")

    fig, axis = plt.subplots(figsize=(15, 6))
    first_trace_length = 0
    for result_index, plane_id in zip(ind_valid_planes, plane_ids, strict=True):
        result = results[result_index]
        if result is None or result["zTrace"] is None:
            raise ValueError(f"Plane {plane_id} has no z-trace to plot.")
        ztrace = result["zTrace"]
        if first_trace_length == 0:
            first_trace_length = len(ztrace)
        axis.plot(
            np.arange(len(ztrace)) / plane_rate,
            ztrace,
            color=plane_color(plane_id),
            label=f"Plane {int(plane_id)}",
        )
    axis.set_xlim(0, first_trace_length / plane_rate)
    axis.invert_yaxis()
    axis.legend()
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Depth (µm)")
    axis.set_title("Z-traces of all roi_planes")
    fig.savefig(os.path.join(plot_path, "zTraces_allPlanes.png"), dpi=300)
    plt.close(fig)
