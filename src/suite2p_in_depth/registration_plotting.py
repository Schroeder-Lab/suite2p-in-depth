"""Diagnostic plotting for multi-plane z-registration outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

CORRELATION_THRESHOLD_STD = 4


def make_plots(
    database: dict[str, Any],
    registration_outputs: dict[str, Any],
    output_directory: str | Path,
) -> None:
    """Write the three standard z-registration quality-control panels.

    Parameters
    ----------
    database
        Suite2p database mapping for the combined ``plane_z`` output.
    registration_outputs
        Registration arrays and reference images. Correlation arrays use
        ``(time, references, planes)`` order and image arrays use ``(y, x)``.
    output_directory
        Directory that receives three PNG files. It is created if necessary.

    Notes
    -----
    This is the only function in this module that writes files. Figures are
    closed after saving and the global Matplotlib style is not modified.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    plane_ids = np.setdiff1d(
        np.arange(database["nplanes"]), database["ignore_flyback_singleplanes"]
    )
    with plt.rc_context({"font.size": 6}):
        plot_reference_images(registration_outputs, plane_ids)
        plt.savefig(destination / "01_reference_images.png", dpi=300)
        plt.close()

        plot_final_ref_and_mean_img(database, registration_outputs)
        plt.savefig(destination / "02_mean_image_zregistration.png", dpi=300)
        plt.close()

        plot_zpos_across_time(database, registration_outputs, plane_ids)
        plt.savefig(destination / "03_z_positions.png", dpi=300)
        plt.close()


def plot_zpos_across_time(
    database: dict[str, Any],
    registration_outputs: dict[str, Any],
    plane_ids: npt.NDArray[np.integer[Any]],
) -> None:
    """Build z-position and correlation panels without writing them.

    Correlations are expected in ``(time, references, planes)`` order. The
    x-axis is frame number and the z-position axis contains imaging-plane IDs.
    NaNs are allowed where registration correlations are unavailable.

    Parameters
    ----------
    database : dict[str, Any]
        Suite2p metadata including frames per experiment.
    registration_outputs : dict[str, Any]
        Correlations, bad-frame flags, and selected reference plane.
    plane_ids : numpy.ndarray
        Physical plane IDs corresponding to reference image order.
    """
    best_reference = np.flatnonzero(plane_ids == registration_outputs["reference_plane"])[0]
    _, axes = plt.subplots(2, 1, figsize=(9, 5))
    planes_per_reference = np.nanargmax(registration_outputs["corrs_time_refs_planes"], axis=2)
    experiment_starts: npt.NDArray[np.integer[Any]] = (
        np.cumsum(database["frames_per_folder"])[:-1] - 1
    )
    colormap = matplotlib.colormaps.get_cmap("tab10").resampled(planes_per_reference.shape[1])
    for plane in range(planes_per_reference.shape[1]):
        offset = 0 if plane == best_reference else (plane - best_reference) * plane_ids.size
        color = (
            "red"
            if plane_ids[plane] == registration_outputs["reference_plane"]
            else mcolors.to_hex(colormap(plane))
        )
        axes[0].axhline(
            y=plane_ids[plane] + offset,
            color=color,
            linewidth=0.5,
            alpha=0.6,
            zorder=0,
        )
        axes[0].scatter(
            range(planes_per_reference.shape[0]),
            plane_ids[planes_per_reference[:, plane]] + offset,
            color=color,
            label=f"Plane {plane_ids[plane]}",
            s=1,
        )
    for experiment_start in experiment_starts:
        axes[0].axvline(x=experiment_start, color="black", linewidth=0.5)
    axes[0].set_title("Z-Positions for all reference images")
    axes[0].set_ylabel("Best imaging plane")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_xlim(0, planes_per_reference.shape[0] - 1)
    axes[0].invert_yaxis()

    correlations = registration_outputs["corrs_time_refs_planes"][:, best_reference, :]
    maximum_correlations = np.nanmax(correlations, axis=1)
    threshold = np.nanmean(maximum_correlations) - CORRELATION_THRESHOLD_STD * np.nanstd(
        maximum_correlations
    )

    start = None
    for frame, is_bad in enumerate(registration_outputs["badframes"]):
        if is_bad and start is None:
            start = frame
        elif not is_bad and start is not None:
            axes[1].axvspan(start - 0.5, frame - 0.5, color="gray", alpha=0.5)
            start = None
    if start is not None:
        axes[1].axvspan(
            start - 0.5,
            len(registration_outputs["badframes"]) - 0.5,
            color="gray",
            alpha=0.5,
        )
    gray_patch = mpatches.Patch(color="gray", alpha=0.5, label="Bad frames")
    axes[1].plot(maximum_correlations, color="red", linewidth=0.5)
    threshold_line = axes[1].axhline(
        y=threshold,
        color="black",
        linestyle="-",
        linewidth=1,
        label="Corr. thresh.",
    )
    for experiment_start in experiment_starts:
        axes[1].axvline(x=experiment_start, color="black", linewidth=0.5)
    axes[1].set_title("Correlation: best reference with frames")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Correlation coefficient")
    axes[1].legend(handles=[gray_patch, threshold_line], loc="upper right", bbox_to_anchor=(1.2, 1))
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_xlim(0, planes_per_reference.shape[0] - 1)
    plt.tight_layout()


def plot_final_ref_and_mean_img(
    database: dict[str, Any], registration_outputs: dict[str, Any]
) -> None:
    """Build the reference-versus-mean-image panel without writing it.

    Images use ``(y, x)`` order and retain their native pixel units. Both
    channels are shown when registration was aligned using channel 2.

    Parameters
    ----------
    database : dict[str, Any]
        Suite2p metadata, including the settings file path.
    registration_outputs : dict[str, Any]
        Final reference and registered mean images.
    """
    settings = np.load(database["settings_path"], allow_pickle=True).item()
    align_by_channel_two = settings["registration"]["align_by_chan2"]
    figure_size = (11, 4) if align_by_channel_two else (9, 4)
    column_count = 4 if align_by_channel_two else 3
    mean_image = registration_outputs["meanImg_chan2" if align_by_channel_two else "meanImg"]

    _, axes = plt.subplots(1, column_count, figsize=figure_size)
    axes[0].imshow(registration_outputs["refImg"], cmap="gray")
    axes[0].set_title("Final Reference Image\n(weighted neighbors)")
    axes[0].axis("off")

    mean_image = np.clip(mean_image, np.percentile(mean_image, 1), np.percentile(mean_image, 99))
    axes[1].imshow(mean_image, cmap="gray")
    axes[1].set_title("Mean Image\nafter Z-Registration")
    axes[1].axis("off")

    rgb = np.zeros((*mean_image.shape, 3))
    reference = registration_outputs["refImg"]
    reference_normalized = (reference - np.min(reference)) / np.ptp(reference)
    mean_normalized = (mean_image - np.min(mean_image)) / np.ptp(mean_image)
    rgb[..., 0] = reference_normalized
    rgb[..., 1] = mean_normalized
    axes[2].imshow(rgb)
    axes[2].set_title("Mean Image (green)\nvs Reference (red)")
    axes[2].axis("off")

    if align_by_channel_two:
        channel_one = registration_outputs["meanImg"]
        channel_one = np.clip(
            channel_one, np.percentile(channel_one, 1), np.percentile(channel_one, 99)
        )
        axes[3].imshow(channel_one, cmap="gray")
        axes[3].set_title("Mean Image\nChannel 1")
        axes[3].axis("off")


def plot_reference_images(
    registration_outputs: dict[str, Any],
    plane_ids: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.integer[Any]]:
    """Build the per-plane reference-image panel without writing it.

    Parameters
    ----------
    registration_outputs
        Mapping containing ``refImg_singleplanes`` images in ``(plane, y, x)``
        order and the selected ``reference_plane`` ID.
    plane_ids
        Physical plane IDs corresponding to the image order.

    Returns
    -------
    numpy.ndarray
        The unchanged ``plane_ids`` array, retained for compatibility.
    """
    reference_images = registration_outputs["refImg_singleplanes"]
    row_count = int(np.floor(np.sqrt(len(reference_images))))
    column_count = int(np.ceil(len(reference_images) / row_count))
    figure, axes = plt.subplots(row_count, column_count, figsize=(9, 6))
    flat_axes = np.array(axes).ravel()
    for index, current_axis in enumerate(flat_axes[: len(reference_images)]):
        current_axis.imshow(reference_images[index], cmap="gray")
        if plane_ids[index] == registration_outputs["reference_plane"]:
            current_axis.set_title(f"Plane {plane_ids[index]} (best reference)", color="red")
        else:
            current_axis.set_title(f"Plane {plane_ids[index]}")
        current_axis.axis("off")
    for empty_axis in flat_axes[len(reference_images) :]:
        figure.delaxes(empty_axis)
    plt.suptitle("Reference Images per Plane")
    return plane_ids
