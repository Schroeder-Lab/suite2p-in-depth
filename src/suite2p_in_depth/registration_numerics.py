"""Pure numerical operations used by multi-plane z-registration."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
IntegerArray = npt.NDArray[np.integer[Any]]


class FrameSource(Protocol):
    """Random-access source of two-dimensional registered frames."""

    def __getitem__(self, index: int) -> npt.ArrayLike:
        """Read a registered frame at a zero-based time index.

        Parameters
        ----------
        index : int
            Frame index within the source.

        Returns
        -------
        array_like
            Two-dimensional image in ``(y, x)`` order.
        """
        ...


def determine_best_ref(correlations: npt.ArrayLike) -> int:
    """Select the reference plane with the best temporal coverage.

    Parameters
    ----------
    correlations
        Correlation coefficients with shape ``(time, references, planes)``.
        NaNs are permitted, but every time/reference pair must contain at
        least one finite plane correlation.

    Returns
    -------
    int
        Zero-based index of the selected reference plane. Ties in temporal
        coverage are resolved using mean maximum correlation.

    Raises
    ------
    ValueError
        If the input is empty, has the wrong shape, or lacks a finite plane
        correlation for any time/reference pair.
    """
    values = np.asarray(correlations)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("Correlations must have shape (time, references, planes).")
    if values.shape[1] != values.shape[2]:
        raise ValueError("The numbers of reference images and imaging planes must match.")
    if np.isnan(values).all(axis=2).any():
        raise ValueError("Every time/reference pair must contain at least one finite correlation.")
    nframes, _, nplanes = values.shape

    planes_per_reference = np.nanargmax(values, axis=2)
    shifts = planes_per_reference - np.arange(nplanes)
    minimum_shifts = np.nanmin(shifts, axis=1)
    maximum_shifts = np.nanmax(shifts, axis=1)

    reference_indices = np.tile(np.arange(nplanes), (nframes, 1))
    references_in_range = np.concatenate(
        [
            reference_indices + minimum_shifts[:, np.newaxis],
            reference_indices + maximum_shifts[:, np.newaxis],
        ],
        axis=0,
    )
    coverage = np.sum((references_in_range >= 0) & (references_in_range < nplanes), axis=0)
    mean_correlations = np.nanmean(np.nanmax(values, axis=2), axis=0)

    candidates = np.flatnonzero(coverage == np.max(coverage))
    if candidates.size == 1:
        return int(candidates[0])
    return int(candidates[np.nanargmax(mean_correlations[candidates])])


def equalize_frame_numbers(
    correlations: MutableSequence[npt.NDArray[Any]],
    z_positions: MutableSequence[npt.NDArray[Any]],
    databases: Sequence[dict[str, Any]],
) -> None:
    """Trim extra frames so all imaging planes agree per experiment.

    Parameters
    ----------
    correlations
        Mutable sequence of arrays whose first axis is time, one per plane.
    z_positions
        Mutable sequence of one-dimensional z-position arrays, one per plane.
    databases
        Suite2p database mappings. Each must contain ``frames_per_folder`` in
        experiment order.

    Notes
    -----
    The arrays in ``correlations`` and ``z_positions`` are replaced in place.
    Extra frames are removed at experiment boundaries, preserving the
    historical output convention.

    Raises
    ------
    ValueError
        If plane counts, frame metadata, or array lengths are inconsistent.
    """
    if not (len(correlations) == len(z_positions) == len(databases)):
        raise ValueError("Correlation, z-position, and database lists must have the same length.")
    if not databases:
        raise ValueError("At least one imaging plane is required.")

    frames_per_plane = np.stack(
        [np.asarray(database["frames_per_folder"]).ravel() for database in databases],
        axis=0,
    )
    if np.any(frames_per_plane < 0):
        raise ValueError("frames_per_folder values must be non-negative.")
    for plane, (plane_correlations, plane_positions) in enumerate(
        zip(correlations, z_positions, strict=True)
    ):
        expected = int(frames_per_plane[plane].sum())
        if len(plane_correlations) != expected or len(plane_positions) != expected:
            raise ValueError(
                f"Plane {plane} contains {len(plane_correlations)} correlations and "
                f"{len(plane_positions)} z positions; expected {expected}."
            )

    minimum_frames = np.min(frames_per_plane, axis=0)
    cumulative_frames: npt.NDArray[np.integer[Any]] = np.cumsum(minimum_frames)
    for plane in range(len(correlations)):
        for experiment in range(frames_per_plane.shape[1]):
            excess_count = frames_per_plane[plane, experiment] - minimum_frames[experiment]
            if excess_count > 0:
                excess = cumulative_frames[experiment] + np.arange(excess_count)
                correlations[plane] = np.delete(correlations[plane], excess, axis=0)
                z_positions[plane] = np.delete(z_positions[plane], excess)


def average_neighbor_frames(
    neighbor_indices: npt.ArrayLike,
    height: int,
    width: int,
    sources: Sequence[FrameSource],
    t: int,
    weights: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Return a weighted image from neighboring registered planes.

    Parameters
    ----------
    neighbor_indices
        One-dimensional plane indices. Indices outside ``sources`` are
        ignored to support edge planes.
    height, width
        Output dimensions in pixels, corresponding to Suite2p ``Ly`` and
        ``Lx`` respectively.
    sources
        Random-access registered-frame sources, one per imaging plane.
    t
        Time index to read from every valid source.
    weights
        Finite weights corresponding one-to-one with ``neighbor_indices``.

    Returns
    -------
    numpy.ndarray
        Weighted frame with shape ``(height, width)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If dimensions, weights, or a source frame shape are invalid.
    """
    indices = np.asarray(neighbor_indices)
    frame_weights = np.asarray(weights, dtype=float)
    if indices.ndim != 1 or frame_weights.shape != indices.shape:
        raise ValueError("neigh_idxs and weights must be one-dimensional arrays of equal length.")
    if height < 1 or width < 1:
        raise ValueError("Output image dimensions must be positive.")
    if not np.isfinite(frame_weights).all():
        raise ValueError("Neighbor weights must be finite.")

    result: npt.NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    for position, neighbor in enumerate(indices):
        if 0 <= neighbor < len(sources):
            frame = np.asarray(sources[int(neighbor)][t], dtype=np.float32)
            if frame.shape != (height, width):
                raise ValueError(
                    f"Neighbor {neighbor} frame has shape {frame.shape}; "
                    f"expected {(height, width)}."
                )
            result += frame_weights[position] * frame
    return result
