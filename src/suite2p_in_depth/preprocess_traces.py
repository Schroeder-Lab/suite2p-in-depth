"""Fluorescence trace preprocessing algorithms."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

logger = logging.getLogger("suite2p")


def correct_neuropil(
    fluorescence: npt.ArrayLike,
    neuropil: npt.ArrayLike,
    fs: float,
    baseline: npt.ArrayLike | None = None,
    num_neuropil_bins: int = 20,
    min_neuropil_percentile: float = 10,
    max_neuropil_percentile: float = 90,
    fluorescence_percentile: float = 5,
    baseline_percentile: float = 5,
    baseline_window: float = 60,
    frames_per_folder: Sequence[int] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Estimate and subtract neuropil contamination from ROI fluorescence.

    A regression estimates a coefficient for each ROI, yielding corrected
    signal ``C = S - rN`` from measured signal ``S`` and neuropil signal ``N``.

    Parameters
    ----------
    fluorescence : np.ndarray [time, rois]
        Calcium traces (measured signal) of ROIs.
    neuropil : np.ndarray [time, rois]
        Neuropil traces of ROIs.
    fs : float
        Imaging frames per second for one plane.
    baseline : array_like or None, optional
        Precomputed baseline traces; estimated when omitted.
    num_neuropil_bins : int, optional
        Number of bins used to partition the distribution of neuropil values.
        Each bin will be associated with a mean neuropil value and a mean
        signal value. The default is 20.
    min_neuropil_percentile : float, optional
        Minimum values of neuropil considered, expressed in percentile.
        The default is 10.
    max_neuropil_percentile : float, optional
        Maximum values of neuropil considered, expressed in percentile.
        The default is 90.
    fluorescence_percentile : float, optional
        Percentile of the measured signal that will be matched to neuropil.
        The default is 5.
    baseline_percentile : float, optional
        Percentile of the measured signal that will be taken as F0.
        The default is 5.
    baseline_window : float, optional
        The window size for the calculation of F0 for both signal and neuropil.
        The default is 60.
    frames_per_folder : sequence[int] or None, optional
        Experiment frame counts for separate baseline estimation.

    Returns
    -------
    signal : np.ndarray [t x nROIs]
        Neuropil corrected calcium traces.
    regression_pars : np.ndarray [rois, 2]
        Intercept and slope of linear fits of neuropil (N) to measured calcium
        traces (F)
    fluorescence_bin_values : np.ndarray [rois, bins]
        Low-percentile values for each calcium trace bin. These
        values were used for linear regression.
    neuropil_bin_values : np.ndarray [rois, bins]
        Values for each neuropil bin. These values were used for linear
        regression.

    This method is based on the MATLAB ``estimateNeuropil`` function written
    by Mario Dipoppa and Sylvia Schroeder.
    """
    fluorescence = np.asarray(fluorescence)
    neuropil = np.asarray(neuropil)
    if fluorescence.ndim != 2 or neuropil.shape != fluorescence.shape:
        raise ValueError("F and N must be two-dimensional arrays with identical shapes.")
    if fs <= 0:
        raise ValueError("fs must be greater than zero.")
    if num_neuropil_bins < 2:
        raise ValueError("num_neuropil_bins must be at least 2.")
    if not 0 <= min_neuropil_percentile < max_neuropil_percentile <= 100:
        raise ValueError("Neuropil percentiles must be ordered and between 0 and 100.")
    if not 0 <= fluorescence_percentile <= 100 or not 0 <= baseline_percentile <= 100:
        raise ValueError("Fluorescence percentiles must be between 0 and 100.")
    if baseline is not None and np.shape(baseline) != fluorescence.shape:
        raise ValueError("F0 must have the same shape as F.")

    num_timepoints, num_rois = fluorescence.shape
    neuropil_bin_values = np.full((num_rois, num_neuropil_bins), np.nan)
    fluorescence_bin_values = np.full((num_rois, num_neuropil_bins), np.nan)
    regression_pars = np.full((num_rois, 2), np.nan)
    signal = np.full((num_timepoints, num_rois), np.nan)

    # Correct for slow drift in ROI and neuropil traces separately.
    if baseline is None:
        baseline = get_f0(
            fluorescence,
            fs,
            f0_percentile=baseline_percentile,
            window_size=baseline_window,
            frames_per_folder=frames_per_folder,
        )
    neuropil_baseline = get_f0(
        neuropil,
        fs,
        f0_percentile=baseline_percentile,
        window_size=baseline_window,
        frames_per_folder=frames_per_folder,
    )
    centered_fluorescence = fluorescence - baseline
    centered_neuropil = neuropil - neuropil_baseline

    for roi_index in range(num_rois):
        roi_neuropil = centered_neuropil[:, roi_index]
        roi_fluorescence = centered_fluorescence[:, roi_index]
        if np.isnan(roi_neuropil).all() or np.isnan(roi_fluorescence).all():
            continue

        # Partition the central neuropil range into equal-width bins for a robust fit.
        neuropil_range = np.asarray(
            np.nanpercentile(
                roi_neuropil,
                [min_neuropil_percentile, max_neuropil_percentile],
            )
        )
        bin_size = (neuropil_range[1] - neuropil_range[0]) / num_neuropil_bins
        if not np.isfinite(bin_size) or bin_size <= 0:
            logger.warning(
                "Skipping neuropil correction for ROI %d: neuropil range is constant or invalid.",
                roi_index,
            )
            neuropil_bin_values[roi_index] = np.nan
            continue
        neuropil_bin_values[roi_index] = neuropil_range[0] + np.arange(num_neuropil_bins) * bin_size
        neuropil_bin_indices = np.floor((roi_neuropil - neuropil_range[0]) / bin_size)

        # Low fluorescence percentiles suppress spiking activity in the regression.
        for bin_index in range(num_neuropil_bins):
            values = roi_fluorescence[neuropil_bin_indices == bin_index]
            if values.size:
                fluorescence_bin_values[roi_index, bin_index] = np.nanpercentile(
                    values, fluorescence_percentile
                )
        valid_bins = np.flatnonzero(
            np.isfinite(fluorescence_bin_values[roi_index])
            & np.isfinite(neuropil_bin_values[roi_index])
        )
        if valid_bins.size < 2 or np.unique(neuropil_bin_values[roi_index, valid_bins]).size < 2:
            logger.warning(
                "Skipping neuropil correction for ROI %d: fewer than two valid regression bins.",
                roi_index,
            )
            fluorescence_bin_values[roi_index] = np.nan
            neuropil_bin_values[roi_index] = np.nan
            continue
        try:
            slope, intercept = np.polyfit(
                neuropil_bin_values[roi_index, valid_bins],
                fluorescence_bin_values[roi_index, valid_bins],
                1,
            )
        except (ValueError, np.linalg.LinAlgError):
            logger.warning("Skipping neuropil correction for ROI %d: regression failed.", roi_index)
            fluorescence_bin_values[roi_index] = np.nan
            neuropil_bin_values[roi_index] = np.nan
            continue
        slope = np.clip(slope, 0, 2)
        regression_pars[roi_index] = (intercept, slope)

        signal[:, roi_index] = (
            roi_fluorescence - (slope * roi_neuropil + intercept) + baseline[:, roi_index]
        )
    return signal, regression_pars, fluorescence_bin_values, neuropil_bin_values


def correct_zmotion(
    fluorescence: npt.ArrayLike,
    neuropil: npt.ArrayLike,
    fluorescence_profiles: npt.ArrayLike,
    ztrace: npt.ArrayLike,
    reference_depth: int,
    ignore_faults: bool = True,
    frames_per_experiment: Sequence[int] | None = None,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    list[npt.NDArray[np.intp]] | None,
]:
    """
    Correct ROI and neuropil traces for axial motion using z profiles.

    Fluorescence at each resolved depth is normalized to the reference depth.
    Unresolved z positions remain NaN in the corrected traces.

    Parameters
    ----------
    fluorescence, neuropil : array_like
        Single-plane traces in ``(frames, ROIs)`` order.
    fluorescence_profiles : array_like
        Axial ROI profiles in ``(slices, ROIs)`` order.
    ztrace : array_like
        Slice index per frame; NaN means unresolved depth.
    reference_depth : int
        Slice index used to normalize fluorescence profiles.
    ignore_faults : bool, optional
        Mask profile regions separated from the reference by troughs.
    frames_per_experiment : sequence[int] or None, optional
        Experiment lengths; smoothing does not cross their boundaries.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, list[numpy.ndarray] or None]
        Corrected ROI traces, corrected neuropil traces, and per-ROI profile
        peak indices when fault masking is enabled.

    Raises
    ------
    ValueError
        If trace shapes, depth indices, or experiment lengths are invalid.
    """
    fluorescence = np.asarray(fluorescence)
    neuropil = np.asarray(neuropil)
    fluorescence_profiles = np.asarray(fluorescence_profiles)
    ztrace = np.asarray(ztrace)
    if fluorescence.ndim != 2 or neuropil.shape != fluorescence.shape:
        raise ValueError("F and N must be two-dimensional arrays with identical shapes.")
    if fluorescence_profiles.ndim != 2 or fluorescence_profiles.shape[1] != fluorescence.shape[1]:
        raise ValueError("F_profiles must have shape (z, n_rois).")
    if ztrace.ndim != 1 or ztrace.size != fluorescence.shape[0]:
        raise ValueError("ztrace length must match the trace time axis.")
    valid_ztrace = np.isfinite(ztrace)
    finite_ztrace = ztrace[valid_ztrace]
    if not np.equal(finite_ztrace, np.floor(finite_ztrace)).all():
        raise ValueError("Finite ztrace values must be integer z-stack indices.")
    ztrace_indices = np.zeros(ztrace.shape, dtype=int)
    ztrace_indices[valid_ztrace] = finite_ztrace.astype(int)
    if finite_ztrace.size and (
        finite_ztrace.min() < 0 or finite_ztrace.max() >= fluorescence_profiles.shape[0]
    ):
        raise ValueError("ztrace contains indices outside F_profiles.")
    if not 0 <= reference_depth < fluorescence_profiles.shape[0]:
        raise ValueError("reference_depth is outside F_profiles.")
    if frames_per_experiment is not None and sum(frames_per_experiment) != fluorescence.shape[0]:
        raise ValueError("frames_per_experiment must sum to the trace length.")

    # Determine correction factor for F and N for each slice in Z stack.
    fluorescence_factors = fluorescence_profiles / fluorescence_profiles[reference_depth, :]

    # Disregard corrections for ROI traces when depth is outside the ROI's profile.
    if ignore_faults:
        fluorescence_factors, peaks = remove_zcorrected_faults(
            fluorescence_factors, reference_depth, threshold=0.5
        )
    else:
        peaks = None

    # Apply correction to F and N.
    fluorescence_correction = np.full(
        (ztrace.size, fluorescence_profiles.shape[1]), np.nan, dtype=float
    )
    fluorescence_correction[valid_ztrace, :] = fluorescence_factors[ztrace_indices[valid_ztrace], :]

    experiment_lengths = (
        [ztrace.size] if frames_per_experiment is None else list(frames_per_experiment)
    )
    experiment_start = 0
    for experiment_length in experiment_lengths:
        experiment_end = experiment_start + experiment_length
        experiment_valid = valid_ztrace[experiment_start:experiment_end]
        padded_valid = np.pad(experiment_valid, 1)
        run_edges = np.flatnonzero(np.diff(padded_valid.astype(np.int8)))
        for run_start, run_end in run_edges.reshape(-1, 2):
            start = experiment_start + run_start
            end = experiment_start + run_end
            fluorescence_correction[start:end, :] = sp.ndimage.gaussian_filter1d(
                fluorescence_correction[start:end, :], sigma=2, axis=0
            )
        experiment_start = experiment_end

    fluorescence_corrected = fluorescence / fluorescence_correction
    neuropil_corrected = neuropil / fluorescence_correction
    return fluorescence_corrected, neuropil_corrected, peaks


def get_f0(
    signal: npt.ArrayLike,
    frame_rate: float,
    f0_percentile: float = 8,
    window_size: float = 60,
    frames_per_folder: Sequence[int] | None = None,
) -> npt.NDArray[Any]:
    """
    Estimate baseline fluorescence for computing dF/F.

    Parameters
    ----------
    signal : array_like
        ROI traces in ``(frames, ROIs)`` order.
    frame_rate : float
        The frame rate (frames/second/plane).
    f0_percentile : float, optional
        Percentile used within each baseline window.
    window_size : float, optional
        Baseline window width in seconds.
    frames_per_folder : sequence[int] or None, optional
        Frame counts for experiment-wise baseline estimation.

    Returns
    -------
    numpy.ndarray
        Baseline traces with the same shape as ``signal``.

    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError("signal must have shape (time, n_rois).")
    if frame_rate <= 0 or window_size <= 0:
        raise ValueError("frame_rate and window_size must be greater than zero.")
    if not 0 <= f0_percentile <= 100:
        raise ValueError("F0_percentile must be between 0 and 100.")
    if frames_per_folder is not None and sum(frames_per_folder) != signal.shape[0]:
        raise ValueError("frames_per_folder must sum to the signal length.")
    baseline = np.zeros_like(signal)

    # Determine F0 per experiment (overall fluorescence may change abruptly between experiments).
    if frames_per_folder is None:
        # Estimate a rolling low percentile, then smooth it into the F0 baseline.
        window_size = int(round(frame_rate * window_size))
        fluorescence_frame = pd.DataFrame(signal)
        fluorescence_frame = fluorescence_frame.rolling(
            round(window_size / 2), min_periods=1, center=True
        ).quantile(f0_percentile * 0.01)
        fluorescence_quantile = fluorescence_frame.to_numpy()
        baseline = sp.ndimage.gaussian_filter1d(
            fluorescence_quantile, sigma=window_size, axis=0, mode="nearest"
        )
    else:
        last_frame = 0
        for folder_length in frames_per_folder:
            next_frame = last_frame + folder_length
            folder_baseline = get_f0(
                signal[last_frame:next_frame],
                frame_rate,
                f0_percentile,
                window_size,
                frames_per_folder=None,
            )
            baseline[last_frame:next_frame] = folder_baseline
            last_frame = next_frame
    return baseline


def get_delta_f_over_f(
    corrected_fluorescence: npt.ArrayLike, baseline: npt.ArrayLike
) -> npt.NDArray[Any]:
    """
    Calculate dF/F using a floor on the baseline denominator.

    Each frame uses its corresponding baseline value. The denominator is
    clamped to at least one to limit amplification at low fluorescence.

    Parameters
    ----------
    corrected_fluorescence : array_like
        Corrected ROI traces in ``(frames, ROIs)`` order.
    baseline : array_like
        Matching baseline fluorescence traces.

    Returns
    -------
    numpy.ndarray
        dF/F traces in ``(frames, ROIs)`` order.

    """
    corrected_fluorescence = np.asarray(corrected_fluorescence)
    baseline = np.asarray(baseline)
    if corrected_fluorescence.shape != baseline.shape:
        raise ValueError("Fc and F0 must have identical shapes.")
    return (corrected_fluorescence - baseline) / np.fmax(1, baseline)


def remove_zcorrected_faults(
    z_factors: npt.ArrayLike,
    reference_depth: int,
    threshold: float = 0.5,
) -> tuple[npt.NDArray[Any], list[npt.NDArray[np.intp]]]:
    """
    Mask axial correction factors outside each ROI's reference peak.

    Troughs around the reference depth separate likely unrelated profile
    peaks. Correction factors below ``threshold`` are also masked.

    Parameters
    ----------
    z_factors : array_like
        Per-slice correction factors in ``(slices, ROIs)`` order.
    reference_depth : int
        Slice index of the reference image.
    threshold : float, optional
        Minimum permitted correction factor.

    Returns
    -------
    tuple[numpy.ndarray, list[numpy.ndarray]]
        Masked correction factors and detected peak indices for each ROI.

    """
    zprofiles_corrected = np.copy(z_factors)
    peaks = []
    for roi in np.arange(z_factors.shape[1]):
        # Find peaks (presumably different neurons/components) in the z profile.
        peak_inds, properties = sp.signal.find_peaks(
            z_factors[:, roi], distance=7, width=2, wlen=31, rel_height=0.5
        )
        peaks.append(peak_inds)
        if len(peak_inds) == 0:
            # No troughs found, return the original profile.
            continue
        # Find peak closest and below the reference depth.
        peak_below = np.searchsorted(peak_inds, reference_depth, side="right")
        # Trough between enclosing peaks.
        if peak_below == peak_inds.size:  # Reference depth is below all peaks.
            trough_ind = np.argmin(z_factors[peak_inds[-1] :, roi]) + peak_inds[-1]
        elif peak_below == 0:  # Reference depth is between top and first peak.
            trough_ind = np.argmin(z_factors[0 : peak_inds[peak_below] + 1, roi])
        else:
            trough_ind = (
                np.argmin(z_factors[peak_inds[peak_below - 1] : peak_inds[peak_below] + 1, roi])
                + peak_inds[peak_below - 1]
            )
        if trough_ind < reference_depth:
            # If reference depth is below trough, then set all values above trough to NaN.
            zprofiles_corrected[:trough_ind, roi] = np.nan
            # Find the trough below the reference depth.
            if (
                peak_below == peak_inds.size
            ):  # Reference depth is between last trough and bottom of profile.
                trough_below = z_factors.shape[0]
            elif peak_below == peak_inds.size - 1:  # Reference depth is above last peak.
                trough_below = (
                    np.argmin(z_factors[peak_inds[peak_below] :, roi]) + peak_inds[peak_below]
                )
            else:
                trough_below = (
                    np.argmin(z_factors[peak_inds[peak_below] : peak_inds[peak_below + 1] + 1, roi])
                    + peak_inds[peak_below]
                )
            # Set all values below trough to NaN.
            zprofiles_corrected[trough_below + 1 :, roi] = np.nan
        else:
            # If reference depth is above trough, then set all values below trough to NaN.
            zprofiles_corrected[trough_ind + 1 :, roi] = np.nan
            # Find the trough above the reference depth.
            if peak_below == 0:  # Reference depth is between top and first trough.
                trough_above = 0
            elif (
                peak_below == 1
            ):  # Reference depth is between first and second peak, but above trough.
                trough_above = int(np.argmin(z_factors[: peak_inds[peak_below - 1] + 1, roi]))
            else:
                trough_above = (
                    np.argmin(
                        z_factors[peak_inds[peak_below - 2] : peak_inds[peak_below - 1] + 1, roi]
                    )
                    + peak_inds[peak_below - 2]
                )
            # Set all values above trough to NaN.
            zprofiles_corrected[: trough_above + 1, roi] = np.nan

    # Make sure that no correction factor is below threshold.
    zprofiles_corrected[zprofiles_corrected < threshold] = np.nan

    return zprofiles_corrected, peaks
