"""Generate trace-correction and z-profile comparison panels for paper Figure 3."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from suite2p_in_depth import preprocess_traces, suite2p_compat

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

NEURONS_ROOT = DATA_ROOT / "neurons"
DATASETS_CSV = NEURONS_ROOT / "neurons.csv"
OUTPUT_DIR = PLOTS_ROOT / "Fig03"
CURATED_RECORDINGS = (
    ("Quille", "2023-07-24"),
    ("Uma", "2023-12-06"),
    ("Dublin", "2024-05-01"),
    ("Ely", "2024-07-01"),
    ("SS126", "2024-07-30"),
)

TRACE_SUBJECT = "Dublin"
TRACE_DATE = "2024-05-01"
TRACE_PLANE_ID = 2
TRACE_ROI_IDS = (1, 37, 12, 2, 7, 10, 40, 15, 4, 41, 5, 29)
TRACE_DURATION_MIN = 50.0
ROI_CHUNK_SIZE = 64
PERCENTILE_HISTOGRAM_MAX_BINS = 80
TRACE_REMOVAL_LOG_BINS_PER_DECADE = 10
MIN_PROFILE_SAMPLES_PER_DEPTH = 100
F_PROFILE_PERCENTILE = 50.0
EXAMPLE_PROFILE_DEPTH_RANGE_UM = (40.0, 100.0)
WHEEL_SMOOTHING_SIGMA_S = 1.0
RUNNING_THRESHOLD_CM_S = 1.0
RUNNING_STATIONARY_S = 2.0
RUNNING_MINIMUM_DURATION_S = 2.0
RUNNING_BASELINE_S = 2.0
RUNNING_WINDOW_BEFORE_S = 5.0
RUNNING_WINDOW_AFTER_S = 10.0
FIGURE_STYLE = {
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
}
RAW_COLOR = "#6b7280"
CORRECTED_COLOR = "#c23b22"
HISTOGRAM_COLOR = "#4c78a8"
LOWER_PERCENTILE_COLOR = "#3b6fb6"
UPPER_PERCENTILE_COLOR = "#c23b22"
ZSTACK_PROFILE_COLOR = "#000000"
MEDIAN_F_PROFILE_COLOR = "#c23b22"


@dataclass(frozen=True)
class Dataset:
    """Identity of one selected recording."""

    subject: str
    date: str

    @property
    def slug(self) -> str:
        return f"{self.subject}_{self.date}"


@dataclass(frozen=True)
class PlaneMetadata:
    """Saved production settings needed to reproduce trace preprocessing."""

    directory: Path
    badframes: np.ndarray
    frames_per_folder: np.ndarray
    plane_rate_hz: float
    num_rois: int
    num_timepoints: int


@dataclass(frozen=True)
class SelectedRoi:
    """One corrected-array column and its original Suite2p identity."""

    output_index: int
    plane_id: int
    roi_id: int

    @property
    def label(self) -> str:
        return f"Plane {self.plane_id}, ROI {self.roi_id}"


@dataclass(frozen=True)
class FigureData:
    """Fully prepared plotting inputs for one dataset."""

    dataset: Dataset
    neural_time_min: np.ndarray
    wheel_time_min: np.ndarray
    experiment_start_min: np.ndarray
    wheel_velocity_cm_s: np.ndarray
    z_movement_um: np.ndarray
    z_plane_ids: np.ndarray
    raw_fluorescence: np.ndarray
    corrected_fluorescence: np.ndarray
    fluorescence_log_ratio_q05: np.ndarray
    fluorescence_log_ratio_q95: np.ndarray
    z_profile_depth_um: np.ndarray
    zstack_profiles: np.ndarray
    median_f0_profiles: np.ndarray
    z_profile_correlations: np.ndarray
    selected_rois: tuple[SelectedRoi, ...]


@dataclass(frozen=True)
class ZProfileFigureData:
    """Z-stack and raw-fluorescence depth profiles for the example ROIs."""

    dataset: Dataset
    depth_um: np.ndarray
    zstack_profiles: np.ndarray
    median_f_profiles: np.ndarray
    correlations: np.ndarray
    selected_rois: tuple[SelectedRoi, ...]


@dataclass(frozen=True)
class ZProfilePopulationData:
    """Per-ROI z-stack-versus-median-F correlations for one dataset."""

    dataset: Dataset
    correlations: np.ndarray


@dataclass(frozen=True)
class TraceDifferenceData:
    """Per-ROI population metrics for one dataset."""

    dataset: Dataset
    fluorescence_log_ratio_q05: np.ndarray
    fluorescence_log_ratio_q95: np.ndarray
    dff_difference_q05: np.ndarray
    dff_difference_q95: np.ndarray
    z_movement_removed_percent: np.ndarray
    z_profile_correlations: np.ndarray


def discover_datasets(datasets_csv: Path) -> list[Dataset]:
    """Require the curated recordings in the manuscript table, in fixed order."""
    path = Path(datasets_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    available = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Date"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset CSV must contain columns {sorted(required)}: {path}")
        for row in reader:
            available.add((str(row["Name"]).strip(), str(row["Date"]).strip()))
    missing = set(CURATED_RECORDINGS) - available
    if missing:
        raise ValueError(f"Curated recordings missing from {path}: {sorted(missing)}")
    return [Dataset(subject, date) for subject, date in CURATED_RECORDINGS]


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load one saved NumPy dictionary with a useful error on invalid content."""
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return value


def _load_manifest(processed_dir: Path) -> dict[str, Any]:
    """Load the completed correction run manifest for one dataset."""
    path = processed_dir / ".suite2p-in-depth" / "correct-traces" / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Correction run manifest not found: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("completion_state") != "complete":
        raise ValueError(f"Correction run is not complete according to {path}")
    return manifest


def _load_integer_vector(path: Path, description: str) -> np.ndarray:
    """Load an integer-valued one-dimensional NumPy array."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")
    values = np.asarray(np.load(path)).reshape(-1)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        raise ValueError(f"{description} must contain only finite integer values: {path}")
    return values.astype(int)


def _load_vector(path: Path, description: str, *, allow_nan: bool = False) -> np.ndarray:
    """Load a finite one-dimensional numeric array."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if values.size == 0 or np.any(np.isinf(values)) or (not allow_nan and np.any(np.isnan(values))):
        requirement = "non-empty and free of infinities" if allow_nan else "non-empty and finite"
        raise ValueError(f"{description} must be {requirement}: {path}")
    return values


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return unique integer values while preserving their first-seen order."""
    return np.asarray(list(dict.fromkeys(np.asarray(values, dtype=int).tolist())), dtype=int)


def _load_plane_metadata(plane_dir: Path) -> PlaneMetadata:
    """Load the exact database, settings, and bad-frame mask used for correction."""
    if not plane_dir.is_dir():
        raise FileNotFoundError(f"Suite2p plane directory not found: {plane_dir}")
    database = _load_mapping(plane_dir / "db_correcting.npy")
    settings = _load_mapping(plane_dir / "settings_correcting.npy")
    reg_outputs = suite2p_compat.load_outputs(plane_dir)[0]
    if reg_outputs is None or reg_outputs.get("badframes") is None:
        raise ValueError(f"Registration bad-frame mask not found in {plane_dir}")

    fluorescence_path = plane_dir / "F.npy"
    neuropil_path = plane_dir / "Fneu.npy"
    if not fluorescence_path.is_file() or not neuropil_path.is_file():
        raise FileNotFoundError(f"F.npy or Fneu.npy not found in {plane_dir}")
    fluorescence = np.load(fluorescence_path, mmap_mode="r")
    neuropil = np.load(neuropil_path, mmap_mode="r")
    if fluorescence.ndim != 2 or neuropil.shape != fluorescence.shape:
        raise ValueError(
            f"F.npy and Fneu.npy must have identical (rois, time) shapes in {plane_dir}"
        )

    badframes = np.asarray(reg_outputs["badframes"], dtype=bool).reshape(-1)
    if badframes.size != fluorescence.shape[1]:
        raise ValueError(f"Bad-frame mask length does not match traces in {plane_dir}")
    frames_per_folder = np.asarray(database["frames_per_folder"], dtype=int).reshape(-1)
    if frames_per_folder.size == 0 or np.any(frames_per_folder <= 0):
        raise ValueError(f"Invalid frames_per_folder in {plane_dir / 'db_correcting.npy'}")
    if int(frames_per_folder.sum()) != fluorescence.shape[1]:
        raise ValueError(f"frames_per_folder does not reproduce the trace length in {plane_dir}")

    total_frame_rate_hz = float(settings["fs"])
    num_acquired_planes = int(database["nplanes"])
    if total_frame_rate_hz <= 0 or num_acquired_planes <= 0:
        raise ValueError(f"Invalid frame rate or plane count in {plane_dir}")
    return PlaneMetadata(
        directory=plane_dir,
        badframes=badframes,
        frames_per_folder=frames_per_folder,
        plane_rate_hz=total_frame_rate_hz / num_acquired_planes,
        num_rois=int(fluorescence.shape[0]),
        num_timepoints=int(fluorescence.shape[1]),
    )


def _validate_saved_outputs(
    corrected_f: np.ndarray,
    z_traces: np.ndarray,
    roi_ids: np.ndarray,
    roi_planes: np.ndarray,
    neural_time_s: np.ndarray,
) -> None:
    """Validate alignment of the dataset-root outputs used in the example plots."""
    if corrected_f.ndim != 2:
        raise ValueError("2pCalcium.F_zcorrected.npy must have shape (time, rois)")
    if roi_ids.size != corrected_f.shape[1] or roi_planes.size != corrected_f.shape[1]:
        raise ValueError("ROI IDs and plane IDs must align with corrected-array columns")
    if neural_time_s.size != corrected_f.shape[0]:
        raise ValueError("Neural timestamps must align with corrected-array rows")
    if z_traces.ndim != 2 or z_traces.shape[1] != corrected_f.shape[0]:
        raise ValueError("2pPlanes.zTraces.npy must have shape (planes, time)")
    if _ordered_unique(roi_planes).size != z_traces.shape[0]:
        raise ValueError("Saved z-trace rows do not match the ordered ROI plane IDs")
    if np.any(np.diff(neural_time_s) <= 0):
        raise ValueError("Concatenated neural timestamps must be strictly increasing")


def _select_rois(
    roi_ids: np.ndarray,
    roi_planes: np.ndarray,
    plane_metadata: Mapping[int, PlaneMetadata],
    plane_id: int,
    selected_roi_ids: Sequence[int],
) -> tuple[SelectedRoi, ...]:
    """Resolve requested displayed ROI IDs to corrected-array columns."""
    if plane_id not in plane_metadata:
        raise ValueError(f"Requested ROI plane {plane_id} is not present in the dataset")
    if len(set(selected_roi_ids)) != len(selected_roi_ids):
        raise ValueError("Requested ROI IDs must be unique")

    selected = []
    for roi_id in selected_roi_ids:
        matches = np.flatnonzero((roi_planes == plane_id) & (roi_ids == roi_id))
        if matches.size != 1:
            raise ValueError(
                f"Expected one saved ROI {roi_id} in plane {plane_id}; found {matches.size}"
            )
        selected.append(
            SelectedRoi(
                output_index=int(matches[0]),
                plane_id=plane_id,
                roi_id=int(roi_id),
            )
        )
    return tuple(selected)


def _calculate_selected_raw_and_baseline_traces(
    selected_rois: Sequence[SelectedRoi],
    plane_metadata: Mapping[int, PlaneMetadata],
    num_timepoints: int,
    dark_offset: float,
    f0_percentile: float,
    f0_window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected raw fluorescence traces and calculate their F0."""
    raw_selected = np.full((num_timepoints, len(selected_rois)), np.nan)
    baseline_selected = np.full_like(raw_selected, np.nan)
    selected_planes = np.asarray([roi.plane_id for roi in selected_rois], dtype=int)

    for plane_id in _ordered_unique(selected_planes):
        metadata = plane_metadata[int(plane_id)]
        selected_positions = np.flatnonzero(selected_planes == plane_id)
        roi_ids = np.asarray([selected_rois[index].roi_id for index in selected_positions])
        fluorescence_file = np.load(metadata.directory / "F.npy", mmap_mode="r")
        fluorescence = np.asarray(fluorescence_file[roi_ids, :], dtype=float).T
        fluorescence[metadata.badframes, :] = np.nan
        fluorescence -= dark_offset

        baseline = preprocess_traces.get_f0(
            fluorescence,
            metadata.plane_rate_hz,
            f0_percentile=f0_percentile,
            window_size=f0_window_s,
            frames_per_folder=metadata.frames_per_folder,
        )
        raw_selected[:, selected_positions] = fluorescence[:num_timepoints, :]
        baseline_selected[:, selected_positions] = baseline[:num_timepoints, :]
    return raw_selected, baseline_selected


def _median_f0_profiles(
    baseline: np.ndarray,
    z_trace: np.ndarray,
    num_depths: int,
) -> np.ndarray:
    """Aggregate each ROI's F0 by estimated integer z-stack slice."""
    baseline = np.asarray(baseline, dtype=float)
    z_trace = np.asarray(z_trace, dtype=float).reshape(-1)
    if baseline.ndim != 2 or baseline.shape[0] != z_trace.size:
        raise ValueError("F0 must have shape (time, rois) and align with the z trace")
    if num_depths < 1:
        raise ValueError("The z-stack must contain at least one depth")

    valid_z = np.isfinite(z_trace)
    finite_z = z_trace[valid_z]
    if not np.equal(finite_z, np.floor(finite_z)).all():
        raise ValueError("Finite z-trace values must be integer z-stack indices")
    if finite_z.size and (finite_z.min() < 0 or finite_z.max() >= num_depths):
        raise ValueError("The z trace contains indices outside the ROI z profiles")

    depth_indices = np.zeros(z_trace.size, dtype=int)
    depth_indices[valid_z] = finite_z.astype(int)
    profiles = np.full((num_depths, baseline.shape[1]), np.nan)
    for depth in np.unique(depth_indices[valid_z]):
        depth_values = baseline[valid_z & (depth_indices == depth), :]
        for roi_index in range(baseline.shape[1]):
            values = depth_values[:, roi_index]
            values = values[np.isfinite(values)]
            if values.size:
                profiles[depth, roi_index] = np.median(values)
    return profiles


def _fluorescence_percentile_profiles(
    fluorescence: np.ndarray,
    z_trace: np.ndarray,
    num_depths: int,
    percentiles: Sequence[float],
    *,
    min_samples_per_depth: int = 1,
) -> np.ndarray:
    """Calculate raw-fluorescence percentiles at every visited z-stack slice."""
    fluorescence = np.asarray(fluorescence, dtype=float)
    z_trace = np.asarray(z_trace, dtype=float).reshape(-1)
    percentile_values = np.asarray(percentiles, dtype=float).reshape(-1)
    if fluorescence.ndim != 2 or fluorescence.shape[0] != z_trace.size:
        raise ValueError("Fluorescence must have shape (time, rois) and align with the z trace")
    if num_depths < 1:
        raise ValueError("The z-stack must contain at least one depth")
    if min_samples_per_depth < 1:
        raise ValueError("min_samples_per_depth must be at least 1")
    if (
        percentile_values.size == 0
        or not np.all(np.isfinite(percentile_values))
        or np.any((percentile_values < 0) | (percentile_values > 100))
    ):
        raise ValueError("Percentiles must be finite values between 0 and 100")

    valid_z = np.isfinite(z_trace)
    finite_z = z_trace[valid_z]
    if not np.equal(finite_z, np.floor(finite_z)).all():
        raise ValueError("Finite z-trace values must be integer z-stack indices")
    if finite_z.size and (finite_z.min() < 0 or finite_z.max() >= num_depths):
        raise ValueError("The z trace contains indices outside the ROI z profiles")

    depth_indices = np.zeros(z_trace.size, dtype=int)
    depth_indices[valid_z] = finite_z.astype(int)
    profiles = np.full(
        (percentile_values.size, num_depths, fluorescence.shape[1]),
        np.nan,
    )
    for depth in np.unique(depth_indices[valid_z]):
        depth_values = fluorescence[valid_z & (depth_indices == depth), :]
        for roi_index in range(fluorescence.shape[1]):
            values = depth_values[:, roi_index]
            values = values[np.isfinite(values)]
            if values.size >= min_samples_per_depth:
                profiles[:, depth, roi_index] = np.percentile(values, percentile_values)
    return profiles


def _profile_correlations(
    zstack_profiles: np.ndarray,
    median_f0_profiles: np.ndarray,
) -> np.ndarray:
    """Calculate per-ROI Pearson correlations across shared depth bins."""
    zstack_profiles = np.asarray(zstack_profiles, dtype=float)
    median_f0_profiles = np.asarray(median_f0_profiles, dtype=float)
    if zstack_profiles.shape != median_f0_profiles.shape or zstack_profiles.ndim != 2:
        raise ValueError("Profile pairs must have identical (depth, rois) shapes")

    correlations = np.full(zstack_profiles.shape[1], np.nan)
    for roi_index in range(zstack_profiles.shape[1]):
        zstack_profile = zstack_profiles[:, roi_index]
        f0_profile = median_f0_profiles[:, roi_index]
        valid = np.isfinite(zstack_profile) & np.isfinite(f0_profile)
        if np.count_nonzero(valid) < 3:
            continue
        zstack_values = zstack_profile[valid]
        f0_values = f0_profile[valid]
        if np.ptp(zstack_values) == 0 or np.ptp(f0_values) == 0:
            continue
        correlations[roi_index] = np.corrcoef(zstack_values, f0_values)[0, 1]
    return correlations


def _column_percentiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate finite-value 5th and 95th percentiles for each ROI column."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Percentile values must have shape (time, rois)")

    lower = np.full(values.shape[1], np.nan)
    upper = np.full(values.shape[1], np.nan)
    for roi_index in range(values.shape[1]):
        roi_values = values[:, roi_index]
        roi_values = roi_values[np.isfinite(roi_values)]
        if roi_values.size:
            lower[roi_index], upper[roi_index] = np.percentile(roi_values, [5, 95])
    return lower, upper


def _log_ratio_percentiles(
    reference: np.ndarray, comparison: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-ROI tails of log2(comparison / reference)."""
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    valid = np.isfinite(reference) & np.isfinite(comparison) & (reference > 0) & (comparison > 0)
    log_ratio = np.full(reference.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(comparison, reference, out=log_ratio, where=valid)
        np.log2(log_ratio, out=log_ratio, where=valid)
    return _column_percentiles(log_ratio)


def _difference_percentiles(
    reference: np.ndarray, comparison: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-ROI tails of comparison minus reference."""
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    valid = np.isfinite(reference) & np.isfinite(comparison)
    difference = np.full(reference.shape, np.nan)
    np.subtract(comparison, reference, out=difference, where=valid)
    return _column_percentiles(difference)


def _z_movement_removed_percent(
    raw_fluorescence: np.ndarray, corrected_fluorescence: np.ndarray
) -> np.ndarray:
    """Calculate valid raw samples made non-finite by z correction per ROI."""
    if raw_fluorescence.shape != corrected_fluorescence.shape or raw_fluorescence.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    raw_valid = np.isfinite(raw_fluorescence)
    removed = raw_valid & ~np.isfinite(corrected_fluorescence)
    valid_counts = raw_valid.sum(axis=0)
    percentages = np.full(raw_fluorescence.shape[1], np.nan)
    np.divide(
        100.0 * removed.sum(axis=0),
        valid_counts,
        out=percentages,
        where=valid_counts > 0,
    )
    return percentages


def calculate_trace_difference_data(
    dataset: Dataset,
    processed_root: Path,
    suite2p_root: Path,
) -> TraceDifferenceData:
    """Calculate trace-difference and z-profile metrics for every ROI."""
    processed_dir = Path(processed_root) / dataset.subject / dataset.date
    suite2p_dir = Path(suite2p_root) / dataset.subject / dataset.date / "suite2p"
    for directory in (processed_dir, suite2p_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    manifest = _load_manifest(processed_dir)
    delta_f_settings = manifest["configuration"]["delta_F"]
    dark_offset = float(delta_f_settings["absolute_zero"])
    f0_percentile = float(delta_f_settings["F0_percentile"])
    f0_window_s = float(delta_f_settings["F0_window"])

    corrected_f = np.load(processed_dir / "2pCalcium.F_zcorrected.npy", mmap_mode="r")
    corrected_dff = np.load(processed_dir / "2pCalcium.dff.npy", mmap_mode="r")
    z_traces = np.asarray(np.load(processed_dir / "2pPlanes.zTraces.npy"), dtype=float)
    zstack_profiles = np.load(processed_dir / "2pRois.zProfiles.npy", mmap_mode="r")
    roi_ids = _load_integer_vector(processed_dir / "2pRois.ids.npy", "ROI IDs")
    roi_planes = _load_integer_vector(processed_dir / "2pRois.2pPlanes.npy", "ROI plane IDs")
    if corrected_f.ndim != 2 or corrected_f.shape[0] == 0:
        raise ValueError("2pCalcium.F_zcorrected.npy must have shape (time, rois)")
    if corrected_dff.shape != corrected_f.shape:
        raise ValueError("Corrected fluorescence and dF/F arrays must have identical shapes")
    if roi_ids.size != corrected_f.shape[1] or roi_planes.size != corrected_f.shape[1]:
        raise ValueError("ROI IDs and plane IDs must align with corrected-array columns")
    if zstack_profiles.ndim != 2 or zstack_profiles.shape[0] != corrected_f.shape[1]:
        raise ValueError("2pRois.zProfiles.npy must have shape (rois, z)")
    z_plane_ids = _ordered_unique(roi_planes)
    if z_traces.shape != (z_plane_ids.size, corrected_f.shape[0]):
        raise ValueError("2pPlanes.zTraces.npy must have shape (planes, time)")

    fluorescence_q05 = np.full(corrected_f.shape[1], np.nan)
    fluorescence_q95 = np.full(corrected_f.shape[1], np.nan)
    dff_q05 = np.full(corrected_f.shape[1], np.nan)
    dff_q95 = np.full(corrected_f.shape[1], np.nan)
    z_movement_removed_percent = np.full(corrected_f.shape[1], np.nan)
    z_profile_correlations = np.full(corrected_f.shape[1], np.nan)
    for plane_index, plane_id in enumerate(z_plane_ids):
        metadata = _load_plane_metadata(suite2p_dir / f"plane{plane_id}")
        output_columns = np.flatnonzero(roi_planes == plane_id)
        plane_roi_ids = roi_ids[output_columns]
        if np.any(plane_roi_ids < 0) or np.any(plane_roi_ids >= metadata.num_rois):
            raise ValueError(f"Saved ROI IDs fall outside F.npy for plane {plane_id}")
        if metadata.num_timepoints < corrected_f.shape[0]:
            raise ValueError(f"Plane {plane_id} is shorter than the corrected output")

        fluorescence_file = np.load(metadata.directory / "F.npy", mmap_mode="r")
        neuropil_file = np.load(metadata.directory / "Fneu.npy", mmap_mode="r")
        for start in range(0, output_columns.size, ROI_CHUNK_SIZE):
            stop = min(start + ROI_CHUNK_SIZE, output_columns.size)
            chunk_columns = output_columns[start:stop]
            chunk_roi_ids = roi_ids[chunk_columns]
            fluorescence = np.asarray(fluorescence_file[chunk_roi_ids, :], dtype=float).T
            neuropil = np.asarray(neuropil_file[chunk_roi_ids, :], dtype=float).T
            fluorescence[metadata.badframes, :] = np.nan
            neuropil[metadata.badframes, :] = np.nan
            fluorescence -= dark_offset
            neuropil -= dark_offset

            num_timepoints = corrected_f.shape[0]
            corrected_f_chunk = np.asarray(corrected_f[:, chunk_columns], dtype=float)
            chunk_q05, chunk_q95 = _log_ratio_percentiles(
                fluorescence[:num_timepoints, :],
                corrected_f_chunk,
            )
            fluorescence_q05[chunk_columns] = chunk_q05
            fluorescence_q95[chunk_columns] = chunk_q95
            z_movement_removed_percent[chunk_columns] = _z_movement_removed_percent(
                fluorescence[:num_timepoints, :],
                corrected_f_chunk,
            )

            baseline = preprocess_traces.get_f0(
                fluorescence,
                metadata.plane_rate_hz,
                f0_percentile=f0_percentile,
                window_size=f0_window_s,
                frames_per_folder=metadata.frames_per_folder,
            )
            median_f_profiles = _fluorescence_percentile_profiles(
                fluorescence[:num_timepoints, :],
                z_traces[plane_index],
                zstack_profiles.shape[1],
                [F_PROFILE_PERCENTILE],
                min_samples_per_depth=MIN_PROFILE_SAMPLES_PER_DEPTH,
            )[0]
            chunk_zstack_profiles = np.asarray(zstack_profiles[chunk_columns, :], dtype=float).T
            z_profile_correlations[chunk_columns] = _profile_correlations(
                chunk_zstack_profiles,
                median_f_profiles,
            )
            neuropil_corrected = preprocess_traces.correct_neuropil(
                fluorescence,
                neuropil,
                metadata.plane_rate_hz,
                baseline,
                baseline_percentile=f0_percentile,
                baseline_window=f0_window_s,
                frames_per_folder=metadata.frames_per_folder,
            )[0]
            uncorrected_dff = preprocess_traces.get_delta_f_over_f(
                neuropil_corrected,
                baseline,
            )
            corrected_dff_chunk = np.asarray(corrected_dff[:, chunk_columns], dtype=float)
            chunk_q05, chunk_q95 = _difference_percentiles(
                uncorrected_dff[:num_timepoints, :],
                corrected_dff_chunk,
            )
            dff_q05[chunk_columns] = chunk_q05
            dff_q95[chunk_columns] = chunk_q95

    return TraceDifferenceData(
        dataset=dataset,
        fluorescence_log_ratio_q05=fluorescence_q05,
        fluorescence_log_ratio_q95=fluorescence_q95,
        dff_difference_q05=dff_q05,
        dff_difference_q95=dff_q95,
        z_movement_removed_percent=z_movement_removed_percent,
        z_profile_correlations=z_profile_correlations,
    )


def prepare_z_profile_figure_data(
    dataset: Dataset,
    processed_root: Path,
    suite2p_root: Path,
    roi_plane_id: int,
) -> ZProfileFigureData:
    """Load only the inputs needed for the example ROI z-profile comparison."""
    processed_dir = Path(processed_root) / dataset.subject / dataset.date
    suite2p_dir = Path(suite2p_root) / dataset.subject / dataset.date / "suite2p"
    for directory in (processed_dir, suite2p_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    manifest = _load_manifest(processed_dir)
    configuration = manifest["configuration"]
    dark_offset = float(configuration["delta_F"]["absolute_zero"])
    z_spacing_um = float(configuration["z_correction"]["spacing"])
    if z_spacing_um <= 0:
        raise ValueError("z_correction.spacing must be positive")

    z_traces = np.asarray(np.load(processed_dir / "2pPlanes.zTraces.npy"), dtype=float)
    zstack_profiles = np.load(processed_dir / "2pRois.zProfiles.npy", mmap_mode="r")
    roi_ids = _load_integer_vector(processed_dir / "2pRois.ids.npy", "ROI IDs")
    roi_planes = _load_integer_vector(processed_dir / "2pRois.2pPlanes.npy", "ROI plane IDs")
    if zstack_profiles.ndim != 2 or zstack_profiles.shape[1] == 0:
        raise ValueError("2pRois.zProfiles.npy must have shape (rois, z)")
    if roi_ids.size != zstack_profiles.shape[0] or roi_planes.size != zstack_profiles.shape[0]:
        raise ValueError("ROI IDs and plane IDs must align with 2pRois.zProfiles.npy")
    z_plane_ids = _ordered_unique(roi_planes)
    if z_traces.ndim != 2 or z_traces.shape[0] != z_plane_ids.size or z_traces.shape[1] == 0:
        raise ValueError("2pPlanes.zTraces.npy must have shape (planes, time)")

    metadata = _load_plane_metadata(suite2p_dir / f"plane{roi_plane_id}")
    selected_rois = _select_rois(
        roi_ids,
        roi_planes,
        {roi_plane_id: metadata},
        roi_plane_id,
        TRACE_ROI_IDS,
    )
    if metadata.num_timepoints < z_traces.shape[1]:
        raise ValueError(f"Plane {roi_plane_id} is shorter than the saved z trace")

    selected_roi_ids = np.asarray([roi.roi_id for roi in selected_rois], dtype=int)
    fluorescence_file = np.load(metadata.directory / "F.npy", mmap_mode="r")
    fluorescence = np.asarray(fluorescence_file[selected_roi_ids, :], dtype=float).T
    fluorescence[metadata.badframes, :] = np.nan
    fluorescence -= dark_offset
    fluorescence = fluorescence[: z_traces.shape[1], :]

    plane_matches = np.flatnonzero(z_plane_ids == roi_plane_id)
    if plane_matches.size != 1:
        raise ValueError(f"Expected one z trace for plane {roi_plane_id}")
    plane_z_trace = z_traces[int(plane_matches[0])]
    median_f_profiles = _fluorescence_percentile_profiles(
        fluorescence,
        plane_z_trace,
        zstack_profiles.shape[1],
        [F_PROFILE_PERCENTILE],
        min_samples_per_depth=MIN_PROFILE_SAMPLES_PER_DEPTH,
    )[0]
    selected_columns = [roi.output_index for roi in selected_rois]
    selected_zstack_profiles = np.asarray(zstack_profiles[selected_columns, :], dtype=float).T
    correlations = _profile_correlations(selected_zstack_profiles, median_f_profiles)
    return ZProfileFigureData(
        dataset=dataset,
        depth_um=np.arange(zstack_profiles.shape[1]) * z_spacing_um,
        zstack_profiles=selected_zstack_profiles,
        median_f_profiles=median_f_profiles,
        correlations=correlations,
        selected_rois=selected_rois,
    )


def calculate_z_profile_population_data(
    dataset: Dataset,
    processed_root: Path,
    suite2p_root: Path,
) -> ZProfilePopulationData:
    """Calculate median-F-versus-z-stack correlations for every ROI."""
    processed_dir = Path(processed_root) / dataset.subject / dataset.date
    suite2p_dir = Path(suite2p_root) / dataset.subject / dataset.date / "suite2p"
    for directory in (processed_dir, suite2p_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    manifest = _load_manifest(processed_dir)
    dark_offset = float(manifest["configuration"]["delta_F"]["absolute_zero"])
    z_traces = np.asarray(np.load(processed_dir / "2pPlanes.zTraces.npy"), dtype=float)
    zstack_profiles = np.load(processed_dir / "2pRois.zProfiles.npy", mmap_mode="r")
    roi_ids = _load_integer_vector(processed_dir / "2pRois.ids.npy", "ROI IDs")
    roi_planes = _load_integer_vector(processed_dir / "2pRois.2pPlanes.npy", "ROI plane IDs")
    if zstack_profiles.ndim != 2 or zstack_profiles.shape[1] == 0:
        raise ValueError("2pRois.zProfiles.npy must have shape (rois, z)")
    if roi_ids.size != zstack_profiles.shape[0] or roi_planes.size != zstack_profiles.shape[0]:
        raise ValueError("ROI IDs and plane IDs must align with 2pRois.zProfiles.npy")
    z_plane_ids = _ordered_unique(roi_planes)
    if z_traces.ndim != 2 or z_traces.shape[0] != z_plane_ids.size or z_traces.shape[1] == 0:
        raise ValueError("2pPlanes.zTraces.npy must have shape (planes, time)")

    correlations = np.full(roi_ids.size, np.nan)
    for plane_index, plane_id in enumerate(z_plane_ids):
        metadata = _load_plane_metadata(suite2p_dir / f"plane{plane_id}")
        output_columns = np.flatnonzero(roi_planes == plane_id)
        plane_roi_ids = roi_ids[output_columns]
        if np.any(plane_roi_ids < 0) or np.any(plane_roi_ids >= metadata.num_rois):
            raise ValueError(f"Saved ROI IDs fall outside F.npy for plane {plane_id}")
        if metadata.num_timepoints < z_traces.shape[1]:
            raise ValueError(f"Plane {plane_id} is shorter than the saved z trace")

        fluorescence_file = np.load(metadata.directory / "F.npy", mmap_mode="r")
        for start in range(0, output_columns.size, ROI_CHUNK_SIZE):
            stop = min(start + ROI_CHUNK_SIZE, output_columns.size)
            chunk_columns = output_columns[start:stop]
            chunk_roi_ids = roi_ids[chunk_columns]
            fluorescence = np.asarray(fluorescence_file[chunk_roi_ids, :], dtype=float).T
            fluorescence[metadata.badframes, :] = np.nan
            fluorescence -= dark_offset
            fluorescence = fluorescence[: z_traces.shape[1], :]
            median_f_profiles = _fluorescence_percentile_profiles(
                fluorescence,
                z_traces[plane_index],
                zstack_profiles.shape[1],
                [F_PROFILE_PERCENTILE],
                min_samples_per_depth=MIN_PROFILE_SAMPLES_PER_DEPTH,
            )[0]
            chunk_zstack_profiles = np.asarray(zstack_profiles[chunk_columns, :], dtype=float).T
            correlations[chunk_columns] = _profile_correlations(
                chunk_zstack_profiles,
                median_f_profiles,
            )
    return ZProfilePopulationData(dataset=dataset, correlations=correlations)


def _trim_experiment_lengths(lengths: np.ndarray, total_length: int) -> np.ndarray:
    """Trim production experiment lengths to a possibly shortened shared time axis."""
    if int(np.sum(lengths)) < total_length:
        raise ValueError("frames_per_folder is shorter than the saved shared time axis")
    trimmed = []
    remaining = int(total_length)
    for length in np.asarray(lengths, dtype=int):
        if remaining == 0:
            break
        retained = min(int(length), remaining)
        trimmed.append(retained)
        remaining -= retained
    return np.asarray(trimmed, dtype=int)


def _experiment_boundary_times(
    neural_time_s: np.ndarray, frames_per_folder: np.ndarray
) -> np.ndarray:
    """Locate experiment transitions on the concatenated timestamp axis."""
    boundaries = np.cumsum(frames_per_folder)[:-1]
    return np.asarray(
        [0.5 * (neural_time_s[index - 1] + neural_time_s[index]) for index in boundaries],
        dtype=float,
    )


def _experiment_start_times(neural_time_s: np.ndarray, frames_per_folder: np.ndarray) -> np.ndarray:
    """Return the timestamp of the first neural frame in each experiment."""
    start_indices = np.concatenate(([0], np.cumsum(frames_per_folder)[:-1]))
    return neural_time_s[start_indices]


def _smooth_wheel_velocity(
    wheel_time_s: np.ndarray,
    velocity: np.ndarray,
    experiment_boundaries_s: np.ndarray,
) -> np.ndarray:
    """Gaussian-smooth velocity independently within every experiment."""
    if wheel_time_s.shape != velocity.shape:
        raise ValueError("Wheel timestamps and velocity must have identical shapes")
    if np.any(np.diff(wheel_time_s) <= 0):
        raise ValueError("Concatenated wheel timestamps must be strictly increasing")
    split_indices = np.searchsorted(wheel_time_s, experiment_boundaries_s, side="left")
    edges = np.concatenate(([0], split_indices, [wheel_time_s.size]))
    smoothed = np.full(velocity.shape, np.nan)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        segment_time = wheel_time_s[start:stop]
        segment_values = velocity[start:stop]
        if segment_time.size < 2:
            smoothed[start:stop] = segment_values
            continue
        sample_interval_s = float(np.median(np.diff(segment_time)))
        if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
            raise ValueError("Cannot determine the wheel sampling interval")
        sigma_samples = WHEEL_SMOOTHING_SIGMA_S / sample_interval_s
        valid = np.isfinite(segment_values)
        numerator = gaussian_filter1d(
            np.where(valid, segment_values, 0.0),
            sigma=sigma_samples,
            mode="nearest",
        )
        denominator = gaussian_filter1d(
            valid.astype(float),
            sigma=sigma_samples,
            mode="nearest",
        )
        np.divide(
            numerator,
            denominator,
            out=smoothed[start:stop],
            where=denominator > 0,
        )
    return smoothed


def prepare_figure_data(
    dataset: Dataset,
    processed_root: Path,
    suite2p_root: Path,
    bonsai_root: Path,
    roi_plane_id: int,
) -> FigureData:
    """Load and calculate every panel input for one recording."""
    processed_dir = Path(processed_root) / dataset.subject / dataset.date
    suite2p_dir = Path(suite2p_root) / dataset.subject / dataset.date / "suite2p"
    bonsai_dir = Path(bonsai_root) / dataset.subject / dataset.date
    for directory in (processed_dir, suite2p_dir, bonsai_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    manifest = _load_manifest(processed_dir)
    configuration = manifest["configuration"]
    delta_f_settings = configuration["delta_F"]
    dark_offset = float(delta_f_settings["absolute_zero"])
    f0_percentile = float(delta_f_settings["F0_percentile"])
    f0_window_s = float(delta_f_settings["F0_window"])
    z_spacing_um = float(configuration["z_correction"]["spacing"])
    if z_spacing_um <= 0:
        raise ValueError("z_correction.spacing must be positive")

    corrected_f = np.load(processed_dir / "2pCalcium.F_zcorrected.npy", mmap_mode="r")
    z_traces = np.asarray(np.load(processed_dir / "2pPlanes.zTraces.npy"), dtype=float)
    zstack_profiles = np.load(processed_dir / "2pRois.zProfiles.npy", mmap_mode="r")
    roi_ids = _load_integer_vector(processed_dir / "2pRois.ids.npy", "ROI IDs")
    roi_planes = _load_integer_vector(processed_dir / "2pRois.2pPlanes.npy", "ROI plane IDs")
    neural_time_s = _load_vector(bonsai_dir / "2pCalcium.timestamps.npy", "Neural timestamps")
    wheel_time_s = _load_vector(bonsai_dir / "wheel.timestamps.npy", "Wheel timestamps")
    wheel_velocity = _load_vector(
        bonsai_dir / "wheel.velocity.npy", "Wheel velocity", allow_nan=True
    )
    _validate_saved_outputs(
        corrected_f,
        z_traces,
        roi_ids,
        roi_planes,
        neural_time_s,
    )
    if wheel_time_s.shape != wheel_velocity.shape:
        raise ValueError("Concatenated wheel timestamps and velocity do not align")
    if zstack_profiles.ndim != 2 or zstack_profiles.shape[0] != corrected_f.shape[1]:
        raise ValueError("2pRois.zProfiles.npy must have shape (rois, z)")

    z_plane_ids = _ordered_unique(roi_planes)
    plane_metadata = {
        int(plane_id): _load_plane_metadata(suite2p_dir / f"plane{plane_id}")
        for plane_id in z_plane_ids
    }
    selected_rois = _select_rois(
        roi_ids,
        roi_planes,
        plane_metadata,
        roi_plane_id,
        TRACE_ROI_IDS,
    )
    raw_fluorescence, uncorrected_f0 = _calculate_selected_raw_and_baseline_traces(
        selected_rois,
        plane_metadata,
        corrected_f.shape[0],
        dark_offset,
        f0_percentile,
        f0_window_s,
    )
    selected_columns = [roi.output_index for roi in selected_rois]
    selected_corrected_f = np.asarray(corrected_f[:, selected_columns], dtype=float)
    fluorescence_log_ratio_q05, fluorescence_log_ratio_q95 = _log_ratio_percentiles(
        raw_fluorescence,
        selected_corrected_f,
    )
    selected_zstack_profiles = np.asarray(zstack_profiles[selected_columns, :], dtype=float).T
    median_f0_profiles = np.full_like(selected_zstack_profiles, np.nan)
    selected_planes = np.asarray([roi.plane_id for roi in selected_rois], dtype=int)
    for plane_index, plane_id in enumerate(z_plane_ids):
        selected_positions = np.flatnonzero(selected_planes == plane_id)
        if selected_positions.size:
            median_f0_profiles[:, selected_positions] = _median_f0_profiles(
                uncorrected_f0[:, selected_positions],
                z_traces[plane_index],
                selected_zstack_profiles.shape[0],
            )
    z_profile_correlations = _profile_correlations(
        selected_zstack_profiles,
        median_f0_profiles,
    )

    z_medians = np.full((z_traces.shape[0], 1), np.nan)
    for plane_index, trace in enumerate(z_traces):
        if not np.isfinite(trace).any():
            raise ValueError(f"Plane {z_plane_ids[plane_index]} has no finite z estimates")
        z_medians[plane_index, 0] = np.nanmedian(trace)
    z_movement_um = (z_traces - z_medians) * z_spacing_um

    reference_lengths = _trim_experiment_lengths(
        plane_metadata[int(z_plane_ids[0])].frames_per_folder,
        neural_time_s.size,
    )
    experiment_boundaries_s = _experiment_boundary_times(neural_time_s, reference_lengths)
    experiment_starts_s = _experiment_start_times(neural_time_s, reference_lengths)
    smoothed_wheel_velocity = _smooth_wheel_velocity(
        wheel_time_s,
        wheel_velocity,
        experiment_boundaries_s,
    )
    time_origin_s = min(float(neural_time_s[0]), float(wheel_time_s[0]))
    return FigureData(
        dataset=dataset,
        neural_time_min=(neural_time_s - time_origin_s) / 60,
        wheel_time_min=(wheel_time_s - time_origin_s) / 60,
        experiment_start_min=(experiment_starts_s - time_origin_s) / 60,
        wheel_velocity_cm_s=smoothed_wheel_velocity,
        z_movement_um=z_movement_um,
        z_plane_ids=z_plane_ids,
        raw_fluorescence=raw_fluorescence,
        corrected_fluorescence=selected_corrected_f,
        fluorescence_log_ratio_q05=fluorescence_log_ratio_q05,
        fluorescence_log_ratio_q95=fluorescence_log_ratio_q95,
        z_profile_depth_um=np.arange(selected_zstack_profiles.shape[0]) * z_spacing_um,
        zstack_profiles=selected_zstack_profiles,
        median_f0_profiles=median_f0_profiles,
        z_profile_correlations=z_profile_correlations,
        selected_rois=selected_rois,
    )


def _finish_axis(axis: mpl.axes.Axes) -> None:
    """Apply restrained manuscript styling to one axis."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.8)
    axis.margins(x=0)


def _plot_z_movement(
    axis: mpl.axes.Axes,
    time_min: np.ndarray,
    movement_um: np.ndarray,
    plane_ids: np.ndarray,
) -> None:
    """Plot median-centered image-estimated movement for every retained plane."""
    colors = mpl.colormaps["tab10"]
    for plane_index, (plane_id, trace) in enumerate(zip(plane_ids, movement_um, strict=True)):
        axis.plot(
            time_min,
            trace,
            color=colors(plane_index % 10),
            linewidth=0.7,
            label=f"Plane {plane_id}",
        )
    axis.axhline(0, color="0.75", linewidth=0.6, zorder=0)
    axis.set_ylabel(r"Relative z ($\mu$m)")
    axis.legend(frameon=False, loc="upper right", ncol=min(4, plane_ids.size))
    _finish_axis(axis)


def _stack_offsets(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose shared per-ROI offsets and evenly spaced label positions."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    num_rois = first.shape[1]
    references = np.full(num_rois, np.nan)
    spans = np.full(num_rois, np.nan)
    for roi_index in range(num_rois):
        values = np.concatenate((first[:, roi_index], second[:, roi_index]))
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Selected ROI {roi_index} has no finite values to plot")
        references[roi_index] = np.median(values)
        low, high = np.percentile(values - references[roi_index], [1, 99])
        spans[roi_index] = high - low
    spacing = float(np.nanmax(spans)) * 1.25
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0
    levels = (num_rois - 1 - np.arange(num_rois)) * spacing
    return levels, levels - references


def _scale_trace_pairs(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each paired ROI by its shared robust fluorescence range."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    scaled_first = np.asarray(first, dtype=float).copy()
    scaled_second = np.asarray(second, dtype=float).copy()
    for roi_index in range(first.shape[1]):
        values = np.concatenate((first[:, roi_index], second[:, roi_index]))
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Selected ROI {roi_index} has no finite values to scale")
        center = float(np.median(values))
        low, high = np.percentile(values, [1, 99])
        scale = float(high - low)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scaled_first[:, roi_index] = (scaled_first[:, roi_index] - center) / scale
        scaled_second[:, roi_index] = (scaled_second[:, roi_index] - center) / scale
    return scaled_first, scaled_second


def _plot_trace_stack(
    axis: mpl.axes.Axes,
    time_min: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    selected_rois: Sequence[SelectedRoi],
    *,
    first_label: str,
    second_label: str,
    ylabel: str,
    lower_percentiles: np.ndarray,
    upper_percentiles: np.ndarray,
    scale_pair_ranges: bool = False,
) -> None:
    """Plot paired traces with the same additive offset for each ROI."""
    lower_percentiles = np.asarray(lower_percentiles, dtype=float).reshape(-1)
    upper_percentiles = np.asarray(upper_percentiles, dtype=float).reshape(-1)
    if lower_percentiles.size != len(selected_rois) or upper_percentiles.size != len(selected_rois):
        raise ValueError("Two percentile values are required for every selected ROI")
    if scale_pair_ranges:
        first, second = _scale_trace_pairs(first, second)
    levels, offsets = _stack_offsets(first, second)
    for roi_index, offset in enumerate(offsets):
        axis.plot(
            time_min,
            first[:, roi_index] + offset,
            color=RAW_COLOR,
            linewidth=0.45,
            alpha=0.9,
            label=first_label if roi_index == 0 else None,
            zorder=2,
        )
        axis.plot(
            time_min,
            second[:, roi_index] + offset,
            color=CORRECTED_COLOR,
            linewidth=0.45,
            alpha=0.9,
            label=second_label if roi_index == 0 else None,
            zorder=3,
        )
    roi_labels = []
    for roi, lower, upper in zip(selected_rois, lower_percentiles, upper_percentiles, strict=True):
        percentiles = (
            f"[{lower:.2f} {upper:.2f}]"
            if np.isfinite(lower) and np.isfinite(upper)
            else "[n/a n/a]"
        )
        roi_labels.append(f"{roi.label}\n{percentiles}")
    axis.set_yticks(levels, roi_labels)
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, loc="upper right", ncol=2)
    _finish_axis(axis)


def _plot_time_limits(data: FigureData) -> tuple[float, float]:
    """Return the first requested minutes shared by neural and wheel recordings."""
    start = min(float(data.neural_time_min[0]), float(data.wheel_time_min[0]))
    recording_stop = max(float(data.neural_time_min[-1]), float(data.wheel_time_min[-1]))
    stop = min(start + TRACE_DURATION_MIN, recording_stop)
    if stop <= start:
        raise ValueError("The concatenated recording has an invalid time range")
    return start, stop


def _time_window(time_min: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Select timestamps within the inclusive plotting interval."""
    return (time_min >= limits[0]) & (time_min <= limits[1])


def _mark_experiment_starts(
    axes: Sequence[mpl.axes.Axes], experiment_start_min: np.ndarray
) -> None:
    """Mark experiment starts consistently on every time-series axis."""
    for axis in axes:
        for start_min in experiment_start_min:
            axis.axvline(start_min, color="0.7", linewidth=0.7, zorder=1)


def plot_fluorescence_correction_example(data: FigureData, output_dir: Path) -> Path:
    """Write locomotion, z-movement, and raw/corrected fluorescence panels."""
    time_limits = _plot_time_limits(data)
    neural_window = _time_window(data.neural_time_min, time_limits)
    wheel_window = _time_window(data.wheel_time_min, time_limits)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.6, 0.75, 8]},
    )
    axes[0].plot(
        data.wheel_time_min[wheel_window],
        data.wheel_velocity_cm_s[wheel_window],
        color="black",
        linewidth=0.6,
    )
    axes[0].axhline(0, color="0.75", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Velocity\n(cm/s)")
    axes[0].set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    _finish_axis(axes[0])
    _plot_z_movement(
        axes[1],
        data.neural_time_min[neural_window],
        data.z_movement_um[:, neural_window],
        data.z_plane_ids,
    )
    _plot_trace_stack(
        axes[2],
        data.neural_time_min[neural_window],
        data.raw_fluorescence[neural_window],
        data.corrected_fluorescence[neural_window],
        data.selected_rois,
        first_label="Raw F",
        second_label="Z-corrected F",
        ylabel="Scaled fluorescence (a.u.)",
        lower_percentiles=data.fluorescence_log_ratio_q05,
        upper_percentiles=data.fluorescence_log_ratio_q95,
        scale_pair_ranges=True,
    )
    _mark_experiment_starts(axes, data.experiment_start_min)
    axes[-1].set_xlabel("Time (min)")
    axes[-1].set_xlim(*time_limits)
    output_path = output_dir / f"{data.dataset.slug}_fluorescence_correction_example.pdf"
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return output_path


def _running_onset_times(
    wheel_time_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    experiment_boundaries_s: np.ndarray,
) -> np.ndarray:
    """Find sustained running onsets preceded by sustained stationary periods."""
    if wheel_time_s.shape != velocity_cm_s.shape or wheel_time_s.ndim != 1:
        raise ValueError("Wheel timestamps and velocity must be aligned vectors")
    if np.any(np.diff(wheel_time_s) <= 0):
        raise ValueError("Wheel timestamps must be strictly increasing")

    split_indices = np.searchsorted(wheel_time_s, experiment_boundaries_s, side="left")
    edges = np.concatenate(([0], split_indices, [wheel_time_s.size]))
    running = np.isfinite(velocity_cm_s) & (velocity_cm_s > RUNNING_THRESHOLD_CM_S)
    onset_times = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop - start < 2:
            continue
        segment_running = running[start:stop]
        candidates = np.flatnonzero(
            segment_running & ~np.concatenate(([False], segment_running[:-1]))
        )
        for candidate in candidates:
            onset_index = start + int(candidate)
            onset_time_s = float(wheel_time_s[onset_index])
            if (
                onset_time_s - wheel_time_s[start] < RUNNING_STATIONARY_S
                or wheel_time_s[stop - 1] - onset_time_s < RUNNING_MINIMUM_DURATION_S
            ):
                continue
            stationary_start = int(
                np.searchsorted(
                    wheel_time_s,
                    onset_time_s - RUNNING_STATIONARY_S,
                    side="left",
                )
            )
            running_stop = int(
                np.searchsorted(
                    wheel_time_s,
                    onset_time_s + RUNNING_MINIMUM_DURATION_S,
                    side="left",
                )
            )
            stationary_values = velocity_cm_s[stationary_start:onset_index]
            if stationary_start < start or stationary_values.size == 0:
                continue
            stationary = np.isfinite(stationary_values) & (
                stationary_values <= RUNNING_THRESHOLD_CM_S
            )
            if not np.all(stationary) or not np.all(running[onset_index:running_stop]):
                continue
            onset_times.append(onset_time_s)
    return np.asarray(onset_times, dtype=float)


def _align_z_to_running_onsets(
    neural_time_s: np.ndarray,
    z_movement_um: np.ndarray,
    onset_times_s: np.ndarray,
    experiment_boundaries_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline and align the median plane estimate around each running onset."""
    if neural_time_s.ndim != 1 or z_movement_um.ndim != 2:
        raise ValueError("Neural timestamps and z movement must be 1D and 2D")
    if z_movement_um.shape[1] != neural_time_s.size:
        raise ValueError("Z movement must align with neural timestamps")
    sample_interval_s = float(np.median(np.diff(neural_time_s)))
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("Cannot determine the neural sampling interval")

    num_before = int(round(RUNNING_WINDOW_BEFORE_S / sample_interval_s))
    num_after = int(round(RUNNING_WINDOW_AFTER_S / sample_interval_s))
    relative_time_s = np.arange(-num_before, num_after + 1, dtype=float) * sample_interval_s
    baseline = (relative_time_s >= -RUNNING_BASELINE_S) & (relative_time_s < 0)
    collapsed_z_um = np.nanmedian(z_movement_um, axis=0)
    finite_z = np.isfinite(collapsed_z_um)
    if np.count_nonzero(finite_z) < 2:
        raise ValueError("The collapsed z trace has fewer than two finite values")

    segment_edges_s = np.concatenate(
        ([neural_time_s[0]], experiment_boundaries_s, [neural_time_s[-1]])
    )
    aligned = []
    for onset_time_s in onset_times_s:
        segment_index = int(np.searchsorted(experiment_boundaries_s, onset_time_s, side="right"))
        if (
            onset_time_s + relative_time_s[0] < segment_edges_s[segment_index]
            or onset_time_s + relative_time_s[-1] > segment_edges_s[segment_index + 1]
        ):
            continue
        trace = np.interp(
            onset_time_s + relative_time_s,
            neural_time_s[finite_z],
            collapsed_z_um[finite_z],
        )
        baseline_value = float(np.mean(trace[baseline]))
        if np.isfinite(baseline_value):
            aligned.append(trace - baseline_value)
    if not aligned:
        raise ValueError("No running onset has a complete aligned z window")
    return relative_time_s, np.asarray(aligned)


def _mean_and_sem(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the pointwise mean and sample SEM across onset-aligned traces."""
    if traces.ndim != 2 or traces.shape[0] == 0:
        raise ValueError("Onset-aligned traces must be a non-empty 2D array")
    valid = np.isfinite(traces)
    counts = valid.sum(axis=0)
    total = np.where(valid, traces, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        counts,
        out=np.full(traces.shape[1], np.nan),
        where=counts > 0,
    )
    squared_deviation = np.where(valid, (traces - mean) ** 2, 0.0).sum(axis=0)
    variance = np.divide(
        squared_deviation,
        counts - 1,
        out=np.full(traces.shape[1], np.nan),
        where=counts > 1,
    )
    sem = np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.full(traces.shape[1], np.nan),
        where=counts > 1,
    )
    return mean, sem


def plot_running_aligned_z(data: FigureData, output_dir: Path) -> Path:
    """Write the mean onset-aligned z displacement and SEM for one dataset."""
    neural_time_s = data.neural_time_min * 60
    wheel_time_s = data.wheel_time_min * 60
    experiment_boundaries_s = data.experiment_start_min[1:] * 60
    onset_times_s = _running_onset_times(
        wheel_time_s,
        data.wheel_velocity_cm_s,
        experiment_boundaries_s,
    )
    relative_time_s, aligned_z_um = _align_z_to_running_onsets(
        neural_time_s,
        data.z_movement_um,
        onset_times_s,
        experiment_boundaries_s,
    )
    mean_z_um, sem_z_um = _mean_and_sem(aligned_z_um)

    figure, axis = plt.subplots(figsize=(4.5, 3.25), constrained_layout=True)
    axis.fill_between(
        relative_time_s,
        mean_z_um - sem_z_um,
        mean_z_um + sem_z_um,
        color=HISTOGRAM_COLOR,
        alpha=0.25,
        linewidth=0,
        label="SEM",
    )
    axis.plot(
        relative_time_s,
        mean_z_um,
        color=HISTOGRAM_COLOR,
        linewidth=1.2,
        label="Mean",
    )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.axhline(0, color="0.75", linewidth=0.6, zorder=0)
    axis.text(
        0.98,
        0.96,
        f"{aligned_z_um.shape[0]} onsets",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    axis.set_xlabel("Time from running onset (s)")
    axis.set_ylabel(r"Z displacement ($\mu$m)")
    axis.set_xlim(relative_time_s[0], relative_time_s[-1])
    axis.legend(frameon=False, loc="lower right")
    _finish_axis(axis)
    output_path = output_dir / f"{data.dataset.slug}_running.pdf"
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return output_path


def _offset_profile_pairs(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Center paired profiles and place them at evenly spaced x positions."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Profile pairs must have identical (depth, rois) shapes")

    centers = np.full(first.shape[1], np.nan)
    spans = np.full(first.shape[1], np.nan)
    for roi_index in range(first.shape[1]):
        values = np.concatenate((first[:, roi_index], second[:, roi_index]))
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Example ROI {roi_index} has no finite profile values")
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        centers[roi_index] = 0.5 * (minimum + maximum)
        spans[roi_index] = maximum - minimum
    maximum_span = float(np.max(spans))
    spacing = 1.25 * maximum_span if maximum_span > 0 else 1.0
    positions = np.arange(first.shape[1], dtype=float) * spacing
    offsets = positions - centers
    return first + offsets, second + offsets, positions, spacing


def plot_z_profile_examples(data: ZProfileFigureData, output_dir: Path) -> Path:
    """Write all selected z-stack and median-F profile pairs on one axis."""
    num_rois = len(data.selected_rois)
    if num_rois == 0:
        raise ValueError("At least one example ROI is required")
    expected_shape = (data.depth_um.size, num_rois)
    if (
        data.zstack_profiles.shape != expected_shape
        or data.median_f_profiles.shape != expected_shape
        or data.correlations.shape != (num_rois,)
    ):
        raise ValueError("Example z-profile arrays do not align with the selected ROIs")
    depth_min_um, depth_max_um = EXAMPLE_PROFILE_DEPTH_RANGE_UM
    displayed_depths = (data.depth_um >= depth_min_um) & (data.depth_um <= depth_max_um)
    if not np.any(displayed_depths):
        raise ValueError("The requested example depth range is outside the z-stack")
    if not np.any(np.isfinite(data.median_f_profiles[displayed_depths])):
        raise ValueError("No displayed depth has enough fluorescence samples")
    zstack_profiles, median_f_profiles, positions, spacing = _offset_profile_pairs(
        data.zstack_profiles,
        data.median_f_profiles,
    )

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for roi_index in range(num_rois):
        axis.plot(
            zstack_profiles[:, roi_index],
            data.depth_um,
            color=ZSTACK_PROFILE_COLOR,
            linewidth=1.2,
            label="Z-stack profile" if roi_index == 0 else None,
        )
        axis.plot(
            median_f_profiles[:, roi_index],
            data.depth_um,
            color=MEDIAN_F_PROFILE_COLOR,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label=(
                f"Median F (>={MIN_PROFILE_SAMPLES_PER_DEPTH} samples/depth)"
                if roi_index == 0
                else None
            ),
        )
    tick_labels = []
    for roi, correlation in zip(data.selected_rois, data.correlations, strict=True):
        correlation_label = f"{correlation:.2f}" if np.isfinite(correlation) else "n/a"
        tick_labels.append(f"P{roi.plane_id} R{roi.roi_id}\nr = {correlation_label}")
    axis.set_xticks(positions, tick_labels)
    axis.set_xlim(positions[0] - 0.6 * spacing, positions[-1] + 0.6 * spacing)
    axis.set_ylim(depth_max_um, depth_min_um)
    axis.set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    axis.set_xlabel("Example ROI (profiles offset along x)")
    axis.set_ylabel(r"Z-stack depth ($\mu$m)")
    axis.legend(frameon=False, loc="upper right")
    _finish_axis(axis)
    output_path = output_dir / f"{data.dataset.slug}_zprofile_median_f_examples.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _pool_metric(
    datasets: Sequence[TraceDifferenceData],
    attribute: str,
    *,
    minimum_exclusive: float | None = None,
) -> tuple[np.ndarray, int]:
    """Pool finite per-ROI values and count datasets contributing to the pool."""
    pooled = []
    num_datasets = 0
    for data in datasets:
        values = np.asarray(getattr(data, attribute), dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if minimum_exclusive is not None:
            values = values[values > minimum_exclusive]
        if values.size:
            pooled.append(values)
            num_datasets += 1
    if not pooled:
        raise ValueError(f"No valid ROI values are available for {attribute}")
    return np.concatenate(pooled), num_datasets


def plot_z_profile_correlation_histogram(
    datasets: Sequence[ZProfilePopulationData], output_dir: Path
) -> Path:
    """Write the pooled distribution of z-stack-versus-median-F correlations."""
    pooled = []
    num_datasets = 0
    for data in datasets:
        values = np.asarray(data.correlations, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            pooled.append(values)
            num_datasets += 1
    if not pooled:
        raise ValueError("No valid z-profile correlations are available")
    correlations = np.concatenate(pooled)
    if np.any((correlations < -1 - 1e-12) | (correlations > 1 + 1e-12)):
        raise ValueError("Pearson correlations must lie between -1 and 1")
    correlations = np.clip(correlations, -1, 1)
    edges = np.linspace(-1, 1, 41)
    weights = np.full(correlations.size, 100.0 / correlations.size)
    median = float(np.median(correlations))
    lower_quartile, upper_quartile = np.percentile(correlations, [25, 75])

    figure, axis = plt.subplots(figsize=(4.5, 3.25), constrained_layout=True)
    axis.hist(
        correlations,
        bins=edges,
        weights=weights,
        color=HISTOGRAM_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.axvline(0, color="0.75", linewidth=0.7, zorder=0)
    axis.plot(
        median,
        1.02,
        marker="v",
        markersize=6,
        color="black",
        linestyle="none",
        transform=axis.get_xaxis_transform(),
        clip_on=False,
    )
    axis.annotate(
        f"median = {median:.2f}\nIQR = [{lower_quartile:.2f}, {upper_quartile:.2f}]",
        xy=(median, 1.02),
        xycoords=axis.get_xaxis_transform(),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        annotation_clip=False,
    )
    axis.text(
        0.98,
        0.96,
        f"{correlations.size:,} ROIs\n{num_datasets} datasets\n"
        f">={MIN_PROFILE_SAMPLES_PER_DEPTH} samples/depth",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_xlabel("Pearson correlation: z-stack profile vs median F profile")
    axis.set_ylabel("ROIs (%)")
    axis.set_xlim(-1, 1)
    axis.set_xticks([-1, -0.5, 0, 0.5, 1])
    _finish_axis(axis)
    output_path = output_dir / "zprofile_median_f_correlations.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _pool_percentiles(
    datasets: Sequence[TraceDifferenceData],
    lower_attribute: str,
    upper_attribute: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool paired finite per-ROI percentiles across contributing datasets."""
    pooled_lower = []
    pooled_upper = []
    num_datasets = 0
    for data in datasets:
        lower = np.asarray(getattr(data, lower_attribute), dtype=float).reshape(-1)
        upper = np.asarray(getattr(data, upper_attribute), dtype=float).reshape(-1)
        if lower.shape != upper.shape:
            raise ValueError("Lower and upper percentile arrays must have identical shapes")
        valid = np.isfinite(lower) & np.isfinite(upper)
        if np.any(valid):
            pooled_lower.append(lower[valid])
            pooled_upper.append(upper[valid])
            num_datasets += 1
    if not pooled_lower:
        raise ValueError(
            f"No valid ROI percentiles are available for {lower_attribute} and {upper_attribute}"
        )
    return np.concatenate(pooled_lower), np.concatenate(pooled_upper), num_datasets


def _percentile_histogram_bin_edges(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Choose shared percentile bins on a linear scale with zero as an edge."""
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    values = np.concatenate((lower, upper))
    if lower.size == 0 or upper.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Percentile histogram values must be non-empty and finite")

    minimum = min(0.0, float(np.min(values)))
    maximum = max(0.0, float(np.max(values)))
    if minimum == maximum:
        epsilon = np.finfo(float).eps
        return np.asarray([-epsilon, 0.0, epsilon])

    edges = np.histogram_bin_edges(values, bins="auto", range=(minimum, maximum))
    if edges.size - 1 > PERCENTILE_HISTOGRAM_MAX_BINS:
        edges = np.linspace(minimum, maximum, PERCENTILE_HISTOGRAM_MAX_BINS + 1)
    if not np.any(edges == 0.0):
        edges = np.sort(np.append(edges, 0.0))
    return edges


def _histogram_bin_edges(
    values: np.ndarray,
    bin_width: float,
    maximum: float | None = None,
) -> np.ndarray:
    """Choose round, fixed-width bin edges for a dynamic or fixed range."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Histogram values must be non-empty and finite")
    if np.any(values < 0):
        raise ValueError("Histogram values cannot be negative")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Histogram bin width must be positive and finite")

    if maximum is None:
        data_maximum = float(np.max(values))
        num_bins = max(1, int(np.ceil(data_maximum / bin_width)))
        maximum = num_bins * bin_width
    else:
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Histogram maximum must be positive and finite")
        num_bins = int(round(maximum / bin_width))
        if num_bins < 1 or not np.isclose(num_bins * bin_width, maximum):
            raise ValueError("Histogram maximum must be a multiple of its bin width")
    return np.linspace(0.0, maximum, num_bins + 1)


def _log_histogram_bin_edges(values: np.ndarray, bins_per_decade: int) -> np.ndarray:
    """Choose logarithmically spaced edges spanning positive values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Histogram values must be non-empty and finite")
    if np.any(values <= 0):
        raise ValueError("Logarithmic histogram values must be positive")
    if bins_per_decade < 1:
        raise ValueError("Logarithmic bins per decade must be positive")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        minimum /= np.sqrt(10.0)
        maximum *= np.sqrt(10.0)
    num_decades = np.log10(maximum / minimum)
    num_bins = max(1, int(np.ceil(num_decades * bins_per_decade)))
    return np.geomspace(minimum, maximum, num_bins + 1)


def _round_ratio_ticks(log2_minimum: float, log2_maximum: float) -> tuple[np.ndarray, list[str]]:
    """Choose round ratio ticks and return their positions on a log2 axis."""
    ratio_minimum, ratio_maximum = np.exp2([log2_minimum, log2_maximum])
    locator = mpl.ticker.MaxNLocator(
        nbins=8,
        steps=[1, 2, 2.5, 4, 5, 10],
    )
    ratio_ticks = locator.tick_values(ratio_minimum, ratio_maximum)
    tolerance = max(abs(ratio_minimum), abs(ratio_maximum)) * 1e-10
    ratio_ticks = ratio_ticks[
        (ratio_ticks > 0)
        & (ratio_ticks >= ratio_minimum - tolerance)
        & (ratio_ticks <= ratio_maximum + tolerance)
    ]
    if ratio_ticks.size < 2:
        ratio_ticks = np.asarray([ratio_minimum, ratio_maximum])

    step = float(np.min(np.diff(ratio_ticks)))
    precision = 0
    while precision < 4 and not np.isclose(step, round(step, precision)):
        precision += 1
    labels = [f"{tick:.{precision}f}" for tick in ratio_ticks]
    return np.log2(ratio_ticks), labels


def _plot_percentile_histograms(
    lower: np.ndarray,
    upper: np.ndarray,
    num_datasets: int,
    output_path: Path,
    *,
    xlabel: str,
    ratio_tick_labels: bool = False,
    median_precision: int = 3,
    additional_statistics: str | None = None,
) -> Path:
    """Write paired per-ROI 5th- and 95th-percentile distributions."""
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper percentile arrays must have identical shapes")
    edges = _percentile_histogram_bin_edges(lower, upper)
    weights = np.full(lower.size, 100.0 / lower.size)
    lower_median = float(np.median(lower))
    upper_median = float(np.median(upper))

    figure, axis = plt.subplots(figsize=(4.5, 3.25), constrained_layout=True)
    axis.hist(
        lower,
        bins=edges,
        weights=weights,
        histtype="step",
        linewidth=1.4,
        color=LOWER_PERCENTILE_COLOR,
        label="5th percentile",
    )
    axis.hist(
        upper,
        bins=edges,
        weights=weights,
        histtype="step",
        linewidth=1.4,
        color=UPPER_PERCENTILE_COLOR,
        label="95th percentile",
    )
    axis.axvline(0, color="0.75", linewidth=0.7, zorder=0)
    for median, color, height in (
        (lower_median, LOWER_PERCENTILE_COLOR, 1.02),
        (upper_median, UPPER_PERCENTILE_COLOR, 1.08),
    ):
        axis.plot(
            median,
            height,
            marker="v",
            markersize=6,
            color=color,
            linestyle="none",
            transform=axis.get_xaxis_transform(),
            clip_on=False,
        )
        displayed_median = float(np.exp2(median)) if ratio_tick_labels else median
        axis.annotate(
            f"{displayed_median:.{median_precision}f}",
            xy=(median, height),
            xycoords=axis.get_xaxis_transform(),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
            annotation_clip=False,
        )
    statistics = f"{lower.size:,} ROIs\n{num_datasets} datasets"
    if additional_statistics:
        statistics += f"\n{additional_statistics}"
    axis.text(
        0.98,
        0.96,
        statistics,
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("ROIs (%)")
    if ratio_tick_labels:
        tick_positions, tick_labels = _round_ratio_ticks(edges[0], edges[-1])
        axis.set_xticks(tick_positions, tick_labels)
    axis.set_xlim(edges[0], edges[-1])
    axis.legend(frameon=False, loc="upper left")
    _finish_axis(axis)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_metric_histogram(
    values: np.ndarray,
    num_datasets: int,
    output_path: Path,
    *,
    xlabel: str,
    bin_width: float | None = None,
    log_bins_per_decade: int | None = None,
    maximum: float | None = None,
    xticks: np.ndarray | None = None,
    final_tick_label: str | None = None,
    median_precision: int = 3,
    percentage_denominator: int | None = None,
) -> Path:
    """Write one pooled per-ROI distribution with a marked median."""
    if bin_width is not None and log_bins_per_decade is not None:
        raise ValueError("Specify either linear bin width or logarithmic bins")
    if bin_width is not None:
        edges = _histogram_bin_edges(values, bin_width, maximum)
    elif log_bins_per_decade is not None:
        if maximum is not None:
            raise ValueError("A fixed maximum is not supported for logarithmic bins")
        edges = _log_histogram_bin_edges(values, log_bins_per_decade)
    else:
        raise ValueError("Specify either linear bin width or logarithmic bins")
    histogram_values = np.minimum(values, maximum) if maximum is not None else values
    if percentage_denominator is None:
        percentage_denominator = values.size
    if percentage_denominator < values.size:
        raise ValueError("Percentage denominator cannot be smaller than plotted ROI count")
    weights = np.full(values.size, 100.0 / percentage_denominator)
    median = float(np.median(values))
    median_position = min(median, edges[-1])

    figure, axis = plt.subplots(figsize=(4.5, 3.25), constrained_layout=True)
    axis.hist(
        histogram_values,
        bins=edges,
        weights=weights,
        color=HISTOGRAM_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.plot(
        median_position,
        1.02,
        marker="v",
        markersize=6,
        color="black",
        linestyle="none",
        transform=axis.get_xaxis_transform(),
        clip_on=False,
    )
    axis.annotate(
        f"median = {median:.{median_precision}f}",
        xy=(median_position, 1.02),
        xycoords=axis.get_xaxis_transform(),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        annotation_clip=False,
    )
    axis.text(
        0.98,
        0.96,
        f"{values.size:,} ROIs\n{num_datasets} datasets",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("ROIs (%)")
    if log_bins_per_decade is not None:
        axis.set_xscale("log")
    axis.set_xlim(edges[0], edges[-1])
    if xticks is not None:
        tick_labels = [f"{tick:g}" for tick in xticks]
        if final_tick_label is not None:
            tick_labels[-1] = final_tick_label
        axis.set_xticks(xticks, tick_labels)
    elif final_tick_label is not None:
        raise ValueError("A final tick label requires explicit tick positions")
    _finish_axis(axis)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_fluorescence_ratio_percentiles(
    datasets: Sequence[TraceDifferenceData],
    output_dir: Path,
) -> Path:
    """Write pooled fluorescence log-ratio percentile histograms."""
    lower, upper, num_datasets = _pool_percentiles(
        datasets,
        "fluorescence_log_ratio_q05",
        "fluorescence_log_ratio_q95",
    )
    lower_below_threshold_percent = 100.0 * np.mean(lower < np.log2(0.8))
    upper_above_threshold_percent = 100.0 * np.mean(upper > np.log2(1.2))
    return _plot_percentile_histograms(
        lower,
        upper,
        num_datasets,
        output_dir / "fluorescence_ratio_percentiles.pdf",
        xlabel="Corrected / uncorrected fluorescence",
        ratio_tick_labels=True,
        median_precision=2,
        additional_statistics=(
            f"5th percentile < 0.8: {lower_below_threshold_percent:.1f}% of ROIs\n"
            f"95th percentile > 1.2: {upper_above_threshold_percent:.1f}% of ROIs"
        ),
    )


def plot_dff_difference_percentiles(
    datasets: Sequence[TraceDifferenceData],
    output_dir: Path,
) -> Path:
    """Write pooled signed dF/F-difference percentile histograms."""
    lower, upper, num_datasets = _pool_percentiles(
        datasets,
        "dff_difference_q05",
        "dff_difference_q95",
    )
    lower_below_005_percent = 100.0 * np.mean(lower < -0.05)
    lower_below_010_percent = 100.0 * np.mean(lower < -0.1)
    upper_above_005_percent = 100.0 * np.mean(upper > 0.05)
    upper_above_010_percent = 100.0 * np.mean(upper > 0.1)
    return _plot_percentile_histograms(
        lower,
        upper,
        num_datasets,
        output_dir / "dff_difference_percentiles.pdf",
        xlabel="Corrected - uncorrected delta-F-over-F",
        additional_statistics=(
            f"5th percentile < -0.05: {lower_below_005_percent:.1f}% of ROIs\n"
            f"5th percentile < -0.1: {lower_below_010_percent:.1f}% of ROIs\n"
            f"95th percentile > 0.05: {upper_above_005_percent:.1f}% of ROIs\n"
            f"95th percentile > 0.1: {upper_above_010_percent:.1f}% of ROIs"
        ),
    )


def plot_z_movement_removed_percentages(
    datasets: Sequence[TraceDifferenceData],
    output_dir: Path,
) -> Path:
    """Write nonzero percentages of each trace removed by z-movement."""
    all_values, _ = _pool_metric(datasets, "z_movement_removed_percent")
    values, num_datasets = _pool_metric(
        datasets,
        "z_movement_removed_percent",
        minimum_exclusive=0.0,
    )
    return _plot_metric_histogram(
        values,
        num_datasets,
        output_dir / "z_movement_removed_percentages.pdf",
        xlabel="Trace removed by z-movement (%)",
        log_bins_per_decade=TRACE_REMOVAL_LOG_BINS_PER_DECADE,
        median_precision=1,
        percentage_denominator=all_values.size,
    )


def generate_figures(
    datasets_csv: Path = DATASETS_CSV,
    processed_root: Path = NEURONS_ROOT,
    suite2p_root: Path = NEURONS_ROOT,
    bonsai_root: Path = NEURONS_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    style: str | None = None,
) -> list[Path]:
    """Generate the dF/F-difference percentile panel for Figure 3."""
    if style:
        plt.style.use(style)
    mpl.rcParams.update(FIGURE_STYLE)
    output_dir = Path(output_dir)
    if output_dir.drive.upper() == "Z:":
        raise ValueError("Refusing to write Figure 3 outputs to the Z: drive")
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dataset = Dataset(subject=TRACE_SUBJECT, date=TRACE_DATE)
    trace_data = prepare_figure_data(
        trace_dataset,
        processed_root,
        suite2p_root,
        bonsai_root,
        TRACE_PLANE_ID,
    )
    z_profile_data = prepare_z_profile_figure_data(
        trace_dataset,
        processed_root,
        suite2p_root,
        TRACE_PLANE_ID,
    )
    datasets = discover_datasets(datasets_csv)
    z_profile_population_data = [
        calculate_z_profile_population_data(dataset, processed_root, suite2p_root)
        for dataset in datasets
    ]
    trace_difference_data = [
        calculate_trace_difference_data(dataset, processed_root, suite2p_root)
        for dataset in datasets
    ]
    return [
        plot_fluorescence_correction_example(trace_data, output_dir),
        plot_running_aligned_z(trace_data, output_dir),
        plot_z_profile_examples(z_profile_data, output_dir),
        plot_z_profile_correlation_histogram(z_profile_population_data, output_dir),
        plot_fluorescence_ratio_percentiles(trace_difference_data, output_dir),
        plot_dff_difference_percentiles(trace_difference_data, output_dir),
        plot_z_movement_removed_percentages(trace_difference_data, output_dir),
    ]


def main() -> None:
    generate_figures(
        datasets_csv=DATASETS_CSV,
        processed_root=NEURONS_ROOT,
        suite2p_root=NEURONS_ROOT,
        bonsai_root=NEURONS_ROOT,
        output_dir=OUTPUT_DIR,
        style=None,
    )


if __name__ == "__main__":
    main()
