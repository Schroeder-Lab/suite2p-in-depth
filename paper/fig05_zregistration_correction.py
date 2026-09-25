"""Plot reference-plane and z-registered traces for paper Figure 5."""

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

from suite2p_in_depth import suite2p_compat

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

BOUTONS_ROOT = DATA_ROOT / "boutons"
DATASETS_CSV = BOUTONS_ROOT / "boutons.csv"
OUTPUT_DIR = PLOTS_ROOT / "Fig05"
CURATED_RECORDINGS = (
    ("Oephelia", "2023-07-21"),
    ("Memphis", "2023-08-21"),
    ("Styx", "2024-02-26"),
    ("Stereopes", "2024-02-13"),
    ("Vesta", "2024-05-08"),
    ("SS123", "2024-09-12"),
)

TRACE_SUBJECT = "Stereopes"
TRACE_DATE = "2024-02-13"
TRACE_ROI_IDS = (0, 11, 1, 4, 15, 3, 72, 129, 89, 42)
PREPROCESS_MANIFEST = "fig05_preprocess.json"
PREPROCESS_SCHEMA_VERSION = 1
MIN_PROFILE_SAMPLES_PER_DEPTH = 100
ROI_CHUNK_SIZE = 64
PERCENTILE_HISTOGRAM_MAX_BINS = 80
WHEEL_SMOOTHING_SIGMA_S = 1.0
BEST_PLANE_SMOOTHING_SIGMA_POINTS = 10.0
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
REFERENCE_COLOR = "#6b7280"
ZREGISTERED_COLOR = "#c23b22"
HISTOGRAM_COLOR = "#4c78a8"
LOWER_PERCENTILE_COLOR = "#3b6fb6"
UPPER_PERCENTILE_COLOR = "#c23b22"
BOUNDARY_COLOR = "0.72"


@dataclass(frozen=True)
class Dataset:
    """Identity of one selected recording."""

    subject: str
    date: str

    @property
    def slug(self) -> str:
        return f"{self.subject}_{self.date}"


@dataclass(frozen=True)
class SelectedRoi:
    """One corrected-array column and its original plane_z ROI identity."""

    output_index: int
    roi_id: int

    @property
    def label(self) -> str:
        return f"ROI {self.roi_id}"


@dataclass(frozen=True)
class DatasetInputs:
    """Validated memory-mapped inputs shared by example and pooled analyses."""

    dataset: Dataset
    dark_offset: float
    neural_time_s: np.ndarray
    wheel_time_s: np.ndarray
    wheel_velocity_cm_s: np.ndarray
    experiment_boundary_s: np.ndarray
    roi_ids: np.ndarray
    zregistered_fluorescence: np.ndarray
    reference_fluorescence: np.ndarray
    zregistered_dff: np.ndarray
    reference_dff: np.ndarray
    zregistered_badframes: np.ndarray
    reference_badframes: np.ndarray
    best_plane_ids: np.ndarray
    plane_ids: np.ndarray
    reference_plane_id: int


@dataclass(frozen=True)
class FigureData:
    """Fully prepared inputs for the Stereopes example figures."""

    dataset: Dataset
    neural_time_min: np.ndarray
    wheel_time_min: np.ndarray
    experiment_boundary_min: np.ndarray
    wheel_velocity_cm_s: np.ndarray
    best_plane_ids: np.ndarray
    plane_ids: np.ndarray
    reference_plane_id: int
    reference_fluorescence: np.ndarray
    zregistered_fluorescence: np.ndarray
    fluorescence_log_ratio_q05: np.ndarray
    fluorescence_log_ratio_q95: np.ndarray
    selected_rois: tuple[SelectedRoi, ...]


@dataclass(frozen=True)
class TraceDifferenceData:
    """Per-ROI correction measures for one dataset."""

    dataset: Dataset
    fluorescence_log_ratio_q05: np.ndarray
    fluorescence_log_ratio_q95: np.ndarray
    dff_difference_q05: np.ndarray
    dff_difference_q95: np.ndarray


@dataclass(frozen=True)
class ZProfileFigureData:
    """Pooled calcium depth profiles and reference-volume bounds for example ROIs."""

    dataset: Dataset
    depth_planes: np.ndarray
    median_f_profiles: np.ndarray
    sample_counts: np.ndarray
    reference_depth_planes: np.ndarray
    functional_channel: int
    selected_rois: tuple[SelectedRoi, ...]


def discover_datasets(
    datasets_csv: Path = DATASETS_CSV,
) -> list[Dataset]:
    """Require the curated bouton recordings in the manuscript table."""
    datasets_csv = Path(datasets_csv)
    if not datasets_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {datasets_csv}")
    available = set()
    with datasets_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Date"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset CSV must contain columns {sorted(required)}: {datasets_csv}")
        for row in reader:
            available.add((str(row["Name"]).strip(), str(row["Date"]).strip()))
    missing = set(CURATED_RECORDINGS) - available
    if missing:
        raise ValueError(f"Curated recordings missing from {datasets_csv}: {sorted(missing)}")
    return [Dataset(subject, date) for subject, date in CURATED_RECORDINGS]


def _load_vector(path: Path, description: str, *, allow_nan: bool = False) -> np.ndarray:
    """Load a non-empty one-dimensional numeric array."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if values.size == 0 or np.any(np.isinf(values)):
        raise ValueError(f"{description} must be non-empty and free of infinities: {path}")
    if not allow_nan and np.any(np.isnan(values)):
        raise ValueError(f"{description} must be finite: {path}")
    return values


def _load_integer_vector(path: Path, description: str) -> np.ndarray:
    """Load a finite, integer-valued one-dimensional array."""
    values = _load_vector(path, description)
    if not np.all(values == np.round(values)):
        raise ValueError(f"{description} must contain only integer values: {path}")
    return values.astype(int)


def _load_matrix(path: Path, description: str) -> np.ndarray:
    """Memory-map a non-empty two-dimensional NumPy array."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{description} must be a non-empty 2D array: {path}")
    return values


def _strictly_increasing(values: np.ndarray, description: str) -> None:
    """Require a finite, strictly increasing timestamp vector."""
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{description} must be finite and strictly increasing")


def _manifest_dark_offset(manifest: Mapping[str, Any], path: Path) -> float:
    """Read and validate the dark offset recorded with generated data."""
    try:
        dark_offset = float(manifest["dark_offset"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Manifest does not contain a numeric dark_offset: {path}") from error
    if not np.isfinite(dark_offset) or dark_offset > 0:
        raise ValueError(f"Manifest dark_offset must be finite and non-positive: {path}")
    return dark_offset


def _validate_preprocess_manifest(
    backup_dir: Path,
    dataset: Dataset,
    reference_plane: int,
    num_frames: int,
) -> float:
    """Require preprocessing products made for the current dataset and inputs."""
    path = backup_dir / PREPROCESS_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Figure 5 preprocessing manifest not found: {path}. "
            "Run paper/fig05_preprocess.py before plotting."
        )
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "dataset": dataset.slug,
        "reference_plane": reference_plane,
        "num_frames": num_frames,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Figure 5 preprocessing manifest does not match inputs: {mismatches}")
    return _manifest_dark_offset(manifest, path)


def _registration_badframes(
    outputs: Mapping[str, Any],
    num_frames: int,
    description: str,
) -> np.ndarray:
    """Load and validate a Suite2p registration bad-frame mask."""
    if outputs.get("badframes") is None:
        raise ValueError(f"{description} does not contain a badframes mask")
    badframes = np.asarray(outputs["badframes"], dtype=bool).reshape(-1)
    if badframes.size != num_frames:
        raise ValueError(f"{description} badframes length does not match the recording")
    return badframes


def _ordered_physical_plane_ids(database: Mapping[str, Any]) -> np.ndarray:
    """Return acquired physical plane IDs after removing flyback planes."""
    num_planes = int(database["nplanes"])
    if num_planes <= 0:
        raise ValueError("plane_z database nplanes must be positive")
    ignored = database.get(
        "ignore_flyback_singleplanes",
        database.get("ignore_flyback", []),
    )
    ignored_ids = np.asarray(ignored, dtype=int).reshape(-1)
    if (
        np.unique(ignored_ids).size != ignored_ids.size
        or np.any(ignored_ids < 0)
        or np.any(ignored_ids >= num_planes)
    ):
        raise ValueError("plane_z database contains invalid flyback plane IDs")
    plane_ids = np.delete(np.arange(num_planes, dtype=int), ignored_ids)
    if plane_ids.size == 0:
        raise ValueError("No non-flyback imaging planes remain")
    return plane_ids


def _best_plane_trace(
    database: Mapping[str, Any],
    outputs: Mapping[str, Any],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load the saved source-plane trace for the final reference image."""
    plane_ids = _ordered_physical_plane_ids(database)
    if outputs.get("reference_plane") is None:
        raise ValueError("plane_z registration outputs do not contain reference_plane")
    reference_plane = int(outputs["reference_plane"])
    if np.count_nonzero(plane_ids == reference_plane) != 1:
        raise ValueError("The chosen reference plane is not a non-flyback imaging plane")
    if outputs.get("planes_across_time") is None:
        raise ValueError("plane_z registration outputs do not contain planes_across_time")
    saved_plane_ids = np.asarray(outputs["planes_across_time"], dtype=float).reshape(-1)
    if (
        saved_plane_ids.size != num_frames
        or not np.all(np.isfinite(saved_plane_ids))
        or not np.all(saved_plane_ids == np.round(saved_plane_ids))
    ):
        raise ValueError("planes_across_time must be a finite physical plane ID per frame")
    best_plane_ids = saved_plane_ids.astype(int)
    if not np.all(np.isin(best_plane_ids, plane_ids)):
        raise ValueError("planes_across_time contains a flyback or out-of-range plane ID")
    return best_plane_ids, plane_ids, reference_plane


def _experiment_boundary_times(
    neural_time_s: np.ndarray,
    database: Mapping[str, Any],
) -> np.ndarray:
    """Locate experiment transitions on the concatenated timestamp axis."""
    lengths = np.asarray(database["frames_per_folder"], dtype=int).reshape(-1)
    if lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("plane_z frames_per_folder must contain positive lengths")
    if int(lengths.sum()) != neural_time_s.size:
        raise ValueError("plane_z frames_per_folder does not reproduce the neural trace length")
    boundaries = np.cumsum(lengths)[:-1]
    return np.asarray(
        [0.5 * (neural_time_s[index - 1] + neural_time_s[index]) for index in boundaries],
        dtype=float,
    )


def _validate_roi_mapping(plane_z_dir: Path, processed_dir: Path) -> np.ndarray:
    """Validate and return the authoritative accepted plane_z ROI mapping."""
    roi_ids = _load_integer_vector(processed_dir / "2pRois.ids.npy", "ROI IDs")
    roi_planes = _load_integer_vector(
        processed_dir / "2pRois.2pPlanes.npy",
        "ROI plane IDs",
    )
    if roi_planes.shape != roi_ids.shape or not np.all(roi_planes == -1):
        raise ValueError("Z-registration ROI plane IDs must contain one -1 per accepted ROI")
    iscell_path = plane_z_dir / "iscell.npy"
    if not iscell_path.is_file():
        raise FileNotFoundError(f"Accepted ROI file not found: {iscell_path}")
    iscell = np.asarray(np.load(iscell_path))
    if iscell.ndim != 2 or iscell.shape[1] < 1:
        raise ValueError(f"iscell.npy must have shape (rois, columns): {iscell_path}")
    accepted_roi_ids = np.flatnonzero(iscell[:, 0].astype(bool))
    if not np.array_equal(roi_ids, accepted_roi_ids):
        raise ValueError("2pRois.ids.npy does not match the current accepted plane_z ROI order")
    return roi_ids


def _load_dataset_inputs(
    dataset: Dataset,
    processed_root: Path,
    bonsai_root: Path,
) -> DatasetInputs:
    """Load and validate one preprocessed Figure 5 recording."""
    processed_dir = Path(processed_root) / dataset.subject / dataset.date
    plane_z_dir = processed_dir / "suite2p" / "plane_z"
    bonsai_dir = Path(bonsai_root) / dataset.subject / dataset.date
    for directory in (processed_dir, plane_z_dir, bonsai_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    plane_database, _ = suite2p_compat.load_parameters(plane_z_dir)
    plane_outputs, _ = suite2p_compat.load_outputs(plane_z_dir)
    if plane_outputs is None:
        raise FileNotFoundError(f"Registration outputs not found in {plane_z_dir}")
    if plane_outputs.get("reference_plane") is None:
        raise ValueError("plane_z registration outputs do not contain reference_plane")
    reference_plane = int(plane_outputs["reference_plane"])
    backup_dir = plane_z_dir.parent / f"backup{reference_plane}"
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"Reference-plane backup directory not found: {backup_dir}")
    backup_outputs, _ = suite2p_compat.load_outputs(backup_dir)
    if backup_outputs is None:
        raise FileNotFoundError(f"Registration outputs not found in {backup_dir}")

    neural_time_s = _load_vector(
        bonsai_dir / "2pCalcium.timestamps.npy",
        "Concatenated neural timestamps",
    )
    wheel_time_s = _load_vector(
        bonsai_dir / "wheel.timestamps.npy",
        "Concatenated wheel timestamps",
    )
    wheel_velocity_cm_s = _load_vector(
        bonsai_dir / "wheel.velocity.npy",
        "Concatenated wheel velocity",
        allow_nan=True,
    )
    _strictly_increasing(neural_time_s, "Concatenated neural timestamps")
    _strictly_increasing(wheel_time_s, "Concatenated wheel timestamps")
    if wheel_time_s.shape != wheel_velocity_cm_s.shape:
        raise ValueError("Concatenated wheel timestamps and velocity do not align")
    num_frames = neural_time_s.size
    dark_offset = _validate_preprocess_manifest(backup_dir, dataset, reference_plane, num_frames)

    roi_ids = _validate_roi_mapping(plane_z_dir, processed_dir)
    zregistered_fluorescence = _load_matrix(plane_z_dir / "F.npy", "Z-registered fluorescence")
    reference_fluorescence = _load_matrix(backup_dir / "F.npy", "Reference-plane fluorescence")
    zregistered_dff = _load_matrix(processed_dir / "2pCalcium.dff.npy", "Z-registered dF/F")
    reference_dff = _load_matrix(backup_dir / "dff.npy", "Reference-plane dF/F")
    reference_dff_roi_ids = _load_integer_vector(
        backup_dir / "dff_roi_ids.npy", "Reference-plane dF/F ROI IDs"
    )
    if not np.array_equal(reference_dff_roi_ids, roi_ids):
        raise ValueError("Reference-plane dF/F ROI IDs do not match 2pRois.ids.npy")
    if (
        zregistered_fluorescence.shape[1] != num_frames
        or reference_fluorescence.shape[1] != num_frames
        or zregistered_dff.shape != (num_frames, roi_ids.size)
        or reference_dff.shape != (num_frames, roi_ids.size)
    ):
        raise ValueError("Trace arrays do not align with timestamps and the ROI mapping")
    if (
        roi_ids.size == 0
        or roi_ids[-1] >= zregistered_fluorescence.shape[0]
        or roi_ids[-1] >= reference_fluorescence.shape[0]
    ):
        raise ValueError("Accepted ROI IDs fall outside one of the fluorescence arrays")

    zregistered_badframes = _registration_badframes(
        plane_outputs, num_frames, "plane_z registration outputs"
    )
    reference_badframes = _registration_badframes(
        backup_outputs, num_frames, f"backup{reference_plane} registration outputs"
    )
    experiment_boundary_s = _experiment_boundary_times(neural_time_s, plane_database)
    best_plane_ids, plane_ids, saved_reference_plane = _best_plane_trace(
        plane_database, plane_outputs, num_frames
    )
    if saved_reference_plane != reference_plane:
        raise ValueError("Inconsistent reference-plane metadata")
    return DatasetInputs(
        dataset=dataset,
        dark_offset=dark_offset,
        neural_time_s=neural_time_s,
        wheel_time_s=wheel_time_s,
        wheel_velocity_cm_s=wheel_velocity_cm_s,
        experiment_boundary_s=experiment_boundary_s,
        roi_ids=roi_ids,
        zregistered_fluorescence=zregistered_fluorescence,
        reference_fluorescence=reference_fluorescence,
        zregistered_dff=zregistered_dff,
        reference_dff=reference_dff,
        zregistered_badframes=zregistered_badframes,
        reference_badframes=reference_badframes,
        best_plane_ids=best_plane_ids,
        plane_ids=plane_ids,
        reference_plane_id=reference_plane,
    )


def _select_rois(
    roi_ids: np.ndarray,
    selected_roi_ids: Sequence[int],
) -> tuple[SelectedRoi, ...]:
    """Resolve requested original plane_z ROI IDs to output columns."""
    if len(set(selected_roi_ids)) != len(selected_roi_ids):
        raise ValueError("Requested ROI IDs must be unique")
    selected = []
    for roi_id in selected_roi_ids:
        matches = np.flatnonzero(roi_ids == roi_id)
        if matches.size != 1:
            raise ValueError(f"Expected one saved ROI {roi_id}; found {matches.size}")
        selected.append(SelectedRoi(output_index=int(matches[0]), roi_id=int(roi_id)))
    return tuple(selected)


def _masked_fluorescence(
    inputs: DatasetInputs,
    roi_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected control/corrected fluorescence with independent bad-frame masks."""
    reference = np.asarray(inputs.reference_fluorescence[roi_ids, :].T, dtype=float)
    zregistered = np.asarray(inputs.zregistered_fluorescence[roi_ids, :].T, dtype=float)
    reference[inputs.reference_badframes, :] = np.nan
    zregistered[inputs.zregistered_badframes, :] = np.nan
    reference -= inputs.dark_offset
    zregistered -= inputs.dark_offset
    return reference, zregistered


def _masked_dff(
    inputs: DatasetInputs,
    output_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected control/corrected dF/F with independent bad-frame masks."""
    reference = np.asarray(inputs.reference_dff[:, output_columns], dtype=float)
    zregistered = np.asarray(inputs.zregistered_dff[:, output_columns], dtype=float)
    reference[inputs.reference_badframes, :] = np.nan
    zregistered[inputs.zregistered_badframes, :] = np.nan
    return reference, zregistered


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
    reference: np.ndarray,
    comparison: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate tails of log2(comparison / reference) for every ROI."""
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    valid = np.isfinite(reference) & np.isfinite(comparison) & (reference > 0) & (comparison > 0)
    log_ratio = np.full(reference.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(comparison, reference, out=log_ratio, where=valid)
        np.log2(log_ratio, out=log_ratio, where=valid)
    return _column_percentiles(log_ratio)


def _difference_percentiles(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate tails of comparison minus reference for every ROI."""
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("Trace pairs must have identical (time, rois) shapes")
    valid = np.isfinite(reference) & np.isfinite(comparison)
    difference = np.full(reference.shape, np.nan)
    np.subtract(comparison, reference, out=difference, where=valid)
    return _column_percentiles(difference)


def _smooth_wheel_velocity(
    wheel_time_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    experiment_boundaries_s: np.ndarray,
) -> np.ndarray:
    """Gaussian-smooth velocity separately within every experiment."""
    if wheel_time_s.shape != velocity_cm_s.shape:
        raise ValueError("Wheel timestamps and velocity must have identical shapes")
    _strictly_increasing(wheel_time_s, "Concatenated wheel timestamps")
    split_indices = np.searchsorted(wheel_time_s, experiment_boundaries_s, side="left")
    edges = np.concatenate(([0], split_indices, [wheel_time_s.size]))
    smoothed = np.full(velocity_cm_s.shape, np.nan)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        segment_time = wheel_time_s[start:stop]
        segment_values = velocity_cm_s[start:stop]
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
        np.divide(numerator, denominator, out=smoothed[start:stop], where=denominator > 0)
    return smoothed


def prepare_figure_data(
    dataset: Dataset,
    processed_root: Path = BOUTONS_ROOT,
    bonsai_root: Path = BOUTONS_ROOT,
    selected_roi_ids: Sequence[int] = TRACE_ROI_IDS,
) -> FigureData:
    """Load and calculate every example-panel input for one recording."""
    inputs = _load_dataset_inputs(dataset, processed_root, bonsai_root)
    selected_rois = _select_rois(inputs.roi_ids, selected_roi_ids)
    roi_ids = np.asarray([roi.roi_id for roi in selected_rois], dtype=int)
    reference_f, zregistered_f = _masked_fluorescence(inputs, roi_ids)
    fluorescence_q05, fluorescence_q95 = _log_ratio_percentiles(reference_f, zregistered_f)
    smoothed_velocity = _smooth_wheel_velocity(
        inputs.wheel_time_s,
        inputs.wheel_velocity_cm_s,
        inputs.experiment_boundary_s,
    )
    time_origin_s = float(inputs.neural_time_s[0])
    return FigureData(
        dataset=dataset,
        neural_time_min=(inputs.neural_time_s - time_origin_s) / 60,
        wheel_time_min=(inputs.wheel_time_s - time_origin_s) / 60,
        experiment_boundary_min=(inputs.experiment_boundary_s - time_origin_s) / 60,
        wheel_velocity_cm_s=smoothed_velocity,
        best_plane_ids=inputs.best_plane_ids,
        plane_ids=inputs.plane_ids,
        reference_plane_id=inputs.reference_plane_id,
        reference_fluorescence=reference_f,
        zregistered_fluorescence=zregistered_f,
        fluorescence_log_ratio_q05=fluorescence_q05,
        fluorescence_log_ratio_q95=fluorescence_q95,
        selected_rois=selected_rois,
    )


def _median_fluorescence_by_depth(
    fluorescence: np.ndarray,
    effective_depths: np.ndarray,
    badframes: np.ndarray,
    min_samples: int = MIN_PROFILE_SAMPLES_PER_DEPTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool individual measurements across planes before taking each ROI/depth median."""
    fluorescence = np.asarray(fluorescence, dtype=float)
    effective_depths = np.asarray(effective_depths, dtype=float)
    badframes = np.asarray(badframes, dtype=bool)
    if (
        fluorescence.ndim != 3
        or 0 in fluorescence.shape
        or effective_depths.shape != fluorescence.shape[:2]
        or badframes.shape != effective_depths.shape
    ):
        raise ValueError(
            "F must have shape (planes, cycles, rois), aligned with depths and badframes"
        )
    if (
        not np.all(np.isfinite(effective_depths))
        or not np.all(effective_depths == np.round(effective_depths))
        or min_samples < 1
    ):
        raise ValueError(
            "Depths must be finite integer plane steps and min_samples must be positive"
        )
    depths = np.arange(int(effective_depths.min()), int(effective_depths.max()) + 1)
    profiles = np.full((depths.size, fluorescence.shape[2]), np.nan)
    counts = np.zeros(profiles.shape, dtype=int)
    for depth_index, depth in enumerate(depths):
        values = fluorescence[(effective_depths == depth) & ~badframes]
        for roi_index in range(fluorescence.shape[2]):
            samples = values[:, roi_index]
            samples = samples[np.isfinite(samples)]
            counts[depth_index, roi_index] = samples.size
            if samples.size >= min_samples:
                profiles[depth_index, roi_index] = np.median(samples)
    return depths, profiles, counts


def prepare_z_profile_figure_data(
    dataset: Dataset,
    profile_root: Path = BOUTONS_ROOT,
    selected_roi_ids: Sequence[int] = TRACE_ROI_IDS,
) -> ZProfileFigureData:
    """Read the all-plane export without loading unrelated trace or behavior products."""
    directory = Path(profile_root) / dataset.subject / dataset.date / "z_profiles"
    manifest_path = directory / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != dataset.slug:
        raise ValueError("Z-profile export schema or dataset does not match")
    dark_offset = _manifest_dark_offset(manifest, manifest_path)
    roi_ids = _load_integer_vector(directory / "roi_ids.npy", "Exported ROI IDs")
    plane_ids = _load_integer_vector(directory / "plane_ids.npy", "Exported physical plane IDs")
    best_plane_ids = _load_integer_vector(directory / "best_plane_ids.npy", "Best-plane IDs")
    num_cycles = int(manifest["num_cycles"])
    if (
        roi_ids.size != int(manifest["num_rois"])
        or np.unique(roi_ids).size != roi_ids.size
        or np.any(roi_ids < 0)
        or not np.array_equal(plane_ids, manifest["plane_ids"])
        or np.any(np.diff(plane_ids) <= 0)
        or best_plane_ids.size != num_cycles
        or not np.all(np.isin(best_plane_ids, plane_ids))
        or manifest["reference_plane"] not in plane_ids
    ):
        raise ValueError("Z-profile ROI, plane or cycle metadata is inconsistent")
    selected_rois = _select_rois(roi_ids, selected_roi_ids)
    if not selected_rois:
        raise ValueError("At least one example ROI is required")
    rows = np.asarray([roi.output_index for roi in selected_rois], dtype=int)
    effective_depths = _load_matrix(directory / "effective_depth_planes.npy", "Effective depths")
    expected_depths = plane_ids[:, np.newaxis] - best_plane_ids
    if not np.array_equal(effective_depths, expected_depths):
        raise ValueError("Effective depths do not match physical plane minus best-plane IDs")
    reference_depths = _load_integer_vector(
        directory / "reference_depth_planes.npy", "Reference depths"
    )
    if not np.array_equal(reference_depths, plane_ids - int(manifest["reference_plane"])):
        raise ValueError("Reference depths do not match the selected reference plane")
    fluorescence = np.empty((plane_ids.size, num_cycles, rows.size), dtype=float)
    badframes = np.empty((plane_ids.size, num_cycles), dtype=bool)
    for plane_index, plane_id in enumerate(plane_ids):
        plane_directory = directory / f"plane{plane_id}"
        with (plane_directory / "manifest.json").open(encoding="utf-8") as stream:
            plane_manifest = json.load(stream)
        fingerprint = plane_manifest["fingerprint"]
        if (
            fingerprint["dataset"] != dataset.slug
            or fingerprint["plane_id"] != plane_id
            or not np.array_equal(fingerprint["roi_ids"], roi_ids)
            or plane_manifest["shape"] != [roi_ids.size, num_cycles]
        ):
            raise ValueError(f"Plane {plane_id} export identity or ROI order is inconsistent")
        values = _load_matrix(plane_directory / "F.npy", f"Plane {plane_id} fluorescence")
        if values.shape != (roi_ids.size, num_cycles):
            raise ValueError(f"Plane {plane_id} fluorescence dimensions do not match the export")
        fluorescence[plane_index] = np.asarray(values[rows], dtype=float).T - dark_offset
        excluded = np.load(plane_directory / "badframes.npy")
        if excluded.shape != (num_cycles,) or excluded.dtype != np.bool_:
            raise ValueError(f"Plane {plane_id} badframes must be one boolean per cycle")
        badframes[plane_index] = excluded
    depths, profiles, counts = _median_fluorescence_by_depth(
        fluorescence, effective_depths, badframes
    )
    return ZProfileFigureData(
        dataset=dataset,
        depth_planes=depths,
        median_f_profiles=profiles,
        sample_counts=counts,
        reference_depth_planes=reference_depths,
        functional_channel=int(manifest["functional_channel"]),
        selected_rois=selected_rois,
    )


def _reference_relative_profiles(profiles: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Divide each calcium profile by its supported, positive median at depth zero."""
    profiles = np.asarray(profiles, dtype=float)
    depths = np.asarray(depths)
    if profiles.ndim != 2 or depths.shape != (profiles.shape[0],):
        raise ValueError("Profile rows must align with the depth vector")
    reference_indices = np.flatnonzero(depths == 0)
    if reference_indices.size != 1:
        raise ValueError("Profiles must contain exactly one depth-zero bin")
    reference = profiles[int(reference_indices[0])]
    invalid = ~np.isfinite(reference) | (reference <= 0)
    if np.any(invalid):
        raise ValueError(
            "Depth-zero median fluorescence must be supported and positive for every example ROI; "
            f"invalid columns: {np.flatnonzero(invalid).tolist()}"
        )
    return profiles / reference


def plot_z_profile_examples(data: ZProfileFigureData, output_dir: Path) -> Path:
    """Plot median F(z)/F(0) with one shared fluorescence-ratio scale across ROIs."""
    num_rois = len(data.selected_rois)
    if (
        num_rois == 0
        or data.median_f_profiles.shape != (data.depth_planes.size, num_rois)
        or data.sample_counts.shape != data.median_f_profiles.shape
        or data.reference_depth_planes.size == 0
    ):
        raise ValueError("Example profile depths, counts and ROI columns do not align")
    median_profiles = np.where(
        data.sample_counts >= MIN_PROFILE_SAMPLES_PER_DEPTH, data.median_f_profiles, np.nan
    )
    median = _reference_relative_profiles(median_profiles, data.depth_planes)
    finite_ratios = median[np.isfinite(median)]
    spacing = 1.25 * max(1.0, float(np.ptp(finite_ratios)))
    positions = np.arange(num_rois, dtype=float) * spacing
    reference_min = float(data.reference_depth_planes.min())
    reference_max = float(data.reference_depth_planes.max())
    supported_depths = data.depth_planes[np.any(np.isfinite(median), axis=1)]
    depth_min = min(reference_min, float(supported_depths.min()))
    depth_max = max(reference_max, float(supported_depths.max()))
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.axhspan(
        reference_min, reference_max, color="0.94", label="Original reference range", zorder=0
    )
    for boundary in (reference_min, reference_max):
        axis.axhline(boundary, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.7, zorder=1)
    for roi_index, position in enumerate(positions):
        axis.vlines(position, depth_min, depth_max, color="0.82", linestyle=":", linewidth=0.6)
        axis.plot(
            median[:, roi_index] - 1.0 + position,
            data.depth_planes,
            color=ZREGISTERED_COLOR,
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=(
                f"Median calcium F (ch{data.functional_channel}; n >= {MIN_PROFILE_SAMPLES_PER_DEPTH})"
                if roi_index == 0
                else None
            ),
        )
    axis.set_xticks(positions, [roi.label for roi in data.selected_rois])
    scale_start = positions[-1] - 0.5
    scale_end = scale_start + 1.0
    scale_depth = depth_max + 0.5
    axis.plot([scale_start, scale_end], [scale_depth, scale_depth], color="black", linewidth=1.2)
    axis.text(
        0.5 * (scale_start + scale_end),
        scale_depth + 0.1,
        "1.0 F(z) / F(0)",
        ha="center",
        va="top",
        fontsize=8,
    )
    axis.set_xlim(
        min(positions[0] + float(finite_ratios.min()) - 1.0, scale_start) - 0.2 * spacing,
        max(positions[-1] + float(finite_ratios.max()) - 1.0, scale_end) + 0.2 * spacing,
    )
    axis.set_yticks(np.arange(int(np.floor(depth_min)), int(np.ceil(depth_max)) + 1))
    axis.set_ylim(depth_max + 0.9, depth_min - 0.3)
    axis.set_ylabel("Depth relative to reference plane (plane steps)")
    axis.set_xlabel("Example ROI (median F(z) / median F(0), offset; shared scale)")
    axis.set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    _finish_axis(axis)
    output_path = Path(output_dir) / f"{data.dataset.slug}_zprofile_median_f_examples.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def calculate_trace_difference_data(
    dataset: Dataset,
    processed_root: Path = BOUTONS_ROOT,
    bonsai_root: Path = BOUTONS_ROOT,
) -> TraceDifferenceData:
    """Calculate Figure 3-equivalent correction measures for every accepted ROI."""
    inputs = _load_dataset_inputs(dataset, processed_root, bonsai_root)
    num_rois = inputs.roi_ids.size
    fluorescence_q05 = np.full(num_rois, np.nan)
    fluorescence_q95 = np.full(num_rois, np.nan)
    dff_q05 = np.full(num_rois, np.nan)
    dff_q95 = np.full(num_rois, np.nan)
    for start in range(0, num_rois, ROI_CHUNK_SIZE):
        stop = min(start + ROI_CHUNK_SIZE, num_rois)
        output_columns = np.arange(start, stop, dtype=int)
        roi_ids = inputs.roi_ids[output_columns]
        reference_f, zregistered_f = _masked_fluorescence(inputs, roi_ids)
        chunk_q05, chunk_q95 = _log_ratio_percentiles(reference_f, zregistered_f)
        fluorescence_q05[output_columns] = chunk_q05
        fluorescence_q95[output_columns] = chunk_q95

        reference_dff, zregistered_dff = _masked_dff(inputs, output_columns)
        chunk_q05, chunk_q95 = _difference_percentiles(reference_dff, zregistered_dff)
        dff_q05[output_columns] = chunk_q05
        dff_q95[output_columns] = chunk_q95
    return TraceDifferenceData(
        dataset=dataset,
        fluorescence_log_ratio_q05=fluorescence_q05,
        fluorescence_log_ratio_q95=fluorescence_q95,
        dff_difference_q05=dff_q05,
        dff_difference_q95=dff_q95,
    )


def _finish_axis(axis: mpl.axes.Axes) -> None:
    """Apply restrained manuscript styling to one axis."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.8)
    axis.margins(x=0)


def _mark_experiment_boundaries(
    axes: Sequence[mpl.axes.Axes],
    experiment_boundaries_min: np.ndarray,
) -> None:
    """Draw experiment transitions consistently across stacked panels."""
    for axis in axes:
        for boundary in experiment_boundaries_min:
            axis.axvline(
                boundary,
                color=BOUNDARY_COLOR,
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )


def _smooth_best_plane_ids(data: FigureData) -> np.ndarray:
    """Gaussian-smooth best-plane IDs without crossing experiment boundaries."""
    split_indices = np.searchsorted(
        data.neural_time_min,
        data.experiment_boundary_min,
        side="left",
    )
    edges = np.concatenate(([0], split_indices, [data.neural_time_min.size]))
    smoothed = np.empty(data.best_plane_ids.shape, dtype=float)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        smoothed[start:stop] = gaussian_filter1d(
            data.best_plane_ids[start:stop].astype(float),
            sigma=BEST_PLANE_SMOOTHING_SIGMA_POINTS,
            mode="nearest",
        )
    return smoothed


def _plot_best_plane(axis: mpl.axes.Axes, data: FigureData) -> None:
    """Plot the saved physical source-plane trace for the final reference image."""
    axis.axhline(
        data.reference_plane_id,
        color=ZREGISTERED_COLOR,
        linewidth=0.5,
        alpha=0.6,
        zorder=0,
    )
    axis.plot(
        data.neural_time_min,
        data.best_plane_ids,
        color=ZREGISTERED_COLOR,
        linewidth=0.2,
        alpha=0.55,
    )
    axis.plot(
        data.neural_time_min,
        _smooth_best_plane_ids(data),
        color=ZREGISTERED_COLOR,
        linewidth=1.0,
    )
    axis.set_yticks(data.plane_ids)
    axis.set_ylabel("Best imaging plane")
    axis.invert_yaxis()
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
    reference: np.ndarray,
    zregistered: np.ndarray,
    selected_rois: Sequence[SelectedRoi],
    *,
    reference_label: str,
    zregistered_label: str,
    ylabel: str,
    lower_percentiles: np.ndarray,
    upper_percentiles: np.ndarray,
    time_mask: np.ndarray,
) -> None:
    """Plot paired traces with the same additive offset for each ROI."""
    lower_percentiles = np.asarray(lower_percentiles, dtype=float).reshape(-1)
    upper_percentiles = np.asarray(upper_percentiles, dtype=float).reshape(-1)
    if lower_percentiles.size != len(selected_rois) or upper_percentiles.size != len(selected_rois):
        raise ValueError("Two percentile values are required for every selected ROI")
    reference, zregistered = _scale_trace_pairs(reference, zregistered)
    levels, offsets = _stack_offsets(reference, zregistered)
    time_mask = np.asarray(time_mask, dtype=bool).reshape(-1)
    if time_mask.size != time_min.size or not np.any(time_mask):
        raise ValueError("Trace plot mask must select neural time points")
    for roi_index, offset in enumerate(offsets):
        axis.plot(
            time_min[time_mask],
            reference[time_mask, roi_index] + offset,
            color=REFERENCE_COLOR,
            linewidth=0.45,
            alpha=0.9,
            label=reference_label if roi_index == 0 else None,
            zorder=2,
        )
        axis.plot(
            time_min[time_mask],
            zregistered[time_mask, roi_index] + offset,
            color=ZREGISTERED_COLOR,
            linewidth=0.45,
            alpha=0.9,
            label=zregistered_label if roi_index == 0 else None,
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


def _neural_time_limits(data: FigureData) -> tuple[float, float]:
    """Return the complete shared neural time range represented by every trace."""
    start = float(data.neural_time_min[0])
    stop = float(data.neural_time_min[-1])
    if stop <= start:
        raise ValueError("The neural recording has an invalid time range")
    return start, stop


def plot_fluorescence_correction_example(
    data: FigureData,
    output_dir: Path,
) -> Path:
    """Write locomotion, best plane, and matched fluorescence as a PDF."""
    time_limits_min = _neural_time_limits(data)
    trace_mask = (data.neural_time_min >= time_limits_min[0]) & (
        data.neural_time_min <= time_limits_min[1]
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.5, 0.625, 5]},
    )
    axes[0].plot(
        data.wheel_time_min,
        data.wheel_velocity_cm_s,
        color="black",
        linewidth=0.6,
    )
    axes[0].axhline(0, color="0.8", linewidth=0.6, zorder=0)
    axes[0].set_ylabel("Velocity\n(cm/s)")
    axes[0].set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    _finish_axis(axes[0])
    _plot_best_plane(axes[1], data)
    _plot_trace_stack(
        axes[2],
        data.neural_time_min,
        data.reference_fluorescence,
        data.zregistered_fluorescence,
        data.selected_rois,
        reference_label="Reference-plane-only F",
        zregistered_label="Z-registered F",
        ylabel="Scaled fluorescence (a.u.)",
        lower_percentiles=data.fluorescence_log_ratio_q05,
        upper_percentiles=data.fluorescence_log_ratio_q95,
        time_mask=trace_mask,
    )
    _mark_experiment_boundaries(axes, data.experiment_boundary_min)
    axes[-1].set_xlabel("Time (min)")
    axes[-1].set_xlim(*time_limits_min)
    output_path = Path(output_dir) / f"{data.dataset.slug}_fluorescence_correction_example.pdf"
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
    _strictly_increasing(wheel_time_s, "Wheel timestamps")
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


def _align_best_plane_to_running_onsets(
    neural_time_s: np.ndarray,
    best_plane_ids: np.ndarray,
    onset_times_s: np.ndarray,
    experiment_boundaries_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline and align the best-plane trace around each running onset."""
    if neural_time_s.ndim != 1 or best_plane_ids.shape != neural_time_s.shape:
        raise ValueError("Neural timestamps and best-plane IDs must be aligned vectors")
    sample_interval_s = float(np.median(np.diff(neural_time_s)))
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("Cannot determine the neural sampling interval")
    num_before = int(round(RUNNING_WINDOW_BEFORE_S / sample_interval_s))
    num_after = int(round(RUNNING_WINDOW_AFTER_S / sample_interval_s))
    relative_time_s = np.arange(-num_before, num_after + 1, dtype=float) * sample_interval_s
    baseline = (relative_time_s >= -RUNNING_BASELINE_S) & (relative_time_s < 0)
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
            neural_time_s,
            best_plane_ids.astype(float),
        )
        baseline_value = float(np.mean(trace[baseline]))
        if np.isfinite(baseline_value):
            aligned.append(trace - baseline_value)
    if not aligned:
        raise ValueError("No running onset has a complete aligned best-plane window")
    return relative_time_s, np.asarray(aligned)


def _mean_and_sem(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the pointwise mean and sample SEM across onset-aligned traces."""
    if traces.ndim != 2 or traces.shape[0] == 0:
        raise ValueError("Onset-aligned traces must be a non-empty 2D array")
    valid = np.isfinite(traces)
    counts = valid.sum(axis=0)
    total = np.where(valid, traces, 0.0).sum(axis=0)
    mean = np.divide(total, counts, out=np.full(traces.shape[1], np.nan), where=counts > 0)
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


def plot_running_aligned_best_plane(data: FigureData, output_dir: Path) -> Path:
    """Write mean running-onset-aligned best-plane shift and SEM."""
    neural_time_s = data.neural_time_min * 60
    wheel_time_s = data.wheel_time_min * 60
    experiment_boundaries_s = data.experiment_boundary_min * 60
    onset_times_s = _running_onset_times(
        wheel_time_s,
        data.wheel_velocity_cm_s,
        experiment_boundaries_s,
    )
    relative_time_s, aligned_planes = _align_best_plane_to_running_onsets(
        neural_time_s,
        data.best_plane_ids,
        onset_times_s,
        experiment_boundaries_s,
    )
    mean_plane, sem_plane = _mean_and_sem(aligned_planes)
    figure, axis = plt.subplots(figsize=(4.5, 3.25), constrained_layout=True)
    axis.fill_between(
        relative_time_s,
        mean_plane - sem_plane,
        mean_plane + sem_plane,
        color=HISTOGRAM_COLOR,
        alpha=0.25,
        linewidth=0,
        label="SEM",
    )
    axis.plot(
        relative_time_s,
        mean_plane,
        color=HISTOGRAM_COLOR,
        linewidth=1.2,
        label="Mean",
    )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.axhline(0, color="0.75", linewidth=0.6, zorder=0)
    axis.text(
        0.98,
        0.96,
        f"{aligned_planes.shape[0]} onsets",
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_title(f"{data.dataset.subject} {data.dataset.date}", loc="left")
    axis.set_xlabel("Time from running onset (s)")
    axis.set_ylabel("Best-plane shift")
    axis.set_xlim(relative_time_s[0], relative_time_s[-1])
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower right")
    _finish_axis(axis)
    output_path = Path(output_dir) / f"{data.dataset.slug}_running.pdf"
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return output_path


def _pool_percentiles(
    datasets: Sequence[TraceDifferenceData],
    lower_attribute: str,
    upper_attribute: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool paired finite per-ROI percentiles across datasets."""
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
        raise ValueError(f"No valid ROI percentiles are available for {lower_attribute}")
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


def _round_ratio_ticks(
    log2_minimum: float,
    log2_maximum: float,
) -> tuple[np.ndarray, list[str]]:
    """Choose round ratio ticks and return their positions on a log2 axis."""
    ratio_minimum, ratio_maximum = np.exp2([log2_minimum, log2_maximum])
    locator = mpl.ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 4, 5, 10])
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
    return np.log2(ratio_ticks), [f"{tick:.{precision}f}" for tick in ratio_ticks]


def _plot_percentile_histograms(
    lower: np.ndarray,
    upper: np.ndarray,
    num_datasets: int,
    output_path: Path,
    *,
    xlabel: str,
    ratio_tick_labels: bool = False,
    median_precision: int = 3,
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
    axis.text(
        0.98,
        0.96,
        f"{lower.size:,} ROIs\n{num_datasets} datasets",
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


def plot_fluorescence_ratio_percentiles(
    datasets: Sequence[TraceDifferenceData], output_dir: Path
) -> Path:
    """Write pooled fluorescence log-ratio percentile histograms."""
    lower, upper, num_datasets = _pool_percentiles(
        datasets,
        "fluorescence_log_ratio_q05",
        "fluorescence_log_ratio_q95",
    )
    return _plot_percentile_histograms(
        lower,
        upper,
        num_datasets,
        Path(output_dir) / "fluorescence_ratio_percentiles.pdf",
        xlabel="Z-registered / reference-plane-only fluorescence",
        ratio_tick_labels=True,
        median_precision=2,
    )


def plot_dff_difference_percentiles(
    datasets: Sequence[TraceDifferenceData], output_dir: Path
) -> Path:
    """Write pooled signed dF/F-difference percentile histograms."""
    lower, upper, num_datasets = _pool_percentiles(
        datasets,
        "dff_difference_q05",
        "dff_difference_q95",
    )
    return _plot_percentile_histograms(
        lower,
        upper,
        num_datasets,
        Path(output_dir) / "dff_difference_percentiles.pdf",
        xlabel="Z-registered - reference-plane-only delta-F-over-F",
    )


def _find_dataset(
    datasets: Sequence[Dataset],
    subject: str,
    date: str,
) -> Dataset:
    """Resolve one unique example dataset from the Process=True recordings."""
    matches = [
        dataset for dataset in datasets if dataset.subject == subject and dataset.date == date
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Process=True dataset {subject} {date}; found {len(matches)}"
        )
    return matches[0]


def generate_figures(
    datasets_csv: Path = DATASETS_CSV,
    processed_root: Path = BOUTONS_ROOT,
    bonsai_root: Path = BOUTONS_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *,
    style: str | None = None,
    profile_root: Path = BOUTONS_ROOT,
) -> list[Path]:
    """Generate all example and population plots for Figure 5."""
    if style:
        plt.style.use(style)
    mpl.rcParams.update(FIGURE_STYLE)
    output_dir = Path(output_dir)
    if output_dir.drive.upper() == "Z:":
        raise ValueError("Refusing to write Figure 5 outputs to the Z: drive")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(datasets_csv)
    trace_dataset = _find_dataset(datasets, TRACE_SUBJECT, TRACE_DATE)
    data = prepare_figure_data(trace_dataset, processed_root, bonsai_root, TRACE_ROI_IDS)
    z_profile_data = prepare_z_profile_figure_data(trace_dataset, profile_root, TRACE_ROI_IDS)
    trace_difference_data = [
        calculate_trace_difference_data(dataset, processed_root, bonsai_root)
        for dataset in datasets
    ]
    return [
        plot_fluorescence_correction_example(data, output_dir),
        plot_running_aligned_best_plane(data, output_dir),
        plot_z_profile_examples(z_profile_data, output_dir),
        plot_fluorescence_ratio_percentiles(trace_difference_data, output_dir),
        plot_dff_difference_percentiles(trace_difference_data, output_dir),
    ]


def main() -> None:
    outputs = generate_figures(
        datasets_csv=DATASETS_CSV,
        processed_root=BOUTONS_ROOT,
        bonsai_root=BOUTONS_ROOT,
        output_dir=OUTPUT_DIR,
        style=None,
        profile_root=BOUTONS_ROOT,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
