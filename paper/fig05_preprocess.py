"""Prepare matched traces and all-plane z-profile inputs for paper Figure 5."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.registration import phase_cross_correlation
from suite2p import io
from suite2p.extraction import extract, masks

from suite2p_in_depth import (
    general,
    preprocess_traces,
    suite2p_compat,
    suite2p_extensions,
)

try:
    from paper.paths import DATA_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path(
    os.environ.get(
        "SUITE2P_IN_DEPTH_PRIVATE_ROOT", REPOSITORY_ROOT.parent / "suite2p-in-depth_private"
    )
)
DATASETS_CSV = PRIVATE_ROOT / "MethodsPaper" / "boutons.csv"
PREPROCESSING_CSV = PRIVATE_ROOT / "Preprocessing" / "preprocess_boutons.csv"
PROCESSED_ROOT = Path(os.environ.get("SUITE2P_IN_DEPTH_PROCESSED_ROOT", PRIVATE_ROOT / "processed"))
BONSAI_ROOT = Path(os.environ.get("SUITE2P_IN_DEPTH_BONSAI_ROOT", PRIVATE_ROOT / "bonsai"))
REGISTERED_ROOT = Path(
    os.environ.get("SUITE2P_IN_DEPTH_REGISTERED_ROOT", PRIVATE_ROOT / "registered")
)
OUTPUT_ROOT = DATA_ROOT / "boutons"
PROFILE_SCHEMA_VERSION = 1

MANIFEST_NAME = "fig05_preprocess.json"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MAX_REFERENCE_SHIFT_PX = 1.0
CURATED_RECORDINGS = (
    ("Oephelia", "2023-07-21"),
    ("Memphis", "2023-08-21"),
    ("Styx", "2024-02-26"),
    ("Stereopes", "2024-02-13"),
    ("Vesta", "2024-05-08"),
    ("SS123", "2024-09-12"),
)

logger = logging.getLogger("fig05_preprocess")


@dataclass(frozen=True)
class Dataset:
    """Identity and reviewed acquisition metadata for one selected recording."""

    subject: str
    date: str
    frame_rate_hz: float
    dark_offset: float

    @property
    def slug(self) -> str:
        """Return a filesystem-friendly dataset identifier."""
        return f"{self.subject}_{self.date}"


@dataclass(frozen=True)
class TraceParameters:
    """Settings required to reproduce the production trace calculation."""

    frame_rate_hz: float
    plane_rate_hz: float
    frames_per_folder: np.ndarray
    dark_offset: float
    f0_percentile: float
    f0_window_s: float


@dataclass(frozen=True)
class AlignmentQC:
    """Numeric checks relating the control movie to the plane_z coordinates."""

    reference_correlation: float
    reference_shift_y_px: float
    reference_shift_x_px: float
    archived_reference_correlation: float
    final_shift_x_max_abs_px: int
    final_shift_y_max_abs_px: int
    final_shift_checksum: str


@dataclass(frozen=True)
class TraceProducts:
    """Products of the shared non-z-stack fluorescence processing path."""

    dff: np.ndarray
    fluorescence_neuropil_corrected: np.ndarray
    regression_parameters: np.ndarray
    fluorescence_bin_values: np.ndarray
    neuropil_bin_values: np.ndarray


class RigidAlignedMovie:
    """Read a Suite2p binary while applying plane_z's final rigid shifts in memory."""

    def __init__(
        self,
        binary: io.BinaryFile,
        shape: tuple[int, int, int],
        y_offsets: np.ndarray,
        x_offsets: np.ndarray,
    ) -> None:
        self._binary = binary
        self.shape = shape
        self._y_offsets = y_offsets
        self._x_offsets = x_offsets

    def __getitem__(self, item: slice) -> np.ndarray:
        if not isinstance(item, slice):
            raise TypeError("RigidAlignedMovie only supports contiguous slice reads.")
        start, stop, step = item.indices(self.shape[0])
        if step != 1:
            raise ValueError("RigidAlignedMovie only supports a slice step of one.")
        frames = np.asarray(self._binary[start:stop]).copy()
        if not np.any(self._y_offsets[start:stop]) and not np.any(self._x_offsets[start:stop]):
            return frames
        return suite2p_extensions.shift_frames(
            torch.from_numpy(frames),
            yoff=self._y_offsets[start:stop],
            xoff=self._x_offsets[start:stop],
            device=torch.device("cpu"),
        )


def _load_reviewed_metadata(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Load reviewed frame rates and dark offsets from the preprocessing table."""
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing CSV not found: {path}")
    metadata: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Date", "Frame_rate", "Darkest_value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Preprocessing CSV must contain columns {sorted(required)}: {path}")
        for row in reader:
            subject = str(row["Name"]).strip()
            date = str(row["Date"]).strip()
            if not subject or not date:
                continue
            frame_rate = str(row["Frame_rate"]).strip()
            dark_offset = str(row["Darkest_value"]).strip()
            if not frame_rate or not dark_offset:
                continue
            key = (subject, date)
            value = (float(frame_rate), float(dark_offset))
            if key in metadata and metadata[key] != value:
                raise ValueError(f"Conflicting preprocessing metadata for {subject} {date}.")
            metadata[key] = value
    return metadata


def discover_datasets(datasets_csv: Path, preprocessing_csv: Path) -> list[Dataset]:
    """Select curated recordings and attach their reviewed trace metadata."""
    if not datasets_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {datasets_csv}")
    reviewed = _load_reviewed_metadata(preprocessing_csv)
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
    datasets = []
    for subject, date in CURATED_RECORDINGS:
        key = (subject, date)
        if key not in reviewed:
            raise ValueError(f"No reviewed preprocessing metadata for {subject} {date}.")
        frame_rate, dark_offset = reviewed[key]
        if frame_rate <= 0 or dark_offset > 0:
            raise ValueError(f"Invalid reviewed trace metadata for {subject} {date}.")
        datasets.append(Dataset(subject, date, frame_rate, dark_offset))
    return datasets


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load one saved NumPy dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in {path}.")
    return value


def _load_vector(path: Path, description: str, *, allow_nan: bool = False) -> np.ndarray:
    """Load and validate a one-dimensional numeric array."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{description} is empty: {path}")
    if not allow_nan and not np.all(np.isfinite(values)):
        raise ValueError(f"{description} contains non-finite values: {path}")
    return values


def _atomic_save(path: Path, value: Any) -> None:
    """Write one NumPy file atomically within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a JSON manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _ordered_physical_plane_ids(database: dict[str, Any]) -> np.ndarray:
    """Return non-flyback acquired plane IDs in their saved reference order."""
    num_planes = int(database["nplanes"])
    ignored = database.get(
        "ignore_flyback_singleplanes",
        database.get("ignore_flyback", []),
    )
    ignored_ids = np.asarray(ignored, dtype=int).reshape(-1)
    plane_ids = np.delete(np.arange(num_planes, dtype=int), ignored_ids)
    if plane_ids.size == 0:
        raise ValueError("No non-flyback plane IDs remain in plane_z metadata.")
    return plane_ids


def _select_reference_image(value: Any, index: int, description: str) -> np.ndarray:
    """Select a two-dimensional image from a Suite2p reference image field."""
    images = np.asarray(value)
    if images.ndim == 2:
        return np.asarray(images, dtype=float)
    if images.ndim == 3 and 0 <= index < images.shape[0]:
        return np.asarray(images[index], dtype=float)
    raise ValueError(f"{description} does not contain reference image index {index}.")


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate a finite Pearson correlation between two reference images."""
    if first.shape != second.shape:
        raise ValueError(f"Reference image shapes differ: {first.shape} versus {second.shape}.")
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Reference images do not share enough finite pixels.")
    first_values = first[valid]
    second_values = second[valid]
    if np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        raise ValueError("A reference image is constant.")
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _integer_offsets(value: Any, name: str, num_frames: int) -> np.ndarray:
    """Validate one saved sequence of Suite2p rigid offsets."""
    offsets = np.asarray(value, dtype=float).reshape(-1)
    if offsets.size != num_frames or not np.all(np.isfinite(offsets)):
        raise ValueError(f"plane_z {name} must contain one finite value per movie frame.")
    rounded = np.rint(offsets)
    if not np.allclose(offsets, rounded):
        raise ValueError(f"plane_z {name} contains non-integer rigid shifts.")
    return rounded.astype(int)


def _validate_roi_pixels(stats: np.ndarray, height: int, width: int) -> None:
    """Ensure every plane_z ROI mask is valid in the control movie dimensions."""
    for roi_id, stat in enumerate(stats):
        y_pixels = np.asarray(stat["ypix"], dtype=int).reshape(-1)
        x_pixels = np.asarray(stat["xpix"], dtype=int).reshape(-1)
        if y_pixels.size == 0 or y_pixels.shape != x_pixels.shape:
            raise ValueError(f"ROI {roi_id} has an empty or inconsistent pixel mask.")
        if (
            y_pixels.min() < 0
            or y_pixels.max() >= height
            or x_pixels.min() < 0
            or x_pixels.max() >= width
        ):
            raise ValueError(f"ROI {roi_id} is outside the backup movie coordinate system.")


def _alignment_qc(
    plane_database: dict[str, Any],
    plane_outputs: dict[str, Any],
    backup_outputs: dict[str, Any],
    reference_plane: int,
    num_frames: int,
    max_reference_shift_px: float,
) -> tuple[AlignmentQC, np.ndarray, np.ndarray]:
    """Verify the shared coordinate system and recover final plane_z rigid shifts."""
    plane_ids = _ordered_physical_plane_ids(plane_database)
    matches = np.flatnonzero(plane_ids == reference_plane)
    if matches.size != 1:
        raise ValueError(f"Reference plane {reference_plane} is absent from {plane_ids.tolist()}.")
    reference_index = int(matches[0])

    target_reference = _select_reference_image(plane_outputs.get("refImg"), 0, "plane_z refImg")
    backup_reference = _select_reference_image(
        backup_outputs.get("refImg"), reference_index, "backup refImg"
    )
    archived_reference = _select_reference_image(
        plane_outputs.get("refImg_singleplanes"),
        reference_index,
        "plane_z refImg_singleplanes",
    )
    if target_reference.shape != backup_reference.shape:
        raise ValueError("plane_z and backup reference images have different dimensions.")

    archived_correlation = _normalized_correlation(archived_reference, backup_reference)
    if archived_correlation < 0.999:
        raise ValueError(
            "The archived plane_z source reference does not match the selected backup reference "
            f"(r={archived_correlation:.6f})."
        )
    reference_correlation = _normalized_correlation(target_reference, backup_reference)
    shift, _, _ = phase_cross_correlation(
        target_reference,
        backup_reference,
        upsample_factor=10,
    )
    reference_shift = np.asarray(shift, dtype=float).reshape(-1)
    if reference_shift.size != 2 or not np.all(np.isfinite(reference_shift)):
        raise ValueError("Could not estimate the backup-to-plane_z reference image shift.")
    if float(np.linalg.norm(reference_shift)) > max_reference_shift_px:
        raise ValueError(
            "The plane_z masks and backup reference are not in the same pixel coordinates: "
            f"estimated residual shift is {reference_shift.tolist()} pixels."
        )

    y_offsets = _integer_offsets(plane_outputs.get("yoff"), "yoff", num_frames)
    x_offsets = _integer_offsets(plane_outputs.get("xoff"), "xoff", num_frames)
    checksum = hashlib.sha256(
        np.column_stack((y_offsets, x_offsets)).astype(np.int64).tobytes()
    ).hexdigest()
    qc = AlignmentQC(
        reference_correlation=reference_correlation,
        reference_shift_y_px=float(reference_shift[0]),
        reference_shift_x_px=float(reference_shift[1]),
        archived_reference_correlation=archived_correlation,
        final_shift_x_max_abs_px=int(np.max(np.abs(x_offsets), initial=0)),
        final_shift_y_max_abs_px=int(np.max(np.abs(y_offsets), initial=0)),
        final_shift_checksum=checksum,
    )
    return qc, y_offsets, x_offsets


def _mask_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Select the saved Suite2p settings that define ROI and neuropil masks."""
    extraction_settings = settings.get("extraction")
    if not isinstance(extraction_settings, dict):
        raise ValueError("plane_z settings do not contain an extraction mapping.")
    required = {
        "lam_percentile",
        "allow_overlap",
        "neuropil_extract",
        "inner_neuropil_radius",
        "min_neuropil_pixels",
        "circular_neuropil",
        "batch_size",
    }
    missing = required.difference(extraction_settings)
    if missing:
        raise ValueError(f"plane_z extraction settings are missing {sorted(missing)}.")
    if not extraction_settings["neuropil_extract"]:
        raise ValueError("The saved Suite2p settings disable neuropil extraction.")
    return extraction_settings


def _extract_reference_traces(
    backup_directory: Path,
    stats: np.ndarray,
    extraction_settings: dict[str, Any],
    device: torch.device,
    y_offsets: np.ndarray,
    x_offsets: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract all plane_z ROI and neuropil masks from the aligned control movie."""
    data_path = backup_directory / "data.bin"
    if not data_path.is_file():
        raise FileNotFoundError(f"Reference-plane movie not found: {data_path}")
    num_frames = y_offsets.size
    expected_size = num_frames * height * width * np.dtype(np.int16).itemsize
    if data_path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected binary size for {data_path}: expected {expected_size}, "
            f"found {data_path.stat().st_size}."
        )

    cell_masks, neuropil_masks = masks.create_masks(
        stats,
        Ly=height,
        Lx=width,
        lam_percentile=float(extraction_settings["lam_percentile"]),
        allow_overlap=bool(extraction_settings["allow_overlap"]),
        neuropil_extract=True,
        inner_neuropil_radius=int(extraction_settings["inner_neuropil_radius"]),
        min_neuropil_pixels=int(extraction_settings["min_neuropil_pixels"]),
        circular_neuropil=bool(extraction_settings["circular_neuropil"]),
    )
    with io.BinaryFile(
        Ly=height,
        Lx=width,
        filename=str(data_path),
        n_frames=num_frames,
        write=False,
    ) as binary:
        movie = RigidAlignedMovie(
            binary,
            (num_frames, height, width),
            y_offsets,
            x_offsets,
        )
        fluorescence, neuropil = extract.extract_traces(
            movie,
            cell_masks,
            neuropil_masks,
            batch_size=int(extraction_settings["batch_size"]),
            device=device,
        )
    if fluorescence.shape != (stats.size, num_frames) or neuropil.shape != fluorescence.shape:
        raise ValueError("Suite2p extraction returned arrays with unexpected dimensions.")
    return fluorescence, neuropil


def _load_trace_parameters(
    dataset: Dataset,
    plane_directory: Path,
    num_frames: int,
) -> TraceParameters:
    """Recover the exact saved trace settings, with reviewed dark-offset override."""
    correcting_database_path = plane_directory / "db_correcting.npy"
    correcting_settings_path = plane_directory / "settings_correcting.npy"
    correcting_database = _load_mapping(correcting_database_path)
    correcting_settings = _load_mapping(correcting_settings_path)
    saved_delta_f = correcting_settings.get("delta_F")
    if not isinstance(saved_delta_f, dict):
        raise ValueError(f"delta_F settings are missing from {correcting_settings_path}.")
    delta_f = dict(saved_delta_f)
    delta_f["absolute_zero"] = dataset.dark_offset

    frame_rate = float(correcting_settings.get("fs", dataset.frame_rate_hz))
    if not np.isclose(frame_rate, dataset.frame_rate_hz, rtol=0, atol=1e-3):
        logger.warning(
            "%s saved frame rate %.6f Hz differs from reviewed %.6f Hz; using the saved value.",
            dataset.slug,
            frame_rate,
            dataset.frame_rate_hz,
        )
    num_planes = int(correcting_database["nplanes"])
    frames_per_folder = np.asarray(correcting_database["frames_per_folder"], dtype=int).reshape(-1)
    if np.any(frames_per_folder <= 0) or int(frames_per_folder.sum()) != num_frames:
        raise ValueError(
            f"{dataset.slug} frames_per_folder does not sum to the movie length {num_frames}."
        )
    parameters = TraceParameters(
        frame_rate_hz=frame_rate,
        plane_rate_hz=frame_rate / num_planes,
        frames_per_folder=frames_per_folder,
        dark_offset=dataset.dark_offset,
        f0_percentile=float(delta_f["F0_percentile"]),
        f0_window_s=float(delta_f["F0_window"]),
    )
    return parameters


def _calculate_dff(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    accepted_roi_ids: np.ndarray,
    badframes: np.ndarray,
    parameters: TraceParameters,
) -> TraceProducts:
    """Run the project's non-z-stack dF/F path on selected Suite2p ROI rows."""
    if fluorescence.ndim != 2 or neuropil.shape != fluorescence.shape:
        raise ValueError("F and Fneu must have identical (rois, time) shapes.")
    num_frames = fluorescence.shape[1]
    badframes = np.asarray(badframes, dtype=bool).reshape(-1)
    if badframes.size != num_frames:
        raise ValueError("Registration badframes length does not match the fluorescence traces.")
    if accepted_roi_ids.size == 0:
        raise ValueError("No accepted ROIs remain in plane_z/iscell.npy.")
    if accepted_roi_ids.min() < 0 or accepted_roi_ids.max() >= fluorescence.shape[0]:
        raise ValueError("Accepted ROI IDs are outside the fluorescence arrays.")

    selected_fluorescence = np.asarray(fluorescence[accepted_roi_ids, :].T, dtype=float)
    selected_neuropil = np.asarray(neuropil[accepted_roi_ids, :].T, dtype=float)
    selected_fluorescence[badframes, :] = np.nan
    selected_neuropil[badframes, :] = np.nan
    selected_fluorescence -= parameters.dark_offset
    selected_neuropil -= parameters.dark_offset

    baseline = preprocess_traces.get_f0(
        selected_fluorescence,
        parameters.plane_rate_hz,
        f0_percentile=parameters.f0_percentile,
        window_size=parameters.f0_window_s,
        frames_per_folder=parameters.frames_per_folder,
    )
    fluorescence_corrected, regression, fluorescence_bins, neuropil_bins = (
        preprocess_traces.correct_neuropil(
            selected_fluorescence,
            selected_neuropil,
            parameters.plane_rate_hz,
            baseline=baseline,
            baseline_percentile=parameters.f0_percentile,
            baseline_window=parameters.f0_window_s,
            frames_per_folder=parameters.frames_per_folder,
        )
    )
    dff = preprocess_traces.get_delta_f_over_f(fluorescence_corrected, baseline)
    return TraceProducts(dff, fluorescence_corrected, regression, fluorescence_bins, neuropil_bins)


def _validate_zregistered_products(
    processed_directory: Path,
    accepted_roi_ids: np.ndarray,
    num_frames: int,
) -> None:
    """Validate existing authoritative outputs without changing them."""
    dff_path = processed_directory / "2pCalcium.dff.npy"
    roi_ids_path = processed_directory / "2pRois.ids.npy"
    roi_planes_path = processed_directory / "2pRois.2pPlanes.npy"
    for path in (dff_path, roi_ids_path, roi_planes_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing correction output not found: {path}")

    dff = np.load(dff_path, mmap_mode="r")
    if dff.shape != (num_frames, accepted_roi_ids.size):
        raise ValueError(
            f"Existing {dff_path.name} has shape {dff.shape}; expected "
            f"{(num_frames, accepted_roi_ids.size)}."
        )
    saved_roi_ids = np.asarray(np.load(roi_ids_path)).reshape(-1)
    if not np.all(np.isfinite(saved_roi_ids)) or not np.all(
        saved_roi_ids == np.round(saved_roi_ids)
    ):
        raise ValueError(f"Existing ROI IDs are not finite integers: {roi_ids_path}")
    if not np.array_equal(saved_roi_ids.astype(int), accepted_roi_ids):
        raise ValueError("Existing 2pRois.ids.npy does not match the accepted plane_z ROI order.")
    saved_roi_planes = np.asarray(np.load(roi_planes_path)).reshape(-1)
    if saved_roi_planes.size != accepted_roi_ids.size or not np.all(saved_roi_planes == -1):
        raise ValueError("Existing 2pRois.2pPlanes.npy must contain one -1 per accepted ROI.")


def _strictly_increasing(values: np.ndarray, description: str) -> None:
    """Require a finite, strictly increasing timestamp vector."""
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{description} must be finite and strictly increasing.")


def _validate_required_timing(
    bonsai_directory: Path,
    num_frames: int,
) -> None:
    """Validate the required dataset-level timing inputs without changing them."""
    input_paths = {
        "neural": bonsai_directory / "2pCalcium.timestamps.npy",
        "wheel_time": bonsai_directory / "wheel.timestamps.npy",
        "wheel_velocity": bonsai_directory / "wheel.velocity.npy",
    }
    neural_time = _load_vector(input_paths["neural"], "Concatenated neural timestamps")
    wheel_time = _load_vector(input_paths["wheel_time"], "Concatenated wheel timestamps")
    wheel_velocity = _load_vector(
        input_paths["wheel_velocity"], "Concatenated wheel velocity", allow_nan=True
    )
    if neural_time.size != num_frames:
        raise ValueError("Neural timestamps do not match the Suite2p frame count.")
    _strictly_increasing(neural_time, "Concatenated neural timestamps")
    _strictly_increasing(wheel_time, "Concatenated wheel timestamps")
    if wheel_time.shape != wheel_velocity.shape:
        raise ValueError("Concatenated wheel timestamps and velocity do not align.")


def _can_reuse_extraction(
    backup_directory: Path,
    num_rois: int,
    num_frames: int,
    alignment_qc: AlignmentQC,
) -> bool:
    """Return whether complete trace outputs came from the same aligned movie inputs."""
    manifest_path = backup_directory / MANIFEST_NAME
    fluorescence_path = backup_directory / "F.npy"
    neuropil_path = backup_directory / "Fneu.npy"
    if (
        not manifest_path.is_file()
        or not fluorescence_path.is_file()
        or not neuropil_path.is_file()
    ):
        return False
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("final_shift_checksum") != alignment_qc.final_shift_checksum
    ):
        return False
    fluorescence = np.load(fluorescence_path, mmap_mode="r")
    neuropil = np.load(neuropil_path, mmap_mode="r")
    expected_shape = (num_rois, num_frames)
    return fluorescence.shape == expected_shape and neuropil.shape == expected_shape


def preprocess_dataset(
    dataset: Dataset,
    processed_root: Path,
    bonsai_root: Path,
    output_root: Path,
    max_reference_shift_px: float,
    force_extraction: bool,
) -> None:
    """Export matched reference traces to the manuscript Figure 5 data folder."""
    processed_directory = processed_root / dataset.subject / dataset.date
    suite2p_directory = processed_directory / "suite2p"
    plane_directory = suite2p_directory / "plane_z"
    bonsai_directory = bonsai_root / dataset.subject / dataset.date
    for directory in (processed_directory, suite2p_directory, plane_directory, bonsai_directory):
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    plane_database, plane_settings = suite2p_compat.load_parameters(plane_directory)
    plane_outputs, _ = suite2p_compat.load_outputs(plane_directory)
    if plane_outputs is None:
        raise FileNotFoundError(f"Registration outputs not found in {plane_directory}.")
    if "reference_plane" not in plane_outputs:
        raise ValueError(f"reference_plane is missing from {plane_directory / 'reg_outputs.npy'}.")
    reference_plane = int(plane_outputs["reference_plane"])
    backup_directory = suite2p_directory / f"backup{reference_plane}"
    output_directory = (
        output_root / dataset.subject / dataset.date / "suite2p" / f"backup{reference_plane}"
    )
    if not backup_directory.is_dir():
        raise FileNotFoundError(f"Reference-plane backup directory not found: {backup_directory}")
    backup_database, _ = suite2p_compat.load_parameters(backup_directory)
    backup_outputs, _ = suite2p_compat.load_outputs(backup_directory)
    if backup_outputs is None:
        raise FileNotFoundError(f"Registration outputs not found in {backup_directory}.")

    stats = np.load(plane_directory / "stat.npy", allow_pickle=True)
    is_cell = np.asarray(np.load(plane_directory / "iscell.npy"))
    if stats.ndim != 1 or is_cell.ndim != 2 or is_cell.shape[0] != stats.size:
        raise ValueError("plane_z stat.npy and iscell.npy have inconsistent shapes.")
    accepted_roi_ids = np.flatnonzero(is_cell[:, 0].astype(bool))
    height = int(plane_database["Ly"])
    width = int(plane_database["Lx"])
    if (int(backup_database["Ly"]), int(backup_database["Lx"])) != (height, width):
        raise ValueError("plane_z and backup movies have different pixel dimensions.")
    num_frames = int(plane_database["nframes"])
    if int(backup_database["nframes"]) != num_frames:
        raise ValueError("plane_z and backup movies contain different numbers of frames.")
    _validate_roi_pixels(stats, height, width)
    trace_parameters = _load_trace_parameters(
        dataset,
        plane_directory,
        num_frames,
    )
    _validate_zregistered_products(
        processed_directory,
        accepted_roi_ids,
        num_frames,
    )
    _validate_required_timing(bonsai_directory, num_frames)
    plane_badframes = np.asarray(plane_outputs.get("badframes"), dtype=bool).reshape(-1)
    if plane_badframes.size != num_frames:
        raise ValueError("plane_z badframes length does not match the movie.")

    alignment_qc, y_offsets, x_offsets = _alignment_qc(
        plane_database,
        plane_outputs,
        backup_outputs,
        reference_plane,
        num_frames,
        max_reference_shift_px,
    )
    extraction_settings = _mask_settings(plane_settings)
    device = general.assign_torch_device(str(plane_settings.get("torch_device", "cpu")))
    reuse_extraction = not force_extraction and _can_reuse_extraction(
        output_directory,
        stats.size,
        num_frames,
        alignment_qc,
    )
    if reuse_extraction:
        logger.info("%s reusing verified control F.npy and Fneu.npy", dataset.slug)
        control_fluorescence = np.load(output_directory / "F.npy", mmap_mode="r")
        control_neuropil = np.load(output_directory / "Fneu.npy", mmap_mode="r")
    else:
        logger.info(
            "%s extracting %d plane_z masks from backup%d on %s",
            dataset.slug,
            stats.size,
            reference_plane,
            device,
        )
        control_fluorescence, control_neuropil = _extract_reference_traces(
            backup_directory,
            stats,
            extraction_settings,
            device,
            y_offsets,
            x_offsets,
            height,
            width,
        )
        _atomic_save(output_directory / "F.npy", control_fluorescence)
        _atomic_save(output_directory / "Fneu.npy", control_neuropil)

    backup_badframes = np.asarray(backup_outputs.get("badframes"), dtype=bool).reshape(-1)

    control_products = _calculate_dff(
        control_fluorescence,
        control_neuropil,
        accepted_roi_ids,
        backup_badframes,
        trace_parameters,
    )
    _atomic_save(output_directory / "dff.npy", control_products.dff)
    _atomic_save(output_directory / "dff_roi_ids.npy", accepted_roi_ids)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset.slug,
        "reference_plane": reference_plane,
        "source_movie": str(backup_directory / "data.bin"),
        "source_masks": str(plane_directory / "stat.npy"),
        "num_rois": int(stats.size),
        "num_accepted_rois": int(accepted_roi_ids.size),
        "num_frames": num_frames,
        "dark_offset": trace_parameters.dark_offset,
        "frame_rate_hz": trace_parameters.frame_rate_hz,
        "plane_rate_hz": trace_parameters.plane_rate_hz,
        "f0_percentile": trace_parameters.f0_percentile,
        "f0_window_s": trace_parameters.f0_window_s,
        "plane_z_badframes": int(np.count_nonzero(plane_badframes)),
        "backup_badframes": int(np.count_nonzero(backup_badframes)),
        "timing_inputs": "validated_read_only",
        "final_shift_checksum": alignment_qc.final_shift_checksum,
        "alignment_qc": asdict(alignment_qc),
        "outputs": {
            "raw_fluorescence": str(output_directory / "F.npy"),
            "raw_neuropil": str(output_directory / "Fneu.npy"),
            "control_dff": str(output_directory / "dff.npy"),
            "control_dff_roi_ids": str(output_directory / "dff_roi_ids.npy"),
            "zregistered_dff": str(processed_directory / "2pCalcium.dff.npy"),
            "authoritative_roi_ids": str(processed_directory / "2pRois.ids.npy"),
        },
    }
    _atomic_write_json(output_directory / MANIFEST_NAME, manifest)
    logger.info(
        "%s complete: backup%d, %d accepted ROIs, dark offset %.0f",
        dataset.slug,
        reference_plane,
        accepted_roi_ids.size,
        dataset.dark_offset,
    )


def _profile_depth_assignments(
    plane_ids: np.ndarray,
    reference_plane: int,
    best_plane_ids: Any,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign reference-relative depths using one best-plane shift per cycle."""
    best = np.asarray(best_plane_ids, dtype=float).reshape(-1)
    if (
        best.size != num_frames
        or not np.all(np.isfinite(best))
        or not np.all(best == np.round(best))
        or not np.all(np.isin(best, plane_ids))
        or reference_plane not in plane_ids
    ):
        raise ValueError("Best-plane IDs must contain one acquired physical plane ID per cycle.")
    cycle_shifts = reference_plane - best.astype(int)
    effective_depths = plane_ids[:, np.newaxis] - reference_plane + cycle_shifts
    return cycle_shifts, effective_depths


def _file_signature(path: Path, *, hash_content: bool = True) -> dict[str, Any]:
    """Fingerprint small metadata files; identify large movies by path, size and mtime."""
    info = path.stat()
    signature = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if hash_content:
        with path.open("rb") as stream:
            signature["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return signature


def _profile_badframes(outputs: dict[str, Any], num_frames: int) -> np.ndarray:
    """Require a saved registration exclusion mask for every imaging cycle."""
    if outputs.get("badframes") is None:
        raise ValueError("Registration outputs do not contain badframes.")
    badframes = np.asarray(outputs["badframes"], dtype=bool).reshape(-1)
    if badframes.size != num_frames:
        raise ValueError("Registration badframes do not match the imaging cycle count.")
    return badframes


def _can_reuse_profile_plane(
    directory: Path,
    fingerprint: dict[str, Any],
    shape: tuple[int, int],
) -> bool:
    """Only reuse completed plane extractions with matching sources and dimensions."""
    try:
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("fingerprint") != fingerprint:
            return False
        for name in ("F.npy", "Fneu.npy"):
            values = np.load(directory / name, mmap_mode="r")
            if values.shape != shape or values.dtype != np.float32:
                return False
        badframes = np.load(directory / "badframes.npy", mmap_mode="r")
        return badframes.shape == (shape[1],) and badframes.dtype == np.bool_
    except (OSError, ValueError, EOFError):
        return False


def preprocess_z_profiles(
    dataset: Dataset,
    processed_root: Path,
    bonsai_root: Path,
    registered_root: Path,
    output_root: Path,
    max_reference_shift_px: float = DEFAULT_MAX_REFERENCE_SHIFT_PX,
    force_extraction: bool = False,
) -> Path:
    """Export accepted plane_z ROI traces from every registered acquisition plane.

    Movies are read only. F and Fneu retain Suite2p intensity units and bad-frame
    samples; consumers must apply badframes and subtract the manifest dark offset
    before calculating fluorescence-depth profiles. No dF/F is calculated here.
    """
    processed_directory = Path(processed_root) / dataset.subject / dataset.date
    plane_directory = processed_directory / "suite2p" / "plane_z"
    source_directory = Path(registered_root) / dataset.subject / dataset.date / "suite2p"
    output_directory = Path(output_root) / dataset.subject / dataset.date / "z_profiles"
    database, settings = suite2p_compat.load_parameters(plane_directory)
    outputs, _ = suite2p_compat.load_outputs(plane_directory)
    if outputs is None or outputs.get("reference_plane") is None:
        raise ValueError("plane_z registration outputs must specify reference_plane.")
    num_frames, height, width = (int(database[key]) for key in ("nframes", "Ly", "Lx"))
    plane_ids = _ordered_physical_plane_ids(database)
    reference_plane = int(outputs["reference_plane"])
    cycle_shifts, effective_depths = _profile_depth_assignments(
        plane_ids, reference_plane, outputs.get("planes_across_time"), num_frames
    )
    cycle_badframes = _profile_badframes(outputs, num_frames)
    references = np.asarray(outputs.get("refImg_singleplanes"))
    if references.shape != (plane_ids.size, height, width):
        raise ValueError("Archived reference images do not match the acquired planes.")
    stats = np.load(plane_directory / "stat.npy", allow_pickle=True)
    iscell = np.asarray(np.load(plane_directory / "iscell.npy"))
    if stats.ndim != 1 or iscell.ndim != 2 or iscell.shape[0] != stats.size or iscell.shape[1] < 1:
        raise ValueError("plane_z stat.npy and iscell.npy have inconsistent shapes.")
    roi_ids = np.flatnonzero(iscell[:, 0].astype(bool))
    if roi_ids.size == 0:
        raise ValueError("No accepted plane_z ROIs remain.")
    _validate_roi_pixels(stats, height, width)
    _validate_zregistered_products(processed_directory, roi_ids, num_frames)
    timestamp_path = Path(bonsai_root) / dataset.subject / dataset.date / "2pCalcium.timestamps.npy"
    timestamps = _load_vector(timestamp_path, "Imaging cycle timestamps")
    _strictly_increasing(timestamps, "Imaging cycle timestamps")
    if timestamps.size != num_frames:
        raise ValueError("Timestamps do not match the imaging cycle count.")
    frames_per_folder = np.asarray(database["frames_per_folder"], dtype=int)
    if np.any(frames_per_folder <= 0) or int(frames_per_folder.sum()) != num_frames:
        raise ValueError("Experiment lengths do not reproduce the imaging cycle count.")
    extraction_settings = _mask_settings(settings)
    metadata_paths = [
        plane_directory / name
        for name in ("stat.npy", "iscell.npy", "db.npy", "settings.npy", "reg_outputs.npy")
    ]
    common_fingerprint = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "dataset": dataset.slug,
        "roi_ids": roi_ids.tolist(),
        "metadata": [_file_signature(path) for path in metadata_paths],
    }

    # Validate every source before starting the expensive movie reads or writing outputs.
    sources = []
    for plane_id in plane_ids:
        directory = source_directory / f"backup{plane_id}"
        source_db, _ = suite2p_compat.load_parameters(directory)
        source_outputs, _ = suite2p_compat.load_outputs(directory)
        if source_outputs is None:
            raise FileNotFoundError(f"Registration outputs not found in {directory}.")
        expected = {"nframes": num_frames, "Ly": height, "Lx": width, "iplane": int(plane_id)}
        if any(int(source_db[key]) != value for key, value in expected.items()):
            raise ValueError(f"Source movie identity or dimensions do not match: {directory}")
        if not np.array_equal(source_db["frames_per_folder"], frames_per_folder):
            raise ValueError(f"Source experiment lengths do not match: {directory}")
        if not np.array_equal(np.asarray(source_outputs.get("refImg")), references):
            raise ValueError(
                f"Source reference stack differs from the ROI registration: {directory}"
            )
        source_badframes = _profile_badframes(source_outputs, num_frames)
        alignment_qc, y_offsets, x_offsets = _alignment_qc(
            database, outputs, source_outputs, reference_plane, num_frames, max_reference_shift_px
        )
        movie_signature = _file_signature(directory / "data.bin", hash_content=False)
        if movie_signature["bytes"] != num_frames * height * width * np.dtype(np.int16).itemsize:
            raise ValueError(f"Registered binary size does not match the metadata: {directory}")
        fingerprint = {
            **common_fingerprint,
            "plane_id": int(plane_id),
            "movie": movie_signature,
            "source_metadata": [
                _file_signature(directory / name)
                for name in ("db.npy", "settings.npy", "reg_outputs.npy")
            ],
        }
        sources.append(
            (
                directory,
                source_badframes | cycle_badframes,
                alignment_qc,
                y_offsets,
                x_offsets,
                fingerprint,
            )
        )

    device = general.assign_torch_device(str(settings.get("torch_device", "cpu")))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "manifest.json").unlink(missing_ok=True)
    plane_manifests = []
    for plane_id, source in zip(plane_ids, sources, strict=True):
        directory, badframes, alignment_qc, y_offsets, x_offsets, fingerprint = source
        destination = output_directory / f"plane{plane_id}"
        shape = (roi_ids.size, num_frames)
        if not force_extraction and _can_reuse_profile_plane(destination, fingerprint, shape):
            logger.info("%s reusing verified plane%d traces", dataset.slug, plane_id)
        else:
            logger.info(
                "%s extracting %d accepted ROIs from %s", dataset.slug, roi_ids.size, directory
            )
            # Build masks from all detections so neuropil exclusion matches the original extraction.
            fluorescence, neuropil = _extract_reference_traces(
                directory, stats, extraction_settings, device, y_offsets, x_offsets, height, width
            )
            _atomic_save(destination / "F.npy", np.asarray(fluorescence[roi_ids], dtype=np.float32))
            _atomic_save(destination / "Fneu.npy", np.asarray(neuropil[roi_ids], dtype=np.float32))
            _atomic_save(destination / "badframes.npy", badframes)
            del fluorescence, neuropil
            _atomic_write_json(
                destination / "manifest.json",
                {
                    "fingerprint": fingerprint,
                    "shape": list(shape),
                    "axes": ["roi", "cycle"],
                    "badframes": int(badframes.sum()),
                    "alignment_qc": asdict(alignment_qc),
                },
            )
        plane_manifests.append(str(Path(f"plane{plane_id}") / "manifest.json"))

    reference_fluorescence = np.empty((roi_ids.size, plane_ids.size), dtype=np.float32)
    flat_references = references.reshape(plane_ids.size, -1).astype(np.float32)
    for index, roi_id in enumerate(roi_ids):
        pixels, weights = masks.create_cell_mask(
            stats[roi_id], height, width, allow_overlap=bool(extraction_settings["allow_overlap"])
        )
        if pixels.size == 0 or not np.all(np.isfinite(weights)):
            raise ValueError(f"Accepted ROI {roi_id} has an empty or invalid extraction mask.")
        reference_fluorescence[index] = flat_references[:, pixels] @ weights
    arrays = {
        "roi_ids.npy": roi_ids,
        "plane_ids.npy": plane_ids,
        "timestamps.npy": timestamps,
        "frames_per_folder.npy": frames_per_folder,
        "best_plane_ids.npy": reference_plane - cycle_shifts,
        "cycle_shift_planes.npy": cycle_shifts,
        "cycle_badframes.npy": cycle_badframes,
        "effective_depth_planes.npy": effective_depths,
        "reference_depth_planes.npy": plane_ids - reference_plane,
        "registration_reference_profiles.npy": reference_fluorescence,
        "registration_reference_images.npy": references,
    }
    for name, values in arrays.items():
        _atomic_save(output_directory / name, values)
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "dataset": dataset.slug,
        "reference_plane": reference_plane,
        "num_rois": int(roi_ids.size),
        "num_cycles": num_frames,
        "plane_ids": plane_ids.tolist(),
        "dark_offset": dataset.dark_offset,
        "fluorescence_convention": "Raw Suite2p F and Fneu; no dark, neuropil or z correction applied.",
        "valid_samples": "Exclude each plane's badframes.npy (source OR plane_z badframes) and nonfinite F.",
        "profile_statistic": "Median of F - dark_offset, with at least 100 valid samples per ROI/depth.",
        "depth_units": "Acquisition plane steps relative to the selected reference plane; not micrometres.",
        "depth_formula": "effective_depth_planes[plane, cycle] = plane_ids[plane] - best_plane_ids[cycle]",
        "trace_axes": ["roi", "cycle"],
        "timestamp_convention": "One saved neural timestamp per cycle; within-cycle plane delays are not applied.",
        "functional_channel": int(database.get("functional_chan", 1)),
        "registration_reference_channel": (
            2 if settings["registration"].get("align_by_chan2", False) else 1
        ),
        "registration_reference_profile_axes": ["roi", "plane"],
        "registration_reference_profile_convention": (
            "Same normalized ROI masks applied to refImg_singleplanes; raw registration-channel units. "
            "These are not calcium reference profiles when registration uses the other channel."
        ),
        "shared_arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "plane_manifests": plane_manifests,
        "timestamp_source": _file_signature(timestamp_path),
        "common_fingerprint": common_fingerprint,
    }
    _atomic_write_json(output_directory / "manifest.json", manifest)
    logger.info(
        "%s complete: %d ROIs, %d cycles, %d planes -> %s",
        dataset.slug,
        roi_ids.size,
        num_frames,
        plane_ids.size,
        output_directory,
    )
    return output_directory


def main() -> None:
    """Export six matched recordings and Stereopes' all-plane z profiles."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    datasets = discover_datasets(DATASETS_CSV, PREPROCESSING_CSV)
    for dataset in datasets:
        preprocess_dataset(
            dataset,
            PROCESSED_ROOT,
            BONSAI_ROOT,
            OUTPUT_ROOT,
            DEFAULT_MAX_REFERENCE_SHIFT_PX,
            False,
        )
    profile_dataset = next(
        dataset
        for dataset in datasets
        if (dataset.subject, dataset.date) == ("Stereopes", "2024-02-13")
    )
    preprocess_z_profiles(
        profile_dataset,
        PROCESSED_ROOT,
        BONSAI_ROOT,
        REGISTERED_ROOT,
        OUTPUT_ROOT,
        DEFAULT_MAX_REFERENCE_SHIFT_PX,
        False,
    )


if __name__ == "__main__":
    main()
