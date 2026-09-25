"""Generate Figure 3 panels for controlled z-stack ground-truth validation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from suite2p_in_depth import correct_traces, extract_data

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

OUTPUT_DIR = PLOTS_ROOT / "Fig03"

FIGURE_STYLE = {
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DAQ_DATA_PATTERN = "NidaqInput*"
DAQ_CHANNEL_PATTERN = "nidaqChannels*.csv"
PIEZO_CHANNEL = "piezo"
FRAME_CLOCK_CHANNEL = "frameclock"
DAQ_SAMPLING_RATE_HZ = 1000
PIEZO_SMOOTHING_WINDOW_MS = 10
ZTRACE_SMOOTHING_WINDOW_S = 1.0
PLOT_END_S = 180.0
CURATED_EXPERIMENTS = (
    ("Quille", "2023-10-12", 5),
    ("Tara", "2023-11-01", 6),
)


@dataclass(frozen=True)
class Figure3Dataset:
    """Location and identity of one controlled-displacement dataset."""

    data_folder: Path
    subject: str
    date: str
    experiment: int

    @property
    def name(self) -> str:
        return f"{self.subject}_{self.date}"


@dataclass(frozen=True)
class Figure3Inputs:
    """Raw inputs needed for controlled-displacement analysis."""

    piezo_voltage: np.ndarray
    piezo_time_s: np.ndarray
    frame_times_s: np.ndarray
    z_traces_slices: np.ndarray
    z_correlations: np.ndarray
    plane_ids: np.ndarray
    reference_depth_slices: np.ndarray
    zstack_spacing_um: float
    piezo_volts_per_micron: float
    num_acquired_planes: int


def discover_datasets(input_root: Path) -> list[Figure3Dataset]:
    """Locate the two fixed Figure 3 datasets and raw experiments."""
    figure_data_root = Path(input_root) / "Fig03"
    if not figure_data_root.is_dir():
        raise FileNotFoundError(f"Figure 3 data directory not found: {figure_data_root}")

    datasets = []
    for subject, date, experiment in CURATED_EXPERIMENTS:
        data_folder = figure_data_root / f"{subject}_{date}"
        experiment_dir = data_folder / "raw" / str(experiment)
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Raw experiment directory not found: {experiment_dir}")
        datasets.append(
            Figure3Dataset(
                data_folder=data_folder,
                subject=subject,
                date=date,
                experiment=experiment,
            )
        )
    return datasets


def _load_daq_inputs(raw_dataset_dir: Path, experiment: int) -> tuple[np.ndarray, ...]:
    """Load the piezo signal and frame timing using the production DAQ readers."""
    experiment_dir = raw_dataset_dir / str(experiment)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Raw experiment directory not found: {experiment_dir}")

    piezo_voltage, piezo_time_s = extract_data.get_piezo_data(
        experiment_dir,
        data_file=DAQ_DATA_PATTERN,
        channel_name_file=DAQ_CHANNEL_PATTERN,
        piezo_channel=PIEZO_CHANNEL,
        sampling_rate=DAQ_SAMPLING_RATE_HZ,
    )
    frame_times_s = extract_data.get_frame_times(
        experiment_dir,
        data_file=DAQ_DATA_PATTERN,
        channel_name_file=DAQ_CHANNEL_PATTERN,
        clock_channel=FRAME_CLOCK_CHANNEL,
        sampling_rate=DAQ_SAMPLING_RATE_HZ,
    )
    return piezo_voltage, piezo_time_s, frame_times_s


def _plane_id(plane_dir: Path) -> int:
    """Return the integer suffix from a Suite2p ``planeN`` directory."""
    try:
        return int(plane_dir.name.removeprefix("plane"))
    except ValueError as error:
        raise ValueError(f"Invalid Suite2p plane directory name: {plane_dir.name}") from error


def _smooth_finite_trace(trace: np.ndarray, window_samples: int) -> np.ndarray:
    """Hanning-smooth a trace without spreading missing values."""
    values = np.asarray(trace, dtype=float)
    if window_samples < 3:
        return values.copy()
    if window_samples % 2 == 0:
        window_samples += 1
    window = np.hanning(window_samples)
    valid = np.isfinite(values)
    numerator = np.convolve(np.where(valid, values, 0), window, mode="same")
    denominator = np.convolve(valid.astype(float), window, mode="same")
    smoothed = np.full(values.shape, np.nan)
    np.divide(numerator, denominator, out=smoothed, where=denominator > 0)
    return smoothed


def _load_algorithm_inputs(
    processed_dataset_dir: Path,
    suite2p_dataset_dir: Path,
    experiment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Load and select Algorithm A outputs for one experiment."""
    if not processed_dataset_dir.is_dir():
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dataset_dir}")
    if not suite2p_dataset_dir.is_dir():
        raise FileNotFoundError(f"Suite2p directory not found: {suite2p_dataset_dir}")

    z_traces = np.load(processed_dataset_dir / "2pPlanes.zTraces.npy")
    z_correlations = np.load(processed_dataset_dir / "2pPlanes.zCorrelations.npy")
    if z_traces.ndim != 2:
        raise ValueError("2pPlanes.zTraces.npy must have shape (planes, time).")
    if z_correlations.ndim != 3:
        raise ValueError(
            "2pPlanes.zCorrelations.npy must have shape (planes, z-stack slices, time)."
        )
    if z_correlations.shape[0] != z_traces.shape[0]:
        raise ValueError("The z-trace and z-correlation plane counts do not match.")

    plane_dirs = sorted(suite2p_dataset_dir.glob("plane*"), key=_plane_id)
    if len(plane_dirs) != z_traces.shape[0]:
        raise ValueError(
            "The number of Suite2p plane directories does not match the saved z-traces."
        )

    databases = [
        np.load(plane_dir / "db_correcting.npy", allow_pickle=True).item()
        for plane_dir in plane_dirs
    ]
    settings = [
        np.load(plane_dir / "settings_correcting.npy", allow_pickle=True).item()
        for plane_dir in plane_dirs
    ]
    zcorrect_outputs = [
        np.load(plane_dir / "zcorrect_outputs.npy", allow_pickle=True).item()
        for plane_dir in plane_dirs
    ]

    experiment_names = [Path(str(path)).name for path in databases[0]["data_path"]]
    experiment_name = str(experiment)
    if experiment_names.count(experiment_name) != 1:
        raise ValueError(
            f"Expected experiment {experiment_name!r} once in data_path; "
            f"found {experiment_names.count(experiment_name)} matches."
        )
    experiment_index = experiment_names.index(experiment_name)
    frames_per_experiment = np.asarray(databases[0]["frames_per_folder"], dtype=int)
    if frames_per_experiment.sum() != z_traces.shape[1]:
        raise ValueError("frames_per_folder does not reproduce the concatenated z-trace length.")

    for database in databases[1:]:
        if [Path(str(path)).name for path in database["data_path"]] != experiment_names:
            raise ValueError("Experiment order differs between Suite2p planes.")
        if not np.array_equal(database["frames_per_folder"], frames_per_experiment):
            raise ValueError("Experiment frame counts differ between Suite2p planes.")

    spacing_values = np.asarray(
        [plane_settings["z_correction"]["spacing"] for plane_settings in settings],
        dtype=float,
    )
    if not np.allclose(spacing_values, spacing_values[0]):
        raise ValueError("Z-stack spacing differs between Suite2p planes.")
    piezo_calibrations = np.asarray(
        [plane_settings["daq"]["piezo_volt_per_micron"] for plane_settings in settings],
        dtype=float,
    )
    if not np.allclose(piezo_calibrations, piezo_calibrations[0]):
        raise ValueError("Piezo monitor calibration differs between Suite2p planes.")
    num_acquired_planes = int(databases[0]["nplanes"])
    if any(int(database["nplanes"]) != num_acquired_planes for database in databases):
        raise ValueError("The acquired plane count differs between Suite2p planes.")

    start = int(frames_per_experiment[:experiment_index].sum())
    stop = start + int(frames_per_experiment[experiment_index])
    plane_ids = np.asarray([_plane_id(plane_dir) for plane_dir in plane_dirs], dtype=int)
    reference_depths = np.asarray(
        [outputs["reference_depth"] for outputs in zcorrect_outputs],
        dtype=float,
    )
    return (
        z_traces[:, start:stop],
        z_correlations[:, :, start:stop],
        plane_ids,
        reference_depths,
        float(spacing_values[0]),
        float(piezo_calibrations[0]),
        num_acquired_planes,
    )


def load_figure_inputs(
    dataset: Figure3Dataset,
) -> Figure3Inputs:
    """Load the curated manuscript inputs for Figure 3."""
    data_folder = dataset.data_folder
    raw_dataset_dir = data_folder / "raw"
    processed_dataset_dir = data_folder / "processed"
    suite2p_dataset_dir = data_folder / "suite2p"
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Figure 3 data directory not found: {data_folder}")
    piezo_voltage, piezo_time_s, frame_times_s = _load_daq_inputs(
        raw_dataset_dir, dataset.experiment
    )
    (
        z_traces,
        z_correlations,
        plane_ids,
        reference_depths,
        zstack_spacing_um,
        piezo_volts_per_micron,
        num_acquired_planes,
    ) = _load_algorithm_inputs(
        processed_dataset_dir,
        suite2p_dataset_dir,
        dataset.experiment,
    )
    return Figure3Inputs(
        piezo_voltage=np.asarray(piezo_voltage),
        piezo_time_s=np.asarray(piezo_time_s),
        frame_times_s=np.asarray(frame_times_s),
        z_traces_slices=np.asarray(z_traces),
        z_correlations=np.asarray(z_correlations),
        plane_ids=plane_ids,
        reference_depth_slices=reference_depths,
        zstack_spacing_um=zstack_spacing_um,
        piezo_volts_per_micron=piezo_volts_per_micron,
        num_acquired_planes=num_acquired_planes,
    )


def prepare_analysis(inputs: Figure3Inputs) -> dict[str, np.ndarray]:
    """Synchronize one experiment and prepare calibrated displacement traces."""
    num_planes = correct_traces.determine_number_of_planes(
        inputs.piezo_voltage,
        inputs.piezo_time_s,
        inputs.frame_times_s,
    )
    if num_planes != inputs.num_acquired_planes:
        raise ValueError(
            f"DAQ detected {num_planes} planes, but Suite2p recorded {inputs.num_acquired_planes}."
        )
    automated_template_um = correct_traces.make_piezo_trace_per_plane(
        inputs.piezo_voltage,
        inputs.piezo_time_s,
        inputs.frame_times_s,
        num_planes=num_planes,
        volt_per_microns=inputs.piezo_volts_per_micron,
        win_size=PIEZO_SMOOTHING_WINDOW_MS,
    )
    automated_template_v = automated_template_um * inputs.piezo_volts_per_micron

    sample_interval_s = float(np.median(np.diff(inputs.piezo_time_s)))
    window_samples = max(
        1,
        int(PIEZO_SMOOTHING_WINDOW_MS / 1000 / sample_interval_s),
    )
    smoothing_window = np.hanning(window_samples) if window_samples > 2 else np.ones(window_samples)
    smoothing_window /= smoothing_window.sum()
    piezo_smoothed_v = np.convolve(
        inputs.piezo_voltage,
        smoothing_window,
        mode="same",
    )

    samples_per_frame = automated_template_v.shape[0]
    cycle_times_s = []
    manual_offsets_v = []
    for cycle_start in range(
        num_planes,
        len(inputs.frame_times_s) - num_planes,
        num_planes,
    ):
        cycle_residuals = []
        for plane in range(num_planes):
            frame_index = cycle_start + plane
            sample_start = int(
                np.searchsorted(
                    inputs.piezo_time_s,
                    inputs.frame_times_s[frame_index],
                    side="left",
                )
            )
            sample_stop = sample_start + samples_per_frame
            if sample_stop > piezo_smoothed_v.size:
                cycle_residuals = []
                break
            cycle_residuals.append(
                piezo_smoothed_v[sample_start:sample_stop] - automated_template_v[:, plane]
            )
        if not cycle_residuals:
            continue
        cycle_times_s.append(
            0.5
            * (inputs.frame_times_s[cycle_start] + inputs.frame_times_s[cycle_start + num_planes])
        )
        manual_offsets_v.append(float(np.median(np.concatenate(cycle_residuals))))

    if not manual_offsets_v:
        raise ValueError("No complete imaging rounds are available for piezo subtraction.")

    cycle_times_s = np.asarray(cycle_times_s)
    manual_offsets_um_raw = np.asarray(manual_offsets_v) / inputs.piezo_volts_per_micron
    manual_offsets_um = manual_offsets_um_raw - np.nanmedian(manual_offsets_um_raw)
    estimated_times_s = inputs.frame_times_s[::num_planes]
    if estimated_times_s.size != inputs.z_traces_slices.shape[1]:
        raise ValueError(
            f"Experiment has {estimated_times_s.size} imaging-round timestamps but "
            f"{inputs.z_traces_slices.shape[1]} saved z estimates per plane."
        )
    estimated_z_um = (
        inputs.z_traces_slices - inputs.reference_depth_slices[:, np.newaxis]
    ) * inputs.zstack_spacing_um
    estimated_z_um -= np.nanmedian(estimated_z_um, axis=1, keepdims=True)
    estimated_interval_s = float(np.median(np.diff(estimated_times_s)))
    smoothing_samples = max(
        3,
        int(round(ZTRACE_SMOOTHING_WINDOW_S / estimated_interval_s)),
    )
    estimated_z_um = np.vstack(
        [_smooth_finite_trace(trace, smoothing_samples) for trace in estimated_z_um]
    )
    estimated_z_slices = estimated_z_um / inputs.zstack_spacing_um + np.nanmedian(
        inputs.z_traces_slices, axis=1, keepdims=True
    )
    if np.any(inputs.plane_ids < 0) or np.any(inputs.plane_ids >= num_planes):
        raise ValueError("Suite2p plane IDs fall outside the acquired piezo-plane cycle.")
    nominal_plane_depths_um = np.nanmedian(automated_template_um[:, inputs.plane_ids], axis=0)
    manual_at_estimated_times_um = np.interp(
        estimated_times_s,
        cycle_times_s,
        manual_offsets_um,
        left=np.nan,
        right=np.nan,
    )
    relative_piezo_slices = (
        nominal_plane_depths_um[:, np.newaxis] + manual_at_estimated_times_um[np.newaxis, :]
    ) / inputs.zstack_spacing_um
    # Fit one shared offset using the raw correction inputs.
    fit_visible = (estimated_times_s >= 0) & (estimated_times_s <= PLOT_END_S)
    raw_residual_slices = inputs.z_traces_slices - relative_piezo_slices
    plane_offsets = np.nanmedian(
        np.where(np.isfinite(raw_residual_slices), raw_residual_slices, np.nan)[:, fit_visible],
        axis=1,
    )
    if not np.all(np.isfinite(plane_offsets)):
        raise ValueError("Cannot fit the measured piezo trajectory to every estimated plane.")
    global_offset_slices = float(np.median(plane_offsets))
    expected_piezo_slices = (
        global_offset_slices
        + (nominal_plane_depths_um[:, np.newaxis] + manual_offsets_um[np.newaxis, :])
        / inputs.zstack_spacing_um
    )
    if inputs.z_correlations.shape != (
        inputs.plane_ids.size,
        inputs.z_correlations.shape[1],
        estimated_times_s.size,
    ):
        raise ValueError("Z-correlations, plane IDs, and imaging-round timestamps are not aligned.")
    return {
        "manual_piezo_time_s": cycle_times_s,
        "manual_piezo_um": manual_offsets_um,
        "expected_piezo_slices": expected_piezo_slices,
        "estimated_z_time_s": estimated_times_s,
        "estimated_z_um": estimated_z_um,
        "estimated_z_slices": estimated_z_slices,
        "estimated_z_raw_slices": inputs.z_traces_slices.copy(),
        "zstack_spacing_um": np.asarray(inputs.zstack_spacing_um),
        "z_correlations": inputs.z_correlations,
        "plane_ids": inputs.plane_ids,
        "piezo_global_offset_slices": np.asarray(global_offset_slices),
        "num_planes": np.asarray(num_planes),
    }


def calculate_metrics(analysis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calculate unsmoothed RMSE in microns over 0-180 s after shared offset fitting.

    Use finite paired samples only, without extrapolating the measured piezo
    trajectory. Pool squared errors across planes for the dataset RMSE.
    """
    time_s = analysis["estimated_z_time_s"]
    visible = (time_s >= 0) & (time_s <= PLOT_END_S)
    estimates = analysis["estimated_z_raw_slices"][:, visible]
    expected = np.vstack(
        [
            np.interp(
                time_s[visible],
                analysis["manual_piezo_time_s"],
                trace,
                left=np.nan,
                right=np.nan,
            )
            for trace in analysis["expected_piezo_slices"]
        ]
    )
    residual_um = (estimates - expected) * float(analysis["zstack_spacing_um"])
    valid = np.isfinite(residual_um)
    n_valid = valid.sum(axis=1)
    if np.any(n_valid == 0):
        raise ValueError("No finite paired depth samples for every plane in the 0-180 s period.")
    squared_error_sum = np.sum(np.where(valid, residual_um, 0) ** 2, axis=1)
    return {
        "rmse_um": np.sqrt(squared_error_sum / n_valid),
        "n_valid": n_valid,
        "dataset_rmse_um": np.asarray(np.sqrt(squared_error_sum.sum() / n_valid.sum())),
    }


def plot_time_course(
    analysis: dict[str, np.ndarray],
    output_dir: Path,
    dataset_name: str,
    metrics: dict[str, np.ndarray],
) -> None:
    """Plot aligned z-stack correlation maps with depth-trace overlays."""
    time_s = analysis["manual_piezo_time_s"]
    manual_piezo_um = analysis["manual_piezo_um"]
    estimated_time_s = analysis["estimated_z_time_s"]
    estimated_z_um = analysis["estimated_z_um"]
    estimated_z_slices = analysis["estimated_z_slices"]
    expected_piezo_slices = analysis["expected_piezo_slices"]
    z_correlations = analysis["z_correlations"]
    plane_ids = analysis["plane_ids"]
    if time_s.size == 0 or manual_piezo_um.shape != time_s.shape:
        raise ValueError("No aligned manual piezo displacement samples are available.")
    if estimated_z_um.shape != (plane_ids.size, estimated_time_s.size):
        raise ValueError("Estimated z-traces, plane IDs, and timestamps are not aligned.")
    if estimated_z_slices.shape != estimated_z_um.shape:
        raise ValueError("Absolute and relative estimated z-traces are not aligned.")
    if expected_piezo_slices.shape != (plane_ids.size, time_s.size):
        raise ValueError("Expected piezo positions, plane IDs, and timestamps are not aligned.")
    if z_correlations.ndim != 3 or z_correlations.shape != (
        plane_ids.size,
        z_correlations.shape[1],
        estimated_time_s.size,
    ):
        raise ValueError("Z-correlations, plane IDs, and timestamps are not aligned.")

    piezo_visible = time_s <= PLOT_END_S
    estimated_visible = estimated_time_s <= PLOT_END_S
    visible_correlation_maps = []
    finite_slice_ranges = []
    line_depth_ranges = []
    required_slice_ranges = []
    for plane_index, plane_correlations in enumerate(z_correlations):
        visible_correlations = plane_correlations[:, estimated_visible]
        relevant_slices = np.flatnonzero(np.any(np.isfinite(visible_correlations), axis=1))
        if relevant_slices.size == 0:
            raise ValueError("No finite z-stack correlations are available in the plotted period.")
        visible_estimated_slices = estimated_z_slices[plane_index, estimated_visible]
        visible_piezo_slices = expected_piezo_slices[plane_index, piezo_visible]
        line_slices = np.concatenate((visible_estimated_slices, visible_piezo_slices))
        line_slices = line_slices[np.isfinite(line_slices)]
        if line_slices.size == 0:
            raise ValueError("No finite depth estimates are available in the plotted period.")
        line_depth_ranges.append((float(line_slices.min()), float(line_slices.max())))
        required_slice_ranges.append(
            (
                max(0, int(np.floor(line_slices.min() + 0.5))),
                min(
                    plane_correlations.shape[0],
                    int(np.ceil(line_slices.max() + 0.5)),
                ),
            )
        )
        finite_slice_ranges.append((int(relevant_slices[0]), int(relevant_slices[-1]) + 1))
        visible_correlation_maps.append(visible_correlations)

    shared_boundaries = []
    for plane_index in range(plane_ids.size - 1):
        shallow_stop = required_slice_ranges[plane_index][1]
        deep_start = required_slice_ranges[plane_index + 1][0]
        if shallow_stop > deep_start:
            raise ValueError(
                "Adjacent plane trace ranges overlap, so they cannot share a heatmap "
                "boundary without clipping a trace."
            )
        line_gap_midpoint = 0.5 * (
            line_depth_ranges[plane_index][1] + line_depth_ranges[plane_index + 1][0]
        )
        boundary = int(np.round(line_gap_midpoint + 0.5))
        shared_boundaries.append(int(np.clip(boundary, shallow_stop, deep_start)))

    slice_starts = [
        min(finite_slice_ranges[0][0], required_slice_ranges[0][0]),
        *shared_boundaries,
    ]
    slice_stops = [
        *shared_boundaries,
        max(finite_slice_ranges[-1][1], required_slice_ranges[-1][1]),
    ]
    slice_ranges = list(zip(slice_starts, slice_stops, strict=True))
    correlation_maps = [
        values[slice_start:slice_stop]
        for values, (slice_start, slice_stop) in zip(
            visible_correlation_maps, slice_ranges, strict=True
        )
    ]

    finite_correlations = np.concatenate(
        [values[np.isfinite(values)] for values in correlation_maps]
    )
    correlation_norm = mpl.colors.Normalize(
        vmin=float(finite_correlations.min()),
        vmax=float(finite_correlations.max()),
    )
    correlation_cmap = mpl.colormaps["gray_r"].copy()
    correlation_cmap.set_bad("0.85")
    fig, axes = plt.subplots(
        plane_ids.size,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    images = []
    visible_times_s = estimated_time_s[estimated_visible]
    for correlation_ax, plane_id, values, (slice_start, slice_stop) in zip(
        axes, plane_ids, correlation_maps, slice_ranges, strict=True
    ):
        image = correlation_ax.imshow(
            values,
            aspect="auto",
            cmap=correlation_cmap,
            norm=correlation_norm,
            interpolation="nearest",
            origin="lower",
            extent=(
                visible_times_s[0],
                visible_times_s[-1],
                slice_start - 0.5,
                slice_stop - 0.5,
            ),
            rasterized=True,
        )
        images.append(image)
        plane_index = len(images) - 1
        correlation_ax.plot(
            estimated_time_s[estimated_visible],
            estimated_z_slices[plane_index, estimated_visible],
            color="#d62728",
            linewidth=1,
            label="Estimated",
        )
        correlation_ax.plot(
            time_s[piezo_visible],
            expected_piezo_slices[plane_index, piezo_visible],
            color="#1f77b4",
            linewidth=1,
            label="Measured piezo",
        )
        correlation_ax.set_ylabel(f"Plane {plane_id}\nZ-stack slice")
        correlation_ax.set_title(
            f"RMSE (unsmoothed): {metrics['rmse_um'][plane_index]:.2f} $\\mu$m",
            loc="right",
            fontsize=10,
        )
        correlation_ax.set_ylim(slice_stop - 0.5, slice_start - 0.5)
        correlation_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=True))
    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, PLOT_END_S)
    fig.colorbar(images[0], ax=axes, label="Correlation", pad=0.02)
    fig.savefig(
        output_dir / f"piezo_and_estimated_z_{dataset_name}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_figures(
    input_root: Path,
    output_dir: Path,
    *,
    style: str | None = None,
) -> None:
    """Load, analyze, and plot the controlled-displacement validation panels."""
    mpl.rcParams.update(FIGURE_STYLE)
    if style:
        plt.style.use(style)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for dataset in discover_datasets(input_root):
        inputs = load_figure_inputs(dataset)
        analysis = prepare_analysis(inputs)
        metrics = calculate_metrics(analysis)
        plot_time_course(analysis, output_dir, dataset.name, metrics)
        common = {
            "dataset": dataset.name,
            "experiment": dataset.experiment,
            "start_s": 0.0,
            "end_s": PLOT_END_S,
            "fitted_offset_um": float(analysis["piezo_global_offset_slices"])
            * inputs.zstack_spacing_um,
        }
        for plane_id, rmse, n_valid in zip(
            analysis["plane_ids"], metrics["rmse_um"], metrics["n_valid"], strict=True
        ):
            metric_rows.append(
                {
                    **common,
                    "plane_id": int(plane_id),
                    "rmse_um": float(rmse),
                    "n_valid": int(n_valid),
                }
            )
        metric_rows.append(
            {
                **common,
                "plane_id": "all",
                "rmse_um": float(metrics["dataset_rmse_um"]),
                "n_valid": int(metrics["n_valid"].sum()),
            }
        )

    with (output_dir / "depth_estimation_rmse.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dataset",
                "experiment",
                "plane_id",
                "start_s",
                "end_s",
                "fitted_offset_um",
                "rmse_um",
                "n_valid",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)


def main() -> None:
    generate_figures(DATA_ROOT, OUTPUT_DIR, style=None)


if __name__ == "__main__":
    main()
