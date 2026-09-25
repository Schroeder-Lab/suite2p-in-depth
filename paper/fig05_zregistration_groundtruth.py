"""Generate Figure 5 panels for controlled z-registration validation."""

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

OUTPUT_DIR = PLOTS_ROOT / "Fig05"

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
DEFAULT_PIEZO_VOLTS_PER_MICRON = 0.025
PLOT_END_S_BY_DATASET = {
    "Stereopes_2024-01-29": 200.0,
    "Styx_2024-03-05": 120.0,
}
CURATED_EXPERIMENTS = (
    ("Stereopes", "2024-01-29", 3),
    ("Styx", "2024-03-05", 4),
)


@dataclass(frozen=True)
class Figure5Dataset:
    """Location and identity of one controlled-displacement dataset."""

    data_folder: Path
    subject: str
    date: str
    experiment: int

    @property
    def name(self) -> str:
        return f"{self.subject}_{self.date}"


@dataclass(frozen=True)
class Figure5Inputs:
    """Raw DAQ data and selected z-registration outputs for one experiment."""

    piezo_voltage: np.ndarray
    piezo_time_s: np.ndarray
    frame_times_s: np.ndarray
    correlations: np.ndarray
    matched_plane_ids: np.ndarray
    plane_ids: np.ndarray
    reference_plane_id: int
    piezo_volts_per_micron: float
    num_acquired_planes: int


def _figure_data_root(input_root: Path) -> Path:
    """Accept either the manuscript-data root or its ``Fig05`` directory."""
    input_root = Path(input_root)
    return input_root if input_root.name == "Fig05" else input_root / "Fig05"


def discover_datasets(input_root: Path) -> list[Figure5Dataset]:
    """Locate the two fixed Figure 5 datasets and raw experiments."""
    figure_data_root = _figure_data_root(input_root)
    if not figure_data_root.is_dir():
        raise FileNotFoundError(f"Figure 5 data directory not found: {figure_data_root}")

    datasets = []
    for subject, date, experiment in CURATED_EXPERIMENTS:
        data_folder = figure_data_root / f"{subject}_{date}"
        experiment_dir = data_folder / "raw" / str(experiment)
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Raw experiment directory not found: {experiment_dir}")
        datasets.append(
            Figure5Dataset(
                data_folder=data_folder,
                subject=subject,
                date=date,
                experiment=experiment,
            )
        )
    return datasets


def _load_daq_inputs(experiment_dir: Path) -> tuple[np.ndarray, ...]:
    """Load the piezo monitor and frame clock through the production DAQ readers."""
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
    return (
        np.asarray(piezo_voltage, dtype=float),
        np.asarray(piezo_time_s, dtype=float),
        np.asarray(frame_times_s, dtype=float),
    )


def _experiment_slice(database: dict, experiment: int) -> slice:
    """Locate one experiment within concatenated Suite2p time-series outputs."""
    data_paths = [Path(str(path)).name for path in database["data_path"]]
    experiment_name = str(experiment)
    if data_paths.count(experiment_name) != 1:
        raise ValueError(
            f"Expected experiment {experiment_name!r} once in db['data_path']; "
            f"found {data_paths.count(experiment_name)} matches."
        )

    frames_per_experiment = np.asarray(database["frames_per_folder"], dtype=int).reshape(-1)
    if frames_per_experiment.size != len(data_paths):
        raise ValueError("db['frames_per_folder'] and db['data_path'] have different lengths.")
    experiment_index = data_paths.index(experiment_name)
    start = int(frames_per_experiment[:experiment_index].sum())
    stop = start + int(frames_per_experiment[experiment_index])
    return slice(start, stop)


def _load_registration_inputs(
    plane_z_dir: Path,
    experiment: int,
    piezo_calibration_override: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, int]:
    """Load and select the registered-plane evidence for one experiment."""
    if not plane_z_dir.is_dir():
        raise FileNotFoundError(f"Registered plane directory not found: {plane_z_dir}")

    database = np.load(plane_z_dir / "db.npy", allow_pickle=True).item()
    settings = np.load(plane_z_dir / "settings.npy", allow_pickle=True).item()
    reg_outputs = np.load(plane_z_dir / "reg_outputs.npy", allow_pickle=True).item()

    num_acquired_planes = int(database["nplanes"])
    ignored_plane_ids = np.asarray(
        database.get("ignore_flyback_singleplanes", database.get("ignore_flyback", [])),
        dtype=int,
    ).reshape(-1)
    plane_ids = np.setdiff1d(np.arange(num_acquired_planes), ignored_plane_ids)
    if plane_ids.size == 0:
        raise ValueError("No imaging planes remain after excluding flyback planes.")

    all_correlations = np.asarray(reg_outputs["corrs_time_refs_planes"], dtype=float)
    if all_correlations.ndim != 3 or all_correlations.shape[1:] != (
        plane_ids.size,
        plane_ids.size,
    ):
        raise ValueError(
            "corrs_time_refs_planes must have shape (time, valid planes, valid planes)."
        )
    all_matched_plane_ids = np.asarray(reg_outputs["planes_across_time"], dtype=int).reshape(-1)
    if all_matched_plane_ids.size != all_correlations.shape[0]:
        raise ValueError("planes_across_time and corrs_time_refs_planes have different lengths.")

    frames_per_experiment = np.asarray(database["frames_per_folder"], dtype=int).reshape(-1)
    if frames_per_experiment.sum() != all_correlations.shape[0]:
        raise ValueError("db['frames_per_folder'] does not reproduce the registration length.")
    selected = _experiment_slice(database, experiment)

    reference_plane_id = int(reg_outputs["reference_plane"])
    reference_matches = np.flatnonzero(plane_ids == reference_plane_id)
    if reference_matches.size != 1:
        raise ValueError("The selected registration reference is not a valid imaging plane.")
    reference_index = int(reference_matches[0])
    correlations = all_correlations[selected, reference_index, :]
    if np.isnan(correlations).all(axis=1).any():
        raise ValueError("At least one time point has no finite plane correlation.")

    matched_plane_ids = all_matched_plane_ids[selected]
    calculated_matches = plane_ids[np.nanargmax(correlations, axis=1)]
    if not np.array_equal(matched_plane_ids, calculated_matches):
        raise ValueError(
            "Saved planes_across_time does not match the correlation maximum for the reference."
        )

    stored_calibration = settings.get("daq", {}).get("piezo_volt_per_micron")
    if piezo_calibration_override is not None:
        piezo_volts_per_micron = float(piezo_calibration_override)
    elif stored_calibration is not None:
        piezo_volts_per_micron = float(stored_calibration)
    else:
        piezo_volts_per_micron = DEFAULT_PIEZO_VOLTS_PER_MICRON
    if not np.isfinite(piezo_volts_per_micron) or piezo_volts_per_micron <= 0:
        raise ValueError("Piezo volts-per-micron calibration must be positive and finite.")

    return (
        correlations,
        matched_plane_ids,
        plane_ids,
        reference_plane_id,
        piezo_volts_per_micron,
        num_acquired_planes,
    )


def load_figure_inputs(
    dataset: Figure5Dataset,
    *,
    piezo_calibration_override: float | None = None,
) -> Figure5Inputs:
    """Load the curated DAQ and z-registration inputs for one dataset."""
    experiment_dir = dataset.data_folder / "raw" / str(dataset.experiment)
    piezo_voltage, piezo_time_s, frame_times_s = _load_daq_inputs(experiment_dir)
    (
        correlations,
        matched_plane_ids,
        plane_ids,
        reference_plane_id,
        piezo_volts_per_micron,
        num_acquired_planes,
    ) = _load_registration_inputs(
        dataset.data_folder / "suite2p" / "plane_z",
        dataset.experiment,
        piezo_calibration_override,
    )
    return Figure5Inputs(
        piezo_voltage=piezo_voltage,
        piezo_time_s=piezo_time_s,
        frame_times_s=frame_times_s,
        correlations=correlations,
        matched_plane_ids=matched_plane_ids,
        plane_ids=plane_ids,
        reference_plane_id=reference_plane_id,
        piezo_volts_per_micron=piezo_volts_per_micron,
        num_acquired_planes=num_acquired_planes,
    )


def _manual_piezo_displacement(
    inputs: Figure5Inputs,
    automated_template_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the repeating multi-plane trajectory from the piezo monitor."""
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
        inputs.num_acquired_planes,
        len(inputs.frame_times_s) - inputs.num_acquired_planes,
        inputs.num_acquired_planes,
    ):
        cycle_residuals = []
        for plane_id in range(inputs.num_acquired_planes):
            frame_index = cycle_start + plane_id
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
                piezo_smoothed_v[sample_start:sample_stop] - automated_template_v[:, plane_id]
            )
        if not cycle_residuals:
            continue
        cycle_times_s.append(
            0.5
            * (
                inputs.frame_times_s[cycle_start]
                + inputs.frame_times_s[cycle_start + inputs.num_acquired_planes]
            )
        )
        manual_offsets_v.append(float(np.median(np.concatenate(cycle_residuals))))

    if not manual_offsets_v:
        raise ValueError("No complete imaging rounds are available for piezo subtraction.")
    manual_displacement_um = np.asarray(manual_offsets_v) / inputs.piezo_volts_per_micron
    manual_displacement_um -= np.nanmedian(manual_displacement_um)
    return np.asarray(cycle_times_s), manual_displacement_um


def prepare_analysis(inputs: Figure5Inputs) -> dict[str, np.ndarray]:
    """Align measured piezo displacement with the z-registration estimate."""
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
    manual_time_s, manual_displacement_um = _manual_piezo_displacement(
        inputs,
        automated_template_um,
    )

    registration_time_s = inputs.frame_times_s[::num_planes]
    if registration_time_s.size != inputs.correlations.shape[0]:
        raise ValueError(
            f"Experiment has {registration_time_s.size} imaging-round timestamps but "
            f"{inputs.correlations.shape[0]} registration estimates."
        )
    time_origin_s = float(registration_time_s[0])
    registration_time_s = registration_time_s - time_origin_s
    manual_time_s = manual_time_s - time_origin_s

    nominal_plane_depths_um = np.nanmedian(
        automated_template_um[:, inputs.plane_ids],
        axis=0,
    )
    if not np.all(np.isfinite(nominal_plane_depths_um)):
        raise ValueError("Cannot measure a finite piezo depth for every valid imaging plane.")
    reference_index = int(np.flatnonzero(inputs.plane_ids == inputs.reference_plane_id)[0])
    plane_index_by_id = {int(plane_id): index for index, plane_id in enumerate(inputs.plane_ids)}
    try:
        matched_plane_indices = np.asarray(
            [plane_index_by_id[int(plane_id)] for plane_id in inputs.matched_plane_ids],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError(f"Matched plane {error.args[0]} is not a valid imaging plane.") from error

    # A deeper source plane matching the reference implies an equal, opposite piezo displacement.
    plane_displacements_um = nominal_plane_depths_um[reference_index] - nominal_plane_depths_um
    registered_displacement_um = plane_displacements_um[matched_plane_indices]
    return {
        "manual_piezo_time_s": manual_time_s,
        "manual_piezo_um": manual_displacement_um,
        "registration_time_s": registration_time_s,
        "registered_displacement_um": registered_displacement_um,
        "correlations": inputs.correlations,
        "matched_plane_ids": inputs.matched_plane_ids,
        "matched_plane_indices": matched_plane_indices,
        "plane_ids": inputs.plane_ids,
        "reference_plane_id": np.asarray(inputs.reference_plane_id),
        "nominal_plane_depths_um": nominal_plane_depths_um,
        "plane_displacements_um": plane_displacements_um,
    }


def calculate_metrics(
    analysis: dict[str, np.ndarray],
    plot_end_s: float,
) -> dict[str, np.ndarray]:
    """Calculate all-sample and in-volume RMSE over the plotted snippet."""
    registration_time_s = analysis["registration_time_s"]
    registered_displacement_um = analysis["registered_displacement_um"]
    if registered_displacement_um.shape != registration_time_s.shape:
        raise ValueError("Registration displacement and timestamps are not aligned.")

    visible = (registration_time_s >= 0) & (registration_time_s <= plot_end_s)
    measured_piezo_um = np.interp(
        registration_time_s[visible],
        analysis["manual_piezo_time_s"],
        analysis["manual_piezo_um"],
        left=np.nan,
        right=np.nan,
    )
    residual_um = registered_displacement_um[visible] - measured_piezo_um
    valid = np.isfinite(residual_um)
    if not np.any(valid):
        raise ValueError("No finite paired displacement samples in the plotted period.")

    plane_displacements_um = np.asarray(analysis["plane_displacements_um"], dtype=float)
    if plane_displacements_um.ndim != 1 or not np.all(np.isfinite(plane_displacements_um)):
        raise ValueError("Imaging-plane displacements must be a finite one-dimensional array.")
    volume_min_um = float(plane_displacements_um.min())
    volume_max_um = float(plane_displacements_um.max())
    in_volume = valid & (measured_piezo_um >= volume_min_um) & (measured_piezo_um <= volume_max_um)
    if not np.any(in_volume):
        raise ValueError("No finite paired displacement samples fall within the imaged volume.")
    return {
        "rmse_um": np.asarray(np.sqrt(np.mean(residual_um[valid] ** 2))),
        "n_valid": np.asarray(valid.sum()),
        "rmse_in_volume_um": np.asarray(np.sqrt(np.mean(residual_um[in_volume] ** 2))),
        "n_valid_in_volume": np.asarray(in_volume.sum()),
    }


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert ordered bin centres to edges while preserving their spacing."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("Heat-map coordinates must contain at least two finite centres.")
    differences = np.diff(centers)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("Heat-map coordinates must be strictly monotonic.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * differences[0]
    edges[-1] = centers[-1] + 0.5 * differences[-1]
    return edges


def plot_time_course(
    analysis: dict[str, np.ndarray],
    output_dir: Path,
    dataset: Figure5Dataset,
) -> Path:
    """Overlay ground-truth and registered displacement on the correlation map."""
    manual_time_s = analysis["manual_piezo_time_s"]
    manual_piezo_um = analysis["manual_piezo_um"]
    registration_time_s = analysis["registration_time_s"]
    registered_displacement_um = analysis["registered_displacement_um"]
    correlations = analysis["correlations"]
    plane_ids = analysis["plane_ids"]
    nominal_plane_depths_um = analysis["nominal_plane_depths_um"]

    if manual_time_s.size == 0 or manual_piezo_um.shape != manual_time_s.shape:
        raise ValueError("No aligned manual piezo displacement samples are available.")
    if registered_displacement_um.shape != registration_time_s.shape:
        raise ValueError("Registration displacement and timestamps are not aligned.")
    if correlations.shape != (registration_time_s.size, plane_ids.size):
        raise ValueError("Correlation map, timestamps, and valid planes are not aligned.")
    if nominal_plane_depths_um.shape != plane_ids.shape:
        raise ValueError("Nominal piezo depths and valid imaging planes are not aligned.")

    try:
        plot_end_s = PLOT_END_S_BY_DATASET[dataset.name]
    except KeyError as error:
        raise ValueError(f"No Figure 5 plot limit is configured for {dataset.name}.") from error
    if plot_end_s > registration_time_s[-1]:
        raise ValueError(
            f"Configured plot limit {plot_end_s:g} s exceeds the duration of {dataset.name}."
        )
    metrics = calculate_metrics(analysis, plot_end_s)

    plane_displacements_um = analysis["plane_displacements_um"]
    if plane_displacements_um.shape != plane_ids.shape:
        raise ValueError("Imaging-plane displacements and valid planes are not aligned.")
    if not np.all(np.diff(plane_displacements_um) < 0):
        raise ValueError("Valid plane IDs must progress from smaller to larger piezo displacement.")
    time_edges_s = _centers_to_edges(registration_time_s)
    displacement_edges_um = _centers_to_edges(plane_displacements_um[::-1])

    finite_correlations = correlations[np.isfinite(correlations)]
    if finite_correlations.size == 0:
        raise ValueError("No finite correlations are available for the heat map.")
    correlation_min = float(finite_correlations.min())
    correlation_max = float(finite_correlations.max())
    if np.isclose(correlation_min, correlation_max):
        correlation_max = correlation_min + np.finfo(float).eps
    correlation_norm = mpl.colors.Normalize(vmin=correlation_min, vmax=correlation_max)
    correlation_cmap = mpl.colormaps["gray_r"].copy()
    correlation_cmap.set_bad("0.85")

    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = axis.pcolormesh(
        time_edges_s,
        displacement_edges_um,
        correlations[:, ::-1].T,
        cmap=correlation_cmap,
        norm=correlation_norm,
        shading="flat",
        rasterized=True,
    )
    axis.plot(
        manual_time_s,
        manual_piezo_um,
        color="#1f77b4",
        linewidth=1.25,
        label="Measured piezo",
    )
    axis.step(
        registration_time_s,
        registered_displacement_um,
        where="mid",
        color="#d62728",
        linewidth=1,
        label="Best-matching plane",
    )
    tick_labels = [
        rf"Plane {plane_id} ({displacement:+.1f} $\mu$m)"
        for plane_id, displacement in zip(
            plane_ids,
            plane_displacements_um,
            strict=True,
        )
    ]
    axis.set_yticks(plane_displacements_um, tick_labels)
    axis.set_ylabel(r"Imaging plane and Z displacement")
    axis.set_xlabel("Time (s)")
    axis.set_title(
        f"{dataset.subject} {dataset.date}, experiment {dataset.experiment}",
        loc="left",
    )
    axis.set_title(
        f"RMSE (all): {float(metrics['rmse_um']):.2f} $\\mu$m\n"
        f"RMSE (in volume): {float(metrics['rmse_in_volume_um']):.2f} $\\mu$m",
        loc="right",
        fontsize=10,
    )
    axis.set_xlim(0, plot_end_s)
    manual_visible = (manual_time_s >= 0) & (manual_time_s <= plot_end_s)
    y_values = np.concatenate((displacement_edges_um, manual_piezo_um[manual_visible]))
    axis.set_ylim(float(np.nanmin(y_values)), float(np.nanmax(y_values)))
    axis.legend(frameon=False, loc="best")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.colorbar(image, ax=axis, label="Correlation", pad=0.02)

    output_path = output_dir / f"piezo_and_best_plane_{dataset.name}.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_figures(
    input_root: Path,
    output_dir: Path,
    *,
    style: str | None = None,
    piezo_volts_per_micron: float | None = None,
) -> list[Path]:
    """Load, analyze, and plot both controlled z-registration recordings."""
    mpl.rcParams.update(FIGURE_STYLE)
    if style:
        plt.style.use(style)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for dataset in discover_datasets(input_root):
        inputs = load_figure_inputs(
            dataset,
            piezo_calibration_override=piezo_volts_per_micron,
        )
        analysis = prepare_analysis(inputs)
        outputs.append(plot_time_course(analysis, output_dir, dataset))
    return outputs


def main() -> None:
    generate_figures(
        DATA_ROOT,
        OUTPUT_DIR,
        style=None,
        piezo_volts_per_micron=None,
    )


if __name__ == "__main__":
    main()
