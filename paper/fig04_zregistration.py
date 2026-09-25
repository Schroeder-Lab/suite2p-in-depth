"""Generate the Figure 4 z-registration workflow panels."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

OUTPUT_DIR = PLOTS_ROOT / "Fig04"

FIGURE_STYLE = {
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def generate_figures(input_root, output_dir, *, style=None, seed=0):
    """Generate the Figure 4 multi-plane z-registration panels.

    Parameters
    ----------
    input_root
        Manuscript-data root containing ``Fig04/Io_2023-02-13``. Correlation
        arrays use ``(time, references, planes)`` order, piezo depth is in
        micrometres, and time arrays are in seconds.
    output_dir
        Directory receiving reference-image, piezo, and registration panels.
    style
        Optional Matplotlib style name or file applied after manuscript
        defaults.
    seed
        Reserved deterministic seed for stochastic panel elements.
    """
    input_root = Path(input_root)
    data_folder = input_root / "Fig04" / "Io_2023-02-13"
    fig_folder = Path(output_dir)
    plane_folder = data_folder / "plane_z"
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Figure 4 data directory not found: {data_folder}")
    if not plane_folder.is_dir():
        raise FileNotFoundError(f"Registered plane directory not found: {plane_folder}")
    mpl.rcParams.update(FIGURE_STYLE)
    if style:
        plt.style.use(style)
    np.random.default_rng(seed)
    fig_folder.mkdir(parents=True, exist_ok=True)

    db = np.load(plane_folder / "db.npy", allow_pickle=True).item()
    reg_outputs = np.load(plane_folder / "reg_outputs.npy", allow_pickle=True).item()
    frame_times = np.load(data_folder / "frame_times.npy")
    piezo_time = np.load(data_folder / "piezo_time.npy")
    piezo_trace = np.load(data_folder / "piezo_trace.npy")
    piezo_good = np.load(data_folder / "piezo_per_plane_shifted_good_+011.npy")
    piezo_bad = np.load(data_folder / "piezo_per_plane_shifted_bad_+015.npy")

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"][:5]
    planes = np.delete(np.arange(db["nplanes"]), db["ignore_flyback_singleplanes"])
    num_planes = len(planes)
    if num_planes < 1:
        raise ValueError("No valid imaging planes remain for Figure 4.")
    correlations = np.asarray(reg_outputs["corrs_time_refs_planes"])
    if correlations.ndim != 3 or correlations.shape[1:] != (num_planes, num_planes):
        raise ValueError(
            "corrs_time_refs_planes must have shape (time, valid planes, valid planes)."
        )
    planes_time_refs = np.nanargmax(correlations, axis=2)

    experiment_frames = np.cumsum(db["frames_per_folder"])
    if len(experiment_frames) < 4:
        raise ValueError("Figure 4 requires at least four experiments.")
    example_frames = range(experiment_frames[2], experiment_frames[3])
    shifts = planes_time_refs - np.arange(num_planes)
    min_shifts = np.nanmin(shifts, axis=1)
    max_shifts = np.nanmax(shifts, axis=1)

    reference_images = reg_outputs["refImg_singleplanes"]
    for index, image in enumerate(reference_images):
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        fig.savefig(
            fig_folder / f"reference_plane{planes[index]:02d}.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)

    if correlations.shape[0] <= 22:
        raise ValueError("Figure 4 requires at least 23 imaging cycles.")
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    image = ax.imshow(correlations[22], cmap="gray_r")
    ax.set_xlabel("Imaging planes")
    ax.set_ylabel("Reference planes")
    tick_positions = np.arange(num_planes)
    ax.set_xticks(tick_positions, planes)
    ax.set_yticks(tick_positions, planes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(image, ax=ax)
    fig.savefig(fig_folder / "corrs_refs-vs-planes.pdf", bbox_inches="tight")
    plt.close(fig)

    gap = 3
    top = 0
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for reference in range(correlations.shape[1]):
        ax.axhline(
            top + reference,
            color=colors[reference],
            linewidth=2,
            label=f"Ref. {planes[reference]}",
        )
        for plane in range(correlations.shape[2]):
            x = np.where(planes_time_refs[example_frames, reference] == plane)[0]
            ax.scatter(x, np.full_like(x, top + plane), color=colors[plane], marker=".", s=50)
        top += num_planes + gap
    ax.invert_yaxis()
    ax.set_xlim(0, len(example_frames))
    ax.set_xlabel("Imaging cycles")
    ax.set_ylabel("Imaging plane")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_folder / "best_plane_per_reference.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(min_shifts[example_frames], color="blue", label="min_shift")
    ax.plot(max_shifts[example_frames], color="red", label="max_shift")
    ax.set_xlim(0, len(example_frames))
    ax.set_xlabel("Imaging cycles")
    ax.set_ylabel("Shift (planes)")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_folder / "shifts.pdf", bbox_inches="tight")
    plt.close(fig)

    if frame_times.ndim != 1 or piezo_time.ndim != 1 or piezo_trace.shape != piezo_time.shape:
        raise ValueError("Frame and piezo inputs must be one-dimensional and aligned.")
    frame_times_shifted = frame_times - 0.006
    frame_limits = [db["nplanes"], db["nplanes"] * 4 + 1]
    if frame_limits[1] + 1 >= frame_times_shifted.size:
        raise ValueError("Figure 4 piezo panel requires more frame-clock samples.")
    frame_indices = range(frame_limits[0], frame_limits[1])
    time_limits = [frame_times_shifted[frame_limits[0]], frame_times_shifted[frame_limits[1]]]
    sample_limits = [
        np.searchsorted(piezo_time, time_limits[0], side="left"),
        np.searchsorted(piezo_time, time_limits[1], side="left"),
    ]
    if sample_limits[1] >= piezo_time.size:
        raise ValueError("Piezo samples do not cover the selected frame interval.")
    sample_indices = range(sample_limits[0], sample_limits[1])
    grays = ["#BBBBBB", "#777777", "#333333"]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for index, frame in enumerate(frame_indices):
        plane_id = index % db["nplanes"]
        color = grays[plane_id] if plane_id < planes[0] else colors[plane_id - planes[0]]
        ax.axvspan(
            frame_times_shifted[frame],
            frame_times_shifted[frame + 1],
            facecolor=color,
            alpha=0.15,
            edgecolor="none",
        )
    ax.plot(piezo_time[sample_indices], piezo_trace[sample_indices], color="black")
    ax.set_xlim(piezo_time[sample_limits[0]], piezo_time[sample_limits[1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_folder / "piezo_trace.pdf", bbox_inches="tight")
    plt.close(fig)

    frame_duration = np.median(np.diff(frame_times))
    for values, filename in (
        (piezo_good, "piezo_per_plane.pdf"),
        (piezo_bad, "piezo_per_plane_shifted.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.plot(np.arange(values.shape[0]) * frame_duration, values)
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Depth ($\mu$m)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(fig_folder / filename, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    generate_figures(DATA_ROOT, OUTPUT_DIR, style=None, seed=0)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4 z-registration panels.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_figures(args.input_root, args.output_dir, style=args.style, seed=args.seed)


if __name__ == "__main__":
    cli()
