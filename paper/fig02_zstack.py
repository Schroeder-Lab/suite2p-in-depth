"""Generate the Figure 2 z-stack correction workflow panels."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import skimage

from suite2p_in_depth import preprocess_traces

try:
    from paper.paths import DATA_ROOT, PLOTS_ROOT
except ModuleNotFoundError:
    from paths import DATA_ROOT, PLOTS_ROOT

OUTPUT_DIR = PLOTS_ROOT / "Fig02"

FIGURE_STYLE = {
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _normalized_image(image):
    image = np.asarray(image, dtype=np.float32)
    minimum = float(np.nanmin(image))
    maximum = float(np.nanpercentile(image, 99.8))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("Cannot normalize a constant or non-finite figure image.")
    return (np.clip(image, minimum, maximum) - minimum) / (maximum - minimum)


def _finish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_figures(input_root, output_dir, *, style=None, seed=0):
    """Generate the Figure 2 z-stack-correction panels.

    Parameters
    ----------
    input_root
        Manuscript-data root containing ``Fig02/Ely_2024-07-01`` and its
        documented Suite2p arrays. Images use ``(z, y, x)`` or ``(y, x)``
        order; piezo depths are in micrometres and times are in seconds.
    output_dir
        Directory receiving the fixed Figure 2 PNG panels.
    style
        Optional Matplotlib style name or file applied after manuscript
        defaults.
    seed
        Reserved deterministic seed for stochastic panel elements.
    """
    input_root = Path(input_root)
    data_folder = input_root / "Fig02" / "Ely_2024-07-01"
    fig_folder = Path(output_dir)
    plane_folder = data_folder / "plane 1"
    roi_index = 25
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Figure 2 data directory not found: {data_folder}")
    if not plane_folder.is_dir():
        raise FileNotFoundError(f"Figure 2 plane directory not found: {plane_folder}")
    mpl.rcParams.update(FIGURE_STYLE)
    if style:
        plt.style.use(style)
    np.random.default_rng(seed)
    fig_folder.mkdir(parents=True, exist_ok=True)

    piezo_per_plane = np.load(data_folder / "piezo_per_plane.npy")
    piezo_time = np.load(data_folder / "piezo_time.npy")
    ops = np.load(plane_folder / "ops.npy", allow_pickle=True).item()
    plane_stack = skimage.io.imread(plane_folder / "stack_plane1_chan1.tif")
    best_slice = int(np.load(plane_folder / "bestSlice_plane1.npy"))
    zcorr = np.load(plane_folder / "zcorr_plane1.npy")
    db = np.load(plane_folder / "db_zcorrect.npy", allow_pickle=True).item()
    settings = np.load(plane_folder / "settings_zcorrect.npy", allow_pickle=True).item()
    detect_outputs = np.load(plane_folder / "detect_outputs_zcorrect.npy", allow_pickle=True).item()
    zprofiles = np.load(data_folder / "2pRois.zProfiles.npy")
    stat = np.load(plane_folder / "stat.npy", allow_pickle=True)
    roi_ids = np.load(data_folder / "2pRois.ids.npy")
    fluorescence = np.load(plane_folder / "F.npy")
    neuropil = np.load(plane_folder / "Fneu.npy")
    fluorescence_z = np.load(data_folder / "2pCalcium.F_zcorrected.npy")
    neuropil_z = np.load(data_folder / "2pCalcium.N_zcorrected.npy")
    dff = np.load(data_folder / "2pCalcium.dff.npy")

    if piezo_per_plane.ndim != 2 or piezo_time.ndim != 1 or piezo_time.size < 2:
        raise ValueError("Piezo inputs must have shapes (samples, planes) and (time,).")
    time_bin = np.nanmedian(np.diff(piezo_time))
    piezo_plot_time = np.arange(piezo_per_plane.shape[0]) * time_bin * 1000
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    for plane in range(piezo_per_plane.shape[1]):
        ax.plot(piezo_plot_time, piezo_per_plane[:, plane], label=f"Plane {plane}")
    ax.invert_yaxis()
    ax.set_xticks(range(0, 31, 10))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"Piezo position ($\mu$m)")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "piezo_per_plane.pdf", bbox_inches="tight")
    plt.close(fig)

    if plane_stack.ndim != 3 or not 0 <= best_slice < plane_stack.shape[0]:
        raise ValueError("The registered z-stack or best-slice index is invalid.")
    reference_norm = _normalized_image(ops["meanImg"])
    stack_norm = _normalized_image(plane_stack[best_slice])
    if reference_norm.shape != stack_norm.shape:
        raise ValueError("Reference image and matching z-stack slice have different shapes.")

    for image, filename in (
        (reference_norm, "mean_image.pdf"),
        (stack_norm, "zstack_reference_slice.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        fig.savefig(fig_folder / filename, bbox_inches="tight")
        plt.close(fig)

    rgb = np.zeros((*reference_norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.round(stack_norm * 255).astype(np.uint8)
    rgb[..., 1] = np.round(reference_norm * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(rgb)
    ax.axis("off")
    fig.savefig(fig_folder / "overlay_ref_zstack.pdf", bbox_inches="tight")
    plt.close(fig)

    if zcorr.ndim != 2 or zcorr.shape[0] != plane_stack.shape[0]:
        raise ValueError("zcorr must have shape (z-stack slices, time).")
    valid_slices = np.where(~np.isnan(zcorr).any(axis=1))[0]
    if valid_slices.size == 0:
        raise ValueError("No valid z-correlation slices remain.")
    ztrace = np.nanargmax(
        sp.ndimage.gaussian_filter1d(zcorr[valid_slices], 2, axis=0, mode="nearest"),
        axis=0,
    ).astype(int)
    ztrace = valid_slices[ztrace]
    experiment_frames = int(db["frames_per_folder"][0])
    if experiment_frames > ztrace.size:
        raise ValueError("The first experiment is longer than the z-correlation trace.")
    ztrace_first = ztrace[:experiment_frames]
    time = np.arange(experiment_frames) / settings["fs"]
    reference_depth = int(np.nanmedian(ztrace_first))

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, ztrace_first, color="black", linewidth=1)
    ax.axhline(reference_depth, color="darkred", linewidth=2, label="Reference depth")
    ax.invert_yaxis()
    ax.set_title("Depth trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "depth_trace.pdf", bbox_inches="tight")
    plt.close(fig)

    if roi_index >= len(roi_ids) or roi_index >= zprofiles.shape[0]:
        raise ValueError(f"roi_index {roi_index} is unavailable in the prepared figure data.")
    roi_id = int(np.asarray(roi_ids[roi_index]).ravel()[0])
    if roi_id >= len(stat):
        raise ValueError(f"Raw Suite2p ROI {roi_id} is unavailable.")
    roi = stat[roi_id]

    fig, ax = plt.subplots(figsize=(4, 8), constrained_layout=True)
    ax.plot(zprofiles[roi_index], np.arange(zprofiles.shape[1]), color="black", linewidth=1)
    ax.axhline(reference_depth, color="darkred", linewidth=1, label="Ref. depth")
    ax.axhline(best_slice, color="black", linestyle="--", linewidth=1, label="Ref. im. match")
    ax.invert_yaxis()
    ax.set_title("Axial profile")
    ax.set_xlabel("Fluorescence (a.u.)")
    ax.set_ylabel(r"Depth ($\mu$m)")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "axial_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    pixel_height = settings["z_correction"]["spacing"]
    pixel_width = detect_outputs["fov_size"] / db["Lx"]
    center_y, center_x = (int(value) for value in roi["med"])
    x_indices = np.clip(center_x + np.arange(-10, 11), 0, plane_stack.shape[2] - 1)
    fig, ax = plt.subplots(figsize=(4, 8), constrained_layout=True)
    ax.imshow(
        plane_stack[:, center_y, x_indices],
        cmap="gray",
        aspect=pixel_height / pixel_width,
    )
    ax.axhline(reference_depth, color="darkred", linewidth=1, label="Ref. depth")
    ax.axhline(best_slice, color="black", linestyle="--", linewidth=1, label="Ref. im. match")
    ax.set_title("Z-stack")
    ax.set_ylabel(r"Depth ($\mu$m)")
    _finish_axes(ax)
    fig.savefig(fig_folder / "zstack_ROI.pdf", bbox_inches="tight")
    plt.close(fig)

    factors = zprofiles[roi_index] / zprofiles[roi_index, reference_depth]
    correction = factors[ztrace_first]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, correction, color="black", linewidth=1)
    ax.axhline(1, color="darkred", linewidth=2, label="Reference depth")
    ax.set_title("Correction trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correction factor")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "correction_trace.pdf", bbox_inches="tight")
    plt.close(fig)

    zero = settings["delta_F"]["absolute_zero"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(
        time, fluorescence[roi_id, :experiment_frames] - zero, color="black", linewidth=1, label="F"
    )
    ax.plot(
        time,
        fluorescence_z[:experiment_frames, roi_index],
        color="gray",
        linewidth=1,
        label="F_corrected",
    )
    ax.plot(
        time,
        neuropil[roi_id, :experiment_frames] - zero,
        color="darkorange",
        linewidth=1,
        label="N",
    )
    ax.plot(
        time,
        neuropil_z[:experiment_frames, roi_index],
        color="orange",
        linewidth=1,
        label="N_corrected",
    )
    ax.set_title("Calcium trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "calcium_traces.pdf", bbox_inches="tight")
    plt.close(fig)

    fluorescence_roi = fluorescence_z[:experiment_frames, [roi_index]]
    neuropil_roi = neuropil_z[:experiment_frames, [roi_index]]
    baseline_f = preprocess_traces.get_f0(
        fluorescence_roi,
        settings["fs"],
        f0_percentile=settings["delta_F"]["F0_percentile"],
        window_size=settings["delta_F"]["F0_window"],
    )
    baseline_n = preprocess_traces.get_f0(
        neuropil_roi,
        settings["fs"],
        f0_percentile=settings["delta_F"]["F0_percentile"],
        window_size=settings["delta_F"]["F0_window"],
    )
    corrected, regression, f_bins, n_bins = preprocess_traces.correct_neuropil(
        fluorescence_roi,
        neuropil_roi,
        settings["fs"],
        baseline_f,
        baseline_percentile=settings["delta_F"]["F0_percentile"],
        baseline_window=settings["delta_F"]["F0_window"],
    )

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.scatter(neuropil_roi - baseline_n, fluorescence_roi - baseline_f, color="black", s=10)
    ax.scatter(n_bins, f_bins, facecolors="none", edgecolors="red", s=15)
    x_line = np.asarray(ax.get_xlim())
    ax.plot(x_line, regression[0, 0] + regression[0, 1] * x_line, color="red", linewidth=2)
    ax.set_xlabel("Neuropil (z-corrected)")
    ax.set_ylabel("Fluorescence (z-corrected)")
    _finish_axes(ax)
    fig.savefig(fig_folder / "signal-neuropil_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, corrected[:, 0], color="black", linewidth=1, label="F_wo_npil")
    ax.plot(time, baseline_f[:, 0], color="grey", linewidth=1, label="F0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.legend(frameon=False, loc="best")
    _finish_axes(ax)
    fig.savefig(fig_folder / "neuropil-corrected_traces.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, dff[:experiment_frames, roi_index], color="black", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    _finish_axes(ax)
    fig.savefig(fig_folder / "dFF_traces.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    generate_figures(DATA_ROOT, OUTPUT_DIR, style=None, seed=0)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2 z-stack panels.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_figures(args.input_root, args.output_dir, style=args.style, seed=args.seed)


if __name__ == "__main__":
    cli()
