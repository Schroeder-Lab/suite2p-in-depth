"""Plot accepted Suite2p ROI and neuropil masks."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_neuropil_masks(
    plane_directory: str | Path, output_path: str | Path | None = None, show: bool = False
) -> list[Figure]:
    """Plot accepted ROI and neuropil masks from one Suite2p plane directory.

    Masks use Suite2p ``(y, x)`` pixel order. Neuropil pixels are labelled 1
    and ROI pixels 2. Supplying ``output_path`` writes one image and therefore
    requires exactly one accepted ROI; otherwise figures are returned only.

    Parameters
    ----------
    plane_directory : str or Path
        Suite2p plane containing ``stat.npy``, ``iscell.npy``, and metadata.
    output_path : str or Path or None, optional
        Destination image path when exactly one ROI is accepted.
    show : bool, optional
        Display figures interactively when true; otherwise close them.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One figure for each accepted ROI.

    Raises
    ------
    FileNotFoundError
        If required Suite2p outputs are missing.
    ValueError
        If masks or image metadata are invalid, no ROI is accepted, or a
        single output path is requested for multiple accepted ROIs.
    """
    plane_directory = Path(plane_directory)
    stat_path = plane_directory / "stat.npy"
    iscell_path = plane_directory / "iscell.npy"
    ops_path = plane_directory / "ops.npy"
    db_path = plane_directory / "db.npy"
    if not stat_path.is_file() or not iscell_path.is_file():
        raise FileNotFoundError(
            f"Expected stat.npy and iscell.npy in plane directory {plane_directory}."
        )
    if ops_path.is_file():
        image_metadata = np.load(ops_path, allow_pickle=True).item()
    elif db_path.is_file():
        image_metadata = np.load(db_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"Expected ops.npy or db.npy in {plane_directory}.")
    shape = (int(image_metadata["Ly"]), int(image_metadata["Lx"]))
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("Ly and Lx must be positive.")

    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path)
    if len(stat) != len(iscell):
        raise ValueError("stat.npy and iscell.npy contain different numbers of ROIs.")

    accepted = [
        (index, roi)
        for index, (roi, cell) in enumerate(zip(stat, iscell, strict=True))
        if bool(cell[0])
    ]
    if not accepted:
        raise ValueError("No accepted ROIs were found.")

    figures = []

    for index, roi in accepted:
        roi_img: np.ndarray = np.zeros(shape, dtype=np.uint8)
        neuropil_idx = np.asarray(roi["neuropil_mask"], dtype=np.int64).ravel()
        if neuropil_idx.size:
            yy_n, xx_n = np.unravel_index(neuropil_idx, shape)
            roi_img[yy_n, xx_n] = 1
        ypix = np.asarray(roi["ypix"], dtype=np.int64).ravel()
        xpix = np.asarray(roi["xpix"], dtype=np.int64).ravel()
        roi_img[ypix, xpix] = 2

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(roi_img)
        ax.set_title(f"ROI {index}")
        ax.axis("off")
        fig.tight_layout()
        figures.append(fig)

    if output_path is not None:
        output_path = Path(output_path)
        if len(figures) != 1:
            raise ValueError(
                "output_path is supported only when exactly one accepted ROI is present."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figures[0].savefig(output_path)
    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return figures


def main() -> None:
    """Run the neuropil-mask plotting command.

    The command accepts a Suite2p plane directory and optionally writes one
    figure or displays the figures interactively.
    """
    parser = argparse.ArgumentParser(description="Plot Suite2p neuropil masks.")
    parser.add_argument("plane_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_neuropil_masks(args.plane_directory, args.output, args.show)


if __name__ == "__main__":
    main()
