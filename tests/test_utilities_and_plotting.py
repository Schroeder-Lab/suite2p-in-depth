from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cycler import cycler
from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg

from suite2p_in_depth import correct_traces, registration_plotting, zstack
from suite2p_in_depth.utilities import extract_lowest_pixel_values as dark
from suite2p_in_depth.utilities import plot_neuropil_masks as masks
from suite2p_in_depth.utilities import plot_piezo_traces as piezo
from suite2p_in_depth.utilities import recreate_registered_bins as recreate


def _assert_legends_are_inside_and_clear_of_plots(figure, expected_legends: int) -> None:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    legends = [axis.get_legend() for axis in figure.axes if axis.get_legend() is not None]
    plot_axes = [axis for axis in figure.axes if axis.has_data()]

    assert len(legends) == expected_legends
    for legend in legends:
        legend_bounds = legend.get_window_extent(renderer)
        assert figure.bbox.contains(legend_bounds.x0, legend_bounds.y0)
        assert figure.bbox.contains(legend_bounds.x1, legend_bounds.y1)
        for plot_axis in plot_axes:
            plot_bounds = plot_axis.get_window_extent(renderer)
            overlap_x = min(legend_bounds.x1, plot_bounds.x1) - max(
                legend_bounds.x0, plot_bounds.x0
            )
            overlap_y = min(legend_bounds.y1, plot_bounds.y1) - max(
                legend_bounds.y0, plot_bounds.y0
            )
            assert overlap_x <= 0 or overlap_y <= 0


@pytest.mark.parametrize(
    ("n_rois", "maximum", "expected"),
    [
        (5, 0, []),
        (5, 2, [1, 3]),
        (5, 5, [0, 1, 2, 3, 4]),
        (3, 10, [0, 1, 2]),
        (4, "all", [0, 1, 2, 3]),
    ],
)
def test_roi_plot_selection_is_capped_even_and_deterministic(n_rois, maximum, expected) -> None:
    first = correct_traces.select_roi_plot_indices(n_rois, maximum)
    second = correct_traces.select_roi_plot_indices(n_rois, maximum)

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, first)
    assert np.issubdtype(first.dtype, np.integer)


def test_registration_plotting_writes_complete_headless_qc_set(tmp_path: Path) -> None:
    settings = tmp_path / "settings.npy"
    np.save(settings, {"registration": {"align_by_chan2": False}})
    database = {
        "nplanes": 3,
        "ignore_flyback_singleplanes": [],
        "frames_per_folder": [3, 3],
        "settings_path": str(settings),
    }
    image = np.arange(36, dtype=float).reshape(6, 6)
    correlations = np.full((6, 3, 3), 0.1)
    for frame in range(6):
        correlations[frame, :, frame % 3] = 1
    outputs = {
        "reference_plane": 1,
        "refImg_singleplanes": np.stack((image, image + 2, image + 4)),
        "refImg": image + 1,
        "meanImg": image + 3,
        "corrs_time_refs_planes": correlations,
        "badframes": np.array([False, True, True, False, False, True]),
    }
    output = tmp_path / "plots"

    registration_plotting.make_plots(database, outputs, output)

    assert {path.name for path in output.iterdir()} == {
        "01_reference_images.png",
        "02_mean_image_zregistration.png",
        "03_z_positions.png",
    }
    assert all(path.stat().st_size > 100 for path in output.iterdir())

    np.save(settings, {"registration": {"align_by_chan2": True}})
    outputs["meanImg_chan2"] = image + 5
    registration_plotting.plot_final_ref_and_mean_img(database, outputs)
    assert len(plt.gcf().axes) == 4
    plt.close()


def test_dark_level_dataset_integration_handles_skip_and_missing_planes(
    monkeypatch, tmp_path: Path
) -> None:
    suite2p = tmp_path / "suite2p"
    plane0 = suite2p / "plane0"
    plane1 = suite2p / "plane1"
    plane0.mkdir(parents=True)
    plane1.mkdir()
    (plane0 / "data.bin").touch()
    np.save(plane0 / "db.npy", {"Ly": 2, "Lx": 2, "nframes": 3})
    np.save(plane0 / "settings.npy", {})
    np.save(plane1 / "db.npy", {"Ly": 2, "Lx": 2, "nframes": 3})
    np.save(plane1 / "settings.npy", {})
    datasets = pd.DataFrame(
        [
            {"Name": "first", "Date": "1", "Process": True},
            {"Name": "skip", "Date": "2", "Process": False},
        ]
    )
    monkeypatch.setattr(
        dark.correct_traces,
        "make_paths",
        lambda row, directories: {"suite2p": str(suite2p)},
    )
    monkeypatch.setattr(dark, "percentile_first_1000_frames", lambda *args, **kwargs: -7.5)

    result = dark.plane_percentiles_all_datasets({"directories": {}}, datasets)

    assert result.loc[0, "Dark value"] == -7.5
    assert np.isnan(result.loc[1, "Dark value"])
    assert dark._load_ops(str(plane0)) is None
    assert dark._plane_bin_path(str(plane0)) == str(plane0 / "data.bin")
    assert dark._plane_bin_path(str(plane1)) is None
    assert dark._find_plane_folders(str(tmp_path / "missing")) == []


def test_piezo_plot_utility_writes_dataset_named_image(monkeypatch, tmp_path: Path) -> None:
    datasets = pd.DataFrame(
        [
            {"Name": "skip", "Date": "1", "Experiments": ["1"], "Planes": 2, "Process": False},
            {"Name": "mouse", "Date": "2", "Experiments": ["3"], "Planes": 2, "Process": True},
        ]
    )
    monkeypatch.setattr(
        piezo.extract_data,
        "get_frame_times",
        lambda *args, **kwargs: np.arange(20, dtype=float) / 20,
    )
    monkeypatch.setattr(
        piezo.extract_data,
        "get_piezo_data",
        lambda *args, **kwargs: (np.sin(np.arange(1000) / 30), np.arange(1000) / 1000),
    )
    monkeypatch.setattr(
        piezo.correct_traces,
        "make_piezo_trace_per_plane",
        lambda *args, **kwargs: np.column_stack((np.arange(5), np.arange(5) + 1)),
    )
    config = {
        "directories": {"tiffs": str(tmp_path / "{Name}" / "{Date}" / "{Experiments}")},
        "daq": {
            "data": "data.bin",
            "channel_names": "channels.csv",
            "clock_channel_name": "clock",
            "piezo_channel_name": "piezo",
            "sampling_rate": 1000,
            "piezo_volt_per_micron": 0.01,
        },
    }

    piezo.load_piezo_data(config, datasets, tmp_path / "plots")

    output = tmp_path / "plots" / "mouse_2_piezo_per_plane.png"
    assert output.is_file()
    assert output.stat().st_size > 100


def _write_ops(path: Path, *, channels: int = 1, nonrigid: bool = False) -> None:
    np.save(
        path / "ops.npy",
        {
            "nchannels": channels,
            "nonrigid": nonrigid,
            "nframes": 2,
            "Ly": 2,
            "Lx": 2,
            "yoff": np.zeros(2),
            "xoff": np.zeros(2),
            "yoff1": np.zeros((2, 1)),
            "xoff1": np.zeros((2, 1)),
            "data_path": ["old/experiment"],
        },
    )


def test_recreate_binary_planning_and_registration(monkeypatch, tmp_path: Path) -> None:
    suite2p = tmp_path / "processed" / "mouse" / "1" / "suite2p"
    plane0 = suite2p / "plane0"
    plane1 = suite2p / "plane1"
    plane0.mkdir(parents=True)
    plane1.mkdir()
    _write_ops(plane0, channels=2, nonrigid=True)
    _write_ops(plane1)
    (plane1 / "data.bin").touch()
    datasets = pd.DataFrame(
        [
            {"Name": "skip", "Date": "0", "Process": False, "Ignore_planes": []},
            {"Name": "mouse", "Date": "1", "Process": True, "Ignore_planes": [1]},
        ]
    )
    config = {"directories": {"suite2p": str(tmp_path / "processed" / "{Name}" / "{Date}")}}

    targets = recreate.plan_missing_bin_targets(config, datasets)

    assert targets == [str(plane0 / "data.bin"), str(plane0 / "data_chan2.bin")]

    output = tmp_path / "temporary"
    paths = {"output": str(output), "piezo": str(tmp_path / "raw")}
    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()

    def tiff_to_binary(updated_ops):
        temporary = output / "suite2p" / "plane0"
        temporary.mkdir(parents=True)
        (temporary / "data.bin").touch()
        (temporary / "data_chan2.bin").touch()

    class Binary:
        def __init__(self, **kwargs):
            self.filename = kwargs["filename"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = []
    monkeypatch.setattr(recreate.io, "tiff_to_binary", tiff_to_binary)
    monkeypatch.setattr(recreate.io, "BinaryFile", Binary)
    monkeypatch.setattr(
        recreate.registration.register,
        "shift_frames_and_write",
        lambda binary, **kwargs: calls.append((binary.filename, kwargs)),
        raising=False,
    )

    recreate.recreate_registered_binary(ops, paths, [str(plane0)])

    assert (plane0 / "data.bin").is_file()
    assert (plane0 / "data_chan2.bin").is_file()
    assert len(calls) == 2
    assert calls[0][1]["yoff1"] is not None


def test_neuropil_mask_validation_and_multiple_roi_output_policy(tmp_path: Path) -> None:
    plane = tmp_path / "plane0"
    plane.mkdir()
    with pytest.raises(FileNotFoundError, match="stat.npy"):
        masks.plot_neuropil_masks(plane)

    np.save(plane / "ops.npy", {"Ly": 3, "Lx": 3})
    stat = np.array(
        [
            {"neuropil_mask": np.array([0, 1]), "ypix": np.array([1]), "xpix": np.array([1])},
            {"neuropil_mask": np.array([7]), "ypix": np.array([2]), "xpix": np.array([2])},
        ],
        dtype=object,
    )
    np.save(plane / "stat.npy", stat)
    np.save(plane / "iscell.npy", np.ones((2, 2)))
    with pytest.raises(ValueError, match="exactly one"):
        masks.plot_neuropil_masks(plane, tmp_path / "masks.png")


def test_zstack_diagnostic_plotting_helpers_write_headless_outputs(tmp_path: Path) -> None:
    image = np.arange(36, dtype=float).reshape(6, 6)
    zstack.plot_reference_slice(tmp_path, image, image + 2)
    assert (tmp_path / "Reference_and_matchingSlice.png").is_file()

    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    stack = np.stack((image, image + 1, image + 2))
    ztrace = np.array([0.0, 1.0, np.nan, 1.0, 0.0, 1.0])
    profile = np.array([1.0, 2.0, 1.5])
    roi = {"med": np.array([3, 3])}
    zstack.plot_zprofile(
        np.arange(6, dtype=float),
        np.arange(6, dtype=float) / 2,
        ztrace,
        profile,
        np.array([1]),
        roi,
        stack,
        1,
        1,
        2,
        [3, 6],
        20,
        plt.get_cmap("Paired"),
        profile_axis=axes[0],
        stack_axis=axes[1],
        ztrace_axis=axes[2],
    )
    assert all(axis.has_data() for axis in axes)
    assert axes[0].get_legend() is not None
    plt.close(figure)

    results = [
        {"zTrace": np.array([0, 1, 1, 2])},
        {"zTrace": np.array([1, 1, 2, 2])},
    ]
    zstack.plot_ztraces(tmp_path, results, [0, 1], [0, 1], plane_rate=2)
    assert (tmp_path / "zTraces_allPlanes.png").is_file()


def test_zoomed_zstack_roi_plot_has_no_unused_or_overlapping_axis(
    monkeypatch, tmp_path: Path
) -> None:
    saved_figures = {}

    def record_figure(figure, path, **kwargs):
        saved_figures[Path(path).name] = figure

    monkeypatch.setattr(plt.Figure, "savefig", record_figure)
    fluorescence = np.arange(6.0)[:, None] - 7000
    neuropil = np.arange(6.0)[:, None] - 8000
    ztrace = np.array([np.nan, 1.0, 1.0, 1.0, 1.0, np.nan])
    profile = np.array([[0.8], [1.0], [1.2]])
    arguments = (
        tmp_path,
        [{"med": np.array([3, 3])}],
        fluorescence,
        neuropil,
        fluorescence + 1,
        neuropil + 1,
        fluorescence - neuropil,
        np.full_like(fluorescence, 15),
        fluorescence / 20,
        ztrace,
        profile,
        [np.array([1])],
        np.ones((3, 6, 6)),
        1,
        1,
        2,
        [3, 6],
        20,
    )

    correct_traces.plot_signal_per_roi(*arguments)
    correct_traces.plot_signal_per_roi(*arguments, zoom_in=True)

    normal = saved_figures["ROI0000.png"]
    zoomed = saved_figures["ROI0000_zoomed.png"]
    normal_plot_axes = [axis for axis in normal.axes if axis.has_data()]
    zoomed_plot_axes = [axis for axis in zoomed.axes if axis.has_data()]
    assert len(normal_plot_axes) == 6
    assert len(zoomed_plot_axes) == 5
    assert len(normal.axes) == 6
    assert len(zoomed.axes) == 5
    assert normal._suptitle.get_text() == "ROI 0"
    assert zoomed._suptitle.get_text() == "ROI 0 (zoom-in)"
    for figure in (normal, zoomed):
        figure.canvas.draw()
        legends = [axis.get_legend() for axis in figure.axes if axis.get_legend() is not None]
        assert len(legends) == 4
        assert figure.axes[0].get_legend() is legends[0]
        legend_axes = [axis for axis in figure.axes if axis.get_legend() is not None]
        assert legend_axes == [figure.axes[0], *figure.axes[-3:]]
        assert [text.get_text() for text in legends[1].get_texts()] == [
            "F(raw)",
            "N(raw)",
            "F(z-corr.)",
            "N(z-corr.)",
        ]
        raw_label_y = [text.get_window_extent().y0 for text in legends[1].get_texts()]
        assert max(raw_label_y) - min(raw_label_y) < 1
        assert [text.get_text() for text in legends[2].get_texts()] == ["F(Npil corr.)", "F0"]
        assert [text.get_text() for text in legends[3].get_texts()] == ["dF/F"]
        assert [
            line.get_label()
            for line in figure.axes[0].lines
            if not line.get_label().startswith("_")
        ] == [
            "F(recording)",
            "F(stack)",
            "Ref. depth",
            "Ref. im. match",
        ]
        assert all(
            figure.bbox.contains(*corner)
            for legend in legends
            for corner in (
                (legend.get_window_extent().x0, legend.get_window_extent().y0),
                (legend.get_window_extent().x1, legend.get_window_extent().y1),
            )
        )
        profile_bounds = figure.axes[0].get_window_extent()
        profile_legend_bounds = figure.axes[0].get_legend().get_window_extent()
        assert profile_legend_bounds.y0 >= profile_bounds.y1
        assert figure.axes[0].get_position().x0 == pytest.approx(0.06)
        assert all(
            figure.bbox.contains(*corner)
            for axis in figure.axes
            for label in axis.get_yticklabels()
            if label.get_visible() and label.get_text()
            for corner in (
                (label.get_window_extent().x0, label.get_window_extent().y0),
                (label.get_window_extent().x1, label.get_window_extent().y1),
            )
        )

    normal_bounds = [axis.get_position().bounds for axis in normal_plot_axes[:3]]
    normal_gaps = [
        normal_bounds[index + 1][0] - (normal_bounds[index][0] + normal_bounds[index][2])
        for index in range(2)
    ]
    assert max(normal_gaps) < 0.01

    zoomed_bounds = [axis.get_position().bounds for axis in zoomed_plot_axes[:2]]
    zoomed_gap = zoomed_bounds[1][0] - (zoomed_bounds[0][0] + zoomed_bounds[0][2])
    assert zoomed_gap < 0.01

    bounds = [axis.get_position().bounds for axis in zoomed_plot_axes]
    for index, (x0, y0, width, height) in enumerate(bounds):
        for other_x0, other_y0, other_width, other_height in bounds[index + 1 :]:
            overlap_x = min(x0 + width, other_x0 + other_width) - max(x0, other_x0)
            overlap_y = min(y0 + height, other_y0 + other_height) - max(y0, other_y0)
            assert overlap_x <= 0 or overlap_y <= 0


def test_bouton_roi_legends_are_inside_and_clear_in_normal_and_zoomed_figures(
    monkeypatch, tmp_path: Path
) -> None:
    saved_figures = []
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda figure, path, **kwargs: saved_figures.append(figure)
    )
    fluorescence = np.arange(6.0)[:, None] + 20
    neuropil = np.arange(6.0)[:, None] + 5
    arguments = (
        tmp_path,
        [{}],
        fluorescence,
        neuropil,
        None,
        None,
        fluorescence - neuropil,
        np.full_like(fluorescence, 15),
        fluorescence / 20,
        None,
        None,
        None,
        None,
        None,
        None,
        2,
        [3, 6],
        20,
    )

    correct_traces.plot_signal_per_roi(*arguments)
    correct_traces.plot_signal_per_roi(*arguments, zoom_in=True)

    assert len(saved_figures) == 2
    for figure in saved_figures:
        plot_axes = [axis for axis in figure.axes if axis.has_data()]
        assert len(plot_axes) == 3
        assert len(figure.axes) == 3
        _assert_legends_are_inside_and_clear_of_plots(figure, expected_legends=3)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in plot_axes:
            plot_bounds = axis.get_window_extent(renderer)
            legend_bounds = axis.get_legend().get_window_extent(renderer)
            assert plot_bounds.width / figure.bbox.width > 0.8
            assert 0 < legend_bounds.x0 - plot_bounds.x1 < figure.bbox.width * 0.04


@pytest.mark.parametrize("with_zstack", [False, True])
def test_roi_plot_subset_is_reused_for_normal_and_zoomed_outputs(
    monkeypatch, tmp_path: Path, with_zstack: bool
) -> None:
    saved_names = []
    monkeypatch.setattr(
        plt.Figure,
        "savefig",
        lambda figure, path, **kwargs: saved_names.append(Path(path).name),
    )
    n_rois = 5
    fluorescence = np.arange(30.0).reshape(6, n_rois) + 20
    neuropil = fluorescence / 4
    roi_indices = correct_traces.select_roi_plot_indices(n_rois, 2)
    plane_stack = np.ones((3, 6, 6)) if with_zstack else None
    ztrace = np.array([np.nan, 1.0, 1.0, 1.0, 1.0, np.nan]) if with_zstack else None
    profiles = np.ones((3, n_rois)) if with_zstack else None
    peaks = [np.array([1])] * n_rois if with_zstack else None
    stat = [{"med": np.array([3, 3])} for _ in range(n_rois)]
    arguments = (
        tmp_path,
        stat,
        fluorescence,
        neuropil,
        fluorescence + 1,
        neuropil + 1,
        fluorescence - neuropil,
        np.full_like(fluorescence, 15),
        fluorescence / 20,
        ztrace,
        profiles,
        peaks,
        plane_stack,
        1 if with_zstack else None,
        1 if with_zstack else None,
        2,
        [3, 6],
        20,
    )

    correct_traces.plot_signal_per_roi(*arguments, roi_indices=roi_indices)
    correct_traces.plot_signal_per_roi(*arguments, roi_indices=roi_indices, zoom_in=True)

    assert saved_names == [
        "ROI0001.png",
        "ROI0003.png",
        "ROI0001_zoomed.png",
        "ROI0003_zoomed.png",
    ]


def test_ztrace_colors_match_piezo_colors_for_noncontiguous_planes(
    monkeypatch, tmp_path: Path
) -> None:
    closed_figures = []
    monkeypatch.setattr(plt, "close", lambda figure=None: closed_figures.append(figure))

    piezo_values = np.column_stack((np.arange(5.0), np.arange(5.0) + 1, np.arange(5.0) + 2))
    correct_traces.plot_piezo_per_plane(tmp_path, piezo_values, sampling_rate=2)
    results = [
        {"zTrace": np.array([0.0, 1.0, 1.0])},
        None,
        {"zTrace": np.array([2.0, 2.0, 1.0])},
    ]
    zstack.plot_ztraces(tmp_path, results, [0, 2], [0, 2], plane_rate=2)

    piezo_axis = closed_figures[0].axes[0]
    ztrace_axis = closed_figures[1].axes[0]
    piezo_colors = [line.get_color() for line in piezo_axis.lines]
    ztrace_colors = [line.get_color() for line in ztrace_axis.lines]

    assert ztrace_colors == [piezo_colors[0], piezo_colors[2]]
    assert [line.get_label() for line in ztrace_axis.lines] == ["Plane 0", "Plane 2"]


def test_plane_color_accepts_rgb_tuple_cycle() -> None:
    rgb = (0.12, 0.47, 0.71)
    with plt.rc_context({"axes.prop_cycle": cycler(color=[rgb])}):
        color = zstack.plane_color(0)
    assert mcolors.is_color_like(color)
    np.testing.assert_allclose(mcolors.to_rgb(color), rgb, atol=1 / 255)
