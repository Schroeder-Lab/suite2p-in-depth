from __future__ import annotations

import importlib
import inspect
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suite2p_in_depth import correct_traces, extract_data, zstack
from suite2p_in_depth.preprocess_traces import correct_neuropil, correct_zmotion
from suite2p_in_depth.process_tiff import register_zstack
from suite2p_in_depth.utilities.plot_neuropil_masks import plot_neuropil_masks
from suite2p_in_depth.utilities.recreate_registered_bins import _plane_id


def test_zregistration_paths_reject_directories_without_tiffs(
    import_zregister, tmp_path: Path
) -> None:
    raw = tmp_path / "raw" / "mouse-a" / "2024-01-02" / "1"
    raw.mkdir(parents=True)
    dataset = pd.Series(
        {
            "Name": "mouse-a",
            "Date": "2024-01-02",
            "Experiments": ["1"],
        }
    )
    directories = {
        "tiffs": str(tmp_path / "raw" / "{Name}" / "{Date}" / "{Experiments}"),
        "output": str(tmp_path / "output" / "{Name}" / "{Date}"),
    }

    missing = import_zregister.make_paths(dataset, directories)
    assert missing["tiffs"] is None

    (raw / "recording.TIFF").touch()
    present = import_zregister.make_paths(dataset, directories)
    assert present["tiffs"] == [str(raw)]


def test_zstack_path_rejects_ambiguous_tiffs(tmp_path: Path) -> None:
    stack_directory = tmp_path / "stacks" / "mouse-a" / "2024-01-02" / "z1"
    stack_directory.mkdir(parents=True)
    (stack_directory / "a.tif").touch()
    (stack_directory / "b.TIFF").touch()
    dataset = pd.Series(
        {
            "Name": "mouse-a",
            "Date": "2024-01-02",
            "Zstack_folder": "z1",
        }
    )
    directories = {"zstack": str(tmp_path / "stacks" / "{Name}" / "{Date}" / "{Zstack}")}

    with pytest.raises(ValueError, match="Expected one z-stack TIFF"):
        zstack.append_paths(dataset, directories, {})


def test_zstack_path_accepts_one_case_insensitive_tiff(tmp_path: Path) -> None:
    stack_directory = tmp_path / "stacks" / "mouse-a" / "2024-01-02" / "z1"
    stack_directory.mkdir(parents=True)
    stack_path = stack_directory / "stack.TIFF"
    stack_path.touch()
    dataset = pd.Series(
        {
            "Name": "mouse-a",
            "Date": "2024-01-02",
            "Zstack_folder": "z1",
        }
    )

    paths = zstack.append_paths(
        dataset,
        {"zstack": str(tmp_path / "stacks" / "{Name}" / "{Date}" / "{Zstack}")},
        {},
    )

    assert paths["zstack"] == str(stack_path)


def test_cached_zcorrelations_use_configured_smoothing(monkeypatch, tmp_path: Path) -> None:
    cached = np.array(
        [
            [0.1, 0.2, 0.8],
            [0.7, 0.6, 0.1],
            [0.2, 0.3, 0.1],
        ]
    )
    cache_path = tmp_path / "zcorr.npy"
    cache_path.touch()
    observed = {}
    original_filter = zstack.sp.ndimage.gaussian_filter1d

    def recording_filter(values, sigma, **kwargs):
        observed["sigma"] = sigma
        return original_filter(values, sigma, **kwargs)

    monkeypatch.setattr(zstack.np, "load", lambda path: cached)
    monkeypatch.setattr(zstack.sp.ndimage, "gaussian_filter1d", recording_filter)

    correlations, trace = zstack.compute_correlations_reference(
        "data.bin",
        str(cache_path),
        {"nframes": 3},
        {},
        best_slice=1,
        plane_stack=np.zeros((3, 2, 2)),
        stack_step=1,
        sigma=0.75,
        device=None,
    )

    np.testing.assert_array_equal(correlations, cached)
    assert trace.shape == (3,)
    assert observed["sigma"] == 0.75


def test_zcorrelation_boundary_matches_are_nan_for_fresh_and_cached_results(
    monkeypatch, tmp_path: Path
) -> None:
    correlations = np.array(
        [
            [3.0, 1.0, 1.0, 3.0],
            [2.0, 3.0, 2.0, 2.0],
            [1.0, 2.0, 3.0, 1.0],
        ]
    )
    cache_path = tmp_path / "zcorr.npy"
    monkeypatch.setattr(
        zstack.suite2p_extensions,
        "compute_zpos",
        lambda *args, **kwargs: correlations,
    )
    arguments = (
        "data.bin",
        str(cache_path),
        {"nframes": 4, "frames_per_folder": [4]},
        {},
        1,
        np.zeros((3, 2, 2)),
        1,
        0.01,
        None,
    )

    fresh_correlations, fresh_trace = zstack.compute_correlations_reference(*arguments)
    cached_correlations, cached_trace = zstack.compute_correlations_reference(*arguments)

    np.testing.assert_array_equal(fresh_correlations, correlations)
    np.testing.assert_array_equal(cached_correlations, correlations)
    np.testing.assert_allclose(fresh_trace, [np.nan, 1.0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(cached_trace, fresh_trace)
    assert fresh_trace.dtype == np.float64


def test_reference_depth_ignores_nan_and_rejects_empty_reference_interval() -> None:
    assert correct_traces._determine_reference_depth(np.array([np.nan, 2.0, 4.0])) == 3

    with pytest.raises(ValueError, match="No finite z-trace positions"):
        correct_traces._determine_reference_depth(np.array([np.nan, np.nan]), reference_frames=2)


def test_each_plane_saves_its_own_settings(import_zregister, tmp_path: Path) -> None:
    plane_paths = [tmp_path / "plane0", tmp_path / "plane1"]
    for path in plane_paths:
        path.mkdir()
    settings = [{"plane": 0}, {"plane": 1}]

    import_zregister.save_plane_settings([str(path) for path in plane_paths], settings)

    assert np.load(plane_paths[0] / "settings.npy", allow_pickle=True).item() == {"plane": 0}
    assert np.load(plane_paths[1] / "settings.npy", allow_pickle=True).item() == {"plane": 1}


def _plane_result(plane_id: int, length: int) -> dict:
    return {
        "zCorr_stack": np.full((2, length), plane_id, dtype=float),
        "zTrace": np.arange(length),
        "zProfiles": np.full((2, 1), plane_id, dtype=float),
        "F_zcorrected": np.full((length, 1), plane_id, dtype=float),
        "N_zcorrected": np.full((length, 1), plane_id, dtype=float),
        "F_ncorrected": np.full((length, 1), plane_id, dtype=float),
        "N_regression": np.array([[0.0, 0.5]]),
        "F_bin_values": np.zeros((1, 2)),
        "N_bin_values": np.zeros((1, 2)),
        "dff": np.full((length, 1), plane_id, dtype=float),
        "locs": np.array([[1.0, 2.0, 3.0]]),
        "cellId": np.array([[plane_id]]),
    }


def test_process_dataset_numeric_plane_order_and_time_axis_trimming(
    monkeypatch, tmp_path: Path
) -> None:
    suite2p_directory = tmp_path / "suite2p"
    suite2p_directory.mkdir()
    lengths = {2: 4, 10: 5, 9: 3}
    for plane in (10, 2, 9):
        (suite2p_directory / f"plane{plane}").mkdir()
    observed_order = []

    def fake_process_plane(config, plane_directory, *args):
        plane = _plane_id(plane_directory)
        observed_order.append(plane)
        return _plane_result(plane, lengths[plane])

    monkeypatch.setattr(correct_traces, "process_plane", fake_process_plane)
    config = {
        "use_zstack": True,
        "z_correction": {"plot": False, "save_traces_zcorrected": False},
        "delta_F": {"save_F_neuropilcorrected": False},
    }

    correct_traces.process_dataset(
        str(suite2p_directory),
        config,
        tiff_path="unused",
        frame_rate=30,
        zstack_path="stack.tif",
        output_directory=str(tmp_path),
    )

    assert observed_order == [2, 9, 10]
    correlations = np.load(tmp_path / "2pPlanes.zCorrelations.npy")
    assert correlations.shape == (3, 2, 3)
    np.testing.assert_array_equal(correlations[:, 0, 0], np.array([2.0, 9.0, 10.0]))
    np.testing.assert_array_equal(
        np.load(tmp_path / "2pRois.2pPlanes.npy"), np.array([2.0, 9.0, 10.0])
    )


def test_process_dataset_non_zstack_uses_combined_plane(monkeypatch, tmp_path: Path) -> None:
    suite2p_directory = tmp_path / "suite2p"
    (suite2p_directory / "plane3").mkdir(parents=True)
    (suite2p_directory / "plane_z").mkdir()
    observed_directories = []

    def fake_process_plane(config, plane_directory, *args):
        observed_directories.append(Path(plane_directory).name)
        return _plane_result(-1, 4)

    monkeypatch.setattr(correct_traces, "process_plane", fake_process_plane)
    config = {
        "use_zstack": False,
        "delta_F": {"save_F_neuropilcorrected": False},
    }

    correct_traces.process_dataset(
        str(suite2p_directory),
        config,
        tiff_path="unused",
        frame_rate=30,
        output_directory=str(tmp_path),
    )

    assert observed_directories == ["plane_z"]
    np.testing.assert_array_equal(np.load(tmp_path / "2pRois.2pPlanes.npy"), np.array([-1.0]))


def test_process_dataset_fails_when_no_valid_planes(monkeypatch, tmp_path: Path) -> None:
    suite2p_directory = tmp_path / "suite2p"
    (suite2p_directory / "plane0").mkdir(parents=True)
    monkeypatch.setattr(correct_traces, "process_plane", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="No valid planes"):
        correct_traces.process_dataset(
            str(suite2p_directory),
            {"use_zstack": True, "z_correction": {}, "delta_F": {}},
            tiff_path="unused",
            frame_rate=30,
            output_directory=str(tmp_path),
        )


def test_roi_coordinates_scale_y_by_ly_and_x_by_lx() -> None:
    locations = np.array([[50.0, 100.0, 7.0]])

    scaled = correct_traces.scale_roi_locations_to_microns(locations, ly=100, lx=400, fov_size=200)

    np.testing.assert_array_equal(scaled, np.array([[100.0, 50.0, 7.0]]))
    np.testing.assert_array_equal(locations, np.array([[50.0, 100.0, 7.0]]))


def test_missing_or_duplicate_daq_channels_fail_clearly(monkeypatch) -> None:
    data = np.zeros((5, 2))
    time = np.arange(5, dtype=float)

    monkeypatch.setattr(
        extract_data,
        "get_nidaq_data",
        lambda *args, **kwargs: (data, np.array(["frame", "other"]), time),
    )
    with pytest.raises(ValueError, match="piezo channel"):
        extract_data.get_piezo_data(".", "data", "channels", "piezo")

    monkeypatch.setattr(
        extract_data,
        "get_nidaq_data",
        lambda *args, **kwargs: (data, np.array(["frame", "frame"]), time),
    )
    with pytest.raises(ValueError, match="frame-clock channel"):
        extract_data.get_frame_times(".", "data", "channels", "frame")


def test_malformed_scientific_array_shapes_fail_before_calculation() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        correct_neuropil(np.zeros((3, 1)), np.zeros((3, 2)), fs=10)
    with pytest.raises(ValueError, match="ztrace length"):
        correct_zmotion(
            np.ones((3, 1)),
            np.ones((3, 1)),
            np.ones((2, 1)),
            np.array([0, 1]),
            reference_depth=0,
        )


def test_register_zstack_raises_instead_of_exiting(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib.import_module("suite2p_in_depth.process_tiff").skimage.io,
        "imread",
        lambda path: np.zeros((2, 1, 3, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="at least 2 repetitions"):
        register_zstack(
            "stack.tif",
            {},
            {},
            {},
            piezo=None,
            device=None,
        )


def test_register_zstack_accepts_more_than_fifty_repetitions(monkeypatch) -> None:
    process_tiff = importlib.import_module("suite2p_in_depth.process_tiff")
    image = np.arange(2 * 51 * 3 * 4, dtype=np.float32).reshape(2, 51, 3, 4)
    monkeypatch.setattr(process_tiff.skimage.io, "imread", lambda path: image)
    monkeypatch.setattr(
        process_tiff.suite2p_extensions,
        "compute_reference",
        lambda frames, **kwargs: np.mean(frames, axis=0),
        raising=False,
    )

    def fake_register_frames(frames, **kwargs):
        offsets = (np.zeros(frames.shape[0]), np.zeros(frames.shape[0]))
        return None, None, np.mean(frames, axis=0), offsets

    monkeypatch.setattr(
        process_tiff.register, "register_frames", fake_register_frames, raising=False
    )
    monkeypatch.setattr(
        process_tiff,
        "register_zstack_across_planes",
        lambda stack, settings, device: (
            stack,
            np.zeros(stack.shape[0]),
            np.zeros(stack.shape[0]),
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        process_tiff,
        "register_zstack_to_ref",
        lambda stack, reference, settings, other, device: (stack, 0, other),
    )
    settings = {
        "do_bidiphase": False,
        "norm_frames": False,
        "smooth_sigma": 1.15,
        "spatial_taper": 3,
        "maxregshift": 0.1,
        "smooth_sigma_time": 0,
        "snr_thresh": 1.2,
        "maxregshiftNR": 5,
    }

    stack, best_plane, other = register_zstack(
        "stack.tif",
        settings,
        {},
        {},
        piezo=None,
        target_image=np.zeros((3, 4)),
        device=None,
    )

    assert stack.shape == (2, 3, 4)
    assert best_plane == 0
    assert other is None


def test_neuropil_mask_helper_uses_explicit_plane_and_output_paths(tmp_path: Path) -> None:
    plane = tmp_path / "plane0"
    plane.mkdir()
    stat = np.array(
        [
            {
                "neuropil_mask": np.array([0, 1, 5]),
                "ypix": np.array([1]),
                "xpix": np.array([1]),
            }
        ],
        dtype=object,
    )
    np.save(plane / "stat.npy", stat)
    np.save(plane / "iscell.npy", np.array([[1, 0.99]]))
    np.save(plane / "ops.npy", {"Ly": 4, "Lx": 5})
    output = tmp_path / "plots" / "mask.png"

    figures = plot_neuropil_masks(plane, output_path=output)

    assert len(figures) == 1
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("script_name", ["fig02_zstack.py", "fig04_zregistration.py"])
def test_paper_generators_require_explicit_existing_data_paths(script_name, tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "paper" / script_name),
        run_name="paper_import",
    )
    input_root = tmp_path / "paper-data"
    output_directory = tmp_path / "figures"

    with pytest.raises(FileNotFoundError, match="data directory"):
        namespace["generate_figures"](input_root, output_directory)

    assert not output_directory.exists()


@pytest.mark.parametrize(
    "script_name", ["fig01_algorithms.py", "fig02_zstack.py", "fig04_zregistration.py"]
)
def test_paper_generators_accept_explicit_action4_paths(script_name) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "paper" / script_name),
        run_name="paper_import",
    )

    signature = inspect.signature(namespace["generate_figures"])

    assert list(signature.parameters) == ["input_root", "output_dir", "style", "seed"]
    assert signature.parameters["style"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
