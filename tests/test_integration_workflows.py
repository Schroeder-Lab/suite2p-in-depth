from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from suite2p_in_depth import correct_traces, process_tiff, run_suite2p, suite2p_compat, zstack


def _write_plane_arrays(plane: Path, *, frames: int = 120) -> None:
    plane.mkdir(parents=True)
    time = np.arange(frames, dtype=float)
    neuropil = np.vstack((30 + 4 * np.sin(time / 11), 25 + 3 * np.cos(time / 13)))
    fluorescence = (
        180 + 0.6 * neuropil + np.vstack((0.2 * np.cos(time / 7), 0.1 * np.sin(time / 9)))
    )
    np.save(plane / "F.npy", fluorescence)
    np.save(plane / "Fneu.npy", neuropil)
    np.save(plane / "iscell.npy", np.array([[1, 0.9], [0, 0.1]]))
    stat = np.array(
        [
            {
                "med": np.array([2, 3]),
                "ypix": np.array([2, 2]),
                "xpix": np.array([3, 4]),
                "overlap": np.array([False, False]),
            },
            {
                "med": np.array([1, 1]),
                "ypix": np.array([1]),
                "xpix": np.array([1]),
                "overlap": np.array([False]),
            },
        ],
        dtype=object,
    )
    np.save(plane / "stat.npy", stat)


def _trace_config(*, plot: bool = False) -> dict:
    return {
        "pixels_to_microns": False,
        "delta_F": {
            "absolute_zero": -5.0,
            "F0_percentile": 8,
            "F0_window": 2,
            "plot": plot,
            "max_roi_plots": "all",
            "plot_zoomed_traces": True,
        },
        "daq": {},
    }


def _suite2p_plane_metadata(frames: int = 120) -> tuple[dict, dict, dict]:
    db = {
        "Ly": 8,
        "Lx": 10,
        "nplanes": 2,
        "nchannels": 1,
        "functional_chan": 1,
        "frames_per_folder": [frames],
    }
    settings = {
        "torch_device": "cpu",
        "registration": {"align_by_chan2": False},
        "extraction": {"allow_overlap": False},
    }
    outputs = {
        "badframes": np.zeros(frames, dtype=bool),
        "meanImg": np.arange(80).reshape(8, 10),
        "refImg": np.arange(80).reshape(8, 10),
    }
    return db, settings, outputs


def test_trace_plane_integration_without_zstack(monkeypatch, tmp_path: Path) -> None:
    plane = tmp_path / "suite2p" / "plane0"
    _write_plane_arrays(plane)
    db, settings, outputs = _suite2p_plane_metadata()
    monkeypatch.setattr(
        correct_traces.suite2p_compat, "load_parameters", lambda path: (db, settings)
    )
    monkeypatch.setattr(correct_traces.suite2p_compat, "load_outputs", lambda path: (outputs, {}))
    output = tmp_path / "output"

    result = correct_traces.process_plane(
        _trace_config(plot=True),
        str(plane),
        str(tmp_path),
        None,
        str(output),
        None,
        20.0,
        100.0,
    )

    assert result is not None
    assert result["dff"].shape == (120, 1)
    assert result["zTrace"] is None
    np.testing.assert_array_equal(result["locs"], np.array([[2.0, 3.0, 100.0]]))
    assert (plane / "db_correcting.npy").is_file()
    assert (plane / "settings_correcting.npy").is_file()
    assert (plane / "detect_outputs_correcting.npy").is_file()
    assert (output / "2P_processed" / "plane0" / "plots" / "ROIs_and_meanImg.png").is_file()
    assert (output / "2P_processed" / "plane0" / "plots" / "ROIs" / "ROI0000.png").is_file()


@pytest.mark.parametrize(
    ("maximum", "plot_zoomed", "expected_calls"),
    [(0, True, 0), (1, False, 1), (1, True, 2), ("all", False, 1)],
)
def test_roi_plot_cap_and_zoom_toggle_control_expensive_plot_calls(
    monkeypatch, tmp_path: Path, maximum, plot_zoomed: bool, expected_calls: int
) -> None:
    plane = tmp_path / "suite2p" / "plane0"
    _write_plane_arrays(plane)
    np.save(plane / "iscell.npy", np.array([[1, 0.9], [1, 0.8]]))
    db, settings, outputs = _suite2p_plane_metadata()
    monkeypatch.setattr(
        correct_traces.suite2p_compat, "load_parameters", lambda path: (db, settings)
    )
    monkeypatch.setattr(correct_traces.suite2p_compat, "load_outputs", lambda path: (outputs, {}))
    monkeypatch.setattr(correct_traces, "plot_rois_and_reference", lambda *args: None)
    monkeypatch.setattr(correct_traces, "zoom_window", (0, 10))
    calls = []

    def record_roi_plots(*args, **kwargs):
        calls.append((kwargs["roi_indices"].copy(), kwargs.get("zoom_in", False)))

    monkeypatch.setattr(correct_traces, "plot_signal_per_roi", record_roi_plots)
    config = _trace_config(plot=True)
    config["delta_F"]["max_roi_plots"] = maximum
    config["delta_F"]["plot_zoomed_traces"] = plot_zoomed
    output = tmp_path / "output"

    result = correct_traces.process_plane(
        config,
        str(plane),
        str(tmp_path),
        None,
        str(output),
        None,
        20.0,
        100.0,
    )

    assert result is not None
    assert len(calls) == expected_calls
    if expected_calls:
        expected_indices = correct_traces.select_roi_plot_indices(2, maximum)
        for indices, _zoomed in calls:
            np.testing.assert_array_equal(indices, expected_indices)
    if expected_calls == 2:
        assert [zoomed for _indices, zoomed in calls] == [False, True]
    if maximum == 0:
        assert not (output / "2P_processed" / "plane0" / "plots" / "ROIs").exists()


def test_trace_plane_integration_with_mocked_zstack_boundary(monkeypatch, tmp_path: Path) -> None:
    plane = tmp_path / "suite2p" / "plane1"
    _write_plane_arrays(plane)
    db, settings, outputs = _suite2p_plane_metadata()
    monkeypatch.setattr(
        correct_traces.suite2p_compat, "load_parameters", lambda path: (db, settings)
    )
    monkeypatch.setattr(correct_traces.suite2p_compat, "load_outputs", lambda path: (outputs, {}))
    registration = {"align_by_chan2": False}
    monkeypatch.setattr(
        correct_traces.zstack,
        "prep_zstack_parameters",
        lambda *args: (
            "data.bin",
            db,
            registration,
            registration,
            registration,
            registration,
            None,
            1,
        ),
    )
    stack = np.stack([outputs["refImg"] + index for index in range(3)])
    monkeypatch.setattr(correct_traces.zstack, "extract_reference_slice", lambda *args: (1, stack))
    monkeypatch.setattr(
        correct_traces.zstack,
        "compute_correlations_reference",
        lambda *args: (np.ones((3, 120)), np.ones(120, dtype=int)),
    )
    monkeypatch.setattr(
        correct_traces.process_tiff,
        "extract_zprofiles",
        lambda *args, **kwargs: np.array([[0.8], [1.0], [1.2]]),
    )
    config = _trace_config()
    config.update(
        {
            "z_correction": {
                "ref_frames": 0.5,
                "stack_step": 1,
                "sigma_trace": 0.5,
                "reference": "first",
                "remove_outliers": False,
            },
            "suite2p_stack_within": {},
            "suite2p_stack_across": {},
            "suite2p_stack_to_ref": {},
            "suite2p_stack_to_frames": {},
        }
    )

    result = correct_traces.process_plane(
        config,
        str(plane),
        str(tmp_path),
        "stack.tif",
        str(tmp_path / "output"),
        np.zeros((4, 2)),
        20.0,
        50.0,
    )

    assert result is not None
    assert result["zCorr_stack"].shape == (3, 120)
    assert result["zProfiles"].shape == (3, 1)
    assert (plane / "zcorrect_outputs.npy").is_file()


def test_process_all_trace_datasets_covers_piezo_and_no_piezo(monkeypatch, tmp_path: Path) -> None:
    datasets = pd.DataFrame(
        [
            {"Name": "skip", "Date": "1", "Process": False, "Frame_rate": 20.0, "Depth": 0},
            {"Name": "flat", "Date": "2", "Process": True, "Frame_rate": 20.0, "Depth": 2},
            {"Name": "piezo", "Date": "3", "Process": True, "Frame_rate": np.nan, "Depth": np.nan},
        ]
    )
    output = tmp_path / "output"
    output.mkdir()

    def fake_paths(row, directories):
        return {
            "suite2p": str(tmp_path / "suite2p"),
            "tiffs": str(tmp_path),
            "output": str(output),
            "piezo": None if row["Name"] == "flat" else str(tmp_path),
        }

    monkeypatch.setattr(correct_traces, "make_paths", fake_paths)
    monkeypatch.setattr(correct_traces.general, "setup_suite2p_logging", lambda path: None)
    db = {"nplanes": 2, "data_path": [str(tmp_path / "experiment-1")]}
    monkeypatch.setattr(
        correct_traces.suite2p_compat, "load_parameters_recording", lambda path: (db, {"fs": 12.0})
    )
    monkeypatch.setattr(
        correct_traces.extract_data,
        "get_frame_times",
        lambda *args, **kwargs: np.arange(20, dtype=float) / 20,
    )
    monkeypatch.setattr(
        correct_traces.extract_data,
        "get_piezo_data",
        lambda *args, **kwargs: (np.sin(np.arange(1000) / 20), np.arange(1000) / 1000),
    )
    monkeypatch.setattr(
        correct_traces, "make_piezo_trace_per_plane", lambda *args, **kwargs: np.zeros((5, 2))
    )
    observed: list[dict] = []
    monkeypatch.setattr(
        correct_traces, "process_dataset", lambda *args, **kwargs: observed.append(kwargs)
    )
    config = {
        "directories": {},
        "use_zstack": False,
        "daq": {
            "data": "daq.bin",
            "channel_names": "channels.csv",
            "clock_channel_name": "frame",
            "piezo_channel_name": "piezo",
            "sampling_rate": 1000,
            "piezo_volt_per_micron": 0.01,
            "plot_piezo": False,
        },
    }

    correct_traces.process_all_datasets(config, datasets)

    assert [item["frame_rate"] for item in observed] == [20.0, pytest.approx(20.0)]
    assert observed[0]["piezo"] is None
    assert observed[1]["piezo"].shape == (5, 2)
    assert observed[0]["depth"] == 2
    assert observed[1]["depth"] == 0


def test_run_suite2p_processes_selected_valid_dataset(monkeypatch, tmp_path: Path) -> None:
    datasets = pd.DataFrame(
        [
            {"Name": "skip", "Date": "1", "Process": False},
            {"Name": "missing", "Date": "2", "Process": True},
            {"Name": "valid", "Date": "3", "Process": True},
        ]
    )

    def fake_paths(row, directories):
        return {
            "output": str(tmp_path / row["Name"]),
            "tiffs": None if row["Name"] == "missing" else [str(tmp_path)],
        }

    monkeypatch.setattr(run_suite2p.zregister, "make_paths", fake_paths)
    monkeypatch.setattr(
        run_suite2p.zregister, "get_settings", lambda row, config: {"torch_device": "cpu"}
    )
    monkeypatch.setattr(
        run_suite2p.zregister, "get_db", lambda paths, row, config: {"data_path": paths["tiffs"]}
    )
    monkeypatch.setattr(run_suite2p.parameters, "default_settings", lambda: {})
    monkeypatch.setattr(run_suite2p.parameters, "default_db", lambda: {})
    monkeypatch.setattr(
        run_suite2p.parameters, "set_settings", lambda target, values: target.update(values)
    )
    monkeypatch.setattr(
        run_suite2p.parameters, "set_db", lambda target, values: target.update(values)
    )
    monkeypatch.setattr(run_suite2p.run_s2p_mod, "logger_setup", lambda path: None, raising=False)
    calls: list[tuple[dict, dict]] = []
    monkeypatch.setattr(
        run_suite2p.run_s2p_mod,
        "run_s2p",
        lambda *, db, settings: calls.append((db, settings)),
        raising=False,
    )

    run_suite2p.process_all_datasets(
        {"directories": {}, "suite2p_settings": {}, "suite2p_db": {}}, datasets
    )

    assert len(calls) == 1
    assert calls[0][0]["data_path"] == [str(tmp_path)]
    assert calls[0][1]["torch_device"] == "cpu"


def test_suite2p_compatibility_current_legacy_and_missing(monkeypatch, tmp_path: Path) -> None:
    current = tmp_path / "current"
    current.mkdir()
    np.save(current / "db.npy", {"nplanes": 2})
    np.save(current / "settings.npy", {"torch_device": "cpu"})
    assert suite2p_compat.load_parameters(current) == ({"nplanes": 2}, {"torch_device": "cpu"})

    legacy = tmp_path / "legacy" / "plane2"
    legacy.mkdir(parents=True)
    ops = {
        "save_path": str(legacy),
        "reg_file": "data.bin",
        "filelist": ["movie.tif"],
        "first_tiffs": [0],
        "Ly": 4,
        "Lx": 5,
        "nframes": 6,
        "frames_per_file": [6],
        "frames_per_folder": [6],
        "meanImg": np.ones((4, 5)),
        "align_by_chan": 1,
        "badframes": np.zeros(6, dtype=bool),
    }
    np.save(legacy / "ops.npy", ops)
    default_db = {"fast_disk": str(tmp_path), "keep_movie_raw": False, "nchannels": 1}
    default_settings = {"registration": {}}
    monkeypatch.setattr(suite2p_compat.parameters, "default_db", lambda: default_db.copy())
    monkeypatch.setattr(suite2p_compat.parameters, "default_settings", lambda: {"registration": {}})
    monkeypatch.setattr(
        suite2p_compat.parameters,
        "convert_settings_orig",
        lambda ops, db, settings: (db, default_settings.copy(), {}),
    )
    db, settings = suite2p_compat.load_parameters(legacy)
    assert db["iplane"] == 2
    assert settings["registration"]["align_by_chan2"] is False
    registration, detection = suite2p_compat.load_outputs(legacy)
    assert registration is not None
    assert registration["badframes"] is not None
    assert detection is not None

    recording = tmp_path / "recording"
    (recording / "plane0").mkdir(parents=True)
    np.save(recording / "plane0" / "db.npy", {"ok": True})
    np.save(recording / "plane0" / "settings.npy", {"ok": True})
    assert suite2p_compat.load_parameters_recording(recording) == ({"ok": True}, {"ok": True})
    with pytest.raises(FileNotFoundError, match="No plane"):
        suite2p_compat.load_parameters_recording(tmp_path / "current")
    with pytest.raises(FileNotFoundError, match="Neither ops"):
        suite2p_compat.load_parameters(tmp_path / "empty")
    assert suite2p_compat.load_outputs(tmp_path / "empty") == (None, None)


def _registration_settings() -> dict:
    return {
        "do_bidiphase": True,
        "norm_frames": True,
        "smooth_sigma": 1.0,
        "spatial_taper": 3.0,
        "maxregshift": 0.1,
        "smooth_sigma_time": 0.0,
        "snr_thresh": 1.2,
        "maxregshiftNR": 0,
        "nonrigid": False,
        "block_size": [2, 2],
    }


def test_zstack_registration_single_and_two_channel_boundaries(monkeypatch) -> None:
    image = np.arange(3 * 2 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 2, 4, 5)
    monkeypatch.setattr(process_tiff.skimage.io, "imread", lambda path: image)
    monkeypatch.setattr(process_tiff.bidiphase, "compute", lambda frames: 1, raising=False)
    monkeypatch.setattr(process_tiff.bidiphase, "shift", lambda frames, phase: None, raising=False)
    monkeypatch.setattr(
        process_tiff.suite2p_extensions,
        "compute_reference",
        lambda frames, **kwargs: frames.mean(0),
        raising=False,
    )

    def fake_register_frames(frames, *args, **kwargs):
        array = np.asarray(frames)
        count = array.shape[0]
        offsets = (
            np.zeros(count),
            np.zeros(count),
            np.ones(count),
            np.zeros((count, 1)),
            np.zeros((count, 1)),
        )
        return None, None, array.mean(0), offsets, [np.array([0])]

    monkeypatch.setattr(
        process_tiff.register, "register_frames", fake_register_frames, raising=False
    )
    monkeypatch.setattr(
        process_tiff,
        "register_zstack_across_planes",
        lambda stack, settings, device: (stack, np.zeros(3), np.zeros(3), None, None, None),
    )

    def fake_shift_frames(frames, **kwargs):
        return (
            frames.detach().cpu().numpy()
            if isinstance(frames, torch.Tensor)
            else np.asarray(frames)
        )

    monkeypatch.setattr(
        process_tiff.suite2p_extensions, "shift_frames", fake_shift_frames, raising=False
    )
    monkeypatch.setattr(process_tiff, "smooth_zstack", lambda sigma, stack: stack)
    monkeypatch.setattr(
        process_tiff,
        "register_zstack_to_ref",
        lambda stack, target, settings, other, device: (stack, 1, other),
    )
    settings = _registration_settings()

    stack, best, other = process_tiff.register_zstack(
        "stack.tif",
        settings,
        settings,
        settings,
        target_image=np.zeros((4, 5)),
        channel_align=2,
        channel_functional=1,
        n_channels=2,
        sigma=(5, 3, 3),
        device=torch.device("cpu"),
    )

    assert stack.shape == (3, 4, 5)
    assert other is not None
    assert other.shape == (3, 4, 5)
    assert best == 1


def test_zstack_rigid_registration_and_reference_alignment(monkeypatch) -> None:
    stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    def fake_register_frames(frames=None, *args, **kwargs):
        if frames is None:
            frames = kwargs["f_align_in"]
        count = np.asarray(frames).shape[0]
        offsets = (
            np.arange(count),
            np.arange(count) + 1,
            np.arange(count, dtype=float),
            np.zeros((count, 1)),
            np.zeros((count, 1)),
        )
        return None, None, np.asarray(frames).mean(0), offsets, [np.array([0])]

    monkeypatch.setattr(
        process_tiff.register, "register_frames", fake_register_frames, raising=False
    )
    monkeypatch.setattr(
        process_tiff.register,
        "shift_frames",
        lambda frames, **kwargs: (
            frames.detach().cpu().numpy()
            if isinstance(frames, torch.Tensor)
            else np.asarray(frames)
        ),
        raising=False,
    )
    settings = _registration_settings()
    registered, yoff, xoff, yoff1, xoff1, blocks = process_tiff.register_zstack_across_planes(
        stack, settings, torch.device("cpu")
    )
    assert registered.shape == stack.shape
    assert yoff.shape == xoff.shape == (3,)
    assert yoff1 is xoff1 is blocks is None

    aligned, best, other = process_tiff.register_zstack_to_ref(
        stack, stack[1], settings, stack + 100, torch.device("cpu")
    )
    assert aligned.shape == stack.shape
    assert best == 2
    assert other is not None
    assert other.shape == stack.shape


def test_extract_zprofiles_uses_only_accepted_rois(monkeypatch, tmp_path: Path) -> None:
    plane = tmp_path / "plane0"
    plane.mkdir()
    np.save(plane / "stat.npy", np.array([{"id": 0}, {"id": 1}], dtype=object))
    np.save(plane / "iscell.npy", np.array([[1, 1], [0, 0]]))
    monkeypatch.setattr(
        process_tiff.suite2p_compat,
        "load_parameters",
        lambda path: ({}, {"extraction": {"allow_overlap": False}}),
    )
    monkeypatch.setattr(
        process_tiff.masks,
        "create_cell_mask",
        lambda stat, **kwargs: stat["id"],
        raising=False,
    )
    monkeypatch.setattr(
        process_tiff.extract,
        "extract_traces",
        lambda stack, rois, npils, device: (np.array([[10.0, 11.0, 12.0]]), None),
        raising=False,
    )

    profiles = process_tiff.extract_zprofiles(
        plane, np.ones((3, 4, 5)), abs_zero=2, device=torch.device("cpu")
    )

    np.testing.assert_array_equal(profiles, np.array([[8.0], [9.0], [10.0]]))


def test_new_zcorrelation_cache_expands_range_and_is_reusable(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "zcorr.npy"
    calls = 0

    def compute(stack, db, settings, reg_file, device):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.vstack((np.full(4, 3.0), np.full(4, 2.0), np.full(4, 1.0)))
        return np.zeros((stack.shape[0], 4))

    monkeypatch.setattr(zstack.suite2p_extensions, "compute_zpos", compute, raising=False)
    plane_stack = np.arange(5 * 2 * 2).reshape(5, 2, 2)
    correlations, trace = zstack.compute_correlations_reference(
        "data.bin",
        str(cache),
        {"nframes": 4, "frames_per_folder": [4]},
        {},
        2,
        plane_stack,
        1,
        0.1,
        torch.device("cpu"),
    )

    assert calls == 2
    assert cache.is_file()
    assert correlations.shape == (5, 4)
    np.testing.assert_array_equal(trace, np.ones(4, dtype=int))


def test_zstack_parameter_prep_and_reference_slice_cache(monkeypatch, tmp_path: Path) -> None:
    plane = tmp_path / "plane1"
    processed = tmp_path / "processed"
    plane.mkdir()
    processed.mkdir()
    settings = {"registration": {"align_by_chan2": False, "base": 1}}
    config = {
        "pixels_to_microns": True,
        "suite2p_stack_within": {"within": 1},
        "suite2p_stack_across": {"across": 1},
        "suite2p_stack_to_ref": {"ref": 1},
        "suite2p_stack_to_frames": {"frames": 1},
        "z_correction": {"sigma_stack": [5, 2, 3], "spacing": 2},
    }
    monkeypatch.setattr(
        zstack.parameters, "set_settings", lambda target, values: target.update(values)
    )
    values = zstack.prep_zstack_parameters(
        str(plane), config, {"Ly": 10, "functional_chan": 1}, settings, fov_size=20
    )
    assert values[0].endswith("data.bin")
    assert values[-2] == (5, 1.0, 1.5)
    assert values[-1] == 2

    registered = np.linspace(0, 65535, 5 * 4 * 5, dtype=np.uint16).reshape(5, 4, 5)
    monkeypatch.setattr(
        zstack.process_tiff,
        "register_zstack",
        lambda *args, **kwargs: (registered, 1, None),
    )
    best, loaded = zstack.extract_reference_slice(
        "raw.tif",
        str(processed),
        1,
        {"functional_chan": 1, "nchannels": 1},
        {"align_by_chan2": False},
        {},
        {},
        0.5,
        None,
        None,
        1,
        {"refImg": np.ones((4, 5))},
        torch.device("cpu"),
    )
    assert int(best) == 1
    np.testing.assert_array_equal(loaded, registered)
    best_cached, loaded_cached = zstack.extract_reference_slice(
        "raw.tif",
        str(processed),
        1,
        {"functional_chan": 1, "nchannels": 1},
        {"align_by_chan2": False},
        {},
        {},
        0.5,
        None,
        None,
        1,
        {"refImg": np.ones((4, 5))},
        torch.device("cpu"),
    )
    assert int(best_cached) == 1
    np.testing.assert_array_equal(loaded_cached, registered)
