from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _row(**overrides) -> pd.Series:
    values = {
        "Name": "mouse",
        "Date": "2026-01-01",
        "Experiments": ["1"],
        "Planes": 2,
        "Channels": 1,
        "Frame_rate": 20.0,
        "Process": True,
    }
    values.update(overrides)
    return pd.Series(values)


def test_settings_database_and_registration_status(import_zregister, tmp_path: Path) -> None:
    settings = import_zregister.get_settings(
        _row(Channels=2), {"registration": {}, "io": {}, "run": {}}
    )
    assert settings["fs"] == 10
    assert settings["registration"]["align_by_chan2"] is True
    assert settings["io"] == {"combined": False, "delete_bin": False, "move_bin": False}

    settings = import_zregister.get_settings(
        _row(Channels=2), {"registration": {"align_by_chan2": False}}
    )
    assert settings["registration"]["align_by_chan2"] is False

    flybacks = tmp_path / "flybacks.csv"
    flybacks.write_text(
        'Planes,Frame_rate_range,Flyback_planes\n2,"[10, 30]","[0]"\n', encoding="utf-8"
    )
    config = {"suite2p_db": {"functional_chan": 1}, "planes_to_flyback": str(flybacks)}
    db = import_zregister.get_db({"tiffs": ["raw"], "output": "out"}, _row(), config)
    assert db["ignore_flyback"] == [0]
    assert db["data_path"] == ["raw"]
    with pytest.raises(FileNotFoundError):
        import_zregister.get_db(
            {"tiffs": ["raw"], "output": "out"},
            _row(),
            {**config, "planes_to_flyback": "missing.csv"},
        )
    with pytest.raises(ValueError, match="No matching entry"):
        import_zregister.get_db({"tiffs": ["raw"], "output": "out"}, _row(Frame_rate=100), config)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert import_zregister.check_registration_status(empty) == (False, False)

    plane = tmp_path / "registered" / "plane0"
    plane.mkdir(parents=True)
    (plane / "data.bin").touch()
    np.save(plane / "db.npy", {"nplanes": 1, "ignore_flyback": []})
    np.save(plane / "settings.npy", {})
    np.save(plane / "reg_outputs.npy", {"zpos_registration": np.arange(3)})
    assert import_zregister.check_registration_status(plane.parent) == (True, True)


class _MemoryBinary:
    arrays: dict[str, np.ndarray] = {}

    def __init__(self, *, Ly: int, Lx: int, filename, n_frames, write=False):  # noqa: N803
        """Mirror Suite2p's public ``BinaryFile`` keyword names in this test double."""
        self.filename = str(filename)
        self.arrays.setdefault(self.filename, np.zeros((n_frames, Ly, Lx), dtype=np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key):
        return self.arrays[self.filename][key]

    def __setitem__(self, key, value):
        self.arrays[self.filename][key] = value


def test_register_plane_reference_creation_and_alignment(import_zregister, monkeypatch) -> None:
    _MemoryBinary.arrays = {
        "reg.bin": np.arange(20 * 2 * 3, dtype=np.float32).reshape(20, 2, 3),
        "raw.bin": np.ones((20, 2, 3), dtype=np.float32),
        "reg2.bin": np.ones((20, 2, 3), dtype=np.float32) * 2,
        "raw2.bin": np.ones((20, 2, 3), dtype=np.float32) * 3,
    }
    monkeypatch.setattr(import_zregister.io, "BinaryFile", _MemoryBinary)
    monkeypatch.setattr(import_zregister.bidiphase, "compute", lambda frames: 2, raising=False)
    monkeypatch.setattr(
        import_zregister.bidiphase, "shift", lambda frames, phase: None, raising=False
    )
    monkeypatch.setattr(
        import_zregister.register,
        "compute_reference",
        lambda frames, settings, device: frames.mean(0),
        raising=False,
    )
    settings = {"align_by_chan2": False, "nimg_init": 6, "do_bidiphase": True, "bidiphase": 0}
    reference, bidi = import_zregister.create_ref_per_plane(
        {"reg_file": "reg.bin", "nframes": 20, "Ly": 2, "Lx": 3},
        settings,
        np.zeros(20, dtype=bool),
        torch.device("cpu"),
    )
    assert reference.shape == (2, 3)
    assert bidi == 2

    monkeypatch.setattr(
        import_zregister.process_tiff,
        "register_zstack_across_planes",
        lambda zstack, settings_reg, device: (
            zstack + 1,
            np.array([0, 1]),
            np.array([1, 0]),
            None,
            None,
            None,
        ),
    )
    aligned = import_zregister.align_ref_images(
        {"nonrigid": True},
        [np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)],
        torch.device("cpu"),
    )
    assert len(aligned) == 2

    observed = {}

    def registration_wrapper(*args, **kwargs):
        observed.update(kwargs)
        return {"zpos_registration": np.arange(20)}

    monkeypatch.setattr(
        import_zregister.register, "registration_wrapper", registration_wrapper, raising=False
    )
    outputs = import_zregister.register_plane(
        {
            "reg_file": "reg.bin",
            "raw_file": "raw.bin",
            "reg_file_chan2": "reg2.bin",
            "raw_file_chan2": "raw2.bin",
            "nframes": 20,
            "Ly": 2,
            "Lx": 3,
            "nchannels": 2,
            "keep_movie_raw": True,
            "save_path": "save",
        },
        {"align_by_chan2": True},
        aligned,
        np.zeros(20, dtype=bool),
        torch.device("cpu"),
    )
    assert outputs["zpos_registration"].shape == (20,)
    assert observed["f_raw"] is not None
    assert observed["f_reg_chan2"] is not None


def test_prepare_gather_write_and_realign_registered_movie(
    import_zregister, monkeypatch, tmp_path: Path
) -> None:
    plane_paths = []
    db_planes = []
    _MemoryBinary.arrays = {}
    for plane_id in range(2):
        plane = tmp_path / f"plane{plane_id}"
        plane.mkdir()
        plane_paths.append(plane)
        reg_file = str(plane / "data.bin")
        reg_file_chan2 = str(plane / "data_chan2.bin")
        _MemoryBinary.arrays[reg_file] = np.full((3, 2, 2), 10 + plane_id, dtype=np.float32)
        _MemoryBinary.arrays[reg_file_chan2] = np.full((3, 2, 2), 20 + plane_id, dtype=np.float32)
        np.save(
            plane / "reg_outputs.npy",
            {
                "badframes": np.array([False, plane_id == 1, False]),
                "xoff": np.array([plane_id, plane_id + 1, plane_id + 2]),
                "yoff": np.array([plane_id + 3, plane_id + 4, plane_id + 5]),
                "refImg": np.ones((2, 2)) * plane_id,
            },
        )
        db_planes.append(
            {
                "save_path": str(plane),
                "save_path0": str(tmp_path),
                "save_folder": "suite2p",
                "reg_file": reg_file,
                "reg_file_chan2": reg_file_chan2,
                "Ly": 2,
                "Lx": 2,
                "nplanes": 2,
                "nchannels": 2,
                "ignore_flyback": [],
            }
        )
    planes_across_time = np.array([0, 1, 0])
    badframes, n_frames, xoff, yoff = import_zregister.gather_single_plane_results(
        db_planes, planes_across_time
    )
    assert badframes.shape == (3, 3)
    assert n_frames == 3
    assert xoff.shape == yoff.shape == (3, 3)

    new_path = tmp_path / "plane_z"
    new_path.mkdir()
    db_new, outputs = import_zregister.prepare_registration_ops(
        0,
        str(new_path / "data.bin"),
        str(new_path),
        db_planes[0],
        np.array([0, 1]),
        planes_across_time,
    )
    assert outputs["reference_plane"] == 0
    np.testing.assert_array_equal(outputs["planes_across_time"], planes_across_time)

    monkeypatch.setattr(import_zregister.io, "BinaryFile", _MemoryBinary)
    import_zregister.write_new_file(
        badframes,
        n_frames,
        str(new_path / "data.bin"),
        str(new_path),
        db_new,
        db_planes,
        planes_across_time,
        np.array([0.2, 0.6, 0.2]),
    )
    assert _MemoryBinary.arrays[str(new_path / "data.bin")].shape == (3, 2, 2)
    assert _MemoryBinary.arrays[str(new_path / "data_chan2.bin")].shape == (3, 2, 2)

    def registration_wrapper(*args, **kwargs):
        return {
            "badframes": np.zeros(3, dtype=bool),
            "xoff": np.array([0, 1, 0]),
            "yoff": np.array([1, 0, 1]),
            "yrange": [0, 2],
            "xrange": [0, 2],
        }

    monkeypatch.setattr(
        import_zregister.register, "registration_wrapper", registration_wrapper, raising=False
    )
    realigned = import_zregister.realign_frames(
        badframes[:, 1],
        0,
        3,
        db_new,
        {"registration": {"align_by_chan2": True}, "run": {"do_regmetrics": False}},
        {"refImg": [np.ones((2, 2)), np.ones((2, 2)) * 2]},
        np.array([0.2, 0.6, 0.2]),
        xoff,
        yoff,
        torch.device("cpu"),
    )
    assert realigned["refImg_singleplanes"] is not None
    assert realigned["xrange"][1] <= 2


def test_reference_similarity_and_create_registered_file_orchestration(
    import_zregister, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        import_zregister.register,
        "compute_filters_and_norm",
        lambda *args, **kwargs: (torch.ones(1), torch.zeros(1), torch.ones(1)),
        raising=False,
    )
    monkeypatch.setattr(
        import_zregister.rigid,
        "phasecorr",
        lambda *args, **kwargs: (None, None, torch.tensor([1.0, 2.0, 1.0])),
        raising=False,
    )
    weights = import_zregister.determine_ref_similarities(
        [np.ones((2, 2)) * index for index in range(3)],
        1,
        {"smooth_sigma": 1, "spatial_taper": 1, "maxregshift": 0.1, "smooth_sigma_time": 0},
        torch.device("cpu"),
    )
    np.testing.assert_allclose(weights, [0.25, 0.5, 0.25])

    old = tmp_path / "suite2p" / "plane0"
    old.mkdir(parents=True)
    np.save(
        old / "reg_outputs.npy",
        {"refImg": [np.ones((2, 2))], "badframes": np.zeros(3, dtype=bool)},
    )
    db = {
        "save_path": str(old),
        "save_path0": str(tmp_path),
        "save_folder": "suite2p",
        "nplanes": 1,
        "ignore_flyback": [],
        "Ly": 2,
        "Lx": 2,
    }
    monkeypatch.setattr(
        import_zregister,
        "gather_single_plane_results",
        lambda dbs, planes: (np.zeros((3, 3), dtype=bool), 3, np.zeros((3, 3)), np.zeros((3, 3))),
    )
    monkeypatch.setattr(import_zregister, "write_new_file", lambda *args: None)
    monkeypatch.setattr(
        import_zregister,
        "realign_frames",
        lambda *args: {"badframes": np.zeros(3, dtype=bool)},
    )
    monkeypatch.setattr(import_zregister, "handle_single_plane_data", lambda *args: None)
    db_new, outputs = import_zregister.create_registered_file(
        [db],
        [{"registration": {}}],
        0,
        np.ones((3, 1)),
        np.array([0.25, 0.5, 0.25]),
        torch.device("cpu"),
    )
    assert db_new["save_path"].endswith("plane_z")
    assert outputs["badframes"].all()


def test_handle_plane_folders_and_process_all_datasets(
    import_zregister, monkeypatch, tmp_path: Path
) -> None:
    suite2p = tmp_path / "suite2p"
    for index in range(3):
        (suite2p / f"plane{index}").mkdir(parents=True)
    db = {"save_path0": str(tmp_path), "save_folder": "suite2p", "ignore_flyback_singleplanes": [0]}
    import_zregister.handle_single_plane_data(False, db)
    assert not (suite2p / "plane0").exists()
    assert (suite2p / "backup1").is_dir()
    assert (suite2p / "backup2").is_dir()

    datasets = pd.DataFrame([_row(Process=False), _row(Name="missing"), _row(Name="valid")])

    def paths(row, directories):
        output = tmp_path / row["Name"]
        output.mkdir(exist_ok=True)
        return {
            "output": str(output),
            "tiffs": None if row["Name"] == "missing" else [str(tmp_path)],
        }

    monkeypatch.setattr(import_zregister, "make_paths", paths)
    monkeypatch.setattr(import_zregister.general, "setup_suite2p_logging", lambda path: None)
    monkeypatch.setattr(
        import_zregister,
        "get_settings",
        lambda row, config: {
            "run": {"do_registration": True, "do_detection": True},
            "registration": {"delete_singleplanes": False},
            "torch_device": "cpu",
        },
    )
    monkeypatch.setattr(
        import_zregister,
        "get_db",
        lambda paths, row, config: {"save_path0": paths["output"], "save_folder": "suite2p"},
    )
    monkeypatch.setattr(import_zregister.parameters, "default_db", lambda: {})
    monkeypatch.setattr(
        import_zregister.parameters, "set_db", lambda target, values: target.update(values)
    )
    monkeypatch.setattr(
        import_zregister.parameters, "set_settings", lambda target, values: target.update(values)
    )
    plane_z = tmp_path / "valid" / "suite2p" / "plane_z"
    observed = {}

    def register_dataset(db, settings):
        plane_z.mkdir(parents=True)
        return {**db, "save_path": str(plane_z)}, {"zpos_registration": np.arange(3)}

    monkeypatch.setattr(import_zregister, "register_dataset", register_dataset)
    monkeypatch.setattr(import_zregister.registration_plotting, "make_plots", lambda *args: None)
    monkeypatch.setattr(
        import_zregister.suite2p_compat,
        "load_parameters",
        lambda path: ({"save_path": str(plane_z)}, {"registration": {}}),
    )
    monkeypatch.setattr(
        import_zregister.run_s2p_mod,
        "run_s2p",
        lambda *, db, settings: observed.update(db=db, settings=settings),
        raising=False,
    )

    import_zregister.process_all_datasets(
        {"directories": {}, "suite2p_settings": {}, "suite2p_db": {}}, datasets
    )

    assert observed["db"]["save_path"] == str(plane_z)
    assert (plane_z / "db.npy").is_file()
    assert (plane_z / "settings.npy").is_file()


def test_register_dataset_full_uncached_orchestration(
    import_zregister, monkeypatch, tmp_path: Path
) -> None:
    save_folder = tmp_path / "suite2p"
    db0 = {
        "save_path0": str(tmp_path),
        "save_folder": "suite2p",
        "nplanes": 2,
        "nchannels": 1,
        "input_format": "tif",
        "ignore_flyback": [],
        "keep_movie_raw": False,
    }
    user_settings = {
        "torch_device": "cpu",
        "registration": {"delete_singleplanes": False},
    }
    monkeypatch.setattr(import_zregister.parameters, "default_settings", lambda: {})
    monkeypatch.setattr(
        import_zregister.parameters, "set_settings", lambda target, values: target.update(values)
    )
    monkeypatch.setattr(
        import_zregister.run_s2p_mod,
        "get_save_folder",
        lambda db: str(save_folder),
        raising=False,
    )
    monkeypatch.setattr(import_zregister, "check_registration_status", lambda path: (False, False))
    monkeypatch.setattr(
        import_zregister.io, "get_file_list", lambda db: (["movie.tif"], [0]), raising=False
    )

    dbs = []
    for plane_id in range(2):
        plane = save_folder / f"plane{plane_id}"
        plane.mkdir(parents=True)
        dbs.append(
            {
                **db0,
                "save_path": str(plane),
                "reg_file": str(plane / "data.bin"),
                "nframes": 50,
                "Ly": 2,
                "Lx": 3,
                "frames_per_folder": [50],
                "data_path": [str(tmp_path)],
            }
        )
    monkeypatch.setattr(import_zregister.io, "init_dbs", lambda db: dbs, raising=False)
    monkeypatch.setitem(import_zregister.files_to_binary, "tif", lambda values, *args: values)
    monkeypatch.setattr(
        import_zregister.suite2p_compat,
        "load_parameters",
        lambda path: (dbs[int(Path(path).name.removeprefix("plane"))], {"registration": {}}),
    )
    monkeypatch.setattr(
        import_zregister.general, "assign_torch_device", lambda requested: torch.device("cpu")
    )
    monkeypatch.setattr(
        import_zregister,
        "create_ref_per_plane",
        lambda db, settings, bad, device: (
            np.ones((2, 3)) * int(Path(db["save_path"]).name[-1]),
            0,
        ),
    )
    monkeypatch.setattr(
        import_zregister,
        "align_ref_images",
        lambda settings, refs, device: refs,
    )

    def register_plane(db, settings, refs, bad, device):
        plane_id = int(Path(db["save_path"]).name[-1])
        correlations = np.full((50, 2), 0.2)
        correlations[:, plane_id] = 0.9
        return {
            "cmax_registration": correlations,
            "zpos_registration": np.full(50, plane_id),
            "refImg": refs,
            "badframes": np.zeros(50, dtype=bool),
        }

    monkeypatch.setattr(import_zregister, "register_plane", register_plane)
    monkeypatch.setattr(
        import_zregister,
        "determine_ref_similarities",
        lambda *args: np.array([0.2, 0.6, 0.2]),
    )
    combined = save_folder / "plane_z"
    combined.mkdir()

    def create_registered_file(*args):
        db_new = {
            **dbs[0],
            "save_path": str(combined),
            "db_path": str(combined / "db.npy"),
            "settings_path": str(combined / "settings.npy"),
        }
        return db_new, {"badframes": np.zeros(50, dtype=bool)}

    monkeypatch.setattr(import_zregister, "create_registered_file", create_registered_file)

    db_new, outputs = import_zregister.register_dataset(db0, user_settings)

    assert db_new["save_path"] == str(combined)
    assert outputs["corrs_time_refs_planes"].shape == (50, 2, 2)
    assert (combined / "reg_outputs.npy").is_file()
    assert (combined / "settings.npy").is_file()


def test_register_dataset_resumes_cached_planes(
    import_zregister, monkeypatch, tmp_path: Path
) -> None:
    save_folder = tmp_path / "suite2p"
    plane_paths = [save_folder / "plane0", save_folder / "plane1"]
    dbs = []
    for plane_id, plane in enumerate(plane_paths):
        plane.mkdir(parents=True)
        correlations = np.full((4, 2), 0.1)
        correlations[:, plane_id] = 1
        np.save(
            plane / "reg_outputs.npy",
            {
                "cmax_registration": correlations,
                "zpos_registration": np.full(4, plane_id),
                "refImg": [np.ones((2, 2)), np.ones((2, 2)) * 2],
            },
        )
        dbs.append(
            {
                "save_path": str(plane),
                "save_path0": str(tmp_path),
                "save_folder": "suite2p",
                "nplanes": 2,
                "nchannels": 1,
                "ignore_flyback": [],
                "frames_per_folder": [4],
            }
        )
    stale = save_folder / "plane_z"
    stale.mkdir()
    monkeypatch.setattr(
        import_zregister.parameters,
        "default_settings",
        lambda: {"torch_device": "cpu", "registration": {}},
    )
    monkeypatch.setattr(
        import_zregister.parameters, "set_settings", lambda target, values: target.update(values)
    )
    monkeypatch.setattr(
        import_zregister.run_s2p_mod, "get_save_folder", lambda db: str(save_folder), raising=False
    )
    monkeypatch.setattr(import_zregister, "check_registration_status", lambda path: (True, True))
    monkeypatch.setattr(
        import_zregister.suite2p_compat,
        "load_parameters",
        lambda path: (dbs[int(Path(path).name[-1])], {"registration": {}}),
    )
    monkeypatch.setattr(
        import_zregister.general, "assign_torch_device", lambda requested: torch.device("cpu")
    )
    monkeypatch.setattr(
        import_zregister, "determine_ref_similarities", lambda *args: np.array([0.2, 0.6, 0.2])
    )
    combined = save_folder / "plane_z"

    def create_registered_file(*args):
        combined.mkdir()
        return (
            {
                **dbs[0],
                "save_path": str(combined),
                "db_path": str(combined / "db.npy"),
                "settings_path": str(combined / "settings.npy"),
            },
            {"badframes": np.zeros(4, dtype=bool)},
        )

    monkeypatch.setattr(import_zregister, "create_registered_file", create_registered_file)

    db_new, outputs = import_zregister.register_dataset(
        {"nplanes": 2, "ignore_flyback": []},
        {"torch_device": "cpu", "registration": {"delete_singleplanes": False}},
    )

    assert db_new["save_path"] == str(combined)
    assert outputs["corrs_time_refs_planes"].shape == (4, 2, 2)
