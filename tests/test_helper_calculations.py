from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from suite2p_in_depth.utilities import determine_planes_channels_rate as imaging_metadata
from suite2p_in_depth.utilities import extract_lowest_pixel_values as dark_pixels


def test_determine_imaging_config_updates_processed_rows_only(monkeypatch, tmp_path: Path) -> None:
    experiment_dir = tmp_path / "mouse-a" / "2024-01-02" / "1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "one.tif").touch()
    datasets = pd.DataFrame(
        [
            {
                "Name": "mouse-a",
                "Date": "2024-01-02",
                "Experiments": ["1"],
                "Process": True,
            },
            {
                "Name": "mouse-b",
                "Date": "2024-01-03",
                "Experiments": ["1"],
                "Process": False,
            },
        ]
    )
    config = {
        "directories": {"imaging": str(tmp_path)},
        "daq": {
            "sampling_rate": 1000,
            "data": "recording.bin",
            "channel_names": "channels.csv",
            "clock_channel_name": "frame",
            "piezo_channel_name": "piezo",
        },
    }

    monkeypatch.setattr(
        imaging_metadata,
        "get_frame_times",
        lambda *args, **kwargs: np.arange(41, dtype=float) * 0.1,
    )
    monkeypatch.setattr(
        imaging_metadata,
        "get_piezo_data",
        lambda *args, **kwargs: (
            np.floor(np.arange(0, 4, 0.01) / 0.1).astype(int) % 4,
            np.arange(0, 4, 0.01),
        ),
    )

    class FakeTiff:
        def __init__(self, path):
            self.series = [SimpleNamespace(shape=(10, 2, 16, 16))]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(imaging_metadata.tifffile, "TiffFile", FakeTiff)

    result = imaging_metadata.determine_imaging_config(config, datasets)

    assert result is datasets
    assert result.loc[0, "Planes"] == 4
    assert result.loc[0, "Frame_rate"] == pytest.approx(10.0)
    assert result.loc[0, "Channels"] == 2
    assert np.isnan(result.loc[1, "Planes"])
    assert np.isnan(result.loc[1, "Frame_rate"])
    assert np.isnan(result.loc[1, "Channels"])
    assert all(
        result[column].dtype == np.float64 for column in ("Planes", "Frame_rate", "Channels")
    )


def test_find_plane_folders_uses_natural_numeric_order(tmp_path: Path) -> None:
    for name in ("plane10", "plane2", "plane0"):
        (tmp_path / name).mkdir()
    (tmp_path / "combined").mkdir()

    folders = dark_pixels._find_plane_folders(str(tmp_path))

    assert folders == [
        str(tmp_path / "plane0"),
        str(tmp_path / "plane2"),
        str(tmp_path / "plane10"),
    ]


def test_dark_pixel_percentile_reads_bounded_frame_count(monkeypatch, tmp_path: Path) -> None:
    frames = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    observed = {}

    class FakeBinaryFile:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __getitem__(self, key):
            observed["key"] = key
            return frames[key]

    monkeypatch.setattr(dark_pixels.io, "BinaryFile", FakeBinaryFile)

    bin_path = tmp_path / "data.bin"
    bin_path.touch()
    result = dark_pixels.percentile_first_1000_frames(
        str(bin_path), {"Ly": 2, "Lx": 3, "nframes": 5}, percentile=10, nframes=1000
    )

    assert result == pytest.approx(2.9, rel=0, abs=2e-7)
    assert isinstance(result, float)
    assert observed["kwargs"] == {
        "Ly": 2,
        "Lx": 3,
        "filename": str(bin_path),
        "n_frames": 5,
        "write": False,
    }
    assert observed["key"] == (slice(None, 5, None), slice(None), slice(None))


def test_short_piezo_recording_can_be_averaged(import_correct_traces) -> None:
    piezo_time = np.arange(1000, dtype=float) / 1000
    piezo = np.sin(2 * np.pi * piezo_time)
    frame_times = np.arange(12, dtype=float) * 0.05

    result = import_correct_traces.make_piezo_trace_per_plane(
        piezo,
        piezo_time,
        frame_times,
        num_planes=2,
        win_size=10,
        max_avg=1000,
    )

    assert result.shape == (50, 2)
    assert np.isfinite(result).all()
