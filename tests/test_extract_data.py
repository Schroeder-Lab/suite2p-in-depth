from __future__ import annotations

from pathlib import Path

import numpy as np

from suite2p_in_depth.extract_data import assign_frame_time, get_nidaq_data


def test_assign_frame_time_extracts_rising_edges() -> None:
    frame_clock = np.array([0.0, 0.8, 0.9, 0.1, 0.0, 0.7, 0.7, 0.0])

    frame_times = assign_frame_time(frame_clock, th=0.5, sampling_rate=10)

    np.testing.assert_array_equal(frame_times, np.array([0.1, 0.5]))
    assert frame_times.shape == (2,)
    assert frame_times.dtype == np.float64


def test_get_nidaq_data_parses_channels_and_truncates_partial_sample(tmp_path: Path) -> None:
    (tmp_path / "channels.csv").write_text("frame,piezo\n", encoding="utf-8")
    raw = np.arange(7, dtype=np.float64)
    raw.tofile(tmp_path / "recording.bin")

    data, channels, times = get_nidaq_data(
        str(tmp_path), "recording.bin", "channels.csv", sampling_rate=2
    )

    np.testing.assert_array_equal(data, np.arange(6, dtype=np.float64).reshape(3, 2))
    np.testing.assert_array_equal(channels, np.array(["frame", "piezo"]))
    np.testing.assert_array_equal(times, np.array([0.0, 0.5, 1.0]))
    assert data.dtype == np.float64
    assert channels.dtype.kind == "U"
    assert times.dtype == np.float64
