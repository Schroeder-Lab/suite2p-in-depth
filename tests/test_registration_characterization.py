from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_zregistration_dataset_csv_parsing_types(import_zregister, tmp_path: Path) -> None:
    csv_path = tmp_path / "datasets.csv"
    csv_path.write_text(
        "Name,Date,Depth,Experiments,Planes,Frame_rate,Channels,Process\n"
        "mouse-a,2024-01-02,150.5,\"['1', '3']\",4,30.0,2,True\n",
        encoding="utf-8",
    )

    datasets = import_zregister.load_dataset_list(str(csv_path))
    row = datasets.iloc[0]

    assert row["Name"] == "mouse-a"
    assert row["Date"] == "2024-01-02"
    assert row["Experiments"] == ["1", "3"]
    assert datasets["Depth"].dtype == np.dtype("float64")
    assert datasets["Planes"].dtype == pd.Int64Dtype()
    assert datasets["Frame_rate"].dtype == np.dtype("float64")
    assert datasets["Channels"].dtype == pd.Int64Dtype()
    assert datasets["Process"].dtype == np.dtype("bool")


def test_trace_dataset_csv_parsing_types(import_correct_traces, tmp_path: Path) -> None:
    csv_path = tmp_path / "datasets.csv"
    csv_path.write_text(
        "Name,Date,Depth,Zstack_folder,Ignore_planes,Process\n"
        "mouse-a,2024-01-02,150.5,zstack-a,2,True\n",
        encoding="utf-8",
    )

    datasets = import_correct_traces.load_dataset_list(str(csv_path))

    assert datasets.iloc[0].to_dict() == {
        "Name": "mouse-a",
        "Date": "2024-01-02",
        "Depth": 150.5,
        "Zstack_folder": "zstack-a",
        "Ignore_planes": 2,
        "Process": True,
    }
    assert datasets["Ignore_planes"].dtype == pd.Int64Dtype()
    assert datasets["Process"].dtype == np.dtype("bool")


def test_determine_best_reference_uses_correlation_tie_break(import_zregister) -> None:
    correlations = np.full((4, 3, 3), 0.1, dtype=np.float64)
    correlations[:, 0, 0] = 0.6
    correlations[:, 1, 1] = 0.9
    correlations[:, 2, 2] = 0.7

    best_reference = import_zregister.determine_best_ref(correlations)

    assert best_reference == 1
    assert isinstance(best_reference, int)


def test_equalize_frame_numbers_trims_each_experiment_in_place(import_zregister) -> None:
    correlations = [
        np.arange(5 * 2 * 2).reshape(5, 2, 2),
        np.arange(100, 100 + 5 * 2 * 2).reshape(5, 2, 2),
    ]
    zpositions = [np.arange(5), np.arange(10, 15)]
    databases = [
        {"frames_per_folder": [3, 2]},
        {"frames_per_folder": [2, 3]},
    ]

    result = import_zregister.equalize_frame_numbers(correlations, zpositions, databases)

    assert result is None
    assert [array.shape for array in correlations] == [(4, 2, 2), (4, 2, 2)]
    np.testing.assert_array_equal(zpositions[0], np.array([0, 1, 3, 4]))
    np.testing.assert_array_equal(zpositions[1], np.array([10, 11, 12, 13]))
    np.testing.assert_array_equal(
        correlations[0], np.delete(np.arange(20).reshape(5, 2, 2), 2, axis=0)
    )
    np.testing.assert_array_equal(
        correlations[1],
        np.delete(np.arange(100, 120).reshape(5, 2, 2), 4, axis=0),
    )


def test_average_neighbor_frames_records_weighting_shape_and_dtype(import_zregister) -> None:
    sources = [
        np.full((2, 2, 3), 10, dtype=np.int16),
        np.full((2, 2, 3), 20, dtype=np.int16),
        np.full((2, 2, 3), 40, dtype=np.int16),
    ]

    averaged = import_zregister.average_neighbor_frames(
        np.array([0, 1, 2]), 2, 3, sources, t=1, weights=np.array([0.2, 0.3, 0.5])
    )

    np.testing.assert_allclose(averaged, np.full((2, 3), 28.0), rtol=0, atol=1e-6)
    assert averaged.shape == (2, 3)
    assert averaged.dtype == np.float32


def test_average_neighbor_frames_ignores_out_of_range_neighbors(import_zregister) -> None:
    sources = [
        np.full((1, 2, 2), 10, dtype=np.float32),
        np.full((1, 2, 2), 20, dtype=np.float32),
    ]

    averaged = import_zregister.average_neighbor_frames(
        np.array([-1, 0, 1]), 2, 2, sources, t=0, weights=np.array([0.0, 0.25, 0.75])
    )

    np.testing.assert_allclose(averaged, np.full((2, 2), 17.5), rtol=0, atol=1e-6)
    assert averaged.dtype == np.float32
