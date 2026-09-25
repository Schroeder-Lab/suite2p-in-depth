from __future__ import annotations

import numpy as np

from suite2p_in_depth.preprocess_traces import (
    correct_neuropil,
    correct_zmotion,
    get_delta_f_over_f,
    get_f0,
)


def test_get_f0_constant_signal_shape_dtype_and_value() -> None:
    signal = np.full((12, 2), [7.0, 12.0], dtype=np.float64)

    baseline = get_f0(signal, frame_rate=4, f0_percentile=8, window_size=2)

    np.testing.assert_allclose(baseline, signal, rtol=0, atol=1e-12)
    assert baseline.shape == signal.shape
    assert baseline.dtype == np.float64


def test_get_f0_handles_experiments_independently() -> None:
    signal = np.concatenate((np.full((8, 1), 5.0), np.full((8, 1), 50.0)), axis=0)

    baseline = get_f0(signal, frame_rate=2, window_size=2, frames_per_folder=[8, 8])

    np.testing.assert_allclose(baseline[:8], 5.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(baseline[8:], 50.0, rtol=0, atol=1e-12)


def test_delta_f_over_f_records_denominator_floor_and_nan_behavior() -> None:
    fluorescence = np.array([[2.0, 4.0, np.nan], [3.0, 7.0, 5.0]])
    baseline = np.array([[0.5, 2.0, 3.0], [1.0, 5.0, np.nan]])

    delta = get_delta_f_over_f(fluorescence, baseline)

    expected = np.array([[1.5, 1.0, np.nan], [2.0, 0.4, np.nan]])
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-12, equal_nan=True)
    assert delta.shape == fluorescence.shape
    assert delta.dtype == np.float64


def test_correct_neuropil_well_conditioned_characterization() -> None:
    rng = np.random.default_rng(7)
    time = np.arange(400)
    neuropil = np.column_stack((50 + 8 * np.sin(time / 17), 35 + 5 * np.cos(time / 23)))
    activity = np.column_stack((rng.exponential(0.15, 400), rng.exponential(0.2, 400)))
    fluorescence = 200 + neuropil * np.array([0.55, 0.8]) + activity

    corrected, regression, f_bins, n_bins = correct_neuropil(
        fluorescence,
        neuropil,
        fs=10,
        num_neuropil_bins=8,
        min_neuropil_percentile=5,
        max_neuropil_percentile=95,
        fluorescence_percentile=10,
        baseline_window=3,
    )

    assert corrected.shape == (400, 2)
    assert regression.shape == (2, 2)
    assert f_bins.shape == (2, 8)
    assert n_bins.shape == (2, 8)
    assert all(value.dtype == np.float64 for value in (corrected, regression, f_bins, n_bins))
    assert not np.isnan(corrected).any()
    np.testing.assert_allclose(
        regression,
        [[0.06828745, 0.54890913], [0.04059306, 0.79123810]],
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        corrected[:3],
        [
            [228.29983694, 230.76343896],
            [228.35942522, 230.79221434],
            [228.30194145, 230.46963530],
        ],
        rtol=2e-9,
        atol=2e-8,
    )


def test_correct_zmotion_well_conditioned_characterization() -> None:
    fluorescence = np.array(
        [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0], [14.0, 24.0], [15.0, 25.0]]
    )
    neuropil = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0], [3.5, 4.5]])
    profiles = np.array([[0.5, 0.8], [1.0, 1.0], [2.0, 1.2]])
    ztrace = np.array([0, 1, 2, 1, 0, 2])

    corrected_f, corrected_n, peaks = correct_zmotion(
        fluorescence,
        neuropil,
        profiles,
        ztrace,
        reference_depth=1,
        ignore_faults=False,
    )

    expected_f = np.array(
        [
            [10.06549579, 20.88726568],
            [10.41415578, 21.54665126],
            [10.50300348, 22.06813518],
            [10.66662825, 22.67665955],
            [10.96351835, 23.40680929],
            [11.43288061, 24.24041501],
        ]
    )
    expected_n = np.array(
        [
            [1.00654958, 2.08872657],
            [1.42011215, 2.56507753],
            [1.75050058, 3.00929116],
            [2.05127466, 3.45079602],
            [2.34932536, 3.90113488],
            [2.66767214, 4.36327470],
        ]
    )
    np.testing.assert_allclose(corrected_f, expected_f, rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(corrected_n, expected_n, rtol=2e-9, atol=2e-8)
    assert corrected_f.shape == fluorescence.shape
    assert corrected_n.shape == neuropil.shape
    assert corrected_f.dtype == np.float64
    assert corrected_n.dtype == np.float64
    assert peaks is None


def test_correct_zmotion_preserves_nan_ztrace_gaps_without_contaminating_neighbors() -> None:
    fluorescence = np.arange(10.0, 15.0)[:, None]
    neuropil = np.arange(2.0, 7.0)[:, None]
    profiles = np.array([[0.5], [1.0], [2.0]])
    ztrace = np.array([0.0, 1.0, np.nan, 2.0, 1.0])

    corrected_f, corrected_n, _ = correct_zmotion(
        fluorescence,
        neuropil,
        profiles,
        ztrace,
        reference_depth=1,
        ignore_faults=False,
    )
    first_f, first_n, _ = correct_zmotion(
        fluorescence[:2],
        neuropil[:2],
        profiles,
        ztrace[:2],
        reference_depth=1,
        ignore_faults=False,
    )
    last_f, last_n, _ = correct_zmotion(
        fluorescence[3:],
        neuropil[3:],
        profiles,
        ztrace[3:],
        reference_depth=1,
        ignore_faults=False,
    )

    assert np.isnan(corrected_f[2]).all()
    assert np.isnan(corrected_n[2]).all()
    np.testing.assert_allclose(corrected_f[:2], first_f)
    np.testing.assert_allclose(corrected_n[:2], first_n)
    np.testing.assert_allclose(corrected_f[3:], last_f)
    np.testing.assert_allclose(corrected_n[3:], last_n)


def test_correct_neuropil_constant_neuropil_returns_nan_for_affected_rois(caplog) -> None:
    fluorescence = np.arange(20, dtype=float).reshape(10, 2)
    neuropil = np.ones_like(fluorescence)
    corrected, regression, f_bins, n_bins = correct_neuropil(fluorescence, neuropil, fs=2)
    assert np.isnan(corrected).all()
    assert np.isnan(regression).all()
    assert np.isnan(f_bins).all()
    assert np.isnan(n_bins).all()
    assert "neuropil range is constant or invalid" in caplog.text
