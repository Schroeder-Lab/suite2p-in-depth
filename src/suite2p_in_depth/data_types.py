"""Shared structures produced by the dataset path builders."""

from typing import NotRequired, TypedDict

import numpy as np


class RegistrationPaths(TypedDict):
    """Resolved paths for a Suite2p or z-registration dataset."""

    tiffs: str | list[str] | None
    output: str


class TracePaths(TypedDict):
    """Resolved paths for a trace-correction dataset."""

    tiffs: str | None
    suite2p: str | None
    output: str
    piezo: NotRequired[str | None]
    zstack: NotRequired[str | None]


class PlaneTraceResults(TypedDict):
    """Per-plane traces and ROI metadata returned by trace correction."""

    zCorr_stack: np.ndarray | None
    zTrace: np.ndarray | None
    zProfiles: np.ndarray | None
    F_zcorrected: np.ndarray | None
    N_zcorrected: np.ndarray | None
    F_ncorrected: np.ndarray
    N_regression: np.ndarray
    F_bin_values: np.ndarray
    N_bin_values: np.ndarray
    dff: np.ndarray
    locs: np.ndarray
    cellId: np.ndarray
