from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from suite2p_in_depth import suite2p_extensions


def test_pick_initial_reference_honors_requested_frame_count() -> None:
    frames = torch.arange(4 * 8 * 8, dtype=torch.int16).reshape(4, 8, 8)

    reference = suite2p_extensions.pick_initial_reference(frames, n_best=1)

    assert reference.shape == (8, 8)
    assert reference.dtype == np.int16
    with pytest.raises(ValueError, match="n_best"):
        suite2p_extensions.pick_initial_reference(frames, n_best=0)


def test_compute_zpos_batches_and_transposes_correlations(monkeypatch) -> None:
    frames = np.arange(5 * 4 * 6, dtype=np.int16).reshape(5, 4, 6)

    class FakeBinary:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["filename"] == "registered.bin"

        def __enter__(self) -> FakeBinary:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def __getitem__(self, item: slice) -> np.ndarray:
            return frames[item]

    monkeypatch.setattr(suite2p_extensions.io, "BinaryFile", FakeBinary)
    monkeypatch.setattr(
        suite2p_extensions.register,
        "compute_filters_and_norm",
        lambda **kwargs: [object(), object(), object()],
        raising=False,
    )
    batch_starts: list[int] = []

    def compute_shifts(references, batch, **kwargs):
        batch_starts.append(int(batch[0, 0, 0]))
        count = batch.shape[0]
        correlations = np.arange(count * len(references), dtype=np.float32).reshape(
            count, len(references)
        )
        return (None, None, None, None, None, None, None, correlations)

    monkeypatch.setattr(
        suite2p_extensions.register, "compute_shifts", compute_shifts, raising=False
    )
    settings = {
        "norm_frames": False,
        "smooth_sigma": 1.0,
        "spatial_taper": 3.0,
        "batch_size": 2,
        "maxregshift": 0.1,
        "smooth_sigma_time": 0.0,
    }

    result = suite2p_extensions.compute_zpos(
        np.zeros((3, 4, 6), dtype=np.int16),
        {"Ly": 4, "Lx": 6, "nframes": 5, "reg_file": "registered.bin"},
        settings,
        device=torch.device("cpu"),
    )

    assert result.shape == (3, 5)
    assert batch_starts == [0, 48, 96]
    np.testing.assert_array_equal(result[:, :2], np.arange(6).reshape(2, 3).T)


@pytest.mark.parametrize("shape", [(64, 256), (256, 64)])
def test_shift_frames_supports_single_block_grid_dimension(shape) -> None:
    n_frames = 2
    height, width = shape
    if height == 64:
        nblocks = [1, 3]
        yblock = [np.array([0, height])] * 3
        xblock = [
            np.array([0, 128]),
            np.array([64, 192]),
            np.array([128, 256]),
        ]
    else:
        nblocks = [3, 1]
        yblock = [
            np.array([0, 128]),
            np.array([64, 192]),
            np.array([128, 256]),
        ]
        xblock = [np.array([0, width])] * 3
    blocks = [yblock, xblock, nblocks]
    movie = torch.arange(n_frames * height * width, dtype=torch.int32).remainder(1000)
    movie = movie.to(torch.int16).reshape(n_frames, height, width)
    offsets = np.zeros((n_frames, 3), dtype=np.float32)

    shifted = suite2p_extensions.shift_frames(
        movie,
        yoff=np.zeros(n_frames, dtype=int),
        xoff=np.zeros(n_frames, dtype=int),
        yoff1=offsets,
        xoff1=offsets,
        blocks=blocks,
        device=torch.device("cpu"),
    )

    assert shifted.shape == movie.shape
    difference = np.abs(shifted.astype(np.int32) - movie.numpy().astype(np.int32))
    assert difference.max() <= 2
