"""Subprocess smoke checks against the installed, non-stubbed Suite2p release."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPECTED_SUITE2P_COMMIT = "90be8953c03c6f4275dcd392ac1c6f554116169e"
EXPECTED_SUITE2P_URL = "https://github.com/MouseLand/suite2p.git"


def main(work_directory: str) -> None:
    work_path = Path(work_directory).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    # Suite2p creates its settings directory during import. Redirect Path.home
    # before importing it so this smoke check remains isolated in pytest tmp_path.
    Path.home = classmethod(lambda cls: work_path)  # type: ignore[method-assign]

    import suite2p
    from suite2p import parameters
    from suite2p.registration import nonrigid

    from suite2p_in_depth import process_tiff, suite2p_extensions

    direct_url = importlib.metadata.distribution("suite2p").read_text("direct_url.json")
    if direct_url is None:
        raise AssertionError("Suite2p direct_url.json is missing.")
    source = json.loads(direct_url)
    assert suite2p.version == "1.1.0"
    assert source["url"] == EXPECTED_SUITE2P_URL
    assert source["vcs_info"]["commit_id"] == EXPECTED_SUITE2P_COMMIT

    device = torch.device("cpu")
    settings = parameters.default_settings()["registration"]
    settings["batch_size"] = 4
    settings["norm_frames"] = False

    rng = np.random.default_rng(8)
    reference_frames = rng.integers(-500, 500, size=(8, 32, 32), dtype=np.int16)
    reference = suite2p_extensions.compute_reference(
        reference_frames, settings, device=device, n_best=4
    )
    assert reference.shape == (32, 32)
    assert reference.dtype == np.int16

    zstack = rng.integers(-1000, 1000, size=(3, 32, 32), dtype=np.int16)
    expected_planes = np.array([0, 1, 2, 1])
    movie = zstack[expected_planes]
    movie_path = work_path / "registered.bin"
    movie.tofile(movie_path)
    correlations = suite2p_extensions.compute_zpos(
        zstack,
        {
            "Ly": movie.shape[1],
            "Lx": movie.shape[2],
            "nframes": movie.shape[0],
            "reg_file": str(movie_path),
        },
        settings,
        device=device,
    )
    assert correlations.shape == (zstack.shape[0], movie.shape[0])
    assert np.array_equal(np.argmax(correlations, axis=0), expected_planes)

    height, width, n_frames = 64, 256, 2
    yblock, xblock, nblocks, *_ = nonrigid.make_blocks(height, width, (128, 128))
    blocks = [yblock, xblock, nblocks]
    rectangular_movie = rng.integers(-500, 500, size=(n_frames, height, width), dtype=np.int16)
    nonrigid_offsets = np.zeros((n_frames, nblocks[0] * nblocks[1]), dtype=np.float32)
    shifted = suite2p_extensions.shift_frames(
        torch.from_numpy(rectangular_movie),
        yoff=np.zeros(n_frames, dtype=int),
        xoff=np.zeros(n_frames, dtype=int),
        yoff1=nonrigid_offsets,
        xoff1=nonrigid_offsets,
        blocks=blocks,
        device=device,
    )
    assert 1 in nblocks
    assert shifted.shape == rectangular_movie.shape

    settings["block_size"] = [128, 128]
    settings["nonrigid"] = True
    settings["maxregshiftNR"] = 5
    registered_stack = process_tiff.register_zstack_across_planes(
        rectangular_movie[:2], settings, device
    )[0]
    assert registered_stack.shape == rectangular_movie[:2].shape

    aligned_stack, best_slice, _ = process_tiff.register_zstack_to_ref(
        rectangular_movie,
        rectangular_movie[0],
        settings,
        device=device,
    )
    assert aligned_stack.shape == rectangular_movie.shape
    assert 0 <= best_slice < rectangular_movie.shape[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: real_suite2p_smoke.py WORK_DIRECTORY")
    main(sys.argv[1])
