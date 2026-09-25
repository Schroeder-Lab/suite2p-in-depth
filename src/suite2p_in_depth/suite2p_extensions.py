"""suite2p-in-depth registration extensions built on Suite2p's public modules.

These functions keep project-specific axial-registration behavior local so the
package can depend directly on an immutable official Suite2p release.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_f
from suite2p import io
from suite2p.registration import register, rigid

logger = logging.getLogger("suite2p.suite2p_in_depth")
DEFAULT_TORCH_DEVICE = torch.device("cuda")


def pick_initial_reference(frames: torch.Tensor, n_best: int = 20) -> npt.NDArray[np.int16]:
    """Build an initial reference from highly correlated frames.

    Parameters
    ----------
    frames : torch.Tensor
        Movie tensor with shape ``(frames, y, x)``.
    n_best : int, optional
        Maximum number of frames averaged around the best seed.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``int16`` reference image.

    Raises
    ------
    ValueError
        If the movie has fewer than two frames or ``n_best`` is invalid.
    """
    if frames.ndim != 3:
        raise ValueError("frames must have shape (n_frames, Ly, Lx).")
    n_images, height, width = frames.shape
    if n_images < 2:
        raise ValueError("At least two frames are required to select a reference.")
    if n_best < 1:
        raise ValueError("n_best must be at least 1.")
    n_best = min(int(n_best), n_images)

    normalized = frames.clone().reshape(n_images, -1).double()
    normalized -= normalized.mean(dim=1, keepdim=True)
    correlations = normalized @ normalized.T
    norms = torch.diag(correlations).sqrt()
    correlations = correlations / torch.outer(norms, norms)
    sorted_correlations = torch.sort(correlations, dim=1, descending=True).values

    # The first value is the frame's self-correlation. When only one output
    # frame is requested, one other correlation is still needed to select the seed.
    n_seed_correlations = max(2, n_best)
    seed_scores = sorted_correlations[:, 1:n_seed_correlations].mean(dim=1)
    seed = torch.argmax(seed_scores)
    selected = torch.argsort(correlations[seed], descending=True)[:n_best]
    reference = normalized[selected].mean(dim=0).cpu().numpy().astype(np.int16)
    return reference.reshape(height, width)


def compute_reference(
    frames: npt.NDArray[Any],
    settings: Mapping[str, Any],
    device: torch.device,
    n_best: int = 20,
) -> npt.NDArray[np.int16]:
    """Iteratively register frames into a Suite2p-style reference.

    Parameters
    ----------
    frames : numpy.ndarray
        Movie in ``(frames, y, x)`` order.
    settings : Mapping[str, Any]
        Suite2p registration and batching settings.
    device : torch.device
        Device used for phase correlation.
    n_best : int, optional
        Number of correlated frames in the initial reference.

    Returns
    -------
    numpy.ndarray
        Final two-dimensional ``int16`` reference image.

    Raises
    ------
    ValueError
        If the movie shape or batch size is invalid.
    """
    frame_array = np.asarray(frames)
    if frame_array.ndim != 3 or frame_array.shape[0] < 2:
        raise ValueError("frames must have shape (n_frames, Ly, Lx) with at least two frames.")

    registered = torch.from_numpy(frame_array)
    reference = pick_initial_reference(registered, n_best=n_best)
    batch_size = int(settings["batch_size"])
    if batch_size < 1:
        raise ValueError("settings['batch_size'] must be at least 1.")

    for iteration in range(8):
        mask_mul, mask_offset, reference_fft = register.compute_filters_and_norm(
            reference,
            False,
            settings["smooth_sigma"],
            settings["spatial_taper"],
            block_size=None,
            device=device,
        )[:3]
        all_y_shifts: list[torch.Tensor] = []
        all_x_shifts: list[torch.Tensor] = []
        all_correlations: list[torch.Tensor] = []
        for start in range(0, registered.shape[0], batch_size):
            stop = min(start + batch_size, registered.shape[0])
            batch = registered[start:stop].to(device)
            y_shift, x_shift, correlation = rigid.phasecorr(
                batch,
                reference_fft,
                mask_mul,
                mask_offset,
                maxregshift=settings["maxregshift"],
                smooth_sigma_time=settings["smooth_sigma_time"],
            )[:3]
            batch = torch.stack(
                [
                    torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                    for frame, dy, dx in zip(batch, y_shift, x_shift, strict=True)
                ],
                dim=0,
            )
            registered[start:stop] = batch.cpu()
            all_y_shifts.append(y_shift)
            all_x_shifts.append(x_shift)
            all_correlations.append(correlation)

        y_shifts = torch.cat(all_y_shifts)
        x_shifts = torch.cat(all_x_shifts)
        correlations = torch.cat(all_correlations)
        n_average = max(2, int(frame_array.shape[0] * (1 + iteration) / 16))
        selected = torch.argsort(correlations, descending=True)[:n_average].cpu()
        reference_tensor = registered[selected].double().mean(dim=0)

        shift_dtype = torch.float32 if device.type == "mps" else torch.float64
        selected_shifts = selected.to(y_shifts.device)
        dy = int(-torch.round(y_shifts[selected_shifts].to(shift_dtype).mean()).item())
        dx = int(-torch.round(x_shifts[selected_shifts].to(shift_dtype).mean()).item())
        reference = (
            torch.roll(reference_tensor, shifts=(-dy, -dx), dims=(0, 1)).numpy().astype(np.int16)
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    return reference


def compute_zpos(
    zstack: npt.NDArray[Any],
    database: Mapping[str, Any],
    settings: Mapping[str, Any],
    reg_file: str | None = None,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> npt.NDArray[np.float32]:
    """Correlate each registered frame with each reference z-stack plane.

    Parameters
    ----------
    zstack : numpy.ndarray
        Reference stack in ``(planes, y, x)`` order.
    database : Mapping[str, Any]
        Movie dimensions, frame count, and default registered binary path.
    settings : Mapping[str, Any]
        Suite2p correlation and batching settings.
    reg_file : str or None, optional
        Registered binary path overriding the database value.
    device : torch.device, optional
        Device used for registration calculations.

    Returns
    -------
    numpy.ndarray
        Float32 correlations in ``(planes, frames)`` order.

    Raises
    ------
    ValueError
        If stack dimensions, frame count, or batch size are invalid.
    OSError
        If no registered binary path is available.
    """
    stack = np.asarray(zstack)
    height = int(database["Ly"])
    width = int(database["Lx"])
    n_frames = int(database["nframes"])
    if stack.ndim != 3 or stack.shape[1:] != (height, width) or stack.shape[0] < 1:
        raise ValueError("zstack must have shape (n_planes, database['Ly'], database['Lx']).")
    if n_frames < 1:
        raise ValueError("database['nframes'] must be at least 1.")
    movie_path = reg_file if reg_file is not None else database.get("reg_file")
    if not movie_path:
        raise OSError("No registered binary movie was specified.")

    references = register.compute_filters_and_norm(
        refImg=[stack[index] for index in range(stack.shape[0])],
        norm_frames=settings["norm_frames"],
        spatial_smooth=settings["smooth_sigma"],
        spatial_taper=settings["spatial_taper"],
        block_size=None,
        device=device,
    )
    batch_size = int(settings["batch_size"])
    if batch_size < 1:
        raise ValueError("settings['batch_size'] must be at least 1.")
    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info("Correlating %d frames in %d batches to z-stack planes", n_frames, n_batches)

    correlations = np.zeros((stack.shape[0], n_frames), dtype=np.float32)
    with contextlib.ExitStack() as resources:
        binary = resources.enter_context(
            io.BinaryFile(
                Ly=height,
                Lx=width,
                filename=movie_path,
                n_frames=n_frames,
                write=False,
            )
        )
        for batch in range(n_batches):
            start = batch * batch_size
            stop = min((batch + 1) * batch_size, n_frames)
            frames = binary[start:stop].copy()
            frame_tensor = torch.from_numpy(frames)
            if device.type == "cuda":
                frame_tensor = frame_tensor.pin_memory()
            frame_tensor = frame_tensor.to(device)
            outputs = register.compute_shifts(
                references,
                frame_tensor,
                maxregshift=settings["maxregshift"],
                smooth_sigma_time=settings["smooth_sigma_time"],
                nZ=len(references),
            )
            correlations[:, start:stop] = outputs[-1].T
    return correlations


def shift_frames(
    frames: torch.Tensor,
    yoff: Sequence[Any] | torch.Tensor,
    xoff: Sequence[Any] | torch.Tensor,
    yoff1: npt.NDArray[Any] | torch.Tensor | None = None,
    xoff1: npt.NDArray[Any] | torch.Tensor | None = None,
    blocks: Sequence[Any] | None = None,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> npt.NDArray[Any]:
    """Apply Suite2p rigid and optional nonrigid frame shifts.

    A local interpolation path handles nonrigid grids with one row or column.

    Parameters
    ----------
    frames : torch.Tensor
        Movie tensor in ``(frames, y, x)`` order.
    yoff, xoff : sequence or torch.Tensor
        Per-frame rigid offsets in pixels.
    yoff1, xoff1 : numpy.ndarray or torch.Tensor or None, optional
        Per-frame nonrigid block offsets.
    blocks : sequence or None, optional
        Suite2p block-grid metadata.
    device : torch.device, optional
        Requested device for offset arrays.

    Returns
    -------
    numpy.ndarray
        Shifted movie in ``(frames, y, x)`` order.

    Raises
    ------
    ValueError
        If nonrigid y offsets are supplied without x offsets.
    """
    if yoff1 is None or blocks is None or len(blocks) < 3 or 1 not in blocks[2]:
        shifted = register.shift_frames(
            frames,
            yoff=yoff,
            xoff=xoff,
            yoff1=yoff1,
            xoff1=xoff1,
            blocks=blocks,
            device=device,
        )
        return shifted[np.newaxis, ...] if shifted.ndim == 2 else shifted
    if xoff1 is None:
        raise ValueError("xoff1 is required when yoff1 is provided.")

    shifted = torch.stack(
        [
            torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
            for frame, dy, dx in zip(frames, yoff, xoff, strict=True)
        ],
        dim=0,
    )
    y_nonrigid = _offset_tensor(yoff1, shifted.device, device)
    x_nonrigid = _offset_tensor(xoff1, shifted.device, device)
    transformed = _transform_data_single_block(
        shifted, blocks[2], blocks[1], blocks[0], y_nonrigid, x_nonrigid
    )
    return transformed.cpu().numpy()


def _offset_tensor(
    offsets: npt.NDArray[Any] | torch.Tensor,
    frame_device: torch.device,
    requested_device: torch.device,
) -> torch.Tensor:
    """Move nonrigid offsets to the frame tensor's device.

    Parameters
    ----------
    offsets : numpy.ndarray or torch.Tensor
        Per-frame block offsets.
    frame_device : torch.device
        Device on which frames are stored.
    requested_device : torch.device
        Requested processing device; MPS uses float32 offsets.

    Returns
    -------
    torch.Tensor
        Offset tensor on ``frame_device``.
    """
    if isinstance(offsets, torch.Tensor):
        return offsets.to(frame_device)
    tensor = torch.from_numpy(np.asarray(offsets))
    if frame_device.type == "cuda":
        tensor = tensor.pin_memory()
    if requested_device.type == "mps":
        tensor = tensor.to(torch.float32)
    return tensor.to(frame_device)


def _transform_data_single_block(
    data: torch.Tensor,
    nblocks: Sequence[int],
    xblock: Sequence[npt.NDArray[Any]],
    yblock: Sequence[npt.NDArray[Any]],
    ymax1: torch.Tensor,
    xmax1: torch.Tensor,
) -> torch.Tensor:
    """Interpolate nonrigid shifts for a single-row or single-column grid.

    Parameters
    ----------
    data : torch.Tensor
        Rigidly shifted frames in ``(frames, y, x)`` order.
    nblocks : sequence[int]
        Number of blocks along each image axis.
    xblock, yblock : sequence[numpy.ndarray]
        Pixel coordinates of Suite2p registration blocks.
    ymax1, xmax1 : torch.Tensor
        Per-frame nonrigid block offsets in pixels.

    Returns
    -------
    torch.Tensor
        Interpolated and resampled ``int16`` frames.
    """
    _, height, width = data.shape
    device = data.device
    ymax1 = ymax1.reshape(-1, *nblocks)
    xmax1 = xmax1.reshape(-1, *nblocks)
    mesh_y, mesh_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    y_centers = np.asarray(yblock[:: nblocks[1]]).mean(axis=1).astype(int)
    x_centers = np.asarray(xblock[: nblocks[1]]).mean(axis=1).astype(int)
    interpolation_height = max(1, int(y_centers.max() - y_centers.min()))
    interpolation_width = max(1, int(x_centers.max() - x_centers.min()))
    displacement = torch_f.interpolate(
        torch.stack((ymax1, xmax1), dim=1),
        size=(interpolation_height, interpolation_width),
        mode="bilinear",
        align_corners=True,
    )
    pad_top = max(0, int(y_centers.min()))
    pad_bottom = max(0, height - interpolation_height - pad_top)
    pad_left = max(0, int(x_centers.min()))
    pad_right = max(0, width - interpolation_width - pad_left)
    displacement = torch_f.pad(
        displacement, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
    )

    displacement[:, 0] += mesh_y
    displacement[:, 1] += mesh_x
    displacement /= torch.tensor([height - 1, width - 1], device=device).view(1, 2, 1, 1)
    displacement = displacement.mul(2).sub(1).permute(0, 2, 3, 1)

    if device.type == "mps":
        padded = torch_f.pad(data.float().unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        scale = torch.tensor(
            [(width - 1) / (width + 1), (height - 1) / (height + 1)], device=device
        )
        grid = torch.clamp(displacement * scale.view(1, 1, 1, 2), -1, 1)
        result = torch_f.grid_sample(
            padded,
            grid[:, :, :, [1, 0]],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    else:
        result = torch_f.grid_sample(
            data.float().unsqueeze(1),
            displacement[:, :, :, [1, 0]],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return result.squeeze(1).short()
