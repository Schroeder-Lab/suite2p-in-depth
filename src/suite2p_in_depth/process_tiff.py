import logging
import os
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import scipy as sp
import skimage as skimage
import skimage.io as skimage_io
import torch
from numpy import ndarray
from scipy.signal import savgol_filter
from suite2p.extraction import extract, masks
from suite2p.registration import bidiphase, nonrigid, register

from . import suite2p_compat, suite2p_extensions

logger = logging.getLogger("suite2p")
DEFAULT_TORCH_DEVICE = torch.device("cuda")


def reslice_zstack(stack: np.ndarray, piezo: np.ndarray | None, spacing: float = 1) -> np.ndarray:
    """
    Reslice a z-stack to match the within-frame piezo trajectory.

    Raster-scanned movie frames can change depth along y; this operation
    applies that slant to the registered stack.

    Parameters
    ----------
    stack : numpy.ndarray
        Registered image stack in ``(z, y, x)`` order.
    piezo : array [samples] or [samples, 1], optional
        Piezo depth sampled while scanning one imaging plane. If ``None``,
        return an unchanged copy of the stack.
    spacing : float, optional
        Distance between adjacent z-stack slices, in the same units as
        ``piezo``.

    Returns
    -------
    numpy.ndarray
        Resliced stack in ``(z, y, x)`` order.
    """
    # Validate the z-stack geometry first. A missing piezo trace represents
    # horizontal imaging planes, for which reslicing is an identity operation.
    # Convert array-like inputs to NumPy without copying an existing array.
    stack = np.asarray(stack)
    # The interpolation below requires one axis each for depth, rows, and columns.
    if stack.ndim != 3:
        raise ValueError("stack must have shape (z, y, x).")
    # Reject empty stacks before constructing interpolation coordinates.
    if any(size < 1 for size in stack.shape):
        raise ValueError("stack dimensions must all be positive.")
    # Spacing is used as a divisor when converting microns to slice indices.
    if spacing <= 0:
        raise ValueError("spacing must be greater than zero.")
    # Without a piezo slope, every row belongs to the same depth. Return a copy
    # so callers can safely modify the result without modifying their input.
    if piezo is None:
        return stack.copy()

    # Normalize the accepted piezo input forms to one finite vector. Reslicing
    # needs at least two samples to estimate how depth changes across scan lines.
    # Use floating point because conversion to fractional slice indices follows.
    piezo_array = np.asarray(piezo, dtype=float)
    # Accept the common column-vector representation [samples, 1].
    if piezo_array.ndim == 2:
        # Multiple columns would ambiguously describe more than one imaging plane.
        if piezo_array.shape[1] != 1:
            raise ValueError("piezo must be one-dimensional or have shape (samples, 1).")
        # Remove the singleton plane axis, leaving only the time/sample axis.
        piezo_array = piezo_array[:, 0]
    # Interpolation needs a one-dimensional trace and at least two support points.
    if piezo_array.ndim != 1 or piezo_array.size < 2:
        raise ValueError("piezo must contain at least two samples.")
    # NaN or infinite positions would propagate into invalid interpolation points.
    if not np.isfinite(piezo_array).all():
        raise ValueError("piezo must contain only finite values.")

    # Express piezo motion in z-stack slice units, using the top-most piezo
    # position as zero. Interpolate those samples onto image rows, then center
    # the offsets on the middle row so that it stays at the original depth.
    # Unpack the canonical stack shape for readability below.
    planes, height, width = stack.shape
    # Subtract the shallowest position and divide by the physical distance
    # between slices; values can now be used directly as fractional z indices.
    piezo_slices = (piezo_array - np.min(piezo_array)) / spacing
    # A one-row image has no y-axis slope, so its only offset must be zero.
    if height == 1:
        line_offsets = np.zeros(1)
    else:
        # The piezo samples span the acquisition of one complete frame. Map
        # their positions uniformly from the first to the last image row.
        sample_y = np.linspace(0, height - 1, piezo_slices.size)
        # Estimate one depth offset for every image row.
        line_offsets = np.interp(np.arange(height), sample_y, piezo_slices)
    # Define the middle row as the reference depth. Rows scanned earlier or
    # later are shifted relative to it, rather than shifting the whole stack.
    line_offsets -= line_offsets[height // 2]

    # Build an interpolator in the stack's canonical [z, y, x] coordinate
    # system and a fixed grid containing every pixel in one output plane.
    # These arrays contain the valid coordinate values along each source axis.
    z_axis = np.arange(planes)
    y_axis = np.arange(height)
    x_axis = np.arange(width)
    # Nearest-neighbor sampling preserves measured pixel values and the
    # established behavior of selecting the closest physical z-stack slice.
    interpolator = sp.interpolate.RegularGridInterpolator(
        (z_axis, y_axis, x_axis), stack, bounds_error=False, fill_value=None, method="nearest"
    )
    # Create the y/x coordinates shared by every resliced output plane.
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")

    # For each output plane, shift its requested z coordinate by the row-wise
    # piezo offset. Clip at the stack boundaries and sample the nearest source
    # voxel while preserving the original [z, y, x] output shape and dtype.
    # Preallocate rather than append so shape and dtype match the input stack.
    stack_resliced = np.empty_like(stack)
    for plane in range(planes):
        # Add the row-dependent offset to this plane's base z coordinate.
        # Clipping repeats the nearest edge slice when the requested depth lies
        # outside the acquired z-stack volume.
        z_grid = np.clip(plane + line_offsets[:, np.newaxis], 0, planes - 1)
        # The piezo position changes by scan row, not by x pixel, so repeat each
        # row's z coordinate across the full image width.
        z_grid = np.broadcast_to(z_grid, (height, width))
        # RegularGridInterpolator expects one [z, y, x] coordinate per row.
        # Flatten each coordinate grid and combine them into that point list.
        points = np.stack((z_grid.ravel(), y_grid.ravel(), x_grid.ravel()), axis=1)
        # Sample source voxels, restore the 2D image shape, and store this plane.
        stack_resliced[plane] = interpolator(points).reshape(height, width)
    # Return a new stack; neither the input stack nor piezo trace is modified.
    return cast(ndarray, stack_resliced)


def register_zstack_across_planes(
    zstack: ndarray, settings_reg: dict[str, Any], device: torch.device
) -> tuple[ndarray, ndarray, ndarray, ndarray | None, ndarray | None, Any]:
    """
    Align neighboring z-stack planes through successive registration passes.

    Passes run from the middle toward both ends and then from top to bottom.

    Parameters
    ----------
    zstack : numpy.ndarray
        Stack in ``(planes, y, x)`` order.
    settings_reg : dict
        Suite2p registration settings.
    device : torch.device
        Device for registration calculations.

    Returns
    -------
    tuple
        Locally registered stack, rigid y and x offsets, nonrigid y and x
        offsets, and block-grid metadata.

    """
    # Align each slice to its predecessor, then accumulate the pairwise shifts.
    y_off = np.zeros(zstack.shape[0])
    x_off = np.zeros(zstack.shape[0])
    for i in range(zstack.shape[0] - 1):
        registration = register.register_frames(
            f_align_in=np.expand_dims(zstack[i + 1], axis=0).astype(np.float32),
            refImg=zstack[i],
            bidiphase=0,
            norm_frames=settings_reg["norm_frames"],
            smooth_sigma=settings_reg["smooth_sigma"],
            spatial_taper=settings_reg["spatial_taper"],
            nonrigid=False,
            maxregshift=settings_reg["maxregshift"],
            smooth_sigma_time=settings_reg["smooth_sigma_time"],
            device=device,
            upsample_meanImg=False,
        )
        offsets = registration[3]
        y_off[i + 1] = offsets[0][0]
        x_off[i + 1] = offsets[1][0]

    # Add up shifts consecutively to align the whole z stack.
    y_off = np.cumsum(y_off, axis=0)
    x_off = np.cumsum(x_off, axis=0)

    # Minimize total shifts by subtracting the median in each direction.
    y_off = y_off - np.median(y_off)
    x_off = x_off - np.median(x_off)

    # Apply the shifts to the z stack.
    if device.type == "cuda":
        zstack_torch = torch.from_numpy(zstack.astype(np.float32)).pin_memory().to(device)
    else:
        zstack_torch = torch.from_numpy(zstack.astype(np.float32)).to(device)

    zstack = suite2p_extensions.shift_frames(
        zstack_torch,
        yoff=y_off.astype(int),
        xoff=x_off.astype(int),
        yoff1=None,
        xoff1=None,
        device=device,
    )

    # Apply non-rigid registration to further align planes to each other.
    if settings_reg["nonrigid"] and settings_reg["maxregshiftNR"] > 0:
        y_off1 = [None]
        x_off1 = [None]
        zstack_nonrigid = zstack.copy()
        if device.type == "cuda":
            z_torch = torch.from_numpy(zstack).pin_memory().to(device)
        else:
            z_torch = torch.from_numpy(zstack).to(device)

        yoff = torch.zeros((1,), dtype=torch.long, device=device)
        xoff = torch.zeros((1,), dtype=torch.long, device=device)
        for i in range(1, zstack.shape[0] - 1):
            ref_img = (zstack[i - 1] + zstack[i + 1]) / 2
            ref_and_masks = register.compute_filters_and_norm(
                ref_img,
                norm_frames=settings_reg["norm_frames"],
                spatial_smooth=settings_reg["smooth_sigma"],
                spatial_taper=settings_reg["spatial_taper"],
                block_size=settings_reg["block_size"],
                device=device,
            )
            (mask_mul_nr, mask_offset_nr, cf_ref_img_nr, blocks, rmin, rmax) = ref_and_masks[3:]
            ymax1, xmax1 = nonrigid.phasecorr(
                z_torch[i].unsqueeze(0),
                blocks,
                mask_mul_nr,
                mask_offset_nr,
                cf_ref_img_nr,
                settings_reg["snr_thresh"],
                settings_reg["maxregshiftNR"],
            )[:2]
            zstack_nonrigid[i] = suite2p_extensions.shift_frames(
                z_torch[i].unsqueeze(0),
                yoff=yoff,
                xoff=xoff,
                yoff1=ymax1,
                xoff1=xmax1,
                blocks=blocks,
                device=device,
            )[0]
            y_off1.append(ymax1.cpu().numpy())
            x_off1.append(xmax1.cpu().numpy())

        # Treat top and bottom at the end since they only have one neighbour.
        ref_and_masks = register.compute_filters_and_norm(
            zstack_nonrigid[1],
            norm_frames=settings_reg["norm_frames"],
            spatial_smooth=settings_reg["smooth_sigma"],
            spatial_taper=settings_reg["spatial_taper"],
            block_size=settings_reg["block_size"],
            device=device,
        )
        (mask_mul_nr, mask_offset_nr, cf_ref_img_nr, blocks, rmin, rmax) = ref_and_masks[3:]
        ymax1, xmax1 = nonrigid.phasecorr(
            z_torch[0].unsqueeze(0),
            blocks,
            mask_mul_nr,
            mask_offset_nr,
            cf_ref_img_nr,
            settings_reg["snr_thresh"],
            settings_reg["maxregshiftNR"],
        )[:2]
        zstack_nonrigid[0] = suite2p_extensions.shift_frames(
            z_torch[0].unsqueeze(0),
            yoff=yoff,
            xoff=xoff,
            yoff1=ymax1,
            xoff1=xmax1,
            blocks=blocks,
            device=device,
        )[0]
        y_off1[0] = ymax1.cpu().numpy()
        x_off1[0] = xmax1.cpu().numpy()

        ref_and_masks = register.compute_filters_and_norm(
            zstack_nonrigid[-2],
            norm_frames=settings_reg["norm_frames"],
            spatial_smooth=settings_reg["smooth_sigma"],
            spatial_taper=settings_reg["spatial_taper"],
            block_size=settings_reg["block_size"],
            device=device,
        )
        (mask_mul_nr, mask_offset_nr, cf_ref_img_nr, blocks, rmin, rmax) = ref_and_masks[3:]
        ymax1, xmax1, cmax1 = nonrigid.phasecorr(
            z_torch[-1].unsqueeze(0),
            blocks,
            mask_mul_nr,
            mask_offset_nr,
            cf_ref_img_nr,
            settings_reg["snr_thresh"],
            settings_reg["maxregshiftNR"],
        )[:3]
        zstack_nonrigid[-1] = suite2p_extensions.shift_frames(
            z_torch[-1].unsqueeze(0),
            yoff=yoff,
            xoff=xoff,
            yoff1=ymax1,
            xoff1=xmax1,
            blocks=blocks,
            device=device,
        )[0]
        y_off1.append(ymax1.cpu().numpy())
        x_off1.append(xmax1.cpu().numpy())

        zstack = zstack_nonrigid
        y_off1 = np.concatenate(y_off1)
        x_off1 = np.concatenate(x_off1)
    else:
        y_off1 = None
        x_off1 = None
        blocks = None

    return zstack, y_off, x_off, y_off1, x_off1, blocks


def register_zstack_to_ref(
    zstack: ndarray,
    reference_image: ndarray,
    settings_reg: dict[str, Any],
    zstack_other: ndarray | None = None,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> tuple[ndarray, np.integer[Any], ndarray | None]:
    """
    Register a z-stack to a movie reference image.

    Parameters
    ----------
    zstack : numpy.ndarray
        Stack in ``(z, y, x)`` order.
    reference_image : numpy.ndarray
        Suite2p movie reference in ``(y, x)`` order.
    settings_reg : dict
        Suite2p registration settings.
    zstack_other : numpy.ndarray or None, optional
        Second-channel stack to transform with the same shifts.
    device : torch.device, optional
        Device for registration calculations.

    Returns
    -------
    tuple[numpy.ndarray, int, numpy.ndarray or None]
        Registered stack, best-matching slice index, and transformed
        second-channel stack when supplied.

    """
    # Align all slices in the Z stack to the reference image using non-rigid registration.
    registration = register.register_frames(
        zstack.astype(np.float32),
        reference_image,
        bidiphase=0,
        nonrigid=True,
        block_size=settings_reg["block_size"],
        norm_frames=settings_reg["norm_frames"],
        smooth_sigma=settings_reg["smooth_sigma"],
        spatial_taper=settings_reg["spatial_taper"],
        maxregshift=settings_reg["maxregshift"],
        smooth_sigma_time=settings_reg["smooth_sigma_time"],
        snr_thresh=settings_reg["snr_thresh"],
        maxregshiftNR=settings_reg["maxregshiftNR"],
        device=device,
        apply_shifts=False,
    )
    offsets = registration[3]

    # Determine correlations between the reference image and aligned Z stack slices.
    correlations_rigid = offsets[2]

    # Find slice in zstack that matches best with the reference image.
    best_slice = np.argmax(correlations_rigid)

    # Apply non-rigid registration for best matching plane to all planes of Z stack.
    blocks = registration[4]
    if device.type == "cuda":
        zstack_torch = torch.from_numpy(zstack).pin_memory().to(device)
    else:
        zstack_torch = torch.from_numpy(zstack).to(device)

    zstack_registered = suite2p_extensions.shift_frames(
        zstack_torch,
        blocks=blocks,
        device=device,
        yoff=np.tile(offsets[0][best_slice], zstack.shape[0]),
        xoff=np.tile(offsets[1][best_slice], zstack.shape[0]),
        yoff1=np.tile(offsets[3][best_slice].reshape(1, -1), (zstack.shape[0], 1)),
        xoff1=np.tile(offsets[4][best_slice].reshape(1, -1), (zstack.shape[0], 1)),
    )

    if zstack_other is not None:
        if device.type == "cuda":
            zstack_other_torch = torch.from_numpy(zstack_other).pin_memory().to(device)
        else:
            zstack_other_torch = torch.from_numpy(zstack_other).to(device)
        zstack_other_registered = suite2p_extensions.shift_frames(
            zstack_other_torch,
            blocks=blocks,
            device=device,
            yoff=np.tile(offsets[0][best_slice], zstack.shape[0]),
            xoff=np.tile(offsets[1][best_slice], zstack.shape[0]),
            yoff1=np.tile(offsets[3][best_slice].reshape(1, -1), (zstack.shape[0], 1)),
            xoff1=np.tile(offsets[4][best_slice].reshape(1, -1), (zstack.shape[0], 1)),
        )
    else:
        zstack_other_registered = None

    return zstack_registered, best_slice, zstack_other_registered


def register_zstack(
    tiff_path: str | os.PathLike[str],
    settings_reg_within: dict[str, Any],
    settings_reg_across: dict[str, Any],
    settings_reg_ref: dict[str, Any],
    spacing: float = 1,
    piezo: ndarray | None = None,
    target_image: ndarray | None = None,
    channel_align: int = 1,
    channel_functional: int = 1,
    n_channels: int = 1,
    sigma: Sequence[float] | None = None,
    ref_frames: float = 0.5,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> tuple[ndarray, np.integer[Any], ndarray | None]:
    """
    Load, register, average, and optionally reslice a z-stack TIFF.

    Repeated images at each depth are aligned before averaging. When piezo
    positions are supplied, stack slices are resliced to match movie frames.

    Parameters
    ----------
    tiff_path : str
        Source z-stack TIFF with repeated images at each depth.
    settings_reg_within, settings_reg_across, settings_reg_ref : dict
        Registration settings within depths, across depths, and to the target.
    spacing : float, optional
        Axial distance between stack slices in micrometres.
    piezo : numpy.ndarray or None, optional
        Within-frame piezo trajectory in micrometres.
    target_image : numpy.ndarray or None, optional
        Movie reference image in ``(y, x)`` order. Required when registering
        the z-stack to the recording reference.
    channel_align, channel_functional : int, optional
        Channel numbers for alignment and fluorescence extraction.
    n_channels : int, optional
        Number of channels in the source TIFF.
    sigma : sequence[float] or None, optional
        Smoothing scale along z, y, and x.
    ref_frames : float, optional
        Fraction of stack repetitions used as reference frames.
    device : torch.device, optional
        Device for registration calculations.

    Returns
    -------
    tuple[numpy.ndarray, int, numpy.ndarray or None]
        Registered alignment-channel stack, best matching slice, and
        functional-channel stack when different.
    """
    # Load the z-stack and select the requested alignment channel.
    image = skimage_io.imread(tiff_path)
    if n_channels > 1:
        image_align = image[:, :, channel_align - 1, :, :]
    else:
        image_align = image

    planes = image_align.shape[0]
    repetitions = image_align.shape[1]
    if repetitions < 2:
        raise ValueError(f"Expected at least 2 repetitions per z-stack plane; found {repetitions}.")
    resolutiony = image_align.shape[2]
    resolutionx = image_align.shape[3]

    if settings_reg_within["do_bidiphase"]:
        stacked = image_align.reshape(-1, resolutiony, resolutionx)
        bidi_phase = bidiphase.compute(stacked)
        logger.info(f"  Estimated bidiphase offset in zstack: {bidi_phase} pixels")
        if bidi_phase != 0:
            bidiphase.shift(stacked, bidi_phase)
            image_align = stacked.reshape(planes, repetitions, resolutiony, resolutionx)

    zstack = np.zeros((planes, resolutiony, resolutionx))
    yoff_per_plane = np.zeros((planes, repetitions))
    xoff_per_plane = np.zeros((planes, repetitions))
    for i in range(planes):
        # Uses the suite2p registration function to align the repeated frames taken
        # per plane to the average across repetitions.
        ref = suite2p_extensions.compute_reference(
            image_align[i],
            settings=settings_reg_within,
            device=device,
            n_best=round(repetitions * ref_frames),
        )
        res = register.register_frames(
            image_align[i],
            refImg=ref,
            bidiphase=0,
            nonrigid=False,
            norm_frames=settings_reg_within["norm_frames"],
            smooth_sigma=settings_reg_within["smooth_sigma"],
            spatial_taper=settings_reg_within["spatial_taper"],
            maxregshift=settings_reg_within["maxregshift"],
            smooth_sigma_time=settings_reg_within["smooth_sigma_time"],
            snr_thresh=settings_reg_within["snr_thresh"],
            maxregshiftNR=settings_reg_within["maxregshiftNR"],
            device=device,
        )
        # Average registered repetitions into one image per physical plane.
        zstack[i, :, :] = res[2]
        # Saves the shifts in x and y for each plane.
        offsets = res[3]
        yoff_per_plane[i, :] = offsets[0]
        xoff_per_plane[i, :] = offsets[1]

    # Register the averaged z-stack planes to one another.
    zstack, yoff_across_planes, xoff_across_planes, yoff1, xoff1, blocks = (
        register_zstack_across_planes(zstack, settings_reg_across, device=device)
    )

    # If zstack of other channel needs to be registered, apply the same shifts
    zstack_other = None
    if channel_functional != channel_align:
        image_other = image[:, :, channel_functional - 1, :, :]

        # Apply bidiphase correction.
        if settings_reg_within["do_bidiphase"] and bidi_phase != 0:
            stacked = image_other.reshape(-1, resolutiony, resolutionx)
            bidiphase.shift(stacked, bidi_phase)
            image_other = stacked.reshape(planes, repetitions, resolutiony, resolutionx)

        # Register and average repeated frames per plane in Z stack.
        if device.type == "cuda":
            im_torch = torch.from_numpy(image_other).pin_memory().to(device)
        else:
            im_torch = torch.from_numpy(image_other).to(device)

        zstack_other = np.zeros((planes, resolutiony, resolutionx))
        for i in range(planes):
            repeated_frames = suite2p_extensions.shift_frames(
                im_torch[i],
                yoff=yoff_per_plane[i].astype(int),
                xoff=xoff_per_plane[i].astype(int),
                device=device,
            )
            zstack_other[i] = np.mean(repeated_frames, axis=0)

        # Register resulting planes.
        if device.type == "cuda":
            zstack_torch = torch.from_numpy(zstack_other).pin_memory().to(device)
        else:
            zstack_torch = torch.from_numpy(zstack_other).to(device)

        zstack_other = suite2p_extensions.shift_frames(
            zstack_torch,
            yoff=yoff_across_planes.astype(int),
            xoff=xoff_across_planes.astype(int),
            yoff1=yoff1,
            xoff1=xoff1,
            blocks=blocks,
            device=device,
        )

    # Smooth zstack with 3D Gaussian filter.
    if sigma is not None:
        zstack = smooth_zstack(sigma, zstack)
        if zstack_other is not None:
            zstack_other = smooth_zstack(sigma, zstack_other)

    # Reslice the stack to match the row-dependent piezo trajectory.
    zstack = reslice_zstack(zstack, piezo, spacing=spacing)
    if zstack_other is not None:
        zstack_other = reslice_zstack(zstack_other, piezo, spacing=spacing)

    # Register the resliced stack to the recording reference image.
    if target_image is None:
        raise ValueError("target_image is required to register the z-stack to a reference.")
    zstack, best_plane, zstack_other = register_zstack_to_ref(
        zstack, target_image, settings_reg_ref, zstack_other, device
    )
    return zstack, best_plane, zstack_other


def smooth_zstack(sigma: Sequence[float], zstack: ndarray) -> ndarray:
    """Smooth a registered stack along x, y, and z with Savitzky-Golay filters.

    Parameters
    ----------
    sigma : sequence[int]
        Filter window lengths for z, y, and x axes.
    zstack : numpy.ndarray
        Stack in ``(z, y, x)`` order.

    Returns
    -------
    numpy.ndarray
        Smoothed stack with the same shape.
    """
    zstack = savgol_filter(
        zstack, window_length=max(sigma[2], 3), polyorder=2, axis=2, mode="nearest"
    )
    zstack = savgol_filter(
        zstack, window_length=max(sigma[1], 3), polyorder=2, axis=1, mode="nearest"
    )
    zstack = savgol_filter(zstack, window_length=sigma[0], polyorder=3, axis=0, mode="nearest")
    return zstack


def extract_zprofiles(
    plane_date_path: str | os.PathLike[str],
    zstack: ndarray,
    abs_zero: float | None = None,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> ndarray:
    """
    Extract fluorescence profiles for accepted ROIs across z-stack depth.

    Parameters
    ----------
    plane_date_path : str
        Suite2p plane directory containing accepted ROI masks and metadata.
    zstack : numpy.ndarray
        Registered stack in ``(z, y, x)`` order.
    abs_zero : float or None, optional
        Value subtracted from each profile; ``None`` leaves values unchanged.
    device : torch.device, optional
        Device used for ROI mask extraction.

    Returns
    -------
    numpy.ndarray
        ROI fluorescence by depth with shape ``(z, ROIs)``.
    """
    # Load neural data.
    db, settings = suite2p_compat.load_parameters(plane_date_path)
    stats = np.load(os.path.join(plane_date_path, "stat.npy"), allow_pickle=True)
    is_cell = np.load(os.path.join(plane_date_path, "iscell.npy")).astype(bool)
    # Disregard ROIs which are not considered cells.
    stats = stats[is_cell[:, 0]]

    # Get masks for ROIs and neuropil.
    ly = zstack.shape[1]
    lx = zstack.shape[2]
    rois = [
        masks.create_cell_mask(
            stat, Ly=ly, Lx=lx, allow_overlap=settings["extraction"]["allow_overlap"]
        )
        for stat in stats
    ]
    npils = [np.empty((0,)) for _ in rois]

    # Get fluorescence profiles of ROI and neuropil masks across depth of Z stack.
    fluorescence_profiles = extract.extract_traces(zstack, rois, npils, device=device)[0]
    fluorescence_profiles = fluorescence_profiles.T  # (z-slices, ROIs)

    # Subtract absolute zero value of fluorescence signal.
    if abs_zero is not None:
        fluorescence_profiles -= abs_zero

    return cast(ndarray, fluorescence_profiles)
