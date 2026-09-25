# Output schemas

NumPy dictionary files are loaded with `np.load(path, allow_pickle=True).item()`. Array files are
loaded with `np.load(path)`. Treat files as a versioned scientific interface: preserve names,
keys, orientations, units, and NaN meanings when building downstream analyses.

## Algorithm 1: z-stack-based trace correction

Z-stack runs write `2pPlanes.zCorrelations.npy` as `(planes, z, time)`,
`2pPlanes.zTraces.npy` as `(planes, time)`, `2pRois.zProfiles.npy` as `(rois, z)`, and optionally
`2pCalcium.F_zcorrected.npy`/`2pCalcium.N_zcorrected.npy` as `(time, rois)`.

`2pPlanes.zCorrelations.npy` holds the slice-by-time matching evidence,
`2pPlanes.zTraces.npy` is the resulting estimated axial trajectory for each plane, and
`2pRois.zProfiles.npy` contains the ROI axial profiles used to normalize the extracted traces.
Z-trace entries are floating-point stack indices; `NaN` marks frames whose best match is at the
top or bottom of the evaluated stack range, where the true axial position is unresolved.
The workflow also writes the shared corrected-trace outputs below after z-motion correction.

## Shared corrected-trace outputs

Dataset-root outputs are:

| File | Shape/meaning |
| --- | --- |
| `2pCalcium.dff.npy` | `(time, rois)`, dF/F |
| `2pRois.xyz.npy` | `(rois, 3)`, x/y/z ROI locations; spatial values are micrometres when configured |
| `2pRois.ids.npy` | `(rois, 1)`, original per-plane ROI indices |
| `2pRois.2pPlanes.npy` | `(rois,)`, numeric Suite2p plane per ROI |
| `2pCalcium.F_ncorrected.npy` | `(time, rois)`, neuropil-corrected fluorescence; optional |
| `2pCalcium.N_regression.npy` | Per-ROI fitted neuropil regression parameters; optional |
| `2pCalcium.F_bin_values.npy` | Per-ROI binned fluorescence diagnostics; optional |
| `2pCalcium.N_bin_values.npy` | Per-ROI binned neuropil diagnostics; optional |

Both Algorithm 1 and Algorithm 2's optional trace post-processing write these files. In Algorithm
1 they are produced after z-stack-based correction; in Algorithm 2 they summarize traces
extracted from the already z-registered `plane_z`.

Per-plane provenance for both trace modes includes `db_correcting.npy`, `settings_correcting.npy`,
`detect_outputs_correcting.npy`, and, for z-stack correction, `zcorrect_outputs.npy`. QC plots
are under `2P_processed/`.

## Algorithm 2: z-registration

`plane_z/reg_outputs.npy` extends Suite2p registration outputs with:

| Key | Shape/meaning |
| --- | --- |
| `zpos_registration` | `(time,)`, estimated reference-plane position per frame |
| `cmax_registration` | Correlation values for frame/reference matching |
| `reference_plane` | Original numeric plane selected as registration reference |
| `planes_across_time` | `(time,)`, source plane used for each output frame |
| `corrs_time_refs_planes` | `(time, references, planes)`, correlations across valid planes |
| `refImg_singleplanes` | `(planes, y, x)` reference images |
| `xoff_singleplanes` | `(time, planes)` per-plane x offsets, pixels |
| `yoff_singleplanes` | `(time, planes)` per-plane y offsets, pixels |

`db.npy` includes `ignore_flyback_singleplanes`; `settings.npy` records the selected plane's own
settings. Plot outputs include `01_reference_images.png`, `02_mean_image_zregistration.png`, and
`03_z_positions.png`.

After ROI curation, the optional non-z-stack `correct-traces` stage reads `plane_z` and writes the
shared corrected-trace outputs above. It does not produce any of the z-stack-only arrays.

## Run manifest

Each selected dataset receives
`.suite2p-in-depth/<workflow>/run_manifest.json` beneath `directories.output`. Schema version 1 stores
the workflow, completion state (`running`, `complete`, or `failed`), update time, package and
Suite2p versions, Python/platform, device, resolved configuration and SHA-256 digest, dataset row,
resolved inputs, output path, and failure message. The file is replaced atomically where the
filesystem supports `os.replace` semantics.

Because resolved configuration and paths may be sensitive, keep manifests with the private
processed dataset and review them before sharing.
