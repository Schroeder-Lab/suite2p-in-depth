# Algorithm 1: z-stack-based trace correction

Use this algorithm for somata or other larger structures when a separate dense reference z-stack
covers the full expected axial-motion range. Unlike
[multi-plane z-registration](z-registration.md), this route does not combine the acquired imaging
planes into a new movie. It first extracts ROIs and traces with conventional Suite2p, then aligns
registered functional frames to the z-stack and corrects each ROI's extracted fluorescence and
neuropil traces using its axial intensity profile.

The two required processing stages are:

1. `run-suite2p` with `plain_suite2p.yaml` to produce registered movies, ROIs, `F.npy`, and
   `Fneu.npy`.
2. `correct-traces` with `zstack.yaml` and `use_zstack: true` to estimate the axial trajectory and
   correct the extracted traces.

## 1. Create a working configuration directory

From the directory where the private configuration should live, run:

```shell
suite2p-in-depth init analysis-config
```

This creates `analysis-config/` and copies eight editable, sanitized templates into it: four
workflow YAML files, two dataset CSVs, `planes_to_flyback.csv`, and `zoom_to_microns.csv`. It does
not inspect or process imaging data, and all placeholder paths and metadata must be replaced.
The command refuses to write into an existing non-empty directory, so run it once for a new
configuration location.

## 2. Configure the functional recording and reference stack

Edit `datasets_zstack.csv`, `plain_suite2p.yaml`, and `zstack.yaml`. The acquisition columns
`Planes`, `Channels`, and pooled `Frame_rate` are used by the Suite2p prerequisite just as they are
for Algorithm 2. If they are uncertain, use
[`inspect-imaging`](helper-utilities.md#inspect-imaging) while preparing the dataset CSV. Use
[`plot-piezo`](helper-utilities.md#plot-piezo) to inspect per-plane piezo depth across each scanned
frame when the acquisition is sloped.

The z-stack directory must contain exactly one `.tif` or `.tiff`, interpreted in `[z, y, x]`
order. Rectangular images and non-unit slice spacing are supported. Set:

- `Zstack_folder` to the directory substituted into `{Zstack}`;
- `Depth` to the upper acquired plane's depth relative to the chosen reference;
- `Ignore_planes` to any additional zero-based planes that should be excluded; and
- `z_correction.spacing` to the distance between consecutive stack slices in micrometres.

Provide the configured DAQ data for a sloped multi-plane acquisition. The per-plane piezo trace is
used to reslice the stack to the row-dependent acquisition depth and to locate each ROI in z. If
horizontal planes are valid, the supported no-piezo identity path can be used.

## 3. Run conventional Suite2p and curate ROIs

```shell
suite2p-in-depth validate --workflow run-suite2p --config analysis-config/plain_suite2p.yaml
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml --dry-run
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml
```

Review Suite2p registration, masks, and traces, and curate `stat.npy`/`iscell.npy` if needed. The
[`plot-neuropil-masks`](helper-utilities.md#plot-neuropil-masks) PNG shows which accepted ROI and
neuropil pixels will feed trace extraction and neuropil regression. If registered `data.bin`
files were removed, use [`recreate-bin-files`](helper-utilities.md#recreate-bin-files) before the
next stage; z-stack frame-to-slice correlation requires the registered movie.

This conventional Suite2p pass is a prerequisite only for Algorithm 1. It is not required before
`zregister`.

## 4. Calibrate the PMT dark level

`delta_F.absolute_zero` is subtracted from the extracted ROI and neuropil fluorescence before
axial-profile correction, neuropil regression, baseline estimation, and dF/F. Use the helper as a
starting estimate:

```shell
suite2p-in-depth estimate-dark-level --config analysis-config/zstack.yaml --dry-run
suite2p-in-depth estimate-dark-level --config analysis-config/zstack.yaml
```

The reported `Dark value` is the lowest per-plane estimate for each dataset, based on the first
200 registered frames. It is not written into the configuration. Compare it with the acquisition
calibration, then enter the reviewed value in `zstack.yaml`; do not pool recordings with
incompatible dark levels.

## 5. Correct the extracted traces

Keep `use_zstack: true`, then run:

```shell
suite2p-in-depth validate --workflow correct-traces --config analysis-config/zstack.yaml
suite2p-in-depth correct-traces --config analysis-config/zstack.yaml --dry-run
suite2p-in-depth correct-traces --config analysis-config/zstack.yaml
```

For each valid plane, the workflow registers and reslices the stack, correlates registered movie
frames with stack slices, estimates a z trace, builds per-ROI axial profiles, corrects ROI and
neuropil fluorescence for axial movement, fits neuropil regression, computes baseline F0, and
calculates dF/F. A malformed or constant neuropil trace affects only that ROI: its derived values
are NaN and a warning is logged. The workflow fails if no valid plane or accepted ROI remains.

Review the matching-slice/reference overlay, per-plane z traces, per-ROI profiles, corrected
traces, and finite-value rates. `2pPlanes.zTraces.npy` is the estimated plane-wise axial motion;
`2pRois.zProfiles.npy` contains the axial profiles used to correct the traces; and
`2pCalcium.dff.npy` is the final time-by-ROI output. Full shapes and optional intermediate arrays
are in [Outputs](outputs.md#algorithm-1-z-stack-based-trace-correction). The
[Z-stack plot guide](z-stack-plot-guide.md) walks through example figures and what to inspect.

Use `delta_F.max_roi_plots` and `delta_F.plot_zoomed_traces` to limit individual-ROI figure cost
as described in [Trace production](trace-correction.md#settings-and-quality-control-shared-by-both-modes).
