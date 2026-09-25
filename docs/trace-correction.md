# Trace production in the two routes

`correct-traces` has two modes, but they do not represent two additional motion-correction
algorithms. The meaning of the command depends on `use_zstack`:

| Configuration | When it is valid | What it does |
| --- | --- | --- |
| `zstack.yaml`, `use_zstack: true` | After conventional Suite2p has produced per-plane registered movies and extracted traces | Implements Algorithm 1 by estimating z-motion from the reference stack, correcting ROI/neuropil traces, then performing neuropil correction, baseline estimation, and dF/F |
| `trace_correction.yaml`, `use_zstack: false` | Only after Algorithm 2 has produced and, if needed, curated `suite2p/plane_z` | Performs neuropil correction, baseline estimation, and dF/F; it does not correct z-motion |

## Within Algorithm 1: z-stack correction

For this mode and its required preceding `run-suite2p` stage, follow the complete
[z-stack-based trace-correction guide](z-stack-correction.md). Keep `use_zstack: true`; changing it
to false would skip the z-stack alignment and axial-profile correction entirely.

## After Algorithm 2: non-z-stack post-processing

Run this only after [multi-plane z-registration](z-registration.md):

```shell
suite2p-in-depth validate --workflow correct-traces --config analysis-config/trace_correction.yaml
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml --dry-run
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml
```

The input root is `directories.suite2p`, with `/suite2p` appended by the workflow. It should hold
the z-registration output, including `plane_z`; accepted ROIs come from its `stat.npy` and
`iscell.npy`. This step standardizes trace outputs from already z-registered data and must keep
`use_zstack: false`.

## Settings and quality control shared by both modes

`delta_F.absolute_zero` must reflect the acquisition's PMT dark offset. It is subtracted from ROI
and neuropil fluorescence before the later correction steps. Use
[`estimate-dark-level`](helper-utilities.md#estimate-dark-level) as a dataset-level QC estimate,
not as an automatic calibration. F0 is a rolling percentile using `F0_percentile` and `F0_window`,
and saved arrays use time-major orientation.

When `delta_F.plot` is true, `delta_F.max_roi_plots` limits the expensive individual-ROI trace
figures independently for each accepted plane. Set it to a non-negative integer, or to `all` to
plot every accepted ROI. If fewer ROIs are available than an integer cap, all are plotted; a
smaller cap selects deterministic, evenly distributed ROI indices. Set
`delta_F.plot_zoomed_traces` to true to create the additional fixed-window `_zoomed.png` figure
for the same selected ROIs when the recording is long enough. These settings do not limit shared
per-plane QC figures or change any saved numerical array.

Use [`plot-neuropil-masks`](helper-utilities.md#plot-neuropil-masks) before processing to inspect
the accepted ROI/neuropil pixels that generate `F.npy` and `Fneu.npy`. If piezo data supplies ROI
depth or stack reslicing, use [`plot-piezo`](helper-utilities.md#plot-piezo) to verify its shape and
units. These PNGs are human QC outputs; neither is read back into the workflow.

When `pixels_to_microns` is true, ROI y coordinates are scaled by image height (`Ly`) and x
coordinates by width (`Lx`) using the field size selected from `zoom_to_size`. This rectangular
image correction was introduced in Action 2 and can change coordinates relative to older runs.

Do not pool recordings with different dark levels without a reviewed calibration. Inspect the
per-ROI diagnostic plots and NaN warnings before accepting results.
