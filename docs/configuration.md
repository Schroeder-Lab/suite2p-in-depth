# Configuration by algorithm

Run `suite2p-in-depth init analysis-config` to create the supported examples in a new working
directory. The command copies four workflow YAML files, two dataset CSVs, and two lookup CSVs; it
does not infer acquisition values or process data. Relative paths in a YAML file are resolved from
that YAML's directory. Windows drive and UNC paths remain Windows paths even when validation runs
on Linux.

Choose configuration files by scientific route, not by command name alone:

| Route and stage | YAML | Dataset CSV | `use_zstack` |
| --- | --- | --- | --- |
| Algorithm 1: conventional Suite2p prerequisite | `plain_suite2p.yaml` | `datasets_zstack.csv` | Not applicable |
| Algorithm 1: z-stack-based trace correction | `zstack.yaml` | `datasets_zstack.csv` | `true` |
| Algorithm 2: image z-registration | `zregistration.yaml` | `datasets_zregistration.csv` | Not applicable |
| Algorithm 2: optional trace post-processing after `plane_z` | `trace_correction.yaml` | `datasets_zregistration.csv` | `false` |

`run-suite2p` is a prerequisite for Algorithm 1 only. It is not a prerequisite for Algorithm 2,
whose `zregister` command runs its own Suite2p stages. Running `correct-traces` with
`use_zstack: false` on conventional Suite2p planes would omit z-stack correction and is not the
documented Algorithm 1 workflow.

## Acquisition values used by both algorithms

The Suite2p-producing stage of either route reads these dataset CSV values:

| CSV field | Meaning and downstream use |
| --- | --- |
| `Planes` | Total acquired planes, including flyback planes. Sets Suite2p `nplanes`, selects the flyback rule, and helps calculate the per-plane sampling rate. |
| `Channels` | Acquired imaging-channel count. Sets Suite2p `nchannels`; `suite2p_db.functional_chan` identifies the signal channel. |
| `Frame_rate` | Pooled frame rate across all planes in Hz. Selects the flyback rule; Suite2p's per-plane rate is `Frame_rate / Planes`. |

If these values are uncertain, use
[`inspect-imaging`](helper-utilities.md#inspect-imaging). Its printed `Planes`, `Channels`, and
`Frame_rate` columns are intended to be copied into the selected row before the real workflow is
validated. The current command accepts a z-registration-compatible YAML with a `daq` section, so
for Algorithm 1 use a working copy of `zregistration.yaml` pointed at `datasets_zstack.csv` and
the relevant TIFF directories. Because the helper currently runs standard z-registration
validation first, selected rows need provisional positive acquisition values matching one
flyback rule; replace them with the measured output and then validate again.

Use [`plot-piezo`](helper-utilities.md#plot-piezo) to visually check the resulting per-plane depth
trajectories and whether flyback assignments make physical sense. Its output is QC only and does
not change the CSV.

## Algorithm 1 configuration

`plain_suite2p.yaml` controls conventional Suite2p ROI detection and trace extraction. Its output
directory must match the `directories.suite2p` root used later by `zstack.yaml`.

In `zstack.yaml`, keep `use_zstack: true`. `z_correction.ref_frames` is a fraction in `(0, 1]`,
`spacing` is the micrometre distance between stack slices, and `sigma_trace` smooths correlation
across slices. `sigma_stack`, `reference`, `remove_outliers`, `stack_step`,
`save_traces_zcorrected`, and the `suite2p_stack_*` mappings control stack registration and saved
outputs. Preserve z-stack array order `[z, y, x]`.

The `delta_F.absolute_zero` value is subtracted from extracted ROI and neuropil signals and from
the stack-derived profiles before trace corrections. Run
[`estimate-dark-level`](helper-utilities.md#estimate-dark-level) after Suite2p has produced
registered binaries, compare its `Dark value` with the PMT calibration, and enter the reviewed
value manually.

## Algorithm 2 configuration

`zregistration.yaml` controls both the per-plane Suite2p stages and construction of `plane_z`.
Small-structure choices such as `suite2p_settings.diameter`, detection thresholds, registration
channel, and flyback rules therefore belong here. `zregister` writes the Suite2p outputs itself;
do not point it at the output of `plain_suite2p.yaml` as an input stage.

After z-registration, `trace_correction.yaml` reads the resulting Suite2p root and keeps
`use_zstack: false`. Set `delta_F.absolute_zero` before this step using the reviewed acquisition
calibration and [`estimate-dark-level`](helper-utilities.md#estimate-dark-level) as a QC estimate.

## Shared YAML keys

| Key | Meaning |
| --- | --- |
| `datasets` | CSV path for dataset selection and metadata |
| `directories.tiffs` | Raw TIFF directory template |
| `directories.suite2p` | Processed recording root; `correct-traces` appends `suite2p` |
| `directories.output` | Dataset output root and manifest location |
| `planes_to_flyback` | CSV mapping plane count/frame rate to flyback indices |
| `suite2p_settings.torch_device` | `cpu`, `cuda`, or `mps`; only CPU/CUDA are supported release modes |
| `suite2p_settings.diameter` | Expected ROI diameter in pixels, greater than zero |
| `suite2p_settings.run` | Suite2p registration/detection/deconvolution booleans |
| `suite2p_settings.registration.align_by_chan2` | Use channel 2 for registration; the packaged configs default to `false` |
| `suite2p_db.functional_chan` | One-based functional channel index |
| `use_zstack` | Select z-stack correction in `correct-traces`; false is reserved for Algorithm 2 post-processing |
| `pixels_to_microns` | Convert ROI coordinates using `zoom_to_size` |
| `zoom_to_size` | CSV mapping acquisition zoom to field-of-view size |
| `delta_F.absolute_zero` | PMT dark offset in raw intensity units; use `estimate-dark-level` as a QC estimate |
| `delta_F.F0_percentile` | Baseline percentile from 0 through 100 |
| `delta_F.F0_window` | Baseline window in seconds, non-negative |
| `delta_F.save_F_neuropilcorrected` | Save corrected fluorescence and regression diagnostics |
| `delta_F.plot` | Generate per-plane and per-ROI QC plots |
| `delta_F.max_roi_plots` | Maximum per-ROI trace figures per accepted plane, as a non-negative integer or `all` |
| `delta_F.plot_zoomed_traces` | Also generate the fixed-window zoomed trace figure for selected ROIs |

Numeric values must be YAML numbers. Expressions such as `"5 / 400"` are rejected; write
`0.0125` instead.

`max_roi_plots` is applied independently to each processed plane. If fewer accepted ROIs are
available than the configured integer, every ROI is plotted. A smaller cap selects deterministic,
evenly distributed ROI indices; normal and zoomed figures use the same selection. Set it to zero
to skip per-ROI figures while retaining shared per-plane QC, or to `all` to plot every accepted ROI.
Zoomed figures are written only when `plot_zoomed_traces` is true and the recording is long enough
for the fixed zoom window.

## DAQ section

`daq` is required by `inspect-imaging` and `plot-piezo`, and by trace configurations that use
piezo data. `sampling_rate` is in Hz, `piezo_volt_per_micron` is volts per micrometre, and the four
string keys (`channel_names`, `data`, `piezo_channel_name`, and `clock_channel_name`) select DAQ
files and channels. File values may use glob syntax interpreted inside each TIFF experiment
directory.

The piezo trace serves two downstream purposes in `correct-traces`: locating an ROI's z coordinate
from its scan row, and, for a sloped Algorithm 1 acquisition, reslicing the dense reference stack
to match row-dependent depth. The standalone `plot-piezo` helper lets a user check these inputs
before processing.

## Path placeholders

Use `{Name}` and `{Date}` in all per-dataset paths. Raw imaging templates for `run-suite2p` and
`zregister` must also contain `{Experiments}`. Z-stack templates must contain `{Zstack}`. No other
placeholders are accepted.

Keep live configurations outside the source checkout, preferably in the private configuration
repository. Never commit raw paths or identifying dataset rows to this project.
