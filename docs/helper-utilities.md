# Helper utilities: purpose and outputs

The `suite2p-in-depth` CLI exposes five supported helpers. They prepare configuration
values, provide human quality-control figures, or restore a required Suite2p working file; none
selects an axial-correction algorithm on its own.

| Helper | Purpose | Output | What the output is used for |
| --- | --- | --- | --- |
| `inspect-imaging` | Infer basic acquisition dimensions and timing | Printed CSV with `Planes`, `Channels`, and `Frame_rate` | Populate or verify the dataset row used to configure Suite2p and flyback handling in either algorithm |
| `estimate-dark-level` | Estimate the PMT dark offset from registered movies | Printed dataset CSV with `Dark value` | QC/starting value for the manually reviewed `delta_F.absolute_zero` used in both trace-processing modes |
| `plot-piezo` | Visualize the synchronized, calibrated depth trajectory for each plane | One PNG per selected dataset | Human QC of plane count, flyback behaviour, DAQ synchronization, sign/scale, ROI depth, and z-stack reslicing assumptions |
| `plot-neuropil-masks` | Visualize an accepted Suite2p ROI and the pixels in its neuropil mask | One PNG | Human QC of the masks underlying `F.npy`, `Fneu.npy`, neuropil regression, and final dF/F |
| `recreate-bin-files` | Restore missing registered Suite2p movies using saved shifts | `data.bin` and, when applicable, `data_chan2.bin` in plane directories | Re-supply movies needed by Algorithm 1 frame-to-stack correlation and by `estimate-dark-level` |

Config paths are explicit and relative config paths are resolved from the YAML file. The two
commands that print CSV also send logs to standard output, so review and copy the table rather
than blindly redirecting the full stream into a dataset file.

## `inspect-imaging`

```shell
suite2p-in-depth inspect-imaging --config analysis-config/zregistration.yaml --dry-run
suite2p-in-depth inspect-imaging --config analysis-config/zregistration.yaml
```

The helper reads the first configured experiment's DAQ frame clock and piezo trace and inspects
the first available TIFF. It prints selected rows with:

| Column | Meaning | Downstream use |
| --- | --- | --- |
| `Name`, `Date` | Dataset identity | Match the reported values to the source CSV row |
| `Planes` | Plane count inferred from synchronized piezo/frame-clock signals | Suite2p `nplanes`, per-plane frame rate, and the `planes_to_flyback.csv` lookup |
| `Channels` | One or two channels inferred from TIFF dimensions | Suite2p `nchannels` and whether registration aligns by channel 2 |
| `Frame_rate` | Median pooled frame-clock rate across all planes, in Hz | Flyback-rule lookup and Suite2p per-plane rate (`Frame_rate / Planes`) |

Copy the three measured acquisition fields into `datasets_zstack.csv` for Algorithm 1 or
`datasets_zregistration.csv` for Algorithm 2, then rerun workflow validation. The helper does not
change the dataset CSV.

The command currently expects a z-registration-compatible configuration, selected dataset rows,
TIFF metadata, and a valid `daq` section. Standard z-registration validation happens first, so
provisional selected rows must already contain positive acquisition values that match one
flyback rule. For Algorithm 1, use a working copy of `zregistration.yaml` pointed at
`datasets_zstack.csv` and the relevant TIFF directories. `--dry-run` validates and lists the plan
without reading DAQ samples.

## `estimate-dark-level`

```shell
suite2p-in-depth estimate-dark-level --config analysis-config/trace_correction.yaml --dry-run
suite2p-in-depth estimate-dark-level --config analysis-config/trace_correction.yaml
```

For each Suite2p plane, this helper calculates the first percentile across all pixel values in up
to the first 200 registered `data.bin` frames. It prints `Dark value` as the lowest finite
per-plane estimate for each dataset.

Use that value to check the acquisition's PMT absolute-zero calibration before manually setting
`delta_F.absolute_zero` in `trace_correction.yaml` or `zstack.yaml`. The workflow subtracts this
offset from ROI and neuropil traces and from z-stack-derived ROI profiles before motion/neuropil
correction and dF/F calculation, so an incompatible value changes the scientific output. The
helper neither edits the YAML nor proves the physical dark level; review it against acquisition
calibration and keep compatible processing batches together.

The inputs are selected Suite2p plane directories, their metadata, and registered `data.bin`
files. Use `recreate-bin-files` first if a required binary has been removed. `--dry-run` validates
and lists the plan without sampling the movies.

## `plot-piezo`

```shell
suite2p-in-depth plot-piezo --config analysis-config/zregistration.yaml --output-dir qc/piezo
```

Inputs are the DAQ channel-name/data files selected by `daq`, the frame-clock and piezo channels,
`daq.piezo_volt_per_micron`, and the dataset plane count. Each output is named
`<Name>_<Date>_piezo_per_plane.png` and overlays the calibrated micrometre trajectory for every
acquired plane across a synchronized frame.

Use the PNG to check the expected plane count and order, identify suspicious flyback behaviour,
and catch DAQ alignment or voltage-to-distance mistakes. During trace processing the same
trajectory can set each ROI's z coordinate from its scan row; in Algorithm 1 it can also reslice
the reference stack to the sloped acquisition plane. The PNG itself is never ingested by a
workflow. The command refuses an existing output and creates the output directory only after
validation. It has no dry-run flag; use `validate` first.

## `plot-neuropil-masks`

```shell
suite2p-in-depth plot-neuropil-masks --plane-dir processed/suite2p/plane0 --output qc/roi-mask.png
```

The plane directory must contain `stat.npy`, `iscell.npy`, and either `ops.npy` or `db.npy` with
`Ly`/`Lx`. The PNG labels neuropil pixels separately from ROI pixels. Use it after Suite2p
detection/curation and before either trace-processing mode to check whether the pixels underlying
`F.npy` and `Fneu.npy` are plausible; those traces feed neuropil regression and final dF/F.

The current CLI single-file interface succeeds only when the plane contains exactly one accepted
ROI. It refuses to overwrite the output and never changes Suite2p masks or arrays. The PNG is a
human diagnostic, not a later processing input.

## `recreate-bin-files`

```shell
suite2p-in-depth recreate-bin-files --config analysis-config/zstack.yaml
suite2p-in-depth recreate-bin-files --config analysis-config/zstack.yaml --dry-run
suite2p-in-depth recreate-bin-files --config analysis-config/zstack.yaml --overwrite
```

The first two forms only inspect selected Suite2p planes and list missing `data.bin` and, for a
second channel, `data_chan2.bin` targets. Actual reconstruction requires the third form, the
original TIFFs, compatible `ops.npy` shifts from the original Suite2p registration, and temporary
output space.

The reconstructed binaries are registered movies placed back into their Suite2p plane
directories. Algorithm 1 uses them to correlate functional frames with z-stack slices, and
`estimate-dark-level` samples `data.bin`; they are not final trace outputs. The helper does not
re-run ROI detection or regenerate `F.npy`/`Fneu.npy`.

It never deletes source binaries automatically and refuses a temporary path that resolves to the
source Suite2p directory. Back up processed data before maintenance and review the dry-run target
list before allowing reconstruction.

## Exit behavior

Utility configuration/input errors return 2 and processing errors return 1.
