# Algorithm 2: multi-plane z-registration

Use this algorithm for axons, boutons, and other small structures recorded in densely sampled,
closely spaced imaging planes. The functional recording itself supplies the neighbouring depth
information: no separate z-stack is used. Signal should be similar between neighbouring
non-flyback planes, most planes should be approximately parallel, and axial motion must remain
within the acquired plane range.

This route is distinct from [z-stack-based trace correction](z-stack-correction.md). Do not run
`run-suite2p` first: `zregister` runs the configured Suite2p registration and detection stages for
each acquired plane as part of this algorithm.

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

## 2. Configure the acquisition and Suite2p stages

Edit `zregistration.yaml` and `datasets_zregistration.csv`:

- `Experiments` selects the raw TIFF folders.
- `Planes` is the total number of acquired planes, including flyback planes.
- `Channels` is the number of acquired imaging channels. `suite2p_db.functional_chan` selects the
  signal channel. Set `suite2p_settings.registration.align_by_chan2: true` to register using a
  second channel; the packaged config defaults to `false`.
- `Frame_rate` is the pooled rate across all acquired planes, not the per-plane rate. The
  Suite2p rate is calculated as `Frame_rate / Planes`.
- `planes_to_flyback.csv` maps the measured plane count and pooled frame rate to zero-based planes
  that must not contribute to z-registration.

If any of the first three acquisition values are uncertain, run
[`inspect-imaging`](helper-utilities.md#inspect-imaging) and copy its reported `Planes`,
`Channels`, and `Frame_rate` into the dataset row. Use
[`plot-piezo`](helper-utilities.md#plot-piezo) to check that the inferred per-plane piezo
trajectories and flyback assignments are physically plausible; its PNG is a QC figure and is not
read by `zregister`.

Choose Suite2p ROI settings for the small structures being tracked. In particular,
`suite2p_settings.diameter`, detection settings, the functional channel, and registration channel
must match the acquisition. See [Configuration](configuration.md) for all shared keys.

## 3. Run z-registration

```shell
suite2p-in-depth validate --workflow zregister --config analysis-config/zregistration.yaml
suite2p-in-depth zregister --config analysis-config/zregistration.yaml --dry-run
suite2p-in-depth zregister --config analysis-config/zregistration.yaml
```

For each non-flyback plane, the workflow performs the configured Suite2p stages, constructs plane
references, estimates every frame's best matching depth, combines information from neighbouring
planes, and produces one z-registered `suite2p/plane_z`. Numeric plane ordering is used, so
`plane10` follows `plane9`.

## 4. Review and curate `plane_z`

Review `01_reference_images.png`, `02_mean_image_zregistration.png`, and `03_z_positions.png`,
along with Suite2p registration metrics and the registered movie. The main image-registration
mapping is `plane_z/reg_outputs.npy`; fields such as `zpos_registration` and
`planes_across_time` are for checking the inferred axial trajectory and tracing each output frame
back to its acquired source plane. See [Outputs](outputs.md#algorithm-2-z-registration) for the
complete field descriptions. The [z-registration plot guide](z-registration-plot-guide.md) shows
these plots alongside examples of the optional ROI and trace figures.

If desired, curate ROIs using the Suite2p files in `plane_z`. The
[`plot-neuropil-masks`](helper-utilities.md#plot-neuropil-masks) helper can visualize the accepted
ROI and surrounding neuropil pixels before trace post-processing; its output is diagnostic only.

## 5. Optionally produce standardized traces

After z-registration and any curation, `correct-traces` with `trace_correction.yaml` performs
neuropil correction, baseline estimation, and dF/F calculation on `plane_z`:

```shell
suite2p-in-depth validate --workflow correct-traces --config analysis-config/trace_correction.yaml
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml --dry-run
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml
```

Keep `use_zstack: false`. This is post-processing of data whose z-motion was already handled by
`zregister`; it is not another z-motion-correction algorithm. Before running it, use
[`estimate-dark-level`](helper-utilities.md#estimate-dark-level) as a QC estimate for the PMT
dark offset and set the reviewed calibration as `delta_F.absolute_zero`.

Keep a single accepted set of flyback rules and absolute-zero calibration for each processing
batch. Action 2 corrected time-axis trimming, per-plane settings persistence, and configured
z-correlation smoothing; include these differences in comparisons with older outputs.
