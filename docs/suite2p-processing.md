# Conventional Suite2p prerequisite for Algorithm 1

Use `run-suite2p` only to prepare functional recordings for
[Algorithm 1, z-stack-based trace correction](z-stack-correction.md). It generates the registered
movies, ROIs, and extracted fluorescence/neuropil traces that `correct-traces` aligns to a dense
reference stack.

Do not run this command before [Algorithm 2, multi-plane z-registration](z-registration.md):
`zregister` includes its own configured Suite2p stages and creates a separate `plane_z` result.

## Configure and run

Initialize a private configuration, then edit `plain_suite2p.yaml` and `datasets_zstack.csv`. The
dataset's `Planes`, `Channels`, and pooled `Frame_rate` configure Suite2p and select flyback rules.
If these are uncertain, use [`inspect-imaging`](helper-utilities.md#inspect-imaging) while
preparing the dataset row. See [Configuration](configuration.md#algorithm-1-configuration) for
the exact interpretation.

```shell
suite2p-in-depth validate --workflow run-suite2p --config analysis-config/plain_suite2p.yaml
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml --dry-run
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml
```

Validation requires every selected experiment directory to exist and contain at least one `.tif`
or `.tiff`. It checks the dataset schema, flyback lookup, Suite2p settings, path placeholders, and
selected device before any output is created.

The workflow writes Suite2p output under `<directories.output>/suite2p/planeN`. Review reference
and mean images, registration metrics, ROI masks, extracted traces, and binary files. If manual ROI
curation is needed, launch the pinned Suite2p GUI from the same environment:

```shell
python -m suite2p
```

The base `suite2p-in-depth` installation includes Suite2p's GUI and NWB dependencies and a Matplotlib
version compatible with the pinned Suite2p revision. Open a plane's `stat.npy` and preserve the
resulting `stat.npy`/`iscell.npy` pair after curation. The
[`plot-neuropil-masks`](helper-utilities.md#plot-neuropil-masks) helper provides focused QC of the
pixels used for an accepted ROI and its neuropil trace.

`data.bin` (and `data_chan2.bin` when present) contains the registered movie used later for
frame-to-z-stack correlation. If a required binary is missing, use the guarded
[`recreate-bin-files`](helper-utilities.md#recreate-bin-files) utility rather than rerunning or
deleting files ad hoc. Its dry run lists exact targets; the recreated binaries become inputs to
Algorithm 1 and to `estimate-dark-level`.

Treat raw TIFFs as immutable and do not mix recordings with incompatible acquisition parameters.
The command writes a dataset manifest to
`<directories.output>/.suite2p-in-depth/run-suite2p/run_manifest.json`; see
[Safe reruns](safe-reruns.md) before repeating a completed run.
