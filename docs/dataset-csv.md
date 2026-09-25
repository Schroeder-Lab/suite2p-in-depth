# Dataset CSV schemas by algorithm

CSV parsing preserves `Name` and `Date` as strings. `Process` must be `True` or `False`; only true
rows are validated against the filesystem and processed.

## Algorithm 1: `datasets_zstack.csv`

The conventional `run-suite2p` prerequisite reads:

| Column | Type | Meaning |
| --- | --- | --- |
| `Name` | string | Subject/container directory name |
| `Date` | string | Recording date/container name |
| `Process` | boolean | Select this row |
| `Experiments` | list | Functional-recording TIFF folders, for example `"[1, 2]"` |
| `Planes` | positive integer | Acquired planes including flyback planes |
| `Channels` | positive integer | Acquired imaging channels |
| `Frame_rate` | positive number | Pooled frame rate across all acquired planes, Hz |

These values configure Suite2p and the flyback lookup. `inspect-imaging` can infer them, but its
current interface requires a z-registration-compatible inspection YAML; see the
[helper guide](helper-utilities.md#inspect-imaging) for the setup.

The later `correct-traces` stage with `use_zstack: true` additionally requires:

| Column | Type | Meaning |
| --- | --- | --- |
| `Zstack_folder` | string | Folder substituted into `{Zstack}`; it must contain exactly one stack TIFF |
| `Ignore_planes` | list | Zero-based planes excluded in addition to flyback planes |
| `Depth` | number | Depth of the upper acquired plane relative to the chosen reference; use `0` only when an absolute value is unavailable |

Keeping both stages in `datasets_zstack.csv` makes the provenance explicit: `run-suite2p`
prepares the recording, and z-stack-enabled `correct-traces` performs Algorithm 1.

## Algorithm 2: `datasets_zregistration.csv`

The `zregister` stage requires:

| Column | Type | Meaning |
| --- | --- | --- |
| `Name` | string | Subject/container directory name |
| `Date` | string | Recording date/container name |
| `Process` | boolean | Select this row |
| `Experiments` | list | Experiment folders, for example `"[1, 2]"` |
| `Planes` | positive integer | Acquired planes including flyback planes |
| `Channels` | positive integer | Acquired imaging channels |
| `Frame_rate` | positive number | Pooled frame rate across all acquired planes, Hz |
| `Depth` | number, optional | Depth of the top imaging plane, usually micrometres |

Use [`inspect-imaging`](helper-utilities.md#inspect-imaging) to infer `Planes`, `Channels`, and
`Frame_rate` from TIFF/DAQ inputs, then copy the results into this table. The plane count and
frame rate must match exactly one row in `planes_to_flyback.csv`; flyback indices are zero-based
and must be smaller than `Planes`.

The optional non-z-stack `correct-traces` stage reuses this dataset identity and any
`Ignore_planes`, `Depth`, or acquisition fields retained from z-registration. It is valid only as
post-processing of Algorithm 2's `plane_z` output.

## Auxiliary CSVs

`planes_to_flyback.csv` requires `Planes`, `Frame_rate_range`, and `Flyback_planes`.
`Frame_rate_range` is a two-value inclusive range; every selected recording in a Suite2p-producing
stage must match exactly one rule. Use the `inspect-imaging` values and the `plot-piezo` QC output
when reviewing this mapping.

`zoom_to_microns.csv` requires `Zoom` and `FOV_size`; the latter is the full field width in
micrometres and is used when `pixels_to_microns: true` writes ROI x/y locations.

The sanitized rows created by `init` are placeholders and use `Process=False`. Replace them only
in a private working configuration.
