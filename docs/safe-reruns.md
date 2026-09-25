# Safe reruns and overwrite behavior

Validation and `--dry-run` resolve selected datasets, verify input directories and TIFF presence,
select a device, and list planned outputs without writing. Use them before every real run.

A real workflow owns only its manifest directory:
`<output>/.suite2p-in-depth/<workflow>/`. If no manifest exists, the run can start. If a complete
manifest exists, or the existing manifest has a different workflow, dataset identity, or
configuration digest, the command exits with code 2 unless `--overwrite` is supplied.
If a manifest for the same workflow exists under another hidden tool directory, use a new output
root; `--overwrite` does not bypass this check.

`--overwrite` authorizes rerunning the selected workflow; it does not authorize arbitrary source
data deletion. Inspect and back up outputs first. Existing scientific files may be replaced by
Suite2p or workflow code during the rerun, so use a new output root when retaining an old baseline
matters.

Exit codes are:

- `0`: success (including validation and dry-run);
- `2`: configuration or input error; and
- `1`: processing failure.

Failed datasets are recorded in their manifests and the final summary. Do not assume an output is
usable unless `completion_state` is `complete` and QC has passed.

`recreate-bin-files` is a special maintenance command. It is dry-run by default even without the
flag and lists exact missing targets. Recreation occurs only with `--overwrite` and without
`--dry-run`; source binaries are never automatically deleted. See [Helper utilities](helper-utilities.md).
