# Run manifests

Every real workflow writes one JSON manifest per selected dataset beneath
`<directories.output>/.suite2p-in-depth/<workflow>/run_manifest.json`. The path is intentionally
workflow-specific and is the only directory the CLI treats as tool-owned for provenance.

Schema version 1 records:

- `workflow`, `completion_state`, and `updated_at`;
- `suite2p_in_depth_version`, `suite2p_version`, Python, platform, and device;
- the resolved configuration file, full resolved configuration, and `configuration_sha256`;
- the normalized dataset row, resolved input directories, and output root; and
- an error string for failed processing.

The file is written as `running` before processing, then atomically replaced with `complete` or
`failed`. A process terminated outside normal exception handling can leave `running`; inspect
scientific artifacts before deciding whether to rerun.

On a later invocation, workflow, `Name`, `Date`, and the configuration digest determine
compatibility. A complete or incompatible manifest causes exit code 2 unless `--overwrite` is
explicit. This is a conservative conflict check, not a guarantee that every partial Suite2p file
can resume safely. Use a fresh output directory when uncertain.
An existing manifest for the same workflow under another hidden tool directory also requires a
fresh output directory.

Manifests contain resolved local paths and the configuration snapshot. Treat them as private
dataset provenance, redact before sharing, and never commit laboratory manifests to this
repository. See [Output schemas](outputs.md) and [Safe reruns](safe-reruns.md).
