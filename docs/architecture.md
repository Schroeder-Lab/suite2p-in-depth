# Architecture

The package uses the `src/suite2p_in_depth` layout. The root `run_suite2p_in_depth.py` launcher prepends
`src` to the import path and calls `suite2p_in_depth.cli:main` for supported source-checkout execution.
The installed `suite2p-in-depth` console entry point calls the same function.

- `cli.py` owns argument parsing, exit codes, orchestration, logging, and utility dispatch.
- `validation.py` loads YAML/CSV, normalizes cross-platform paths, validates scientific inputs,
  and creates write-free dataset plans.
- `run_manifest.py` handles compatibility checks and atomic provenance manifests.
- `run_suite2p.py`, `zregister.py`, and `correct_traces.py` orchestrate three CLI workflows that
  implement two scientific algorithms. `run-suite2p` followed by z-stack-enabled
  `correct-traces` is Algorithm 1; `zregister` is Algorithm 2. Non-z-stack `correct-traces` is
  only Algorithm 2 trace post-processing.
- `suite2p_compat.py` is the boundary for current and legacy Suite2p mappings.
- `suite2p_extensions.py` contains the small axial-registration extensions absent from the pinned
  official Suite2p release, including the single-block nonrigid compatibility path.
- `registration_numerics.py`, `preprocess_traces.py`, and related numerical modules contain
  isolatable array operations.
- `registration_plotting.py` and utility plotting modules own visualization.
- `utilities/` contains implementations promoted from retired helper scripts.
- `paper/` contains import-safe, configurable figure scripts and is excluded from the wheel.

Configuration templates are package data in `src/suite2p_in_depth/config`. Laboratory configuration,
raw data, manuscript data, and generated figures do not belong in the public repository.

Data flows from validated YAML/CSV into a `ValidatedWorkflow`, then into one selected dataset at a
time. Suite2p-specific untyped dictionaries remain at the adapter/workflow boundary. Manifests
record the resolved plan and completion state. Pure numerical functions should remain independent
of CLI and filesystem state.

Compatibility rules are conservative: documented `.npy` filenames, keys, array orientations,
units, and NaN behavior are stable unless a confirmed defect has a focused regression test.
Scientific formula changes require explicit review and baseline comparison.

Suite2p is consumed directly from the official repository at an immutable release commit. Local
extensions must remain focused, tested, and documented in
[Suite2p compatibility](suite2p-compatibility.md); do not recreate a long-lived Suite2p fork.
