# suite2p-in-depth

`suite2p-in-depth` is a research preprocessing tool for two-photon calcium imaging. It builds on a
pinned official Suite2p release and implements two fundamentally different approaches to axial
brain-motion correction:

| | Algorithm 1: z-stack-based trace correction | Algorithm 2: multi-plane z-registration |
| --- | --- | --- |
| Intended data | Larger structures such as somata | Small structures such as axons and boutons |
| Required acquisition | A conventional functional recording plus a separate dense reference z-stack | The functional recording itself densely samples closely spaced imaging planes |
| What is aligned/corrected | Suite2p-extracted ROI and neuropil traces are corrected using frame-to-stack alignment and per-ROI axial profiles | Images from neighbouring acquired planes are registered into one `plane_z` movie |
| Commands | `run-suite2p`, then `correct-traces` with `use_zstack: true` | `zregister`, then optionally `correct-traces` with `use_zstack: false` for trace post-processing |
| Role of conventional Suite2p | `run-suite2p` is a prerequisite that produces the ROIs, traces, and registered movies used by correction | Suite2p stages are run inside `zregister`; do **not** run `run-suite2p` first |

These routes are alternatives, not consecutive axial-correction stages. In particular,
`correct-traces` with `use_zstack: false` does not estimate or correct z-motion: it is only the
neuropil/baseline/dF/F post-processing step for already z-registered `plane_z` data.

The trace workflow can also fit per-ROI neuropil correction and calculate dF/F. The software is
intended for reproducible research processing, not acquisition control, clinical use, or an
automatic substitute for Suite2p quality control.

The dependency is pinned directly to official Suite2p `v1.1.0` at immutable commit
`90be8953c03c6f4275dcd392ac1c6f554116169e`. Depth-specific reference construction,
z-correlation, and a narrow single-block nonrigid compatibility fix live in this repository; no
Suite2p fork is required. The commit pin prevents upstream changes from silently changing
scientific results.

## Installation

Most users should [install the repository as a package](#install-as-a-package). This provides the
`suite2p-in-depth` command used throughout this README and the linked guides. Developers who need
source changes to take effect without reinstalling should instead
[run from a source checkout](#run-from-a-source-checkout).

### Install as a package

Install a reviewed GitHub release tag into an isolated Python 3.12 environment. The installed
`suite2p-in-depth` command can be used outside the repository checkout:

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Schroeder-Lab/suite2p-in-depth.git@v0.1.0"
suite2p-in-depth --version
suite2p-in-depth --help
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`. A downloaded release wheel can instead
be installed with `python -m pip install path/to/suite2p_in_depth-0.1.0-py3-none-any.whl`. This project
is not published to PyPI.

### Run from a source checkout

Clone the repository, create an isolated Python 3.12 environment, and install its runtime
dependencies. The project itself does not need to be installed, and edits under `src/` take
effect on the next run:

```shell
git clone https://github.com/Schroeder-Lab/suite2p-in-depth.git
cd suite2p-in-depth
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_suite2p_in_depth.py --version
```

Run source-checkout commands from the repository root. For every command below and in the linked
guides, replace the `suite2p-in-depth` prefix with `python run_suite2p_in_depth.py`; all subcommands and
arguments remain the same. On PowerShell, activate with `.venv\Scripts\Activate.ps1`.

See [Installation](docs/installation.md) for CUDA setup and troubleshooting.

## Five-minute starts

First create a private working configuration directory:

```shell
suite2p-in-depth init analysis-config
suite2p-in-depth --help
```

`init` creates `analysis-config/` relative to the current directory and copies eight editable,
sanitized templates into it: four workflow YAML files, two example dataset CSVs, the flyback-plane
lookup, and the zoom-to-microns lookup. It does not inspect or process data and does not fill in
acquisition settings. Replace the placeholder paths and metadata before running a workflow.
For safety, `init` refuses to write into a directory that already exists and is non-empty.

### Route 1: correct extracted traces with a z-stack

Edit `datasets_zstack.csv`, `plain_suite2p.yaml`, and `zstack.yaml`, and set `Process` to `True`
only for the rows to run. First run conventional Suite2p and review or curate its ROIs. Then run
z-stack-based trace correction.

```shell
suite2p-in-depth validate --workflow run-suite2p --config analysis-config/plain_suite2p.yaml
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml --dry-run
suite2p-in-depth run-suite2p --config analysis-config/plain_suite2p.yaml
suite2p-in-depth validate --workflow correct-traces --config analysis-config/zstack.yaml
suite2p-in-depth correct-traces --config analysis-config/zstack.yaml --dry-run
suite2p-in-depth correct-traces --config analysis-config/zstack.yaml
```

See [Z-stack-based trace correction](docs/z-stack-correction.md) for complete configuration,
calibration, and quality-control instructions.

### Route 2: z-register densely sampled planes

Edit `datasets_zregistration.csv` and `zregistration.yaml`, and set `Process` to `True` only for
the rows to run. `zregister` includes the required per-plane Suite2p processing.

```shell
suite2p-in-depth validate --workflow zregister --config analysis-config/zregistration.yaml
suite2p-in-depth zregister --config analysis-config/zregistration.yaml --dry-run
suite2p-in-depth zregister --config analysis-config/zregistration.yaml
```

Review and, if needed, curate the resulting `suite2p/plane_z`. To produce neuropil-corrected
traces and dF/F from that z-registered movie, run the separate non-z-stack post-processing step:

```shell
suite2p-in-depth validate --workflow correct-traces --config analysis-config/trace_correction.yaml
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml --dry-run
suite2p-in-depth correct-traces --config analysis-config/trace_correction.yaml
```

This final step uses `use_zstack: false`; it does not perform further axial correction. See
[Multi-plane z-registration](docs/z-registration.md) for complete instructions.

Validation and dry-run do not write outputs. A real run creates a `run_manifest.json` under the
tool-owned `.suite2p-in-depth/<workflow>/` directory for each selected dataset. Completed or
incompatible runs are refused unless `--overwrite` is explicit; inspect the output before using
that flag.

Set `suite2p_settings.torch_device` to `cpu` or `cuda` in the relevant YAML. Trace correction is
CPU-only. Always compare CPU and CUDA outputs on representative de-identified data before a
release or environment change.

## Guides

Start with the guide for the algorithm you chose:

- **Algorithm 1:** [Z-stack-based trace correction](docs/z-stack-correction.md), whose first stage
  is [conventional Suite2p processing](docs/suite2p-processing.md); see the
  [Z-stack plot guide](docs/z-stack-plot-guide.md) for illustrated quality-control examples
- **Algorithm 2:** [Multi-plane z-registration](docs/z-registration.md), followed only if needed by
  [non-z-stack trace post-processing](docs/trace-correction.md#after-algorithm-2-non-z-stack-post-processing);
  see the [z-registration plot guide](docs/z-registration-plot-guide.md) for illustrated examples

Configuration and reference material:

- [Configuration](docs/configuration.md) and [dataset CSV schemas](docs/dataset-csv.md)
- [Helper utilities and what their outputs are for](docs/helper-utilities.md)
- [Outputs](docs/outputs.md) and [run manifests](docs/manifests.md)
- [Safe reruns](docs/safe-reruns.md)
- [Paper figure reproducibility](docs/paper-reproducibility.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture and testing](docs/architecture.md)
- [Suite2p compatibility](docs/suite2p-compatibility.md)
- [Contributing](docs/contributing.md)
- [Security reporting](SECURITY.md)
- [Acknowledgements](ACKNOWLEDGEMENTS.md)

## Support matrix

| Platform | Python | Execution | Status |
| --- | --- | --- | --- |
| Ubuntu Linux | 3.12 | CPU | Tested in CI |
| Windows | 3.12 | CPU | Tested in CI |
| Linux/Windows | 3.12 | CUDA | Supported when a compatible PyTorch/CUDA stack is installed; verify locally |
| macOS | 3.12 | CPU/MPS | Development convenience only; not in the supported release matrix |

## Development checks

Install the development dependencies, then run the same formatting, linting, naming, typing, and
test checks used during development:

```shell
python -m ruff format --check run_suite2p_in_depth.py src paper tests
python -m ruff check run_suite2p_in_depth.py src paper tests
python -m mypy
python -m pytest
```

Python identifiers follow PEP 8 `snake_case`; external Suite2p field names are retained where they
form part of Suite2p's persisted data schema. Comments explain intent or scientific constraints,
while obsolete disabled code is removed rather than left commented out.

Report reproducible defects with the [bug report template](https://github.com/Schroeder-Lab/suite2p-in-depth/issues/new?template=bug_report.yml).
Do not attach raw laboratory data, identifying metadata, credentials, or private paths. Scientific
questions and enhancement proposals may use the
[feature request template](https://github.com/Schroeder-Lab/suite2p-in-depth/issues/new?template=feature_request.yml).

## License, acknowledgements, and citation

Copyright 2026 University of Sussex. This software is licensed under
[GPL-3.0-only](LICENSE). Development contributions from Sylvia Schröder, Liad J Baruchin, and
Andrew Mikhniak are acknowledged, along with funding from the Wellcome Trust. See
[Acknowledgements](ACKNOWLEDGEMENTS.md).

Citation metadata for the software is in [CITATION.cff](CITATION.cff). A paper citation will be
added only after its bibliographic metadata is approved. The planned release is `v0.1.0`.
