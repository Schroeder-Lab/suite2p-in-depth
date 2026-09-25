# Installation

## Requirements

Use 64-bit Python 3.12 on Ubuntu Linux or Windows. CPU execution is the tested baseline. Git is
required because Suite2p is installed from an immutable Git commit. A compiler may be required if
your platform cannot use binary wheels for a dependency.

The package is GitHub-first and is not published to PyPI. Use only a reviewed tag or commit.

## Install as a package

Create a fresh environment and install a reviewed GitHub release tag:

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Schroeder-Lab/suite2p-in-depth.git@v0.1.0"
suite2p-in-depth --version
suite2p-in-depth --help
```

PowerShell activation is `.venv\Scripts\Activate.ps1`. A downloaded release wheel can instead be
installed with `python -m pip install path/to/suite2p_in_depth-0.1.0-py3-none-any.whl`.

## Run from a source checkout

Developers can install only the runtime dependencies and execute the checked-out source directly:

```shell
git clone https://github.com/Schroeder-Lab/suite2p-in-depth.git
cd suite2p-in-depth
git checkout v0.1.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_suite2p_in_depth.py --version
```

Run source-checkout commands from the repository root. Replace the `suite2p-in-depth` prefix in the
documented commands with `python run_suite2p_in_depth.py`; all subcommands and arguments remain the
same. `requirements.txt` contains the same direct runtime dependencies as `pyproject.toml`, and CI
checks that they remain identical.

## Official Suite2p dependency

Package metadata pins Suite2p `v1.1.0` directly from `MouseLand/suite2p` at commit
`90be8953c03c6f4275dcd392ac1c6f554116169e`. Do not replace it with an unpinned release: the
package relies on its settings/database split and registration behavior. Project-specific axial
registration extensions are part of `suite2p-in-depth`, not a Suite2p fork. To confirm the installed
Suite2p source and revision:

```shell
python -c "import importlib.metadata as m; print(m.distribution('suite2p').read_text('direct_url.json'))"
```

The output must name `https://github.com/MouseLand/suite2p.git` and the commit above. See
[Suite2p compatibility](suite2p-compatibility.md) for the upstream comparison and local extension
policy.

## CPU and CUDA

Set `suite2p_settings.torch_device: "cpu"` for the portable baseline. Set it to `"cuda"` only
after installing a PyTorch build compatible with the local NVIDIA driver and verifying
`python -c "import torch; print(torch.cuda.is_available())"` returns `True`. The CLI reports the
selected device during validation and dry-run. Trace correction always uses CPU.

CUDA availability is environment-specific and is not exercised by hosted CI. Compare representative
CPU and CUDA outputs before accepting a CUDA environment.

## Verify the installation

```shell
suite2p-in-depth --help
suite2p-in-depth init analysis-config
suite2p-in-depth validate --workflow run-suite2p --config analysis-config/plain_suite2p.yaml
```

The initialized CSV rows have `Process=False`, so validation succeeds without laboratory data.
`init` will not overwrite a non-empty directory.
