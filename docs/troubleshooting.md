# Troubleshooting

## Unsure which command sequence to use

For larger structures with a separate dense reference stack, run `run-suite2p` first and then
z-stack-enabled `correct-traces`. For small structures recorded in densely spaced planes, run
`zregister`; it includes its own Suite2p stages. Use non-z-stack `correct-traces` only afterward if
standardized neuropil-corrected traces and dF/F are needed. See the decision table in the
[README](../README.md) before diagnosing configuration errors.

## Exit code 2 before processing

This is an input/configuration problem. Read the final error: common causes are a missing required
key, expression-like numeric YAML, unknown path placeholder, missing TIFF, non-existent Suite2p
directory, ambiguous z-stack TIFFs, or a dataset that does not match exactly one flyback rule.
Run `validate` again after correction; it writes nothing.

## No selected datasets

Initialized examples deliberately have `Process=False`. Edit the private CSV and select only rows
whose inputs are ready. CSV booleans must be `True` or `False`.

## CUDA is requested but unavailable

Set `torch_device: "cpu"` for a portable run, or install the PyTorch build appropriate for the
local driver. Confirm `torch.cuda.is_available()` before validation. Never compare a new CUDA
environment only by runtime; use the scientific acceptance metrics in the release checklist.

## Existing output is refused

Open `run_manifest.json` and compare workflow, dataset, configuration digest, and completion
state. Use a new output root to preserve the previous result. Use `--overwrite` only after review;
it is not a resume promise for incompatible partial artifacts.

## No valid planes or ROIs

Check flyback/ignore-plane indices, `iscell.npy`, Suite2p outputs, and manual curation. The tool
fails rather than writing misleading empty arrays.

## NaNs in one ROI

Constant or insufficient neuropil regression data produces NaNs for that ROI and a warning while
other ROIs continue. Inspect its signal, neuropil mask, finite-value rate, and diagnostic plots.

## Z-stack mismatch

Confirm a single TIFF, `[z, y, x]` ordering, rectangular image dimensions, positive spacing, and
that axial motion stays within the stack. Check piezo units: `piezo_volt_per_micron` is volts per
micrometre.

## Fonts or display errors in paper scripts

Set `MPLBACKEND=Agg`. Missing Arial warnings can change typography; provide the approved font or a
reviewed `--style`. Paper scripts require explicit existing input roots and never download data.

When filing an issue, include the package/Suite2p versions, OS, Python, command, redacted config,
manifest metadata, traceback, and the smallest synthetic reproducer. Do not share raw lab data.
