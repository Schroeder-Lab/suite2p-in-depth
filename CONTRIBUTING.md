# Contributing

Thank you for improving `suite2p-in-depth`. Scientific correctness and reproducibility take priority
over broad rewrites.

## Development setup

Use Python 3.12 in a clean virtual environment:

```shell
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev]"
```

Read [the architecture guide](docs/architecture.md) and [testing guide](docs/testing.md). Create a
focused branch and keep unrelated formatting or numerical changes out of the same review.

## Change requirements

- Preserve documented filenames, keys, shapes, orientations, units, and NaN behavior.
- Add a deterministic regression test for every defect fix.
- Do not change a scientific formula without rationale, focused tests, and an accepted baseline
  comparison.
- Keep pure numerical code independent of Suite2p and filesystem boundaries where practical.
- Update user documentation and describe user-visible changes in the pull request.
- Use only synthetic, non-identifying test fixtures. Never commit raw lab data, private paths,
  secrets, manifests containing sensitive paths, or manuscript data.

Run the complete local gate from [Testing](docs/testing.md). A pull request should describe
behavior changes, tests and platforms run, numerical tolerances, output compatibility, and any
manual scientific validation.

## Reporting defects

Use the repository issue templates and redact all subject identifiers and infrastructure paths.
Security concerns must follow [SECURITY.md](SECURITY.md), not a public issue.

By contributing, you confirm you have the right to submit the change. Contributions are accepted
under the project's [GPL-3.0-only license](LICENSE).
