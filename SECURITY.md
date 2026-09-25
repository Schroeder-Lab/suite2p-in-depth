# Security policy

## Supported versions

The project is pre-release. Security fixes target the latest commit on `main` until `v0.1.0` is
released; afterward, only the latest release is supported unless maintainers announce otherwise.

## Report privately

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability
reporting for `Schroeder-Lab/suite2p-in-depth`, or contact the Schroeder Lab maintainers through an
institutionally published private channel if that feature is unavailable. Include affected
version/commit, platform, impact, reproduction steps using synthetic data, and suggested
mitigation. Do not send raw laboratory data, credentials, private repository contents, or subject
identifiers.

Maintainers should acknowledge a report within seven days, assess scope, coordinate a fix and
disclosure date, and credit the reporter if requested. Timelines depend on scientific validation
and dependency coordination.

## Scope

Relevant concerns include unsafe path handling, unintended overwrite/deletion, malicious
configuration or NumPy payload handling, dependency compromise, and disclosure through manifests
or logs. NumPy files loaded with `allow_pickle=True` must be treated as trusted inputs; never open
untrusted Suite2p artifacts. General scientific correctness reports belong in the bug template.
