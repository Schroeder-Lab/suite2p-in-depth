# Suite2p compatibility

suite2p-in-depth depends directly on the official
[`MouseLand/suite2p`](https://github.com/MouseLand/suite2p) repository. The supported dependency is
Suite2p `v1.1.0`, commit `90be8953c03c6f4275dcd392ac1c6f554116169e`. Package metadata, CI,
and release checks all verify that immutable source revision. A Schroeder Lab Suite2p fork is no
longer required.

## Comparison with the former fork

The former fork was compared with official Suite2p `v1.1.0`. Upstream already contains the
post-`v1.0.0.1` registration, MPS, GUI, extraction, flyback-plane, and documentation changes that
the old fork lacked. Three fork behaviors were not present in `v1.1.0` and remain necessary for
the depth workflows:

| Former fork behavior | Upstream `v1.1.0` | suite2p-in-depth implementation |
| --- | --- | --- |
| Configurable number of highly correlated frames for z-stack reference initialization | Suite2p uses a fixed count of 20 | `suite2p_extensions.compute_reference` |
| Batchwise correlation of registered frames against every z-stack plane | `zalign.compute_zpos` is a placeholder | `suite2p_extensions.compute_zpos` |
| Nonrigid interpolation when the block grid has one row or one column | Upstream interpolation requests a zero-sized dimension | `suite2p_extensions.shift_frames` for that edge case only |

The generic single-block correction should ultimately be contributed upstream. Until it appears
in a reviewed Suite2p release, the local compatibility path is limited to the z-stack operations
that require it. All other registration calls use official Suite2p directly.

## Upgrade policy

For a future Suite2p release:

1. Compare the release tag with the pinned commit and review changes to `parameters`,
   `registration.register`, `registration.nonrigid`, `registration.zalign`, and output mappings.
2. Check whether each local extension is now provided upstream with compatible shapes and
   numerical behavior. Remove redundant local code instead of preserving two implementations.
3. Run unit and synthetic real-Suite2p tests, then complete the manual scientific CPU/CUDA
   comparison on representative de-identified data before changing the supported revision.
4. Update the dependency commit, CI `direct_url.json` assertion, release checklist, changelog, and
   this page together.

Never use an unpinned Suite2p branch for a release or silently change the dependency during a
scientific rerun.
