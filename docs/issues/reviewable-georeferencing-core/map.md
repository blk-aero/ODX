# Reviewable georeferencing core

## Destination

Replace the broad direct-georeferencing integration with a lean, reviewable
core for ordinary single-project ODX jobs: canonical processing, a minimal
persisted exact contract, exact public point/mesh export, and direct
orthophoto rendering with clear preconditions.

## Notes

- Maintain correctness at the public geometry boundary; delete speculative or
  unused features rather than preserving them by default.
- Work from `7eed3705afc149b67d2fe9b129118cf02c3c0c18` on
  `codex/georeferencing-direct-raster`. Do not disturb the pre-existing
  deletion of `AGENTS.md`.
- Use `/ponytail` (full), `/grilling`, and `/domain-modeling`. A direct
  orthophoto is intentionally distinct from a raster coordinate warp.
- GitHub Issues are disabled here, so this local Markdown tracker is the
  canonical map.

## Decisions so far

<!-- Closed tickets appear here as one-line linked gists. -->
- [Ticket 01: minimum coordinate contract](issues/01-minimum-coordinate-contract.md) — use an immutable ENU-to-automatic-UTM/UPS contract; defer alignment, split/merge, and secondary artifacts.
- [Ticket 02: `--align` boundary](issues/02-align-boundary.md) — preserve stock alignment until direct exports are integrated, then reject `--align` at direct-export entry rather than add partial post-CRS alignment.
- [Ticket 03: split/merge boundary](issues/03-split-merge-boundary.md) — retain stock split/merge until direct integration; then reject georeferenced submodels at direct-export entry before contracts or artifacts, deferring output selection and merge validation.
- [Ticket 04: secondary-artifact boundary](issues/04-secondary-artifact-boundary.md) — retain stock secondary consumers through the late affine reconstruction view; point/mesh are exact, orthophoto is direct, and boundary selection is deferred with a fail-closed guard.

## Required implementation

- [Ticket 07: integrate the minimum coordinate contract into direct exports](issues/07-integrate-direct-exports.md) — wire the resolved core boundaries into the ordinary single-project path.

## Not yet specified

- After the core boundary is fixed, identify any remaining code that cannot be
  classified as core, legacy compatibility, or a separately scoped feature.

## Out of scope

- Keeping the unused GDAL exact-raster warp; it is dead production code and
  should be deleted with its tests.
- Release/benchmark framework and acceptance-process documentation.
- The new split/merge coordinate-contract protocol and derivative manifests.
- Rich contract provenance, broad secondary-artifact exactness, and audit
  reporting beyond fields needed by the core runtime.
- The unrelated removal of repository GitHub workflows from the current PR.
