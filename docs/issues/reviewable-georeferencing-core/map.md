# Reviewable georeferencing core

## Destination

Replace the broad direct-georeferencing integration with a lean, reviewable
core for ordinary single-project ODX jobs: canonical processing, a minimal
persisted exact contract, exact public point/mesh export, and direct
orthophoto rendering with clear preconditions.

## Notes

- Maintain correctness at the public geometry boundary; delete speculative or
  unused features rather than preserving them by default.
- Build on the current master-based branch; use
  `7eed3705afc149b67d2fe9b129118cf02c3c0c18` only as a read-only reference.
  Do not disturb the pre-existing deletion of `AGENTS.md`.
- Use `/ponytail` (full), `/grilling`, and `/domain-modeling`. A direct
  orthophoto is intentionally distinct from a raster coordinate warp.
- GitHub Issues are disabled here, so this local Markdown tracker is the
  canonical map.

## Decisions so far

<!-- Closed tickets appear here as one-line linked gists. -->
- [Ticket 01: minimum coordinate contract](issues/01-minimum-coordinate-contract.md) — use an immutable ENU-to-automatic-UTM/UPS contract; defer alignment, split/merge, and secondary artifacts.
- [Ticket 02: `--align` boundary](issues/02-align-boundary.md) — keep alignment on the stock path; do not add post-CRS alignment state to the focused contract.
- [Ticket 03: split/merge boundary](issues/03-split-merge-boundary.md) — keep georeferenced split submodels on the stock split/merge path; defer output selection and merge validation.
- [Ticket 04: secondary-artifact boundary](issues/04-secondary-artifact-boundary.md) — retain stock secondary consumers through the late affine reconstruction view; exact point/mesh and direct orthophoto are best effort, and boundary options stay stock.
- [Ticket 07: direct-export integration](issues/07-integrate-direct-exports.md) — persist the existing GCP/GPS vertical state, transform OBJ normals locally, and use exact canonical geometry when available before late affine compatibility publication.
- [Ticket 05: minimum behavioural evidence](issues/05-minimum-behavioural-evidence.md) — retain seven behavioural checks for exact extent, contract reload/failure, vertical controls, stock options, direct geometry, and direct orthophoto rendering.
- [Ticket 06: reviewable commit series](issues/06-reviewable-commit-series.md) — retain the independently compiling contract, decision, integration, correction, and evidence commits; exclude broad-PR subsystems and workflow deletion.

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
