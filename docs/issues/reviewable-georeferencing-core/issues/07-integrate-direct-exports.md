# Integrate the minimum coordinate contract into direct exports

Type: task
Status: open
Blocked by: 01, 02, 03, 04

## Goal

Wire the retained coordinate contract into the ordinary single-project direct
export path, using the boundaries established by tickets 01--04. This is the
first production seam for the focused core; do not add deferred split/merge,
secondary-artifact, or post-CRS-alignment systems with it.

## Acceptance criteria

- The direct path preserves canonical topocentric processing and uses the
  persisted exact contract for its retained public exports.
- A georeferenced direct job snapshots the canonical reconstruction at the
  existing `reconstruction.topocentric.json` path and keeps it authoritative
  through geometry work. Before any later OpenSfM action, rematerialize that
  canonical file at OpenSfM's active path; never use the prior affine
  compatibility view as an input. If the canonical copy is missing or
  incompatible, fail with an instruction to rerun from reconstruction.
- After the exact public point cloud and mesh are exported, invoke stock
  OpenSfM geocoordinates export unchanged to restore its affine compatibility
  view at the established active reconstruction path for later stock
  consumers. Do not add compatibility provenance, report fields, or other
  metadata that labels that view or its consumers exact.
- The direct orthophoto calculates its working resolution from the canonical
  reconstruction and renders only from the persisted contract and nonempty,
  compatible public mesh. Before renderer invocation, fail clearly if either
  prerequisite is missing, unreadable, empty, or incompatible; do not fall
  back to a topocentric mesh or a raster warp.
- At direct-path entry in `ODMGeoreferencingStage.process`, `--align` fails
  clearly before a contract or any direct public artifact is published.
- At that same entry, raise a clear `system.ExitException` before contract
  resolution or direct artifact publication when
  `reconstruction.is_georeferenced() and is_submodel(tree.opensfm)`. Do not
  use `outputs["large"]`: it is false in submodel child processes because
  split dispatch removes `--split`.
- At that same entry, reject `--boundary` and `--auto-boundary` for a
  georeferenced direct job before contract resolution or direct artifact
  publication. The stock offset-only boundary path cannot select canonical
  geometry through the exact operation. Do not add boundary inversion here.
- Leave stock point-cloud derivatives, bounds/crop/cutline handling, DEMs and
  their tiles/COGs, reports and camera/GCP exports, glTF/3D Tiles, and copy
  outputs unchanged. They may use their existing public-file or compatibility
  reconstruction paths, but receive no coordinate-contract integration or
  exactness claim in this ticket.
- The direct path creates no output-selection manifest, shared-observation or
  derivative manifest, and performs no merge validation; those are deferred.
- The stock path remains unchanged until this direct path is selected.
