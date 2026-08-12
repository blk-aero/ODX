# Integrate the minimum coordinate contract into direct exports

Type: task
Status: resolved
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
  through geometry work when the snapshot exists. Before any later OpenSfM
  action, rematerialize that canonical file at OpenSfM's active path; never
  intentionally replace it with the prior affine compatibility view. If the
  snapshot is unavailable, keep the stock path available.
- After exact public point cloud and mesh export, invoke stock OpenSfM
  geocoordinates export unchanged to restore its affine compatibility view at
  the established active reconstruction path for later stock consumers. If an
  exact adapter fails, log a warning and keep the stock export path available.
  Do not add compatibility provenance, report fields, or other metadata that
  labels that view or its consumers exact.
- The direct orthophoto uses the canonical reconstruction, persisted contract,
  and public mesh when they are available. If the exact inputs are missing or
  invalid, keep the existing stock reconstruction/georeferencing path instead
  of adding a new fail-closed guard or raster warp.
- `--align`, `--boundary`, and `--auto-boundary` remain available and retain
  stock stage behavior. They do not select the focused exact contract.
- Georeferenced split submodels remain on the stock split path. Do not use
  `outputs["large"]` as a new direct-export signal; split dispatch removes
  `--split` from child processes.
- Leave stock point-cloud derivatives, bounds/crop/cutline handling, DEMs and
  their tiles/COGs, reports and camera/GCP exports, glTF/3D Tiles, and copy
  outputs unchanged. They may use their existing public-file or compatibility
  reconstruction paths, but receive no coordinate-contract integration or
  exactness claim in this ticket.
- The direct path creates no output-selection manifest, shared-observation or
  derivative manifest, and performs no merge validation; those are deferred.
- The stock path remains unchanged until this direct path is selected.

## Outcome

- `ODMOpenSfMStage` snapshots and rematerializes the canonical reconstruction,
  and `ODMMvsTexStage` retains the canonical textured OBJ for exact export.
- `ODMGeoreferencingStage` attempts the Ticket 01 contract for ordinary
  georeferenced jobs, publishes exact LAZ and XY-local textured OBJ geometry
  when those adapters succeed, and otherwise continues with stock exports.
  It then late-publishes only stock OpenSfM's affine reconstruction view.
- `ODMOrthoPhotoStage` takes its resolution from the canonical reconstruction
  and reaches the existing renderer only with the persisted matching contract
  and public mesh containing vertices and faces.
- Contract creation derives the vertical state from the controls OpenSfM
  actually uses: non-checkpoint GCP height, or usable GPS altitude when GPS is
  selected. Vertically uncontrolled jobs persist `unreferenced` and retain
  canonical relative Z.
- Textured OBJ normals use the inverse-transpose local Jacobian at each
  referenced vertex. Shared normal indices expand only where the nonlinear
  operation makes their directions location-dependent; topology, UVs,
  materials, and textures remain intact.
- Canonical textured OBJs are used by the exact adapter when present; a
  missing or invalid copy leaves the stock textured model in place.
- `tests/test_direct_georeferencing.py` exercises stock option availability,
  GCP/GPS/unreferenced vertical selection, streamed point and textured-mesh
  adapters, transformed normals, and direct rendering. Contract resolution
  and reload are exercised in `tests/test_georeferencing.py`.

## Answer

The minimum contract is integrated only at the ordinary georeferenced direct
boundary. Canonical geometry remains authoritative when the exact adapters are
available; the stock affine reconstruction is generated afterward solely for
untouched legacy consumers. Direct orthophoto rendering uses the public mesh
and persisted contract when available, without a raster warp, and otherwise
keeps the stock path. Alignment, boundaries, submodels, and auto-boundaries
remain vanilla. The follow-up persists the existing GCP/GPS vertical-control
signal and transforms normal directions according to the local nonlinear
operation without adding broader validation machinery.

## Follow-up acceptance criteria

- Production selects and persists the correct vertical-reference state rather
  than always defaulting to ellipsoidal height; unreferenced input preserves
  relative Z.
- Public OBJ normals remain valid for the exact transformed surface while UVs,
  faces, materials, and textures remain intact.
- A missing or invalid canonical textured mesh leaves stock mesh and
  compatibility publication available.
