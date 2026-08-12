# Set the secondary-artifact compatibility boundary

Type: grilling
Status: resolved
Blocked by: 01

## Question

Which non-core public products should remain on their existing compatibility
path without new coordinate-contract code (for example reports, bounds,
tiles, and DEM-related metadata), and which need a clear fail-closed guard?
The answer must keep the focused exact point/mesh/orthophoto guarantee honest
without carrying broad report, boundary, tile, and provenance integration.

## Acceptance criteria

- The only new direct core outputs are the exact public point cloud and
  textured mesh, plus an orthophoto rendered directly from that validated mesh
  and the persisted contract. The orthophoto makes no exact-raster-mapping
  claim, and no secondary artifact gains an exactness claim.
- Ticket 07 leaves the existing report, GCP/camera, bounds, crop/cutline,
  point-cloud derivative, DEM/COG, raster-tile, glTF/3D-Tile, and copy-output
  consumers on their stock paths. It adds no coordinate-contract tags,
  report fields, derivative manifests, raster evidence, or compatibility
  provenance for them.
- For a direct georeferenced job, the canonical reconstruction remains the
  source for geometry work and direct exports. Only after those core exports
  may Ticket 07 invoke stock OpenSfM geocoordinates export to restore its
  existing affine compatibility reconstruction at the active path for later
  stock consumers. A rerun restores the canonical copy first; it never
  projects the prior compatibility view.
- Ticket 07 rejects a direct georeferenced job using `--boundary` or
  `--auto-boundary` before it resolves or persists a contract or publishes a
  direct public artifact. The current offset-only boundary conversion cannot
  safely select canonical geometry. `--crop` and ordinary secondary options
  otherwise retain their stock behavior.
- Missing or incompatible core inputs fail only at the core boundary: a
  direct export cannot use a missing canonical reconstruction or contract,
  cannot publish a topocentric working mesh as public geometry, and cannot
  invoke the direct orthophoto renderer without a valid contract and nonempty,
  compatible public mesh. Secondary products retain their existing optional
  warning/continuation behavior.

## Answer

Keep the stock secondary paths; do not make them contract-aware. The public
LAZ and textured mesh are the exact geometry boundary, and the orthophoto is
the direct render of that mesh. Existing derivatives may continue to consume
their usual files, but this change neither proves nor labels them as exact.

The one compatibility adapter worth retaining is the existing reconstruction
path: snapshot `reconstruction.topocentric.json` after reconstruction, keep it
authoritative through geometry work, and late-publish stock OpenSfM's affine
view at `reconstruction.json`. Restore the canonical file before any later
OpenSfM action or fail with an instruction to rerun from reconstruction. Do
not add a compatibility-provenance file.

`--boundary` and `--auto-boundary` are the narrow fail-closed exception. They
feed the pre-export filter through an affine offset conversion, so accepting
them on a direct job would make a selected direct artifact appear to honor an
output-CRS boundary when it cannot. Exact boundary inversion and all secondary
artifact integration are deferred.
