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
- For an ordinary direct georeferenced job, the canonical reconstruction is
  used when the contract and exact adapters are available. The stock affine
  reconstruction is still published for later consumers; if an exact adapter
  is unavailable, the stage logs a warning and continues with the stock
  export path.
- `--boundary` and `--auto-boundary` remain available. Their filtering keeps
  the stock offset-based behavior; the focused core does not add boundary
  inversion or reject the job. `--crop` and other secondary options likewise
  retain their stock behavior.
- Missing or incompatible exact inputs are best-effort conditions, not new
  pipeline-wide guards. The direct adapter falls back to the stock point,
  mesh, reconstruction, or orthophoto path where possible.

## Answer

Keep the stock secondary paths; do not make them contract-aware. The public
LAZ and textured mesh are the exact geometry boundary, and the orthophoto is
the direct render of that mesh. Existing derivatives may continue to consume
their usual files, but this change neither proves nor labels them as exact.

The one compatibility adapter worth retaining is the existing reconstruction
path: snapshot `reconstruction.topocentric.json` after reconstruction, keep it
authoritative through geometry work when present, and late-publish stock
OpenSfM's affine view at `reconstruction.json`. Restore the canonical file
before any later OpenSfM action when the snapshot exists; otherwise retain the
stock path. Do not add a compatibility-provenance file.

`--boundary` and `--auto-boundary` stay vanilla. They feed the existing
pre-export filter through its affine offset conversion, and the focused core
does not claim to improve selected-area accuracy for those options. Exact
boundary inversion and all secondary-artifact integration remain deferred.
