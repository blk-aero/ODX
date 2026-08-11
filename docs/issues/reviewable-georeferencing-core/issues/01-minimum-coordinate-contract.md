# Define the minimum exact coordinate contract

Type: grilling
Status: resolved

## Question

What is the smallest persisted coordinate contract and
`opendm.georeferencing` public surface that can correctly support ordinary,
georeferenced single-project exports? Decide which values and operations are
runtime necessities versus provenance, split/merge, alignment, or
future-facing scaffolding.

## Acceptance criteria

- A frozen, versioned contract persists only the canonical topocentric anchor,
  automatic output CRS, exact PROJ operation, XY storage offset, and vertical
  reference state.
- Its public API resolves the ordinary WGS 84 UTM/UPS selection, transforms
  non-empty ENU point arrays exactly in batches, and applies no Z storage
  offset. Absolute consumers keep the default coordinates; mesh consumers can
  explicitly request the existing XY-local storage convention.
- Invalid anchors, offsets, coordinates, output areas, operations, and
  manifests fail with a georeferencing error. Missing, stale, or incompatible
  manifests instruct the operator to rerun from reconstruction.
- Reload validates the exact persisted operation rather than silently
  resolving a replacement.
- Raster warping, alignment, split/merge output selection, control
  normalization, provenance/report/boundary/tile integration, and
  format-specific point-cloud or mesh export remain out of this ticket.

## Answer

`opendm.georeferencing` now exposes `TopocentricAnchor`,
`resolve_coordinate_contract`, `CoordinateContract.transform_points`, and
`load_coordinate_contract`. The contract has exactly five runtime values:
anchor, output CRS WKT, operation, XY offset, and vertical-reference state.
Its JSON schema is intentionally strict so a rerun can only reload the same
contract or fail closed. The direct orthophoto path receives this contract in a
later ticket; it does not introduce a raster coordinate warp here.
