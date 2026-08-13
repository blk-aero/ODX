# Georeferencing glossary

- **Canonical topocentric frame** — the persisted metric local ENU frame in
  which ODX reconstructs and processes geometry before publishing it.
- **Coordinate contract** — the immutable, persisted exact mapping from a
  canonical topocentric frame to a selected output CRS, including the storage
  offset and vertical-reference state needed to decode public geometry.
- **Public geometry** — a point cloud or textured mesh published at ODX's
  established output path, expressed through the coordinate contract.
- **Direct orthophoto** — an orthophoto rendered from validated public
  geometry and its coordinate contract; it is not an exact raster warp.
- **Exact export** — evaluating the coordinate contract for the represented
  geometry, rather than applying one scene-wide affine approximation.
