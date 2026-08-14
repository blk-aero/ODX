# Technical debt

- [ ] Make exact mesh export failure handling atomic. If mesh export fails
  after an exact point-cloud export, restore the stock outputs and keep the
  coordinate-contract state consistent; fail closed if restoration fails.
- [ ] Expand georeferencing coverage with integration tests for the deployed
  PDAL CLI fallback (including GCP VLR serialization), mesh rollback, and
  consistency across point-cloud, mesh, DEM, and orthophoto outputs.
- [ ] Document the vertical-datum semantics restored for GCP Z values versus
  GPS/photo altitudes, including whether finite GCP elevations may be treated
  as WGS84 ellipsoidal heights.
