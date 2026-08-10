# Georeferencing release acceptance

Routine validation runs the deterministic `tests.test_georeferencing`,
`tests.test_splitmerge_georeferencing`, and
`tests.test_georeferencing_acceptance` suites. These small fixtures cover the
coordinate/control/hemisphere/polar/antimeridian/alignment/split matrix,
mathematical extents and rejection behavior without requiring images or a
Docker daemon.

Large benchmarks and representative photogrammetry runs are release gates, not
routine CI. Record the 1-million and 10-million point clouds, 100,000 and
1-million vertex meshes, 25 and 100 megapixel rasters, ordinary and
forced-subdivision boundaries, one GCP project, and one GPS-only project using
the JSON schema enforced by `contrib/georeferencing/benchmark.py`. Use the same
idle machine, pinned inputs, options, grids, worker count, and container digests.
Run one warm-up for each variant, then at least three measured runs alternating
stock and corrected. Preserve raw records, commands, peak RSS, output sizes,
counts, throughput, and the generated median summary.

Release thresholds are: normalized export scaling no worse than `2.5x`, total
representative job regression no greater than 10%, and peak RSS or disk growth
no greater than 25% unless investigation proves the difference is within
measured run-to-run noise. Shared-runner timings are informational.

For manual stock-versus-corrected acceptance, process the same known failing
dataset and options against the pinned stock and corrected services. Record
task IDs, image and registered-image counts, dense-point counts, coverage,
GCP/checkpoint residuals, raster resolution and extent, expected artifact
inventory, container digests, and coordinate-contract provenance. Decode the
point cloud, orthophoto, reports, boundaries, and map overlays and compare each
with the surveyed GCP coordinates. Accept only when corrected outputs agree
better without unexplained loss of registered images, points, coverage, or
expected products. Do not report this gate as passed from fixture-only tests.

Reproducibility compares operation selection, decoded coordinates, counts,
topology, CRS WKT, storage offset, vertical-reference status, and alignment metadata. Timestamps,
compression layout, generated identifiers, and byte ordering are deliberately
excluded when decoded behavior remains within the declared representation
tolerance.
