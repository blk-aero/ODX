# Define the minimum behavioural evidence

Type: grilling
Status: resolved
Blocked by: 01, 02, 03, 04, 07

## Question

What is the smallest set of real behavioural tests that establishes the
focused guarantee: non-affine extent, exact point and mesh export, direct
orthophoto prerequisites, and contract reload/failure? Remove source-text
checks, tests that orchestrate other tests, and simulated lineage checks that
do not observe retained production seams.

## Acceptance criteria

- A long ENU extent is compared with an independently evaluated PROJ
  transformation, so a scene-wide affine approximation cannot pass; the same
  check covers automatic UTM plus north/south UPS selection and rejects
  non-finite and out-of-area coordinates.
- A persisted coordinate contract reloads unchanged, preserving its
  unreferenced vertical state and relative Z; missing or changed manifests
  fail with the instruction to rerun from reconstruction.
- Real PDAL in the project container writes and decodes exact public LAZ
  coordinates; the mesh export writes exact XY-local vertices while retaining
  its OBJ material, UV, and face structure.
- The direct stage keeps `--align`, `--boundary`, `--auto-boundary`, and
  georeferenced split submodels on stock behavior; ordinary jobs persist the
  vertical state exposed by the selected GCP or GPS controls and preserve
  relative Z when that state is unreferenced.
- Exact point, mesh, and orthophoto adapters are best effort. Missing or
  invalid exact inputs log a warning and leave the stock export path available.
- When the direct prerequisites are available, orthophoto rendering uses the
  persisted contract and public mesh; otherwise it uses the stock
  reconstruction/georeferencing inputs.
- The suite contains no source-text checks, test-running-test orchestration,
  fake cross-artifact lineage, or replacement coordinate math.

## Outcome

`tests/test_georeferencing.py` now contains the exact-extent, automatic
UTM/UPS selection, validation, and persisted-contract checks.
`tests/test_direct_georeferencing.py` contains the stock-option matrix,
vertical selection, real-PDAL LAZ, exact visual mesh and normals, and direct
orthophoto tests. The LAZ test skips explicitly when PDAL's Python bindings
are unavailable outside the project container.

## Answer

Seven focused tests cover the retained core only: two contract tests and five
direct-export tests. They replace duplicate stage-contract checks, the fake
PDAL stream, simulated compatibility-publication lifecycle coverage, and
fail-closed option tests that no longer describe the vanilla stock path.
