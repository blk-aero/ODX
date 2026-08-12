# Define the reviewable commit series

Type: grilling
Status: resolved
Blocked by: 05

## Question

How should the resulting focused diff be arranged into independently
buildable, reviewable commits, and which deferrals need explicit follow-up
notes? The answer should leave unrelated workflow deletions out of the series.

## Acceptance criteria

- Retain the current master-based commit order unless a commit boundary breaks
  independent compilation or introduces out-of-scope production behavior.
- Give reviewers a small reading order that pairs the map with the contract,
  groups the three direct-path boundary decisions, and reads direct integration
  with its focused evidence changes.
- Keep the direct orthophoto distinct from the excluded GDAL exact-raster warp.
  Exclude the reference PR's benchmark/release framework, split/merge manifest
  protocol, post-CRS alignment integration, broad secondary-artifact
  integration, and unrelated GitHub-workflow deletion.
- State the remaining environment-backed test limitation without treating an
  unavailable host dependency or container daemon as a passing behavioral run.

## Outcome

No history rewrite is needed. The current order preserves independently
compiling code boundaries and makes the scope reductions reviewable:

1. Read `e221f79` (`Add georeferencing refactor map`) with `144cddf` (`Add
   persisted coordinate contract`). The first supplies the vocabulary and
   boundary; the second adds the standalone contract and its numerical tests.
2. Read `53ed7fe` (`Choose direct export alignment boundary`), `2114fb8`
   (`Define split/merge export boundary`), and `204477c` (`Define secondary
   artifact boundary`) together. They are documentation-only decisions that
   constrain the following integration; they add no dead pre-integration guard.
3. Read `ccc34b9` (`Integrate direct georeferencing exports`) with
   `a7ec757` (`Reduce georeferencing behavioral evidence`), `711465b`
   (`Cover coordinate contract edge cases`), and `ce422dc` (`Fix direct
   georeferencing correctness`). The first is the direct runtime seam; the
   next two remove simulated coverage and retain the specific automatic-CRS
   and validation cases. The correction makes the vertical state live,
   transforms OBJ normals correctly, and fails closed on a missing canonical
   mesh without widening the seam.

`144cddf`, `ccc34b9`, `a7ec757`, `711465b`, and `ce422dc` each compile at
their own commit after their already-introduced dependencies. The intervening
boundary commits are docs-only. Rewriting would not repair a build or scope
boundary; it would only hide the intentional evidence reduction and its
review-driven correctness follow-up, so the series remains as committed.

The read-only broad reference contains work that is deliberately absent here:
the unused GDAL exact-raster warp and its tests; benchmark/release tooling;
the split/merge output-selection and derivative-manifest protocol; a full
post-CRS alignment system; and contract-aware reports, bounds, tiles, DEMs,
and other secondary artifacts. The retained direct orthophoto renders the
validated public mesh and contract, not a raster coordinate warp. Alignment,
boundaries, and georeferenced split submodels instead fail closed at the
direct entry; stock secondary consumers retain their compatibility path. The
reference branch's separate removal of repository GitHub workflows is unrelated
to georeferencing and is not in this series.

## Validation limitation

Static compilation passed for every code-bearing commit. A temporary pinned
environment ran the two contract tests and six native-independent direct
tests; the real LAZ test skips explicitly because host PDAL Python bindings
are absent. The available Docker CLI also has no running daemon, so the
project container could not supply those bindings. The real LAZ check must
still run in that project container.

## Answer

Keep the commits in this order. The tracker records the review-driven
correction after the original closure instead of hiding it in a history
rewrite. The series remains the smallest scope-preserving history: contract,
decisions, direct integration, correction, and behavioral evidence remain
distinct, and no broad-PR subsystem or workflow deletion leaks into it.
