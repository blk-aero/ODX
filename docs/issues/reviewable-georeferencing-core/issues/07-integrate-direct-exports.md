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
- At direct-path entry in `ODMGeoreferencingStage.process`, `--align` fails
  clearly before a contract or any direct public artifact is published.
- The stock path remains unchanged until this direct path is selected.
