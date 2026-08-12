# Choose the `--align` boundary

Type: grilling
Status: resolved
Blocked by: 01

## Question

`--align` predates this change, but making it a post-CRS operation across all
public artifacts substantially widens the patch. For corrected single-project
jobs, should the focused change retain exact alignment everywhere, reject it
early with a clear message, or use another explicitly safe boundary? It must
not silently produce disagreeing public artifacts.

## Answer

Do not add post-CRS alignment to the focused direct-georeferencing core.
Until direct exports are wired into production, retain the stock `--align`
path unchanged: the new coordinate-contract module is not reached by a stage,
so a guard now would be dead code, while a parser-level rejection would change
the existing option's behavior.

When the direct-export path is introduced, keep `--align` on the stock path.
The focused contract has no aligned state, so aligned jobs use stock
geocoordinate export and stock artifact handling instead of attempting an
exact direct export. A derived aligned contract, matrix helper, and
cross-artifact alignment protocol remain deferred.

## Acceptance criteria

- Before direct-export integration, `--align` retains the stock parser and
  stage behavior, including its existing unaligned-cloud output.
- The focused `CoordinateContract` has no post-CRS alignment state, matrix, or
  derived-contract API.
- [Ticket 07](07-integrate-direct-exports.md) leaves `--align` available and
  uses stock geocoordinate export for aligned jobs; it does not persist a
  direct contract for that path.
- An aligned job continues through the stock point-cloud, mesh, and
  compatibility paths rather than being rejected by the focused core.
