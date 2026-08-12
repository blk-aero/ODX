# Choose the split/merge boundary

Type: grilling
Status: resolved
Blocked by: 01

## Question

Without the new output-selection and derivative-manifest protocol, a
georeferenced split project can merge independently anchored submodels into
one public artifact incorrectly. Should the focused change add an explicit
early rejection for this unsupported combination, or retain some narrower,
safe behavior? Do not keep the new split/merge validation system merely to
avoid making this choice.

## Answer

Do not retain output-selection manifests, shared-observation derivatives, or
merge validation. Until Ticket 07 selects direct exports, the stock split path
remains unchanged.

When Ticket 07 reaches a georeferenced submodel, keep the stock split path.
Submodels do not select the focused direct contract, and stock geocoordinate
export keeps the active reconstruction compatible with the existing merge
flow. `is_submodel` is the real signal: child commands remove `--split`, so
`outputs["large"]` is false in each submodel process.

## Acceptance criteria

- Before Ticket 07, stock split/merge behavior remains unchanged; no direct
  guard is added now.
- Ticket 07 leaves georeferenced submodels on the stock path and does not
  resolve a direct contract for them.
- No output-selection, shared-observation, or derivative manifest is added;
  the existing parent merge remains responsible for split outputs.
- Output-selection propagation and merge validation are deferred to a
  separately scoped split/merge change.
