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

When Ticket 07 selects a direct export for a georeferenced reconstruction,
reject a submodel at the first direct-path entry in
`ODMGeoreferencingStage.process` when
`reconstruction.is_georeferenced() and is_submodel(tree.opensfm)`. The guard
must fail clearly before resolving or persisting a coordinate contract or
publishing a direct artifact. `is_submodel` is the real signal: child commands
remove `--split`, so `outputs["large"]` is false in each submodel process.
This leaves ordinary single-project jobs and the pre-integration stock split
path alone, while ensuring the parent never merges independently anchored
direct outputs.

## Acceptance criteria

- Before Ticket 07, stock split/merge behavior remains unchanged; no direct
  guard is added now.
- Ticket 07 raises a clear `system.ExitException` before contract resolution
  or direct artifact publication when
  `reconstruction.is_georeferenced() and is_submodel(tree.opensfm)`.
- The rejection creates no output-selection manifest, shared-observation or
  derivative manifest, direct contract, or direct public artifact, and the
  parent merge does not combine local frames.
- Output-selection propagation and merge validation are deferred to a
  separately scoped split/merge change.
