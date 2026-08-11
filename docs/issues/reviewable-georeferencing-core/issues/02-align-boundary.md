# Choose the `--align` boundary

Type: grilling
Status: open
Blocked by: 01

## Question

`--align` predates this change, but making it a post-CRS operation across all
public artifacts substantially widens the patch. For corrected single-project
jobs, should the focused change retain exact alignment everywhere, reject it
early with a clear message, or use another explicitly safe boundary? It must
not silently produce disagreeing public artifacts.
