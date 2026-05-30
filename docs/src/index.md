# EulerAP

EulerAP is a compact 2D relaxation-Euler example built around a finite-volume
discretization, with different fluxes, and an implicit backward-Euler time step.

The manual is organized as a small library reference. Start with the exported
types and then move through the flux, boundary-condition, operator, Jacobian,
solver, and statistics sections.

```@meta
CurrentModule = EulerAP
```

```@contents
Pages = [
	"library/types.md",
	"library/fluxes.md",
	"library/boundary_conditions.md",
	"library/operators.md",
	"library/jacobian.md",
	"library/build_problem.md",
	"library/solver.md",
	"library/stats.md",
]
Depth = 1
```
