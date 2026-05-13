# EulerAP — README

This repository contains a compact 2D relaxation-Euler example implemented in
`relaxation_euler_2d.jl` using ClimaTimeSteppers for IMEX integration. The
script is intentionally minimal and uses a finite-volume spatial discretization
on the periodic domain $[0,1]\times[0,1]$.

## PDE / model

The code solves a relaxation-style Euler system in conservative variables
$(\rho, m_x, m_y)$ where $m_x=\rho u_x$, $m_y=\rho u_y$ are the momentum
densities. The PDE in dimensional form used in the code is

$$
\begin{aligned}
\rho_t + \partial_x m_x + \partial_y m_y &= 0, \\
m_{x,t} + \partial_x\left(\frac{m_x^2}{\rho}+\frac{\rho}{\varepsilon}\right) + \partial_y\left(\frac{m_x m_y}{\rho}\right) &= -\frac{m_x}{\varepsilon}, \\
m_{y,t} + \partial_x\left(\frac{m_x m_y}{\rho}\right) + \partial_y\left(\frac{m_y^2}{\rho}+\frac{\rho}{\varepsilon}\right) &= -\frac{m_y}{\varepsilon}.
\end{aligned}
$$

Here $\varepsilon$ is the relaxation parameter (named `eps` in the code). As
$\varepsilon\to 0$ the pressure term $\rho/\varepsilon$ becomes stiff and the
right-hand-side relaxation terms become dominant — motivating IMEX splitting.

The mixed terms  
$$
\partial_y\left(\frac{m_x m_y}{\rho}\right)
\quad\text{and}\quad
\partial_x\left(\frac{m_x m_y}{\rho}\right)
$$  

appear only in 2D because momentum is transported in both coordinate
directions. In 1D, all $\partial_y(\cdot)$ terms vanish, so these cross-fluxes
do not appear.

## Main parameters (in code)

- `RelaxationParams` — struct holding PDE/mesh parameters:
  - `eps` (Float64): relaxation parameter $\varepsilon$.
  - `nx` (Int): number of cells in the x-direction.
  - `ny` (Int): number of cells in the y-direction.
  - `dx` (Float64): cell width in x (usually `1.0/nx`).
  - `dy` (Float64): cell height in y (usually `1.0/ny`).

These parameters are created by `build_problem(; nx, ny, eps, tspan)` and stored
in the `prob` object returned.

## Key functions and what they do

- `cell_index(i,j,p::RelaxationParams)` — converts 2D indices `(i,j)` to a
  global cell index (1-based). Used to index flattened arrays of length
  `nx*ny`.

- `rusanov_flux_x(...)` and `rusanov_flux_y(...)` — compute a Rusanov (local
  Lax–Friedrichs) numerical flux across a vertical (`_x`) or horizontal
  (`_y`) interface. The formula used is

  $$F_{\text{num}} = \tfrac12(F_L+F_R) - \tfrac12\alpha (U_R-U_L)$$

  where $\alpha$ is a local maximum wave speed (estimated as
  $|u|+c$, with $c=\sqrt{1/\varepsilon}$ in the model) and $F$ are the
  physical flux components for each conserved field.

- `explicit_part!(du, u, p, t)` — builds the explicit spatial flux
  divergence contribution (finite-volume flux differences) for all cells and
  stores it in `du`. It implements periodic boundary conditions by wrapping
  indices when computing neighbor fluxes in x and y directions.

- `implicit_part!(du, u, p, t)` — evaluates the stiff relaxation source
  terms. For this model the only implicit terms are the momentum relaxation
  terms, implemented as

  $$\partial_t m_x = -m_x/\varepsilon,\qquad \partial_t m_y = -m_y/\varepsilon,$$

  while the density equation has no implicit source here.

- `wfact!(w,u,p,dtgamma,t)` — fills the matrix/factor (`Wfact`) that the
  ClimaTimeSteppers Newton solver uses during implicit stage solves. The
  function is written to accept either a `SparseMatrixCSC` or a `Diagonal`
  matrix prototype and sets the diagonal entries appropriate for the linear
  part of the implicit operator (it places `-1` in density diagonal entries
  and `-(1 + dtgamma/eps)` for momentum entries; `dtgamma` is the method
  coefficient). This helps the Newton linear solver reuse/initialize factor
  structure efficiently.

- `initial_condition(x,y)` — returns the initial `(rho, mx, my)` at a point.
  The example uses smooth sin/cos perturbations so the solver exercise is
  numerically well-behaved.

- `build_problem(; nx, ny, eps, tspan)` — assembles the flattened initial
  condition vector `u0`, constructs a Jacobian prototype (`Diagonal` by
  default for efficiency), constructs `CTS.ODEFunction` and `CTS.ClimaODEFunction`
  wrappers and returns `(prob, x, y, p)` where `prob` is a `ClimaTimeSteppers`
  `ODEProblem` ready to solve.

## Time integrator setup

- The code constructs an IMEX algorithm using `CTS.IMEXAlgorithm(CTS.ARS343(),
  CTS.NewtonsMethod(...))`. This picks an ARS(3,4,3)-type IMEX scheme for
  time-discretization; the stiff substep is solved with a Newton method
  (`NewtonsMethod`). The Jacobian template (`jac_prototype`) and `Wfact`
  are provided so ClimaTimeSteppers can prepare linear solvers efficiently.

## I/O / plotting

- After the solve the code reshapes the flattened solution into grids and
  writes three PNG files in the working directory:
  - `rho_final_2d.png` — final density heatmap
  - `ux_final_2d.png` — final x-velocity heatmap (computed as `mx ./ rho`)
  - `uy_final_2d.png` — final y-velocity heatmap (computed as `my ./ rho`)

## How momentum / mass relate (units and recovery)

- The code stores conserved variables as densities per unit area (volume):
  - density: $\rho$ (mass per unit area),
  - momentum density: $m_x=\rho u_x$, $m_y=\rho u_y$.

- To recover velocities use `u_x = m_x / \rho` and `u_y = m_y / \rho`.

- Per-cell quantities (if you need them) are obtained by multiplying by cell
  area `V = dx*dy`: cell mass $M = \rho V$, cell momentum
  $P_x = m_x V = M u_x$.

## Where to look in the source

- Implementation and function definitions: [relaxation_euler_2d.jl](relaxation_euler_2d.jl)
- Top-level runner / defaults are in the same file; edit `build_problem` or
  the call-site near the bottom of `relaxation_euler_2d.jl` to change grid
  size, `eps`, or final time.

## Quick run (from repo root)

```bash
julia --project=. relaxation_euler_2d.jl
```

## References

- Randall J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge University Press, 2002.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd edition, Springer, 2009.

# EulerAP.jl
