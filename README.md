# EulerAP — README

This repository contains a compact 2D relaxation-Euler example implemented in
`relaxation_euler2d.jl`. The script is intentionally minimal and uses a
finite-volume spatial discretization on the periodic domain $[0,1]\times[0,1]$.

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
right-hand-side relaxation terms become dominant — motivating implicit
treatment of the stiff terms.

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

 - `implicit_part!(du, u, p, t)` — builds the finite-volume flux-divergence
   contributions (both x and y interfaces) and accumulates them into `du`.
   This function is also used inside the backward-Euler residual evaluation
   so the stiff relaxation source for the momentum components is accounted
   for when assembling the residual.

 - `initial_condition(x,y)` — returns the initial `(rho, mx, my)` at a point.
   The example uses smooth sin/cos perturbations so the solver exercise is
   numerically well-behaved.

 - `build_problem(; nx, ny, eps, tspan)` — assembles the flattened initial
   condition vector `u0`, coordinate vectors `x`, `y`, constructs the
   `RelaxationParams` (`p`) and a sparse Jacobian prototype (`jac_prototype`)
   and returns `(u0, x, y, p, jac_prototype)`.

## Time integrator setup

The implementation uses an implicit backward-Euler time-stepping handled via
the `NonlinearSolve` ecosystem. The function `solve_backward_euler` builds a
`NonlinearProblem` around the residual implemented in
`backward_euler_residual!` and solves each implicit step with a
Newton-style method (`NewtonRaphson`) using `AutoForwardDiff` for Jacobian
information. A fixed step size `dt` is used by default; the solver advances
until the final time in `tspan`.

## Jacobian Matrix Layouts for 2D (3-Variable System)

## 1. Baseline: Single Variable (Scalar 2D)
If $U$ is just a single scalar (e.g., density $\rho$), grid has $N = N_x \times N_y$ total cells.
*   **State Vector:** $\mathbf{U} = [\rho_1, \rho_2, \dots, \rho_N]^T$
*   **Stencil Impact:** Because of the 2D 5-point stencil (Center, Left, Right, Bottom, Top), the Jacobian is an $N \times N$ matrix with exactly **5 non-zero bands**:
    *   **Main diagonal:** The cell itself ($C$).
    *   **Two adjacent off-diagonals:** Left ($L$) and Right ($R$) neighbors.
    *   **Two far off-diagonals:** Bottom ($B$) and Top ($T$) neighbors, located $\pm N_x$ rows away.

This 5-banded structure is often referred to as a **block-tridiagonal matrix** (where the blocks are $N_x \times N_x$ matrices representing entire rows of the grid).

---

## 2. 3 Variables
Now, $U$ represents 3 quantities: $[\rho, m_x, m_y]$. Total number of unknowns is $3N$. The Jacobian becomes $3N \times 3N$ matrix. They way those 5 original bands map into this matrix depends on the memory layout.

### Layout 1: Interleaved (Array of Structs)
Memory is ordered cell-by-cell:
$$ \mathbf{U} = [\rho_1, m_{x1}, m_{y1}, \rho_2, m_{x2}, m_{y2}, \dots, \rho_N, m_{xN}, m_{yN}]^T $$

*   **Structure:** The matrix looks exactly like the scalar block-tridiagonal matrix, but every single scalar entry **"inflates"** into a dense $3 \times 3$ block.
*   **Mapping:** Where the scalar matrix had a scalar linking cell $c$ to its right neighbor $r$, the interleaved matrix has a $3 \times 3$ dense block mapping the 3 equations of cell $c$ to the 3 variables of cell $r$:

$$ \text{Scalar Entry} \rightarrow \begin{bmatrix} \frac{\partial R^\rho}{\partial \rho} & \frac{\partial R^\rho}{\partial m_x} & \frac{\partial R^\rho}{\partial m_y} \\[6pt] \frac{\partial R^{m_x}}{\partial \rho} & \frac{\partial R^{m_x}}{\partial m_x} & \frac{\partial R^{m_x}}{\partial m_y} \\[6pt] \frac{\partial R^{m_y}}{\partial \rho} & \frac{\partial R^{m_y}}{\partial m_x} & \frac{\partial R^{m_y}}{\partial m_y} \end{bmatrix} $$

*   **Result:** A pentadiagonal/block-tridiagonal matrix where the bands are 3 elements thick.

### Layout 2: Stacked (Struct of Arrays) — *Used in code*
Memory is ordered variable-by-variable (grouping all densities, then all x-momentums, then all y-momentums):
$$ \mathbf{U} = [\rho_1 \dots \rho_N, \;\; m_{x1} \dots m_{xN}, \;\; m_{y1} \dots m_{yN}]^T $$

*   **Structure:** The $3 \times 3$ blocks are shattered. The entire Jacobian is partitioned into a $3 \times 3$ grid of $N \times N$ sub-matrices:

$$ \mathbf{J} = \begin{bmatrix} \mathbf{J}_{\rho, \rho} & \mathbf{J}_{\rho, m_x} & \mathbf{J}_{\rho, m_y} \\[6pt] \mathbf{J}_{m_x, \rho} & \mathbf{J}_{m_x, m_x} & \mathbf{J}_{m_x, m_y} \\[6pt] \mathbf{J}_{m_y, \rho} & \mathbf{J}_{m_y, m_x} & \mathbf{J}_{m_y, m_y} \end{bmatrix} $$

*   **Sparsity Benefit:** Every single one of those 9 sub-matrices ($\mathbf{J}_{\rho,\rho}$, $\mathbf{J}_{\rho,m_x}$, etc.) shares the **exact same 5-banded pentadiagonal sparsity pattern** as the single-variable scalar case.
*   **Example:** $\mathbf{J}_{m_x, \rho}$ is an $N \times N$ pentadiagonal matrix describing how the x-momentum equation in every cell reacts to changes in density in neighboring cells.

---

## 3. Our `jac_prototype` Code
Our code uses the **Stacked Layout** ($u = [\rho; m_x; m_y]$), we must explicitly build that $3 \times 3$ grid of sub-matrices using loop offsets:

```julia
row = row_var * ncells + cell 
col = col_var * ncells + neighbor
```

*   `cell` and `neighbor` trace out the standard 5-banded pentadiagonal structure within an $N \times N$ space.
*   `row_var * ncells` shifts the row index down into the correct equation block (e.g., row block 0 for $\rho$, block 1 for $m_x$).
*   `col_var * ncells` shifts the column index right into the correct variable block (e.g., col block 0 for $\rho$, block 1 for $m_x$).

**Summary:** Moving from 1 to 3 variables either **inflates the bands** to be 3-elements thick (Interleaved) or **duplicates the banded structure** into a $3 \times 3$ grid of identical sparsity patterns (Stacked).


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

- Implementation and function definitions: [relaxation_euler2d.jl](relaxation_euler2d.jl)
- Top-level runner / defaults are in the same file; edit `build_problem` or
  the call-site near the bottom of `relaxation_euler2d.jl` to change grid
  size, `eps`, or final time.

## Quick run (from repo root)

```bash
julia --project=. relaxation_euler2d.jl
```

## References

- Randall J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge University Press, 2002.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  3rd edition, Springer, 2009.

# EulerAP.jl
