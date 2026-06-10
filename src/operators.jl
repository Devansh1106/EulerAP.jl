using StaticArrays

"""
    _global_state(u, idx, ncells, Val(NVAR))

Extract the complete multi-variable state vector for a single cell from the component-blocked 
(stacked) global state vector `u`.

This function serves as the data unpacking layer for individual cells. It translates a single flat 
cell index into the memory offsets required to gather all associated conservation and relaxation 
variables into a contiguous, type-stable `SVector`.

# Memory Layout & Indexing Invariant
The global array `u` is structured such that data is clustered by variable rather than contiguously 
by cell. For a system with `NVAR` variables and `ncells` active grid cells, the global vector is divided 
into `NVAR` contiguous blocks of length `ncells`. 

The function unrolls the following index stride mapping at compile time:

    u_local[v] = u[(v - 1) * ncells + idx]  for v = 1, 2, ..., NVAR

# Arguments
- `u::AbstractVector`: The component-blocked global state vector.
- `idx::Int`: The flat, 1D global linear index of the target cell.
- `ncells::Int`: The total number of active interior grid cells (the block stride length).
- `::Val{NVAR}`: A compile-time dispatch value indicating the number of equations/variables per cell.

# Returns
- `SVector{NVAR, eltype(u)}`: A static vector containing all variable states for cell `idx`.
"""
@inline function _global_state(u, 
                               idx::Int, 
                               ncells::Int, 
                               ::Val{NVAR}) where {NVAR}

    return SVector{NVAR}(ntuple(v -> 
                                u[(v - 1) * ncells + idx], 
                                NVAR))
end

"""
    _step_index(I, axis, sign)

Shift a multi-dimensional coordinate index by one unit along a specified spatial axis.

This utility function computes a neighbor's coordinate position in a dimension-agnostic 
manner. It isolates the given coordinate `axis` and adds the directional offset `sign`, 
while leaving all other coordinate components untouched.

# Mathematical Behavior
Given a coordinate vector \$\\mathbf{I} = (x_1, \\dots, x_d, \\dots, x_N)^T\$, targeting 
axis \$d\$ with a directional step \$s \\in \\{-1, 1\\}\$ produces:

    _step_index(\\mathbf{I}, d, s) = (x_1, \\dots, x_d + s, \\dots, x_N)^T

# Arguments
- `I::CartesianIndex{NDIMS}`: The base multi-dimensional coordinate index (e.g., cell center).
- `axis::Int`: The 1-based index of the target coordinate dimension to alter (e.g., 1 for x, 2 for y).
- `sign::Int`: The directional step indicator, typically `-1` (negative direction) or `1` (positive direction).

# Returns
- `CartesianIndex{NDIMS}`: A new coordinate index offset along the designated axis, 
  frequently used to look up adjacent cell states or probe ghost cell regions.
"""
@inline function _step_index(I::CartesianIndex{NDIMS}, 
                             axis::Int, 
                             sign::Int) where {NDIMS}

    coords = Tuple(I)
    return CartesianIndex(ntuple(d -> 
                                 d == axis ? coords[d] + sign : coords[d], 
                                 NDIMS))
end

"""
    implicit_part!(du, u, p::RelaxationParams{NDIMS}, t; resolved_flux::F) where {NDIMS, F}

Assemble the semi-discrete finite-volume spatial operator and stiff relaxation source terms 
into the rate-of-change vector `du`.

This function computes the right-hand side (RHS) of a hyperbolic relaxation system in a 
dimension-agnostic framework. It handles interior numerical fluxes, applies boundary conditions 
via ghost-state extrapolation, and appends algebraic source terms.

# Mathematical Formulation
For each cell \$I\$, the semi-discrete conservation law is evaluated as:

```math
\\frac{d\\mathbf{u}_I}{dt} = - \\sum_{d=1}^{\\text{NDIMS}} \\frac{\\mathbf{F}_{I+1/2, d} - \\mathbf{F}_{I-1/2, d}}{\\Delta x_d} + \\mathbf{S}(\\mathbf{u}_I)

```

where:

* \$\\mathbf{F}_{I\\pm1/2, d}\$ are the interface numerical fluxes computed via the pre-resolved `resolved_flux` object.
* \$\\Delta x_d\$ is the grid spacing along dimension \$d\$.
* \$\\mathbf{S}(\\mathbf{u}_I) = [0, -m_1/\\epsilon, \\dots, -m_d/\\epsilon]^T\$ is the stiff relaxation
source term driving momenta toward equilibrium.

# Memory & Layout Invariants

The function operates on a component-blocked (stacked) vector format where data is clustered
by variable across the total active grid cells \$N_{\text{cells}}\$:

Index Mapping: global_idx = (v - 1) * N_cells + cell_idx

Conservative flux matching is enforced across interior faces by performing a dual-update
(subtracting from the left cell, adding to the right cell) using a single flux evaluation.

# Type Specialization & Performance

By asserting the keyword type parameter `resolved_flux::F`, the Julia compiler generates specialized,
bare-metal machine code for the exact `FluxPair` closure configuration passed down from the
implicit time-stepper. This ensures that the numerical flux calculations are fully inlined
inside the multi-dimensional grid loops with zero function-call overhead.

# Arguments

* `du::AbstractVector`: Destination vector holding the evaluated spatial residuals.
* `u::AbstractVector`: Stacked global state vector input.
* `p::RelaxationParams{NDIMS}`: Structural solver configuration parameters.
* `t::Real`: Current physical simulation time.

# Keyword Arguments

* `resolved_flux::F`: A pre-compiled `FluxPair` object holding the resolved numerical flux functions.
Passing a raw `Symbol` is prohibited to maintain strict type stability.
"""
function implicit_part!(du, 
                        u, 
                        p::RelaxationParams{NDIMS}, 
                        t; 
                        resolved_flux::F) where {NDIMS, F}

    # resolved_flux = resolve_flux(flux)
    _ncells       = ncells(p)
    nvars         = Val(NDIMS + 1)

    fill!(du, 0.0)

    for I in CartesianIndices(p.size)
        idx    = cell_index(I, p)
        center = _global_state(u, idx, _ncells, nvars)
        coords = Tuple(I)

        for axis in 1:NDIMS
            inv_dx = inv(p.dx[axis])

            if coords[axis] == 1
                left_state = get_boundary_state(_step_index(I, axis, -1), 
                                                p, 
                                                u, 
                                                t)

                face_flux  = resolved_flux.flux(left_state, 
                                                center, 
                                                axis, 
                                                p.eps)

                @inbounds for v in 1:(NDIMS + 1) # variables = NDIMS + 1 (rho, mx, my ...)
                    du[(v - 1) * _ncells + idx] += face_flux[v] * inv_dx
                end
            end

            right_idx = neighbor_index(I, p, axis, 1)
            if right_idx != 0
                right_state = right_idx == idx ? 
                              get_boundary_state(_step_index(I, axis, 1), p, u, t) : 
                              _global_state(u, 
                                            right_idx, 
                                            _ncells, 
                                            nvars)

                face_flux = resolved_flux.flux(center, 
                                               right_state, 
                                               axis, 
                                               p.eps)

                @inbounds for v in 1:(NDIMS + 1)
                    du[(v - 1) * _ncells + idx] -= face_flux[v] * inv_dx
                    if right_idx != idx
                        du[(v - 1) * _ncells + right_idx] += face_flux[v] * inv_dx
                    end
                end
            else # Dirichlet case 
                right_state = get_boundary_state(_step_index(I, axis, 1), 
                                                 p, 
                                                 u, 
                                                 t)

                face_flux = resolved_flux.flux(center, 
                                               right_state, 
                                               axis, 
                                               p.eps)

                @inbounds for v in 1:(NDIMS + 1)
                    du[(v - 1) * _ncells + idx] -= face_flux[v] * inv_dx
                end
            end
        end

        # Source term for relaxation
        @inbounds for v in 2:(NDIMS + 1)
            du[(v - 1) * _ncells + idx] -= u[(v - 1) * _ncells + idx] / p.eps
        end
    end

    return nothing
end

"""
    gather_local_state(u, I, p, t=0.0)

Collect and serialize a \$(2 \\cdot \\text{NDIMS} + 1)\$-point spatial stencil around cell `I` 
into a single, type-stable `SVector`.

This function serves as the data extraction layer for cell-by-cell Jacobian assembly and residual 
evaluations. It queries the multi-dimensional neighbor states (including boundary ghost regions) 
and reshapes the local patch data into a contiguous vector structure.

# Mathematical & Memory Layout
For an environment with \$M = \\text{NDIMS} + 1\$ variables, a total of \$B = 2 \\cdot \\text{NDIMS} + 1\$ 
stencil blocks are pulled. The resulting static vector has a total length of \$B \\cdot M\$ and is packed 
contiguously by cell:

    U_stencil = [ u_center(1:M)..., u_left(1:M)..., u_right(1:M)..., u_bottom(1:M)..., u_top(1:M)... ]

Index slicing formulas `div((i - 1), M) + 1` and `(i - 1) % M + 1` are unrolled at compile time 
to map the flat vector index `i` back to its respective block and physical variable identifiers.

# Automatic Differentiation Interaction
When evaluated during an implicit step, the input vector `u` contains `ForwardDiff.Dual` numbers. 
By pulling states through `get_boundary_state`, internal derivative tags are maintained across 
algebraic boundary updates, ensuring correct sensitivity mappings are delivered to the global matrix.

# Arguments
- `u::AbstractVector`: The component-blocked global state vector.
- `I::CartesianIndex{NDIMS}`: The multi-dimensional coordinate index of the target center cell.
- `p::RelaxationParams{NDIMS}`: Structural solver configuration parameters.
- `t::Float64`: Current physical simulation time (defaults to `0.0`).

# Returns
- `SVector{stencil_size, eltype(u)}`: A packed, static vector containing all primitive/relaxation 
  variables within the spatial stencil, where `stencil_size = (2*NDIMS + 1) * (NDIMS + 1)`.
"""
function gather_local_state(u, 
                            I::CartesianIndex{NDIMS}, 
                            p::RelaxationParams{NDIMS}, 
                            t::Float64 = 0.0) where {NDIMS}

    nvars        = NDIMS + 1
    num_blocks   = 2 * NDIMS + 1
    stencil_size = num_blocks * nvars

    # 1. Fetch all states in the stencil (Returns a tuple of SVectors)
    # Using Val(num_blocks) guarantees zero allocations
    blocks = ntuple(Val(num_blocks)) do block
        if block == 1
            return get_boundary_state(I, p, u, t)
        end

        axis = ((block - 2) ÷ 2) + 1
        sign = isodd(block) ? 1 : -1
        return get_boundary_state(_step_index(I, axis, sign), 
                                  p, 
                                  u, 
                                  t)
    end

    # 2. Flatten the tuple of SVectors into a single SVector type-stably
    return SVector{stencil_size, eltype(u)}(ntuple(Val(stencil_size)) do i
        block_idx = div((i - 1), nvars) + 1
        var_idx = (i - 1) % nvars + 1
        return blocks[block_idx][var_idx]
    end)
end

function gather_local_state(u, 
                            i::Int, 
                            j::Int, 
                            p::RelaxationParams{2}, 
                            t::Float64 = 0.0)

    return gather_local_state(u, 
                              CartesianIndex(i, j), 
                              p, 
                              t)
end

"""
    local_residual(local_u, p::RelaxationParams; flux=:rusanov)

Compute the localized algebraic residual vector for a single interior cell using its isolated 
\$(2 \\cdot \\text{NDIMS} + 1)\$-point patch vector.

This function acts as the mathematical core for localized cell-by-cell Automatic Differentiation (AD) 
matrix assembly. By containing the spatial operator math within an isolated patch input, it allows 
differentiation tools (e.g., `ForwardDiff.jl`) to isolate small-scale dense sensitivities without 
touching the global sparse state dimensions.

# Mathematical Formulation
Given the localized block variables containing center, axial left, and axial right values, 
the cell residual vector \$\\mathbf{R}\$ is computed element-by-element across all dimensions as:

    R[v] = - ∑_{d=1}^{NDIMS} \\frac{F_{right, d}[v] - F_{left, d}[v]}{\\Delta x_d} - \\delta_{v} \\frac{u_{center}[v]}{\\epsilon}

where \$\\delta_{v} = 0\$ if \$v = 1\$ (mass conservation is unpenalized) and \$\\delta_{v} = 1\$ 
for all momentum equations (\$v \\ge 2\$).

# Memory & Input Layout
The input vector `local_u` must be flat and contiguously packed by cell blocks, tracking the sequence:

    local_u = [ Center..., Left_x..., Right_x..., Bottom_y..., Top_y... ]

The mapping expression `2 * (axis - 1) + [2 or 3]` is parsed at compile time to automatically 
route indices to their corresponding spatial direction relative to the center cell.

# Arguments
- `local_u::AbstractVector`: A packed local patch vector of length `(2*NDIMS + 1) * (NDIMS + 1)`. 
  Often contains `ForwardDiff.Dual` types during implicit assembly routines.
- `p::RelaxationParams{NDIMS}`: Structural solver configuration parameters.
- `flux::Symbol`: Shortcut identifier for the numerical interface flux formulation (defaults to `:rusanov`).

# Returns
- `SVector{NDIMS + 1, eltype(local_u)}`: The localized residual vector evaluated for the center cell.
"""
function local_residual(local_u::AbstractVector{T}, 
                        p::RelaxationParams{NDIMS}; 
                        flux = :rusanov) where {NDIMS, T}

    resolved_flux = resolve_flux(flux)
    nvars         = NDIMS + 1

    # local_u has layout: 1D vector rho_centre, mx_centre, my_centre, rho_left, mx_l, my_l, right, bottom, top
    @inline block_state(block::Int) = SVector{nvars}(ntuple(v -> 
                                                            local_u[(block - 1) * nvars + v], 
                                                            nvars))

    center = block_state(1)
    residual = zeros(MVector{NDIMS + 1, eltype(local_u)})

    for axis in 1:NDIMS
        left    = block_state(2 * (axis - 1) + 2)
        right   = block_state(2 * (axis - 1) + 3)
        f_left  = resolved_flux.flux(left, 
                                     center, 
                                     axis, 
                                     p.eps)

        f_right = resolved_flux.flux(center, 
                                     right, 
                                     axis, 
                                     p.eps)

        inv_dx  = inv(p.dx[axis])

        @inbounds for v in 1:nvars
            residual[v] -= (f_right[v] - f_left[v]) * inv_dx
        end
    end

    @inbounds for v in 2:nvars
        residual[v] -= center[v] / p.eps
    end

    return SVector{NDIMS + 1, eltype(local_u)}(residual)
end
