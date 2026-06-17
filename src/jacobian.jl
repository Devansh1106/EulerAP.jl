"""
    _local_stencil_indices(I, p)

Generate a compile-time unrolled `NTuple` containing the flat global indices of a 
dimension-agnostic \$(2 \\cdot \\text{NDIMS} + 1)\$-point stencil centered at `I`.

This function maps a linear sequence of `block` IDs to a structured spatial coordinate 
stencil. The mapping is dimension-agnostic and relies on integer arithmetic to alter 
coordinate axes sequentially.

# Block Mapping Rules
The returned tuple orders the global linear memory offsets as follows:
- `block = 1`: The **Center** cell index itself.
- `block = 2`: **Left** neighbor (axis 1, step -1) -> \$(i-1, j, \\dots)\$
- `block = 3`: **Right** neighbor (axis 1, step +1) -> \$(i+1, j, \\dots)\$
- `block = 4`: **Bottom** neighbor (axis 2, step -1) -> \$(i, j-1, \\dots)\$
- `block = 5`: **Top** neighbor (axis 2, step +1) -> \$(i, j+1, \\dots)\$
- `block = 2*d / 2*d+1`: Alternating negative and positive neighbor steps along axis \$d\$.

# Arguments
- `I::CartesianIndex{NDIMS}`: The active interior cell's multi-dimensional coordinate index.
- `p::RelaxationParams{NDIMS}`: The structural solver configuration parameters.

# Returns
- `NTuple{2*NDIMS + 1, Int}`: A type-stable tuple of the flat, 1D global positions of the 
  stencil cluster, ready to be mapped straight into the global Jacobian assembly slots.
"""

@inline function _local_stencil_indices(I::CartesianIndex{NDIMS}, 
                                        p::RelaxationParams{NDIMS}) where {NDIMS}
    return ntuple(block -> begin
        if block == 1
            return cell_index(I, p)
        end

        axis = div((block - 2), 2) + 1
        sign = isodd(block) ? 1 : -1
        return neighbor_index(I, p, axis, sign)
    end, 2 * NDIMS + 1)
end

"""
    SparseJacobianCache{T, Ti, TIn, TOut, TJ, TCfg, F}

Lightweight container that holds the pre-built global sparse matrix `J`, a
compact `positions` mapping from local (cell,local-col,row-var) indices to
entries inside `J.nzval`, and pre-allocated buffers used by ForwardDiff.

- `J`: the sparse Jacobian prototype (SparseMatrixCSC) with the correct
    sparsity pattern and writable `nzval` storage.
- `positions`: an Int[3,15,_ncells] mapping allowing direct in-place updates
    of `J.nzval` from a 3×15 local Jacobian without searching the global
    structure at runtime.
- `x_cache`, `y_cache`, `Jloc_cache`, `cfg`, `f_closure`: ForwardDiff
    pre-allocations so we can compute each cell's local Jacobian in-place with
    zero (or minimal) allocations.

This structure exists to enable a low-allocation, high-performance local-
AD assembly strategy where the 3×15 local Jacobian for a cell is written
directly into the global matrix via the `positions` lookup.
"""
struct SparseJacobianCache{T, Ti, TIn, TOut, TJ, TCfg, F, FFlux}
    J::SparseMatrixCSC{T, Ti}
    positions::Array{Int, 3}

    # Pre-allocated buffers for ForwardDiff-based local Jacobian assembly
    x_cache::TIn
    y_cache::TOut
    Jloc_cache::TJ
    cfg::TCfg
    f_closure::F
    resolved_flux::FFlux
end

# --- Cache Builder ---
"""
    build_jacobian_cache(p; resolved_flux)

Build the sparse Jacobian prototype and cached ForwardDiff buffers used by the
backward-Euler solve.
"""
function build_jacobian_cache(p::RelaxationParams{NDIMS}; 
                              resolved_flux::FluxPair) where {NDIMS}

    _ncells      = ncells(p)
    nvars        = NDIMS + 1
    stencil_size = (2 * NDIMS + 1) * nvars
    n            = nvars * _ncells

    # Pre-allocate sparse matrix builder arrays
    I     = Int[]
    J_col = Int[]

    sizehint!(I, 
              stencil_size * nvars * _ncells)
    sizehint!(J_col, 
              stencil_size * nvars * _ncells)

    # Pass 1: Build the global sparsity pattern
    for Icell in CartesianIndices(p.size)
        center    = cell_index(Icell, p)
        neighbors = _local_stencil_indices(Icell, p)

        for row_var in 0:(nvars - 1)
            row = row_var * _ncells + center
            for neighbor in neighbors
                neighbor == 0 && continue
                for col_var in 0:(nvars - 1)
                    col = col_var * _ncells + neighbor
                    push!(I, row)
                    push!(J_col, col)
                end
            end
        end
    end

    J        = sparse(I, J_col, ones(Float64, length(I)), n, n)
    J.nzval .= 0.0

    # Pass 2: Map local elements directly to J.nzval indices
    positions = zeros(Int, 
                      nvars, 
                      stencil_size, 
                      _ncells)

    for Icell in CartesianIndices(p.size)
        center    = cell_index(Icell, p)
        neighbors = _local_stencil_indices(Icell, p)

        for row_var in 0:(nvars - 1)
            row = row_var * _ncells + center
            for (neighbor_idx, neighbor) in enumerate(neighbors)
                neighbor == 0 && continue

                for col_var in 0:(nvars - 1)
                    col           = col_var * _ncells + neighbor
                    local_col_idx = (neighbor_idx - 1) * nvars + col_var + 1

                    col_start = J.colptr[col]
                    col_end   = J.colptr[col + 1] - 1

                    col_rows = @view(J.rowval[col_start:col_end])
                    rel_idx  = findfirst(==(row), col_rows)
                    if rel_idx === nothing
                        positions[row_var + 1, local_col_idx, center] = 0
                    else
                        positions[row_var + 1, local_col_idx, center] = col_start + rel_idx - 1
                    end
                end
            end
        end
    end

# --- ForwardDiff Pre-allocations ---
    x_cache    = zeros(Float64, stencil_size)              # Used for Input (i.e. local_u)
    y_cache    = zeros(Float64, nvars)                     # Used for Output (i.e. drho, dmy)
    Jloc_cache = zeros(Float64, nvars, stencil_size)    # Used for Output (i.e. Jacobian)

    # Create an in-place mutating function: f!(output, input)
    f! = (y, x) -> begin
        res = local_residual(x, 
                             p; 
                             resolved_flux = resolved_flux)
        for v in 1:nvars
            y[v] = res[v]
        end
        return nothing
    end

    cfg = ForwardDiff.JacobianConfig(f!, 
                                     y_cache, 
                                     x_cache, 
                                     ForwardDiff.Chunk(x_cache))

    return SparseJacobianCache(J, 
                               positions, 
                               x_cache, 
                               y_cache, 
                               Jloc_cache, 
                               cfg, 
                               f!,
                               resolved_flux)
end


"""
    assemble_global_jacobian!(cache, u, p, dt, t)

Assemble the backward-Euler Jacobian `I - dt * dF/du` into the cached sparse
matrix.
"""
function assemble_global_jacobian!(cache::SparseJacobianCache, 
                                   u, 
                                   p::RelaxationParams{NDIMS}, 
                                   dt::Float64, 
                                   t::Float64) where {NDIMS}
    nz        = cache.J.nzval
    positions = cache.positions
    nz       .= 0.0 

    _ncells = ncells(p)
    nvars   = NDIMS + 1

    for Icell in CartesianIndices(p.size)
        center  = cell_index(Icell, p)
        local_u = gather_local_state(u, Icell, p, t)

        for k in 1:length(cache.x_cache)
            cache.x_cache[k] = local_u[k]
        end

        ForwardDiff.jacobian!(cache.Jloc_cache, 
                              cache.f_closure, 
                              cache.y_cache, 
                              cache.x_cache, 
                              cache.cfg)

        for row_var in 0:(nvars - 1)
            for local_col_idx in 1:length(cache.x_cache)
                pos = positions[row_var + 1, 
                                local_col_idx, 
                                center]
                if pos == 0
                    continue
                end

                val = -dt * cache.Jloc_cache[row_var + 1, local_col_idx]
                if local_col_idx == row_var + 1
                    val += 1.0
                end
                nz[pos] += val
            end
        end
    end
    return cache.J
end