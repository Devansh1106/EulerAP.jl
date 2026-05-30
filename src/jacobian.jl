# --- Boundary Condition Logic ---
@inline function neighbor_index(i, j, p, side)
    bcfg = get_bc_config(p)

    if side == :left
        if i == 1
            if isa(bcfg.left, PeriodicBC)
                return cell_index(p.nx, j, p)
            elseif isa(bcfg.left, NeumannBC)
                return cell_index(1, j, p)
            else
                return 0
            end
        end
        return cell_index(i - 1, j, p)

    elseif side == :right
        if i == p.nx
            if isa(bcfg.right, PeriodicBC)
                return cell_index(1, j, p)
            elseif isa(bcfg.right, NeumannBC)
                return cell_index(p.nx, j, p)
            else
                return 0
            end
        end
        return cell_index(i + 1, j, p)

    elseif side == :bottom
        if j == 1
            if isa(bcfg.bottom, PeriodicBC)
                return cell_index(i, p.ny, p)
            elseif isa(bcfg.bottom, NeumannBC)
                return cell_index(i, 1, p)
            else
                return 0
            end
        end
        return cell_index(i, j - 1, p)

    elseif side == :top
        if j == p.ny
            if isa(bcfg.top, PeriodicBC)
                return cell_index(i, 1, p)
            elseif isa(bcfg.top, NeumannBC)
                return cell_index(i, p.ny, p)
            else
                return 0
            end
        end
        return cell_index(i, j + 1, p)
    end

    error("Unknown side")
end


# --- Cache Structure ---
"""
    SparseJacobianCache{T, Ti, TIn, TOut, TJ, TCfg, F}

Lightweight container that holds the pre-built global sparse matrix `J`, a
compact `positions` mapping from local (cell,local-col,row-var) indices to
entries inside `J.nzval`, and pre-allocated buffers used by ForwardDiff.

- `J`: the sparse Jacobian prototype (SparseMatrixCSC) with the correct
    sparsity pattern and writable `nzval` storage.
- `positions`: an Int[3,15,ncells] mapping allowing direct in-place updates
    of `J.nzval` from a 3×15 local Jacobian without searching the global
    structure at runtime.
- `x_cache`, `y_cache`, `Jloc_cache`, `cfg`, `f_closure`: ForwardDiff
    pre-allocations so we can compute each cell's local Jacobian in-place with
    zero (or minimal) allocations.

This structure exists to enable a low-allocation, high-performance local-
AD assembly strategy where the 3×15 local Jacobian for a cell is written
directly into the global matrix via the `positions` lookup.
"""
struct SparseJacobianCache{T, Ti, TIn, TOut, TJ, TCfg, F}
    J::SparseMatrixCSC{T, Ti}
    positions::Array{Int, 3}

    # Pre-allocated buffers for ForwardDiff-based local Jacobian assembly
    x_cache::TIn
    y_cache::TOut
    Jloc_cache::TJ
    cfg::TCfg
    f_closure::F
end

# --- Cache Builder ---
"""
    build_jacobian_cache(p; flux=:rusanov)

Build the sparse Jacobian prototype and cached ForwardDiff buffers used by the
backward-Euler solve.
"""
function build_jacobian_cache(p::RelaxationParams; flux = :rusanov)
    ncells = p.nx * p.ny
    n = 3 * ncells

    # Pre-allocate sparse matrix builder arrays
    I = Int[]
    J_col = Int[]
    sizehint!(I, 45 * ncells)
    sizehint!(J_col, 45 * ncells)

    # Pass 1: Build the global sparsity pattern
    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            neighbors = (
                center, 
                neighbor_index(i, j, p, :left),
                neighbor_index(i, j, p, :right),
                neighbor_index(i, j, p, :bottom),
                neighbor_index(i, j, p, :top)
            )

            for row_var in 0:2
                row = row_var * ncells + center
                for neighbor in neighbors
                    neighbor == 0 && continue
                    for col_var in 0:2
                        col = col_var * ncells + neighbor
                        push!(I, row)
                        push!(J_col, col)
                    end
                end
            end
        end
    end

    J = sparse(I, J_col, ones(Float64, length(I)), n, n)
    J.nzval .= 0.0

    # Pass 2: Map local elements directly to J.nzval indices
    positions = zeros(Int, 3, 15, ncells)
    
    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            neighbors = (
                center, 
                neighbor_index(i, j, p, :left),
                neighbor_index(i, j, p, :right),
                neighbor_index(i, j, p, :bottom),
                neighbor_index(i, j, p, :top)
            )

            for row_var in 0:2
                row = row_var * ncells + center
                for (neighbor_idx, neighbor) in enumerate(neighbors)
                    neighbor == 0 && continue
                    
                    for col_var in 0:2
                        col = col_var * ncells + neighbor
                        local_col_idx = (neighbor_idx - 1) * 3 + col_var + 1
                        
                        col_start = J.colptr[col]
                        col_end = J.colptr[col+1] - 1
                        
                        # Search within the column for the exact row index.
                        # Use findfirst to ensure we only map when an exact entry exists.
                        col_rows = @view(J.rowval[col_start:col_end])
                        rel_idx = findfirst(==(row), col_rows)
                        if rel_idx === nothing
                            # leave as zero (no entry in this column for that row)
                            positions[row_var+1, local_col_idx, center] = 0
                        else
                            positions[row_var+1, local_col_idx, center] = col_start + rel_idx - 1
                        end
                    end
                end
            end
        end
    end

# --- ForwardDiff Pre-allocations ---
    resolved_flux = resolve_flux(flux)
    
    x_cache = zeros(Float64, 15)       # Input (local_u)
    y_cache = zeros(Float64, 3)        # Output (drho, dmx, dmy)
    Jloc_cache = zeros(Float64, 3, 15) # Output Jacobian

    # Create an in-place mutating function: f!(output, input)
    f! = (y, x) -> begin
        drho, dmx, dmy = local_residual(x, p; flux = resolved_flux)
        y[1] = drho
        y[2] = dmx
        y[3] = dmy
        return nothing
    end

    # Pre-allocate the dual numbers locking the Chunk size to 15
    cfg = ForwardDiff.JacobianConfig(f!, y_cache, x_cache, ForwardDiff.Chunk{15}())

    return SparseJacobianCache(J, positions, x_cache, y_cache, Jloc_cache, cfg, f!)
end


"""
    assemble_global_jacobian!(cache, u, p, dt, t; flux=:rusanov)

Assemble the backward-Euler Jacobian `I - dt * dF/du` into the cached sparse
matrix.
"""
function assemble_global_jacobian!(cache::SparseJacobianCache, u, p::RelaxationParams, dt::Float64, t::Float64; flux = :rusanov)
    nz = cache.J.nzval
    positions = cache.positions
    nz .= 0.0 

    ncells = p.nx * p.ny

    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            local_u = gather_local_state(u, i, j, p, t)
            
            # 1. Copy the gathered tuple into our pre-allocated input cache
            for k in 1:15
                cache.x_cache[k] = local_u[k]
            end

            # 2. Compute the local Jacobian IN-PLACE (Zero Allocations!)
            ForwardDiff.jacobian!(cache.Jloc_cache, cache.f_closure, cache.y_cache, cache.x_cache, cache.cfg)

            for row_var in 0:2
                for local_col_idx in 1:15
                    pos = positions[row_var+1, local_col_idx, center]
                    if pos == 0
                        continue
                    end

                    # 3. Use the cached Jacobian
                    val = -dt * cache.Jloc_cache[row_var + 1, local_col_idx]
                    if local_col_idx == row_var + 1
                        val += 1.0
                    end
                    nz[pos] += val
                end
            end
        end
    end
    return cache.J
end