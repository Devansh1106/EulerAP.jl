# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function stencil_indices(I::CartesianIndex{1},
                                 semi::AbstractSemidiscretization)

    center = I

    # (I, semi, axis, sign); `x`: axis = 1
    left   = neighbor_index(I, semi, 1, -1)
    right  = neighbor_index(I, semi, 1, 1)

    return (center, left, right)
end

function neighbor_index(I::CartesianIndex{1},
                        semi::AbstractSemidiscretization,
                        axis::Int,
                        sign::Int)
    
    @assert axis == 1
    shifted = I[1] + sign

    # Interior cell
    if 1 <= shifted && shifted <= size(semi.mesh, 1)
        return CartesianIndex(shifted)
    end

    nx = size(semi.mesh, 1)
    bc = _side_bc(semi.boundary_conditions,
                  axis,
                  sign)

    if isa(bc, PeriodicBC)
        wrapped = _wrap_index(shifted, nx)
        return CartesianIndex(wrapped)

    elseif bc isa NeumannBC || bc isa ExtrapolateBC
        # Ghost state depends on interior state
        return CartesianIndex(clamp(shifted, 1, nx))

    elseif bc isa DirichletBC
        # Ghost state independent of interior state
        return CartesianIndex(0)

    else
        error("Unknown boundary condition type $(typeof(bc))")
    end 
end

@inline function _side_bc(bcfg,
                          axis::Int,
                          sign::Int)

    @assert axis == 1
    return sign < 0 ? bcfg.left : bcfg.right
end

@inline function _wrap_index(i::Int, nx::Int)
    return mod1(i, nx)
end

@inline function _state_at(u::AbstractVector,
                           I::CartesianIndex{1},
                           semi::AbstractSemidiscretization)

    nvars = nvariables(semi.equations)
    cell  = cell_index(I, semi)

    return SVector{nvars}(ntuple(v ->
                                 u[global_dof(cell, v, nvars)],
                                 nvars))
end

@inline function boundary_side(I::CartesianIndex{1},
                               semi::AbstractSemidiscretization)

    i = I[1]
    if i < 1
        return :left
    elseif i > size(semi.mesh, 1)
        return :right
    else
        return nothing
    end
end

function rhs!(du, u,
              solver::FVSolver{1, TFlux},
              semi::SemidiscretizationHyperbolic,
              t) where {TFlux}

    cache = semi.cache
    fill!(du, zero(eltype(du)))
    nvars = nvariables(semi.equations)
    for I in eachcell(semi.mesh)
        gather_local_state!(cache.x_cache,
                            u,
                            I,
                            semi,
                            t)

        local_residual!(cache.residual_buffer,
                        cache.x_cache,
                        semi)
        cell = cell_index(I, semi)

        @inbounds for v in 1:nvars
            du[global_dof(cell, v, nvars)] = cache.residual_buffer[v]
        end
    end

    return nothing
end

@inline function gather_local_state!(x, u, I::CartesianIndex{1},
                                     semi::AbstractSemidiscretization,
                                     t)

    nvars = nvariables(semi.equations)
    center, left, right = stencil_indices(I, semi)

    offset = 0

    for cell in (center, left, right)
        state = cell_state(u, cell, semi, t)

        @inbounds for v in 1:nvars
            x[offset + v] = state[v]
        end
        offset += nvars
    end

    return nothing
end

@inline function cell_state(u, I::CartesianIndex{1},
                            semi::AbstractSemidiscretization,
                            t)

    nx = size(semi.mesh, 1)

    # --------------------------------------------------
    # Interior cell
    # --------------------------------------------------
    if 1 <= I[1] <= nx
        return extract_cell_state(u, I, semi)
    end

    # --------------------------------------------------
    # Ghost cell
    # --------------------------------------------------

    side = boundary_side(I, semi)

    bc = side === :left ?
         semi.boundary_conditions.left :
         semi.boundary_conditions.right

    return apply_bc(bc, u, I, semi, t)
end

@inline function extract_cell_state(u, I::CartesianIndex{1},
                                    semi::AbstractSemidiscretization)

    nvars = nvariables(semi.equations)
    cell  = cell_index(I, semi)

    return SVector{nvars}(ntuple(v -> 
                                 u[global_dof(cell, v, nvars)],
                                 nvars))
end

@inline function local_residual!(y, x, semi::AbstractSemidiscretization)

    equations = semi.equations
    source_terms = semi.source_terms
    flux = semi.solver.flux

    # --------------------------------------------------
    # Extract stencil states
    # --------------------------------------------------

    u_center = local_state(x, 1, equations)
    u_left   = local_state(x, 2, equations)
    u_right  = local_state(x, 3, equations)

    # --------------------------------------------------
    # Numerical fluxes
    # --------------------------------------------------

    flux_left = flux(u_left, u_center, 1, equations)

    flux_right = flux(u_center, u_right, 1, equations)

    dx = semi.mesh.dx[1]

    # --------------------------------------------------
    # FV divergence
    # --------------------------------------------------

    @inbounds for v in eachindex(y)
        y[v] = -(flux_right[v] - flux_left[v]) / dx
    end

    # --------------------------------------------------
    # Source terms
    # --------------------------------------------------

    if source_terms !== nothing
        src = source_terms(u_center,
                           equations)

        @inbounds for v in eachindex(y)
            y[v] += src[v]
        end
    end

    return nothing
end

@inline function local_state(x, local_cell::Int, equations)

    nvars = nvariables(equations)
    first = (local_cell - 1) * nvars + 1
    last  = local_cell * nvars

    return SVector{nvars}(@view x[first:last])
end

@inline function apply_bc(bc::PeriodicBC,
                          u,
                          I,
                          semi,
                          t)

    error("PeriodicBC should never reach apply_bc")
end

@inline function apply_bc(bc::ExtrapolateBC,
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    interior_i = clamp(I[1],
                       1,
                       size(semi.mesh,1))

    return extract_cell_state(u, CartesianIndex(interior_i), semi)
end

@inline function apply_bc(bc::DirichletBC,
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    x = coordinates(I, semi.mesh)

    return bc.boundary_value(x, t, semi.equations)
end

@inline function apply_bc(bc::NeumannBC,
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    interior_i = clamp(I[1],
                       1,
                       size(semi.mesh,1))

    interior = extract_cell_state(u,
                                  CartesianIndex(interior_i),
                                  semi)

    grad = bc.boundary_gradient(coordinates(I, semi.mesh),
                                t,
                                semi.equations)

    dx = semi.mesh.dx[1]

    return interior + dx * grad
end


end # @muladd