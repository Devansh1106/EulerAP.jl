# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function stencil_indices(I::CartesianIndex{2},
                                 semi::AbstractSemidiscretization)

    center = I

    # x-axis: axis = 1
    left   = neighbor_index(I, semi, 1, -1)
    right  = neighbor_index(I, semi, 1,  1)

    # y-axis: axis = 2
    bottom = neighbor_index(I, semi, 2, -1)
    top    = neighbor_index(I, semi, 2,  1)

    return (center, left, right, bottom, top)
end

function neighbor_index(I::CartesianIndex{2},
                        semi::AbstractSemidiscretization,
                        axis::Int,
                        sign::Int)

    shifted = I[axis] + sign
    nx = size(semi.mesh, axis)
    side = axis == 1 ? (sign < 0 ? :left : :right) : (sign < 0 ? :bottom : :top)
    bc = getproperty(semi.boundary_conditions, side)

    # Interior cell
    if 1 <= shifted <= nx
        return _shifted_cartesian(I, axis, shifted)
    end

    if isa(bc, PeriodicBC)
        wrapped = _wrap_index(shifted, nx)
        return _shifted_cartesian(I, axis, wrapped)

    elseif bc isa NeumannBC || bc isa ExtrapolateBC
        clamped = clamp(shifted, 1, nx)
        return _shifted_cartesian(I, axis, clamped)

    elseif bc isa DirichletBC
        # Ghost cell outside domain; boundary_side will identify it
        return _shifted_cartesian(I, axis, shifted)

    else
        error("Unknown boundary condition type $(typeof(bc))")
    end
end

@inline function _shifted_cartesian(I::CartesianIndex{2}, axis::Int, val::Int)
    if axis == 1
        return CartesianIndex(val, I[2])
    else
        return CartesianIndex(I[1], val)
    end
end

@inline function _state_at(u::AbstractVector,
                           I::CartesianIndex{2},
                           semi::AbstractSemidiscretization)

    nvars = nvariables(semi.equations)
    cell  = cell_index(I, semi)

    return SVector{nvars}(ntuple(v ->
                                 u[global_dof(cell, v, nvars)],
                                 nvars))
end

@inline function boundary_side(I::CartesianIndex{2},
                               semi::AbstractSemidiscretization)

    i, j = Tuple(I)
    nx, ny = size(semi.mesh)

    if i < 1
        return :left
    elseif i > nx
        return :right
    elseif j < 1
        return :bottom
    elseif j > ny
        return :top
    else
        return nothing
    end
end

function rhs!(du, u,
              solver::FVSolver{2, TFlux},
              semi::SemidiscretizationHyperbolic,
              t;
              dt=0.0) where {TFlux}

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
                        semi.solver,
                        semi;
                        dt=dt)
        cell = cell_index(I, semi)

        @inbounds for v in 1:nvars
            du[global_dof(cell, v, nvars)] = cache.residual_buffer[v]
        end
    end

    return nothing
end

@inline function gather_local_state!(x, u, I::CartesianIndex{2},
                                     semi::AbstractSemidiscretization,
                                     t)

    nvars = nvariables(semi.equations)
    center, left, right, bottom, top = stencil_indices(I, semi)

    offset = 0

    for cell in (center, left, right, bottom, top)
        state = cell_state(u, cell, semi, t)

        @inbounds for v in 1:nvars
            x[offset + v] = state[v]
        end
        offset += nvars
    end

    return nothing
end

@inline function cell_state(u, I::CartesianIndex{2},
                            semi::AbstractSemidiscretization,
                            t)

    nx, ny = size(semi.mesh)

    # Interior cell
    if 1 <= I[1] <= nx && 1 <= I[2] <= ny
        return extract_cell_state(u, I, semi)
    end

    # Ghost cell
    side = boundary_side(I, semi)

    bc = _bc_from_side(semi.boundary_conditions, side)

    return apply_bc(bc, u, I, semi, t)
end

@inline function _bc_from_side(bc, side)
    if side === :left
        return bc.left
    elseif side === :right
        return bc.right
    elseif side === :bottom
        return bc.bottom
    elseif side === :top
        return bc.top
    else
        error("Unknown boundary side: $side")
    end
end

@inline function extract_cell_state(u, I::CartesianIndex{2},
                                    semi::AbstractSemidiscretization)

    nvars = nvariables(semi.equations)
    cell  = cell_index(I, semi)

    return SVector{nvars}(ntuple(v ->
                                 u[global_dof(cell, v, nvars)],
                                 nvars))
end

@inline function local_residual!(y, x, solver::FVSolver{2, TFlux}, semi::AbstractSemidiscretization; dt=0.0) where {TFlux}

    equations = semi.equations
    source_terms = semi.source_terms
    flux = semi.solver.flux

    # --------------------------------------------------
    # Extract stencil states (5 stencil points for 2D)
    # --------------------------------------------------

    u_center  = local_state(x, 1, equations)
    u_left    = local_state(x, 2, equations)
    u_right   = local_state(x, 3, equations)
    u_bottom  = local_state(x, 4, equations)
    u_top     = local_state(x, 5, equations)

    dx = semi.mesh.dx[1]
    dy = semi.mesh.dx[2]

    # --------------------------------------------------
    # Numerical fluxes
    # --------------------------------------------------

    flux_left   = flux(u_left, u_center, 1, equations, dt)
    flux_right  = flux(u_center, u_right, 1, equations, dt)
    flux_bottom = flux(u_bottom, u_center, 2, equations, dt)
    flux_top    = flux(u_center, u_top, 2, equations, dt)

    # --------------------------------------------------
    # FV divergence
    # --------------------------------------------------

    @inbounds for v in eachindex(y)
        y[v] = -((flux_right[v] - flux_left[v]) / dx +
                 (flux_top[v] - flux_bottom[v]) / dy)
    end

    # --------------------------------------------------
    # Source terms
    # --------------------------------------------------

    if source_terms !== nothing
        src = source_terms(u_center, equations)

        @inbounds for v in eachindex(y)
            y[v] += src[v]
        end
    end

    return nothing
end

@inline function apply_bc(bc::ExtrapolateBC{2},
                          u,
                          I::CartesianIndex{2},
                          semi,
                          t)

    nx, ny = size(semi.mesh)
    interior_i = clamp(I[1], 1, nx)
    interior_j = clamp(I[2], 1, ny)

    return extract_cell_state(u, CartesianIndex(interior_i, interior_j), semi)
end

@inline function apply_bc(bc::DirichletBC{2},
                          u,
                          I::CartesianIndex{2},
                          semi,
                          t)

    x = coordinates(I, semi.mesh)

    return bc.boundary_value(x, t, semi.equations)
end

@inline function apply_bc(bc::NeumannBC{2},
                          u,
                          I::CartesianIndex{2},
                          semi,
                          t)

    nx, ny = size(semi.mesh)
    interior_i = clamp(I[1], 1, nx)
    interior_j = clamp(I[2], 1, ny)

    interior = extract_cell_state(u,
                                  CartesianIndex(interior_i, interior_j),
                                  semi)

    grad = bc.boundary_gradient(coordinates(I, semi.mesh),
                                t,
                                semi.equations)

    dx = semi.mesh.dx[1]
    dy = semi.mesh.dx[2]
    d = I[1] < 1 || I[1] > nx ? dx : dy

    return interior + d * grad
end


end # @muladd