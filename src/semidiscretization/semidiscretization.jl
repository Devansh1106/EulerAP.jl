# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    EulerAPSolution

Container storing the final state and time returned by
a custom time integrator.

Unlike SciML's `ODESolution`, this stores only the final
solution state.
"""
struct EulerAPSolution{TU,T}
    u::TU
    t::T
end

mutable struct FVCache{TJacobian, TPosition, TX, TY, TJlocal, TResidualBuffer}
    # Jacobian infrastructure
    jac_prototype::TJacobian
    positions::TPosition

    # ForwardDiff pre-allocations
    x_cache::TX
    y_cache::TY
    J_local_cache::TJlocal
    config::Union{Nothing, ForwardDiff.JacobianConfig}
    local_residual!::Union{Nothing, Function}
    residual_buffer::TResidualBuffer
end

function create_cache(mesh::AbstractMesh,
                      equations::AbstractEquations,
                      solver)

    nvars = nvariables(equations)
    local_stencil = 2 * ndims(mesh) + 1
    ncells = ndofs(mesh)
    n = ncells * nvars
    T = eltype(mesh.dx)

    x_cache = zeros(T, local_stencil * nvars)
    residual_buffer = zeros(T, nvars)

    # Pre-allocate with proper types so FVCache type parameters are not locked to `Nothing`.
    # These will be properly filled in `build_jacobian_cache!` when Jacobian is needed.
    jac_prototype = spzeros(T, n, n)
    positions = zeros(Int, nvars, local_stencil * nvars, ncells)
    y_cache = zeros(T, nvars)
    J_local_cache = zeros(T, nvars, local_stencil * nvars)

    return FVCache(jac_prototype, # jac_prototype
                   positions,     # positions
                   x_cache,       # x_cache
                   y_cache,       # y_cache
                   J_local_cache, # J_local_cache
                   nothing,       # config
                   nothing,       # local_residual!
                   residual_buffer) # residual_buffer
end

abstract type AbstractBC end

struct PeriodicBC <: AbstractBC end

struct ExtrapolateBC <: AbstractBC end

struct DirichletBC{F} <: AbstractBC
    boundary_value::F
end

struct NeumannBC{F} <: AbstractBC
    boundary_gradient::F
end

"""
    ndofs(semi::AbstractSemidiscretization)

Return the number of degrees of freedom associated with each scalar variable.
"""
@inline function ndofs(semi::AbstractSemidiscretization)
    return prod(size(semi.mesh))
end

@inline ndofs(mesh::AbstractMesh) = ncells(mesh)

@inline global_dof(cell::Int, var::Int, nvars::Int) = (cell - 1) * nvars + var

# Select the right-hand side function corresponding to the semidiscretization `semi`.
@inline default_rhs(::AbstractSemidiscretization) = rhs!

"""
    coordinates(I, mesh)

Return the physical coordinates of the center of cell `I`.
"""
@inline function coordinates(I::CartesianIndex{NDIMS},
                             mesh::CartesianMesh{NDIMS}) where {NDIMS}

    return ntuple(d ->
                  mesh.coordinates_min[d] + (I[d] - 0.5) * mesh.dx[d],
                  NDIMS)
end

"""
    semidiscretize(semi::AbstractSemidiscretization, tspan;
                   jac_prototype::Bool = false)

Wrap the semidiscretization `semi` as an ODE problem in the time interval `tspan` that
can be passes to `solve` from the [SciML ecosystem](https://docs.sciml.ai/DiffEqDocs/latest/).

Optional keyword arguments:
- `jac_prototype`: This will be built manually in the function `build_jac_prototype`. Specifies the sparsity structure of the Jacobian to enable e.g. efficient implicit time stepping.
"""
function semidiscretize(semi::AbstractSemidiscretization, tspan;
                        jac_prototype::Bool = false)

    u0_ode = initial_condition(first(tspan), semi)
    rhs_semi! = default_rhs(semi)

    iip = true # is-inplace, i.e., we modify a vector when callig `rhs_semi!`
    specialize = SciMLBase.FullSpecialize # specialize on `rhs_semi!` and parameters (semi)

    # Check if Jacobian prototype is provided for sparse Jacobian
    if jac_prototype
        # Build Jacobian cache, positions array and jacobian prototype
        build_jacobian_cache!(semi)

        # J_prototype = build_jacobian_prototype(semi)
        cache = semi.cache

        # Convert `jac_prototype` to type of `u0_ode`.
        ode = SciMLBase.ODEFunction(rhs_semi!,
                                    jac = jacobian!,
                                    jac_prototype = convert.(eltype(u0_ode),
                                                             cache.jac_prototype))

        return ODEProblem{iip, specialize}(ode, u0_ode, tspan, semi)
    else
        # We could also construct an `ODEFunction` explicitly without the Jacobian here,
        # but we stick to the lean direct in-place function `rhs_semi!` and
        # let OrdinaryDiffEq.jl handle the rest
        return ODEProblem{iip, specialize}(rhs_semi!, u0_ode, tspan, semi)
    end
end

function build_jacobian_cache!(semi::AbstractSemidiscretization)
    cache     = semi.cache
    mesh      = semi.mesh
    equations = semi.equations
    solver    = semi.solver
    ncells    = ndofs(semi)
    nvars     = nvariables(equations)
    local_stencil = stencil_size(semi)
    n             = ncells * nvars

    nnz_estimate = local_stencil * n
    T = eltype(mesh.dx)

    # Pre-allocating sparse matrix builder arrays
    I     = Int[]
    J_col = Int[]

    sizehint!(I, nnz_estimate)
    sizehint!(J_col, nnz_estimate)

    # ------------------------------------------------
    # Building sparsity pattern
    # ------------------------------------------------
    for cell in eachcell(mesh)
        center = cell_index(cell, mesh)
        neighbors = stencil_indices(cell, semi)

        for row_var in 1:nvars
            row = global_dof(center, row_var, nvars)

            for neighbor in neighbors
                # neighbor == 0 && continue
                if neighbor isa CartesianIndex{1}
                    if !(1 <= neighbor[1] <= size(mesh,1))
                        continue
                    end
                end
                neighbor_cell = cell_index(neighbor, mesh)

                for col_var in 1:nvars
                    col = global_dof(neighbor_cell, col_var, nvars)
                    push!(I, row)
                    push!(J_col, col)
                end
            end
        end
    end

    jac_prototype = sparse(I, J_col,
                           fill(one(T), length(I)),
                           n, n)
    fill!(jac_prototype.nzval, zero(T))

    # -----------------------------------------------
    # Local to Global mapping
    # -----------------------------------------------
    positions = zeros(Int, nvars, local_stencil * nvars, ncells)

    for cell in eachcell(mesh)
        center = cell_index(cell, mesh)
        neighbors = stencil_indices(cell, semi)

        for row_var in 1:nvars
            row = global_dof(center, row_var, nvars)

            for (neighbor_idx, neighbor) in enumerate(neighbors)
                # neighbor == 0 && continue
                if neighbor isa CartesianIndex{1}
                    if !(1 <= neighbor[1] <= size(mesh,1))
                        continue
                    end
                end
                neighbor_cell = cell_index(neighbor, mesh)

                for col_var in 1:nvars
                    col = global_dof(neighbor_cell, col_var, nvars)

                    local_col = (neighbor_idx - 1) * nvars + col_var

                    col_start = jac_prototype.colptr[col]

                    col_end = jac_prototype.colptr[col + 1] - 1

                    rows = @view jac_prototype.rowval[col_start:col_end]

                    rel_idx = findfirst(==(row), rows)

                    positions[row_var, local_col, center] = rel_idx === nothing ?
                                                              0 :
                                                              col_start + rel_idx - 1
                end
            end
        end
    end

    # --------------------------------------------------
    # ForwardDiff buffers
    # --------------------------------------------------

    y_cache = zeros(T, nvars)

    J_local_cache = zeros(T, nvars, local_stencil * nvars)

    local_residual_closure = (y, x) -> 
    begin
        local_residual!(y, x, semi)
        return nothing
    end

    config = ForwardDiff.JacobianConfig(local_residual_closure, y_cache, cache.x_cache)

    # --------------------------------------------------
    # Store in cache
    # --------------------------------------------------
    # `x_cache` and `residual_buffer` are already filled in `create_cache()`
    cache.jac_prototype = jac_prototype
    cache.positions     = positions
    cache.y_cache       = y_cache
    cache.J_local_cache = J_local_cache
    cache.config        = config
    cache.local_residual! = local_residual_closure

    return nothing
end

function jacobian!(J, u, semi::AbstractSemidiscretization, t)
    cache = semi.cache
    assemble_jacobian!(J, u, semi, cache, t)

    return nothing
end

function assemble_jacobian!(J, u, semi::AbstractSemidiscretization,
                            cache::FVCache, t)

    fill!(J.nzval, zero(eltype(J)))
    for cell in eachcell(semi.mesh)
        centre = cell_index(cell, semi)
        gather_local_state!(cache.x_cache, u, cell, semi, t)

        ForwardDiff.jacobian!(cache.J_local_cache,
                              cache.local_residual!,
                              cache.y_cache,
                              cache.x_cache,
                              cache.config)

        scatter_local_jacobian!(J.nzval,
                                cache.positions,
                                cache.J_local_cache,
                                centre)
    end

    return nothing
end

function scatter_local_jacobian!(nzval, positions,
                                 J_local, center)

    @inbounds for row_var in axes(J_local, 1)
        for local_col_idx in axes(J_local, 2)
            pos = positions[row_var, local_col_idx, center]

            pos == 0 && continue

            nzval[pos] += J_local[row_var, local_col_idx]
        end
    end

    return nothing
end

"""
    initial_condition(t, semi)

Construct the global ODE state vector at time `t`
using the user-supplied initial condition function.

The global state vector is stored in cell-major ordering:

    [ρ₁, m₁, ρ₂, m₂, ...]

for a 1D two-variable system.
"""
function initial_condition(t,
                           semi::SemidiscretizationHyperbolic)

    mesh      = semi.mesh
    equations = semi.equations

    nvars = nvariables(equations)
    T = eltype(mesh.dx)
    u0 = zeros(T, nvars * ndofs(mesh))

    for I in eachcell(mesh)
        cell = cell_index(I, semi)
        x = coordinates(I, mesh)

        state = semi.initial_condition(x, t, equations)

        @inbounds for v in 1:nvars
            u0[global_dof(cell, v, nvars)] = state[v]
        end
    end

    return u0
end

@inline semi(context::CallbackContext) = context.simulation.semi

# ============================================================================
# Display
# ============================================================================

@inline Base.show(io::IO, ::SemidiscretizationHyperbolic) = print(io, "Hyperbolic semidiscretization")
# ============================================================================
# Display
# ============================================================================

@inline Base.show(io::IO, ::PeriodicBC) = print(io, "Periodic")

@inline Base.show(io::IO, ::DirichletBC) = print(io, "Dirichlet")

@inline Base.show(io::IO, ::NeumannBC) = print(io, "Neumann")

@inline Base.show(io::IO, ::ExtrapolateBC) = print(io, "Extrapolation")

end # @muladd