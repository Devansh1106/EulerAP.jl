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

"""
    EllipticCache

Cache storing pre-allocated arrays for the elliptic implicit solver in the IMEX scheme.
All arrays avoid re-allocation on every time step.
"""
# Separate struct holding NewtonCache for scalability
mutable struct EllipticCache{TV, TJ, TN}
    # Thomas algorithm work arrays (used in assemble_nonlinear_jacobian!)
    dl::TV
    d::TV
    du::TV

    # Nonlinear residual
    residual::TV

    # Jacobian prototype
    jacobian::TJ

    # Newton solver cache (holds all Newton-related state)
    newton_cache::TN
end

"""
    NewtonParameters

Parameters passed to the Newton solver for the elliptic implicit equation.
"""
mutable struct NewtonParameters{TRhs, TCoeff, TTime}
    rhs::TRhs
    laplacian_coeff::TCoeff
    t::TTime
end

"""
    NewtonCache

Cache holding all Newton-related state for the elliptic solver.
"""
mutable struct NewtonCache{TP}
    params::TP
    nonlinear_cache::Union{Nothing, Any}
end

function create_newton_cache(mesh::CartesianMesh)
    nx = ncells(mesh)
    T = eltype(mesh.dx)

    # Create initial parameters struct (will be updated during solve)
    params = NewtonParameters(zeros(T, nx), zero(T), zero(T))

    # NonlinearProblem is created later in set_newton_semi!
    # once the semidiscretization is available
    return NewtonCache(params, nothing)
end

function set_newton_semi!(newton_cache::NewtonCache, semi::AbstractSemidiscretization)
    mesh = semi.mesh
    T = eltype(mesh.dx)
    nx = ncells(mesh)

    # Create closures that capture semi
    function residual!(F, phi, params)
        assemble_nonlinear_residual!(F, phi, params, semi)
    end
    function jacobian!(J, phi, params)
        assemble_nonlinear_jacobian!(J, phi, params, semi)
    end

    jac_prototype = semi.cache_elliptic.jacobian
    nonlinear_function = NonlinearFunction(residual!, jac = jacobian!,
                                           jac_prototype = jac_prototype)
    u0 = zeros(T, nx)
    prob = NonlinearProblem(nonlinear_function, u0, newton_cache.params)
    newton_cache.nonlinear_cache = init(prob, NewtonRaphson(); linsolve_kwargs = (linsolve = linsolve,))

    return nothing
end

# function create_elliptic_cache(mesh::CartesianMesh{1},
#                                equations_elliptic::AbstractEquations,
#                                solver_elliptic)
#     nx = ncells(mesh)
#     T = eltype(mesh.dx)

#     jacobian = Tridiagonal(
#         zeros(T, nx - 1),
#         zeros(T, nx),
#         zeros(T, nx - 1),
#     )

#     # Create the Newton cache (nonlinear problem created later in set_newton_semi!
#     # once the semidiscretization is fully constructed)
#     newton_cache = create_newton_cache(mesh)

#     return EllipticCache(zeros(T, nx),      # dl
#                          zeros(T, nx),      # d
#                          zeros(T, nx),      # du
#                          zeros(T, nx),      # residual
#                          jacobian,
#                          newton_cache)
# end

function create_elliptic_cache(mesh::CartesianMesh{1},
                               equations_elliptic::AbstractEquations,
                               solver_elliptic)
    nx = ncells(mesh)
    T = eltype(mesh.dx)

    # Sparse Jacobian prototype with an explicit tridiagonal + periodic-corner
    # sparsity pattern. A plain `LinearAlgebra.Tridiagonal` cannot store the
    # (1, nx) / (nx, 1) entries needed for periodic boundary conditions, which
    # silently made the Newton Jacobian inconsistent with the (correctly
    # periodic) residual in `assemble_nonlinear_residual!`.
    rows = Int[]
    cols = Int[]
    for i in 1:nx
        push!(rows, i); push!(cols, i)              # diagonal
        if i > 1
            push!(rows, i); push!(cols, i - 1)       # sub-diagonal
        end
        if i < nx
            push!(rows, i); push!(cols, i + 1)       # super-diagonal
        end
    end
    if nx > 2
        push!(rows, 1);  push!(cols, nx)             # periodic corner
        push!(rows, nx); push!(cols, 1)              # periodic corner
    end
    jacobian = sparse(rows, cols, zeros(T, length(rows)), nx, nx)

    # Create the Newton cache (nonlinear problem created later in set_newton_semi!
    # once the semidiscretization is fully constructed)
    newton_cache = create_newton_cache(mesh)

    return EllipticCache(zeros(T, nx),      # dl (unused now, kept for struct compatibility)
                         zeros(T, nx),      # d  (unused now, kept for struct compatibility)
                         zeros(T, nx),      # du (unused now, kept for struct compatibility)
                         zeros(T, nx),      # residual
                         jacobian,
                         newton_cache)
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

    # Statistics
    stats::Union{Nothing, CallbackStats}
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
                   residual_buffer, # residual_buffer
                   nothing)       # stats
end

abstract type AbstractBC{NDIMS} end

struct PeriodicBC{NDIMS} <: AbstractBC{NDIMS} end

struct ExtrapolateBC{NDIMS} <: AbstractBC{NDIMS} end

struct DirichletBC{NDIMS, F} <: AbstractBC{NDIMS}
    boundary_value::F
end

# Convenience constructor: DirichletBC{1}(f) => DirichletBC{1, typeof(f)}(f)
function DirichletBC{NDIMS}(boundary_value::F) where {NDIMS, F}
    return DirichletBC{NDIMS, F}(boundary_value)
end

struct NeumannBC{NDIMS, F} <: AbstractBC{NDIMS}
    boundary_gradient::F
end

# Convenience constructor: NeumannBC{1}(f) => NeumannBC{1, typeof(f)}(f)
function NeumannBC{NDIMS}(boundary_gradient::F) where {NDIMS, F}
    return NeumannBC{NDIMS, F}(boundary_gradient)
end

"""
    MixedBC{NDIMS}(bcs...)

Composite boundary condition that applies a **different** [`AbstractBC`](@ref)
to each solution variable on the same side, e.g.

```julia
MixedBC{1}(ExtrapolateBC{1}(), DirichletBC{1}((x, t, equations) -> 0.0))
```

applies `ExtrapolateBC` to variable 1 (e.g. density) and a zero Dirichlet
condition to variable 2 (e.g. momentum) on that side.

`bcs` must contain exactly `nvariables(equations)` entries, in the same order
as the state vector `u`. Each entry can be any existing `AbstractBC{NDIMS}`,
and the ghost value for variable `v` is computed by dispatching to that
entry's own `apply_bc` method and keeping only component `v` — so the
numerics for each sub-condition are identical to using that condition on its
own.

!!! note
    The Jacobian sparsity pattern built in `build_jacobian_cache!` currently
    makes a single dependent-on-interior-cell / independent-of-interior-cell
    decision per side (see `neighbor_index`), not per variable. A `MixedBC`
    is routed through the "independent" branch so that `apply_bc` is always
    consulted; this is correct for the residual (`rhs!`) evaluation, but if a
    variable inside a `MixedBC` uses `ExtrapolateBC`/`NeumannBC` (which *do*
    depend on the neighboring interior cell) and you later build an implicit
    Jacobian (`jac_prototype = true`) for a system using `MixedBC`, that
    dependency edge will be missing from the sparsity pattern. This is not
    exercised by the current explicit/IMEX examples, but should be revisited
    before using `MixedBC` with an implicit hyperbolic solve.
"""
struct MixedBC{NDIMS, N, T <: Tuple} <: AbstractBC{NDIMS}
    bcs::T
end

function MixedBC{NDIMS}(bcs::Vararg{AbstractBC{NDIMS}, N}) where {NDIMS, N}
    return MixedBC{NDIMS, N, typeof(bcs)}(bcs)
end

struct BoundaryConditions1D
    sides::Tuple{AbstractBC{1}, AbstractBC{1}}
end

function BoundaryConditions1D(left::AbstractBC{1}, right::AbstractBC{1})
    return BoundaryConditions1D((left, right))
end

function Base.getproperty(bc::BoundaryConditions1D, side::Symbol)
    sides = getfield(bc, :sides)
    if side === :left
        return sides[1]
    elseif side === :right
        return sides[2]
    else
        error("Unknown boundary side: $side for 1D")
    end
end

struct BoundaryConditions2D
    sides::Tuple{AbstractBC{2}, AbstractBC{2}, AbstractBC{2}, AbstractBC{2}}
end

function BoundaryConditions2D(left::AbstractBC{2}, right::AbstractBC{2},
                              bottom::AbstractBC{2}, top::AbstractBC{2})
    return BoundaryConditions2D((left, right, bottom, top))
end

function Base.getproperty(bc::BoundaryConditions2D, side::Symbol)
    sides = getfield(bc, :sides)
    if side === :left
        return sides[1]
    elseif side === :right
        return sides[2]
    elseif side === :bottom
        return sides[3]
    elseif side === :top
        return sides[4]
    else
        error("Unknown boundary side: $side for 2D")
    end
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
                if !(neighbor isa CartesianIndex{ndims(mesh)})
                    continue
                end
                neighbor_cell = cell_index(neighbor, mesh)
                # Skip ghost cells outside domain (e.g. DirichletBC ghost cells at index 0 or nx+1).
                # Their state does not depend on interior DOFs, so they contribute zero to the Jacobian.
                # NOTE: `MixedBC` is also routed here (see its docstring) even when one of its
                # per-variable sub-conditions is an `ExtrapolateBC`/`NeumannBC` that *does* depend
                # on the interior neighbor; that dependency edge is currently dropped from the
                # sparsity pattern in that case.
                if neighbor_cell < 1 || neighbor_cell > ncells
                    continue
                end

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
                if !(neighbor isa CartesianIndex{ndims(mesh)})
                    continue
                end
                neighbor_cell = cell_index(neighbor, mesh)

                # Skip ghost cells outside domain (same as sparsity pattern building above)
                if neighbor_cell < 1 || neighbor_cell > ncells
                    continue
                end

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
        local_residual!(y, x, semi.solver, semi; dt=0.0)
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

The global state vector is stored in cell-major ordering.
"""
function initial_condition(t,
                           semi::AbstractSemidiscretization)

    mesh = semi.mesh
    nvars = nvariables(semi)

    # TODO: Need to settle this. This is a temporary setup by calling IC for EPB system on EulerPressureLess1D type which should be changed to a common type for EPB system (hyper+elliptic)
    # # For coupled hyperbolic-elliptic systems, pass the semidiscretization
    # # so initial conditions can return the full state (including elliptic vars).
    # # For purely hyperbolic systems, pass just the equations.
    # if semi isa SemidiscretizationHyperbolicElliptic
    #     eq_or_semi = semi
    # else
    #     eq_or_semi = semi.equations
    # end

    T = eltype(mesh.dx)
    u0 = zeros(T, nvars * ndofs(mesh))

    for I in eachcell(mesh)
        cell = cell_index(I, semi)
        x = coordinates(I, mesh)

        state = semi.initial_condition(x, t, semi.equations)

        @inbounds for v in 1:nvars
            u0[global_dof(cell, v, nvars)] = state[v]
        end
    end

    return u0
end

@inline semi(context::CallbackContext) = context.simulation.semi

# For 1D Cartesian Mesh 
function check_periodicity_mesh_boundary_conditions(mesh::CartesianMesh{1}, bcs::BoundaryConditions1D)
    if mesh.periodicity[1]
        if !(bcs.left isa PeriodicBC{1} &&
             bcs.right isa PeriodicBC{1})

            throw(ArgumentError(
                "Periodic x-direction requires PeriodicBC on both left and right boundaries."))
        end
    end
    return nothing
end

# For 2D Cartesian Mesh 
function check_periodicity_mesh_boundary_conditions(mesh::CartesianMesh{2}, bcs::BoundaryConditions2D)
    if mesh.periodicity[1]
        if !(bcs.left isa PeriodicBC{2} &&
             bcs.right isa PeriodicBC{2})

            throw(ArgumentError(
                "Periodic x-direction requires PeriodicBC on both left and right boundaries."))
        end
    end

    if mesh.periodicity[2]
        if !(bcs.bottom isa PeriodicBC{2} &&
             bcs.top isa PeriodicBC{2})

            throw(ArgumentError(
                "Periodic y-direction requires PeriodicBC on both bottom and top boundaries."))
        end
    end
    return nothing
end

@inline Base.ndims(semi::AbstractSemidiscretization) = ndims(semi.mesh)


# ============================================================================
# Display
# ============================================================================

# 1D 
@inline Base.show(io::IO, ::PeriodicBC{1})  = print(io, "Periodic")
@inline Base.show(io::IO, ::DirichletBC{1}) = print(io, "Dirichlet")
@inline Base.show(io::IO, ::NeumannBC{1})   = print(io, "Neumann")
@inline Base.show(io::IO, ::ExtrapolateBC{1}) = print(io, "Extrapolation")

# 2D
@inline Base.show(io::IO, ::PeriodicBC{2})  = print(io, "Periodic")
@inline Base.show(io::IO, ::DirichletBC{2}) = print(io, "Dirichlet")
@inline Base.show(io::IO, ::NeumannBC{2})   = print(io, "Neumann")
@inline Base.show(io::IO, ::ExtrapolateBC{2}) = print(io, "Extrapolation")

end # @muladd