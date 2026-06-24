# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationHyperbolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic conservation law.
"""
mutable struct SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition,
                                            BoundaryConditions,
                                            SourceTerms, Solver, Cache} <: AbstractSemidiscretization

    mesh::Mesh
    equations::Equations
    const initial_condition::InitialCondition

    const boundary_conditions::BoundaryConditions
    const source_terms::SourceTerms
    const solver::Solver
    cache::Cache
end

"""
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions)

Construct a semidiscretization of a hyperbolic PDE.

Boundary conditions must be provided explicitly either as a `NamedTuple`.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                      source_terms = nothing,
                                      boundary_conditions)

    @assert ndims(mesh) === ndims(equations)
    cache = create_cache(mesh, equations, solver)
    check_periodicity_mesh_boundary_conditions(mesh, boundary_conditions)

    return SemidiscretizationHyperbolic{typeof(mesh), typeof(equations),
                                        typeof(initial_condition),
                                        typeof(boundary_conditions),
                                        typeof(source_terms),
                                        typeof(solver), typeof(cache)}(mesh, equations,
                                                                       initial_condition,
                                                                       boundary_conditions,
                                                                       source_terms,
                                                                       solver, cache)    
end

# For 1D Cartesian Mesh 
function check_periodicity_mesh_boundary_conditions(mesh::CartesianMesh{1}, bcs)
    if mesh.periodicity[1]
        if !(bcs.left isa PeriodicBC &&
             bcs.right isa PeriodicBC)

            throw(ArgumentError(
                "Periodic x-direction requires PeriodicBC on both left and right boundaries."))
        end
    end
    return nothing
end

# For 2D Cartesian Mesh 
function check_periodicity_mesh_boundary_conditions(mesh::CartesianMesh{2}, bcs)
    if mesh.periodicity[1]
        if !(bcs.left isa PeriodicBC &&
             bcs.right isa PeriodicBC)

            throw(ArgumentError(
                "Periodic x-direction requires PeriodicBC on both left and right boundaries."))
        end
    end

    if mesh.periodicity[2]
        if !(bcs.bottom isa PeriodicBC &&
             bcs.top isa PeriodicBC)

            throw(ArgumentError(
                "Periodic y-direction requires PeriodicBC on both bottom and top boundaries."))
        end
    end
    return nothing
end

@inline Base.ndims(semi::SemidiscretizationHyperbolic) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolic) = nvariables(semi.equations)

# SciML rhs! function
function rhs!(du_ode, u_ode, 
              semi::SemidiscretizationHyperbolic,
              t)

    # defined in solvers/ folder
    rhs!(du_ode, u_ode,
         semi.solver,
         semi,
         t)

    return nothing
end

end # @muladd