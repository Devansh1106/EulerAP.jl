# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationHyperbolicElliptic

A struct containing everything needed to describe a spatial semidiscretization
of a mixed hyprebolic-elliptic system such as Euler-Poisson-Boltzmann system.
"""

struct SemidiscretizationHyperbolicElliptic{Mesh, Equations, EquationsElliptic,
                                            InitialCondition, 
                                            BoundaryConditions,
                                            BoundaryConditionsElliptic,
                                            SourceTerms, SourceTermsElliptic 
                                            Solver, SolverElliptic,
                                            Cache, CacheElliptic} <: AbstractSemidiscretization

    mesh::Mesh

    equations::Equations
    equations_elliptic::EquationsElliptic

    initial_condition::InitialCondition

    boundary_conditions::BoundaryConditions
    boundary_conditions_elliptic::BoundaryConditionsElliptic

    source_terms::SourceTerms
    source_terms_elliptic::SourceTermsElliptic

    solver::Solver
    solver_elliptic::SolverElliptic

    cache::Cache
    cache_elliptic::CacheElliptic
end

"""
    SemidiscretizationHyperbolicElliptic(mesh,
                                         equations,
                                         initial_condition,
                                         solver;
                                         elliptic_solver,
                                         source_terms = nothing,
                                         boundary_conditions)

Construct a semidiscretization for coupled hyperbolic-elliptic systems.

The hyperbolic and elliptic equations are supplied as a tuple

    (hyperbolic_equations, elliptic_equations)

Similarly, boundary conditions are supplied as

    (hyperbolic_bc, elliptic_bc)

The hyperbolic operator is discretized using the finite-volume solver,
while the elliptic equation is handled by the specified elliptic solver.
"""
function SemidiscretizationHyperbolicElliptic(mesh,
                                              equations::NamedTuple,
                                              initial_condition,
                                              solver;
                                              elliptic_solver,
                                              source_terms = nothing,
                                              boundary_conditions)

    hyperbolic_equations, elliptic_equations = equations

    hyperbolic_bc, elliptic_bc = boundary_conditions

    cache = create_cache(mesh,
                         hyperbolic_equations,
                         solver)

    elliptic_cache = create_elliptic_cache(mesh,
                                           elliptic_equations,
                                           elliptic_solver)

    check_periodicity_mesh_boundary_conditions(mesh, hyperbolic_bc)
    check_periodicity_mesh_boundary_conditions(mesh, elliptic_bc)

    return SemidiscretizationHyperbolicElliptic{typeof(mesh),
                                                typeof(hyperbolic_equations),
                                                typeof(elliptic_equations),
                                                typeof(initial_condition),
                                                typeof(hyperbolic_bc),
                                                typeof(elliptic_bc),
                                                typeof(source_terms),
                                                typeof(solver),
                                                typeof(elliptic_solver),
                                                typeof(cache),
                                                typeof(elliptic_cache)}(mesh,
                                                                        hyperbolic_equations,
                                                                        elliptic_equations,
                                                                        initial_condition,
                                                                        hyperbolic_bc,
                                                                        elliptic_bc,
                                                                        source_terms,
                                                                        solver,
                                                                        elliptic_solver,
                                                                        cache,
                                                                        elliptic_cache)
end

@inline nvariables(semi::SemidiscretizationHyperbolicElliptic) = nvariables(semi.equations) + 1 # +1 is for Φ from Poisson-Boltzmann

# SciML rhs! function
function rhs!(du_ode, u_ode, 
              semi::SemidiscretizationHyperbolicElliptic,
              t)

    # defined in solvers/ folder
    rhs!(du_ode, u_ode,
         semi.solver,
         semi,
         t)

    return nothing
end

@inline Base.show(io::IO, ::SemidiscretizationHyperbolicElliptic) = print(io, "Hyperbolic-Elliptic Semidiscretization")


end # @muladd