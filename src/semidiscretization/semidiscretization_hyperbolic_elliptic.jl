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
                                            SourceTerms, SourceTermsElliptic, 
                                            Solver,
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
                                              equations::Tuple,
                                              initial_condition,
                                              solver;
                                              source_terms = nothing,
                                              elliptic_solver = EllipticSolver(),
                                              source_terms_elliptic = nothing,
                                              boundary_conditions)

    hyperbolic_equations, elliptic_equations = equations

    hyperbolic_bc, elliptic_bc = boundary_conditions

    cache = create_cache(mesh,
                         hyperbolic_equations,
                         solver)

    elliptic_cache = create_elliptic_cache(mesh,
                                           elliptic_equations,
                                           EllipticSolver())

    check_periodicity_mesh_boundary_conditions(mesh, hyperbolic_bc)
    check_periodicity_mesh_boundary_conditions(mesh, elliptic_bc)

    return SemidiscretizationHyperbolicElliptic{typeof(mesh),
                                                typeof(hyperbolic_equations),
                                                typeof(elliptic_equations),
                                                typeof(initial_condition),
                                                typeof(hyperbolic_bc),
                                                typeof(elliptic_bc),
                                                typeof(source_terms),
                                                typeof(source_terms_elliptic),
                                                typeof(solver),
                                                typeof(cache),
                                                typeof(elliptic_cache)}(mesh,
                                                                        hyperbolic_equations,
                                                                        elliptic_equations,
                                                                        initial_condition,
                                                                        hyperbolic_bc,
                                                                        elliptic_bc,
                                                                        source_terms,
                                                                        source_terms_elliptic,
                                                                        solver,
                                                                        cache,
                                                                        elliptic_cache)
end

@inline nvariables(semi::SemidiscretizationHyperbolicElliptic) = nvariables(semi.equations) + 1
@inline Base.show(io::IO, ::SemidiscretizationHyperbolicElliptic) = print(io, "Hyperbolic-Elliptic Semidiscretization")

# ============================================================================
# Elliptic ghost value helper (for IMEX scheme)
# ============================================================================

"""
    elliptic_ghost_value(x_elliptic, i, nx, semi, t)

Return the ghost cell value for the elliptic variable at index `i`
(which is outside the domain `1:nx`), using the elliptic boundary
conditions stored in `semi.boundary_conditions_elliptic`.
"""
@inline function elliptic_ghost_value(x_elliptic, i, nx, semi, t)
    bc = semi.boundary_conditions_elliptic

    if i < 1
        # Left boundary
        side_bc = bc.left
        if isa(side_bc, PeriodicBC)
            return x_elliptic[_wrap_index(i, nx)]
        elseif isa(side_bc, DirichletBC)
            x = coordinates(CartesianIndex(1), semi.mesh)
            return side_bc.boundary_value(x, t, semi.equations_elliptic)
        elseif isa(side_bc, NeumannBC)
            interior = x_elliptic[1]
            grad = side_bc.boundary_gradient(coordinates(CartesianIndex(1), semi.mesh),
                                             t, semi.equations_elliptic)
            grad = first(grad)
            return interior + semi.mesh.dx[1] * grad
        else
            # ExtrapolateBC or default: copy interior
            return x_elliptic[1]
        end
    else
        # Right boundary
        side_bc = bc.right
        if isa(side_bc, PeriodicBC)
            return x_elliptic[_wrap_index(i, nx)]
        elseif isa(side_bc, DirichletBC)
            x = coordinates(CartesianIndex(nx), semi.mesh)
            return side_bc.boundary_value(x, t, semi.equations_elliptic)
        elseif isa(side_bc, NeumannBC)
            interior = x_elliptic[nx]
            grad = side_bc.boundary_gradient(coordinates(CartesianIndex(nx), semi.mesh),
                                             t, semi.equations_elliptic)
            grad = first(grad)
            return interior + semi.mesh.dx[1] * grad
        else
            # ExtrapolateBC or default: copy interior
            return x_elliptic[nx]
        end
    end
end

@inline function _hyperbolic_ghost_state(u_hyper, I::CartesianIndex{NDIMS}, semi, t) where {NDIMS}
    side = boundary_side(I, semi)
    if side !== nothing
        bc = getproperty(semi.boundary_conditions, side)
        if bc isa PeriodicBC
            I = CartesianIndex(ntuple(d -> _wrap_index(I[d], size(semi.mesh, d)), NDIMS))
        end
    end
    return cell_state(u_hyper, I, semi, t)
end

@inline function _elliptic_var(x_elliptic,
                               I::CartesianIndex{NDIMS},
                               semi,
                               t) where {NDIMS}

    i = cell_index(I, semi)

    if 1 <= i <= length(x_elliptic)
        @inbounds return x_elliptic[i]
    else
        return elliptic_ghost_value(x_elliptic,
                                    i,
                                    length(x_elliptic),
                                    semi,
                                    t)
    end
end


end # @muladd