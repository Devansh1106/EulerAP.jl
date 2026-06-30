using EulerAP

# --------------------------------------------------
# Barenblatt convergence test
# --------------------------------------------------

function barenblatt(x, t, Γ, γ)
    t_eff = Float64(t)
    β = 1.0 / (γ + 1.0)
    ξ = x / (t_eff^β)
    factor = (γ - 1.0) / (2.0 * γ * (γ + 1.0))
    bracket = Γ - factor * (ξ^2)
    positive = max(bracket, 0.0)
    return t_eff^(-β) * (positive^(1.0 / (γ - 1.0)))
end

function exact_solution_barenblatt(x, t, equations)
    γ = equations.gamma
    ρ = barenblatt(x[1], t, 1.0, γ)
    β = 1.0 / (γ + 1.0)
    mx = ρ * β * x[1] / t
    return (ρ, mx)
end

# Build semi for a given grid size
function make_semi(N)
    mesh = CartesianMesh((N,), (-6.0,), (6.0,))
    equations = RelaxationEulerEquations1D(gamma=3.0, epsilon=1e-1)
    solver = FVSolver(flux=FluxRusanov(), ndims=1)
    bc = BoundaryConditions1D(ExtrapolateBC{1}(), ExtrapolateBC{1}())
    return SemidiscretizationHyperbolic(
        mesh, equations,
        initial_condition_barenblatt,
        solver;
        source_terms=source_terms,
        boundary_conditions=bc
    )
end

# Run convergence test
convergence_test(
    make_semi,
    [100, 200, 400, 800],
    (1.0, 1.2),
    ImplicitEulerCustom();
    exact_solution=exact_solution_barenblatt
)