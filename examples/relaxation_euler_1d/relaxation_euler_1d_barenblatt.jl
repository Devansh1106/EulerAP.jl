using EulerAP

# t has to be positive for Barenblatt solution
# The result from this IC is not what we are expecting. With ϵ=1, we expect the solution to be far away from exact sol (BarenBlatt is 
# an exact sol) and as ϵ ≪ 1, we expect the solution to approach the exact sol. But currently the behavior is exactly opposite, it 
# matches with ϵ=1 and moves away as we reduce ϵ ≪ 1.

# Why we expect whatever is mentioned above: Since as ϵ→0, the relaxed Euler system converges to Porous Medium Equation (PME) whose
# exact solution is the BarenBlatt hence for ϵ ≪ 1, we expect it go closer to the BarenBlatt exact solution and be far away for ϵ=1.

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (1000,),
    (-6.0,),
    (6.0,)
    # periodicity = (true,)
)

# --------------------------------------------------
# Equations
# --------------------------------------------------

equations = RelaxationEulerEquations1D(;
    gamma = 3.0,
    epsilon = 1e-4
)

# --------------------------------------------------
# Solver
# --------------------------------------------------

solver = FVSolver(
    flux = FluxRusanov(),
    ndims = 1
)

# --------------------------------------------------
# Boundary conditions
# --------------------------------------------------

boundary_conditions = BoundaryConditions1D(
    ExtrapolateBC{1}(),
    ExtrapolateBC{1}()
)

# --------------------------------------------------
# Exact solution (Barenblatt)
# --------------------------------------------------

function barenblatt(x, t, Γ, γ)
    t_eff = Float64(t)
    β = 1.0 / (γ + 1.0)
    ξ = x / (t_eff^β)
    factor = (γ - 1.0) / (2.0 * γ * (γ + 1.0))
    bracket = Γ - factor * (ξ^2)
    positive = max(bracket, 0.0)
    ρ = t_eff^(-β) * (positive^(1.0 / (γ - 1.0)))
    return ρ
end

function exact_solution_barenblatt(x, t, equations)
    gamma = equations.gamma
    ρ = barenblatt(x[1], t, 1.0, gamma)
    β = 1.0 / (gamma + 1.0)
    u = β * x[1] / t
    mx = ρ * u
    return (ρ, mx)
end

# --------------------------------------------------
# Semidiscretization
# --------------------------------------------------

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition_barenblatt,
    solver;
    source_terms = source_terms,
    boundary_conditions = boundary_conditions
)

# --------------------------------------------------
# Time integration
# --------------------------------------------------

tspan = (1.0, 1.2)

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = CallbackSet(
    AliveCallback(),
    PerformanceCallback(),
    AnalysisCallback(exact_solution = exact_solution_barenblatt),
    SummaryCallback()
)

# --------------------------------------------------
# Output
# --------------------------------------------------

const OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")
eps_str  = equations.epsilon

initial_filename =
    "relaxation_euler_1d_barenblatt_$(mesh_str)_$(eps_str)_initial.h5"

solution_filename =
    "relaxation_euler_1d_barenblatt_$(mesh_str)_$(eps_str).h5"

# --------------------------------------------------
# Save initial condition
# --------------------------------------------------

save_initial_condition(
    semi,
    joinpath(OUTPUT_DIR, initial_filename);
    t = first(tspan)
)

# --------------------------------------------------
# Solve
# --------------------------------------------------

sol = solve(
    semi,
    tspan,
    ImplicitEulerCustom();
    callbacks = callbacks
)

# --------------------------------------------------
# Save final solution
# --------------------------------------------------

save_solution(
    sol,
    semi,
    joinpath(OUTPUT_DIR, solution_filename)
)
