using EulerAP

# --------------------------------------------------
# Initial condition
# --------------------------------------------------

# function initial_condition(x, t, equations)

#     ρ = 1.0 + 0.2 * sin(2π * x[1])
#     m = 0.0

#     return (ρ, m)
# end

const rho_L = 1.0
const rho_R = 0.0
const x_m   = 0.5

function initial_condition(x, t, equations)
    rho0 = x < x_m ? rho_L : rho_R

    return (rho0, 0.0)
end

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (100,),
    (0.0,),
    (1.0,),
    periodicity = (true,)
)

# --------------------------------------------------
# Equations
# --------------------------------------------------

equations = RelaxationEulerEquations1D(
    gamma = 2.0,
    epsilon = 1.0
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

boundary_conditions = (
    left  = PeriodicBC(),
    right = PeriodicBC()
)

# --------------------------------------------------
# Semidiscretization
# --------------------------------------------------

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition,
    solver;
    boundary_conditions = boundary_conditions
)

# --------------------------------------------------
# Time integration
# --------------------------------------------------

tspan = (0.0, 0.01)

sol = solve(
    semi,
    tspan,
    ImplicitEulerCustom() # dt = dx by default; can be overwritten
)

# --------------------------------------------------
# Output
# --------------------------------------------------

save_solution(
    sol,
    semi,
    "relaxation_euler_1d_periodic.h5"
)

println("Simulation complete.")