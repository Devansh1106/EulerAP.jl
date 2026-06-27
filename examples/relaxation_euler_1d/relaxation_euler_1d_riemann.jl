using EulerAP

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
    initial_condition_single_box,
    solver;
    source_terms = source_terms,
    boundary_conditions = boundary_conditions
)

# --------------------------------------------------
# Save initial condition
# --------------------------------------------------

mesh_str = join(mesh.cells_per_dimension, "x")
init_filename = "relaxation_euler_1d_riemann_$(mesh_str)_$(equations.epsilon)_initial.h5"
save_initial_condition(semi,
                       joinpath("data_new", init_filename))

# --------------------------------------------------
# Time integration
# --------------------------------------------------

tspan = (0.0, 0.01)

sol = EulerAP.solve(
    semi,
    tspan,
    ImplicitEulerCustom() # dt = dx by default; can be overwritten
)

# --------------------------------------------------
# Output
# --------------------------------------------------

mesh_str = join(mesh.cells_per_dimension, "x")
filename = "relaxation_euler_1d_riemann_$(mesh_str)_$(equations.epsilon).h5"
save_solution(sol,
              semi,
              joinpath("data_new", filename))

println("Simulation complete.")