using EulerAP

# tspan = (1.0, 1.2)
# t has to be positive for Barenblatt solution

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
    epsilon = 1e-6
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
    left  = ExtrapolateBC(),
    right = ExtrapolateBC()
)

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

sol = EulerAP.solve(
    semi,
    tspan,
    ImplicitEulerCustom() # dt = dx by default; can be overwritten
)

# --------------------------------------------------
# Save initial condition
# --------------------------------------------------

# mesh_str = join(mesh.cells_per_dimension, "x")
# init_filename = "relaxation_euler_1d_barenblatt_$(mesh_str)_$(equations.epsilon)_initial.h5"
# save_initial_condition(semi,
#                        joinpath("data_new", init_filename);
#                        t = first(tspan))


# --------------------------------------------------
# Output
# --------------------------------------------------

mesh_str = join(mesh.cells_per_dimension, "x")
filename = "relaxation_euler_1d_barenblatt_$(mesh_str)_$(equations.epsilon).h5"
save_solution(sol,
              semi,
              joinpath("data_new", filename))

println("Simulation complete.")