using EulerAP

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (100,),
    (-6.0,),
    (6.0,)
    # periodicity = (true,)
)

# --------------------------------------------------
# Equations
# --------------------------------------------------

equations = RelaxationEulerEquations1D(
    gamma = 3.0,
    epsilon = 1.0e-7
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
    initial_condition_single_box,
    solver;
    source_terms = source_terms,
    boundary_conditions = boundary_conditions
)

# --------------------------------------------------
# Time integration
# --------------------------------------------------

tspan = (0.0, 0.01)

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = CallbackSet(
    AliveCallback(),
    PerformanceCallback(),
    SummaryCallback()
)

# --------------------------------------------------
# Output
# --------------------------------------------------

const OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")
eps_str  = equations.epsilon

initial_filename =
    "relaxation_euler_1d_single_box_$(mesh_str)_$(eps_str)_initial.h5"

solution_filename =
    "relaxation_euler_1d_single_box_$(mesh_str)_$(eps_str).h5"

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