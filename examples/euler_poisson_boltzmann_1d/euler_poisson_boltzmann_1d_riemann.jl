using EulerAP

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (100,),
    (0.0,),
    (1.0,)
    # periodicity = (true,)
)
lambda = 1e-0

# --------------------------------------------------
# Equations
# --------------------------------------------------

# Hyperbolic: pressure-less Euler
equations_hyperbolic = EulerPressureLess1D(
    gamma = 1.4
)

# Elliptic: Poisson-Boltzmann
equations_elliptic = PoissonBoltzmann(
    lambda = lambda
)

# --------------------------------------------------
# Solver
# --------------------------------------------------

solver = FVSolver(
    flux = FluxEnergyStable(1.0),
    ndims = 1
)

# --------------------------------------------------
# Boundary conditions
# --------------------------------------------------

# function dir_bc(x, t, equations)
#     return (0.0, 0.0)
# end

# function dir_bc_hyp(x, t, equations)
#     return 0.0
# end

boundary_conditions = (
    # hyperbolic case 1D
    BoundaryConditions1D(
        PeriodicBC{1}(),
        PeriodicBC{1}()
    ),
    # elliptic case 1D
    BoundaryConditions1D(
        PeriodicBC{1}(),
        PeriodicBC{1}()
    )
)

# --------------------------------------------------
# Semidiscretization
# --------------------------------------------------

semi = SemidiscretizationHyperbolicElliptic(
    mesh,
    (equations_hyperbolic, equations_elliptic),
    initial_condition_riemann_epb,
    solver; # solver_elliptic is default to NewtonRaphson() from NLS
    source_terms = source_terms_hyperbolic,
    source_terms_elliptic = nothing, # elliptic source term is internally constructed for this system
    boundary_conditions = boundary_conditions # tuple for hyperbolic and elliptic cases
)

# --------------------------------------------------
# Time integration
# --------------------------------------------------

# IMEX integrator with first-order 3-stage scheme
integrator = IMEXIntegrator(
    FirstOrderThreeStagesIMEX()
)

tspan = (0.0, 0.1)

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

OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")

initial_filename =
    "euler_poisson_boltzmann_1d_riemann_$(mesh_str)_initial.h5"

solution_filename =
    "euler_poisson_boltzmann_1d_riemann_$(mesh_str)_$(lambda).h5"

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
    integrator;
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