using EulerAP


u0, eta = 1.2, -1e-6 # parameters required for soliton test case calculation
initial_condition_soliton, L = make_initial_condition_soliton(u0, eta)

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (800,),
    (0.0,),
    (L,),
    periodicity = (true,) # For hyperbolic part only
)
lambda = 1.0
t = L/u0
tspan = (0.0, t)

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

boundary_conditions = (
    BoundaryConditions1D(
        PeriodicBC{1}(),
        PeriodicBC{1}()
    ),
    # elliptic case 1D (periodic)
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
    initial_condition_soliton,
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

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = CallbackSet(
    AliveCallback(interval=20),
    PerformanceCallback(),
    SummaryCallback(),
    AnalysisCallback(interval=20)
)

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")

initial_filename =
    "euler_poisson_boltzmann_1d_soliton_$(mesh_str)_initial.h5"

solution_filename =
    "euler_poisson_boltzmann_1d_soliton_$(mesh_str)_$(lambda).h5"

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

sol = solve(semi,
            tspan,
            integrator;
            callbacks = callbacks)

# --------------------------------------------------
# Save final solution
# --------------------------------------------------

save_solution(
    sol,
    semi,
    joinpath(OUTPUT_DIR, solution_filename)
)