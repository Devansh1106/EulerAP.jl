using EulerAP


u0, eta = 1.2, -1e-6 # parameters required for soliton test case calculation
initial_condition_soliton, L = make_initial_condition_soliton(u0, eta)
exact_solution_soliton, _    = make_soliton_solution(u0, eta)
# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (2000,),
    (0.0,),
    (L,),
    periodicity = (true,) # For hyperbolic part only
)
lambda = 1e0
t = L/u0
# t_L = t/5.0
# t_L = (2.0*t)/5.0
t_L = t
tspan = (0.0, t_L)

# --------------------------------------------------
# Equations
# --------------------------------------------------

# Hyperbolic: pressure-less Euler
equations_hyperbolic = EulerPressureLess1D(
    gamma = 3.0
)

# Elliptic: Poisson-Boltzmann
equations_elliptic = PoissonBoltzmann(
    lambda = lambda
)

# --------------------------------------------------
# Solver
# --------------------------------------------------

solver = FVSolver(
    flux = FluxEnergyStable(0.0),
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
    # SecondOrderFiveStagesIMEX()
)

# Slope limiter used by the second-order scheme's reconstruction. Choices:
#
#   minmod              default; classical minmod, TVD
#   nolimiter           plain central slope (Uᵢ₊₁ - Uᵢ₋₁) / (2Δx); no limiting
#                       at all — second order, but not TVD, so it can oscillate
#                       and lose positivity of ρ on discontinuous data
#   MinmodTheta(θ)      generalized minmod, θ ∈ [1, 2]; θ = 1 reduces to
#                       `minmod`, θ = 2 is the most compressive TVD choice
#   CWENO(ε)            nonlinear φ-weighted average of the one-sided slopes,
#                       φ(s) = (ε + s²)⁻², default ε = 1e-6
#
# The parameterized ones are structs and always need the parentheses, even for
# their defaults: `MinmodTheta()`, `CWENO()`. Ignored by the first-order scheme,
# which reconstructs nothing.
limiter = minmod

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = CallbackSet(
    AliveCallback(interval=20),
    PerformanceCallback(),
    SummaryCallback(),
    AnalysisCallback(exact_solution = exact_solution_soliton)
)

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")

initial_filename =
    "euler_poisson_boltzmann_1d_soliton_$(mesh_str)_initial.h5"

solution_filename =
    "euler_poisson_boltzmann_1d_soliton_$(mesh_str)_$(lambda)_$(t_L).h5"

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
            limiter = limiter,
            callbacks = callbacks)

# --------------------------------------------------
# Save final solution
# --------------------------------------------------

save_solution(
    sol,
    semi,
    joinpath(OUTPUT_DIR, solution_filename)
)