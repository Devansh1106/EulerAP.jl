using EulerAP

# --------------------------------------------------
# Smooth manufactured test case
# --------------------------------------------------
#
# Initial state (see `manufactured_ic`):
#
#     ρ(0, x) = 1 + 0.1 sin(2πx)
#     v(0, x) = 0.1 cos(2πx)
#
# on the unit interval with periodic boundaries in *all* variables — both the
# hyperbolic (ρ, m) part and the elliptic potential φ. φ(0, ·) is not
# prescribed: as for every initial condition of this coupled system it is
# obtained by solving -λ²Δφ + e^φ = ρ(0, ·) at t = 0.
#
# Everything here is smooth and bounded away from vacuum (ρ ∈ [0.9, 1.1]),
# which is what makes it useful as a reference run for the second-order scheme:
# no discontinuity for the limiter to clip, no near-vacuum region to stress
# positivity. Note that no analytic solution (and no compensating source term)
# is defined for this state, so this file is a plain forward run — to measure
# an EOC you need a reference solution, e.g. a fine-grid run or an added source
# term making a chosen profile exact.

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (200,),
    (0.0,),
    (1.0,),
    periodicity = (true,) # For hyperbolic part only
)

lambda = 1e0

t_final = 1.0
tspan = (0.0, t_final)

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
    # hyperbolic case 1D (periodic)
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
    manufactured_ic,
    solver; # solver_elliptic is default to NewtonRaphson() from NLS
    source_terms = source_terms_hyperbolic,
    source_terms_elliptic = nothing, # elliptic source term is internally constructed for this system
    boundary_conditions = boundary_conditions # tuple for hyperbolic and elliptic cases
)

# --------------------------------------------------
# Time integration
# --------------------------------------------------

integrator = IMEXIntegrator(
    # FirstOrderThreeStagesIMEX()
    SecondOrderFiveStagesIMEX()
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
#
# The solution here is smooth, so `nolimiter` is the choice that shows the
# scheme's formal second order; `minmod` clips the slope at the extrema of the
# sine and drags the observed order back towards 1.
limiter = nolimiter

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
    "euler_poisson_boltzmann_1d_manufactured_sol_$(mesh_str)_initial.h5"

solution_filename =
    "euler_poisson_boltzmann_1d_manufactured_sol_$(mesh_str)_$(lambda)_$(t_final).h5"

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
