using EulerAP

# --------------------------------------------------
# Soliton convergence test
# --------------------------------------------------
#
# Sect. 5.1 of Degond, Liu, Savelief & Vignal (J. Sci. Comput. 2012): the
# EPB soliton is a travelling wave that keeps its shape and speed u0 exactly,
# so the exact solution at time t is just the t = 0 profile translated by a
# distance u0 * t (their Eq. (5.1)). This gives an analytical reference
# solution to measure the convergence order of the scheme against, the same
# way `initial_condition_barenblatt`/`exact_solution_barenblatt` do for the
# relaxation Euler Barenblatt test case.

u0, eta = 1.2, -1e-6   # soliton parameters (Bohm criterion: 1 < u0 < 1.6)

initial_condition_soliton, L = make_initial_condition_soliton(u0, eta)
exact_solution_soliton, _    = make_soliton_solution(u0, eta)

lambda = 1e-2   # dispersive regime (Debye length of order the domain size)

# --------------------------------------------------
# Build semi for a given grid size
# --------------------------------------------------

function make_semi(N)
    mesh = CartesianMesh(
        (N,),
        (0.0,),
        (L,),
        periodicity = (true,)   # for the hyperbolic part only
    )

    equations_hyperbolic = EulerPressureLess1D(gamma = 3.0)
    equations_elliptic   = PoissonBoltzmann(lambda = lambda)

    solver = FVSolver(flux = FluxEnergyStable(0.0), ndims = 1)

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

    return SemidiscretizationHyperbolicElliptic(
        mesh,
        (equations_hyperbolic, equations_elliptic),
        initial_condition_soliton,
        solver;
        source_terms = source_terms_hyperbolic,
        source_terms_elliptic = nothing,
        boundary_conditions = boundary_conditions
    )
end

# --------------------------------------------------
# Run convergence test
# --------------------------------------------------

t_L = L / u0   # time for the soliton to cross the whole periodic domain

# Small grid sizes and a short tspan (t_L/20 rather than a full domain
# crossing) — keeps this test fast to iterate on; the CFL-limited IMEX
# schemes here get expensive fast at N >= 400-800.
#
# The `limiter` keyword picks the slope limiter used by the second-order
# scheme's reconstruction. Available choices:
#
#   minmod              default; TVD, but clips the slope at smooth extrema,
#                       which drags the measured EOC back towards 1
#   nolimiter           plain central slope (ρ_{i+1} - ρ_{i-1}) / (2Δx); no
#                       limiting at all — second order, but not TVD
#   MinmodTheta(θ)      generalized minmod, θ ∈ [1, 2]: θ = 1 is the classical
#                       minmod, θ = 2 the most compressive TVD choice. Reduces
#                       to the central slope on smooth data
#   CWENO(ε)            nonlinear φ-weighted average of the one-sided slopes,
#                       φ(s) = (ε + s²)⁻², default ε = 1e-6; central on smooth
#                       data, suppresses the steep side at a discontinuity
#
# The soliton here is smooth, so `nolimiter` measures the scheme's formal order
# without limiter interference. Use `minmod`, `MinmodTheta(...)` or `CWENO()`
# for any test case with discontinuities.
convergence_test(
    make_semi,
    [100, 200, 400, 800],
    (0.0, t_L / 5),
    # IMEXIntegrator(FirstOrderThreeStagesIMEX());
    IMEXIntegrator(SecondOrderFiveStagesIMEX());
    exact_solution = exact_solution_soliton,
    limiter = nolimiter
)
