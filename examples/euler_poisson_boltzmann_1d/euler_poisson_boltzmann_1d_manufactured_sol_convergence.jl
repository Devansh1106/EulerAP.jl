using EulerAP

# --------------------------------------------------
# Self-referenced convergence test for the smooth manufactured state
# --------------------------------------------------
#
# `manufactured_ic` has no analytic time-dependent solution, so the error is
# measured *self-referentially* instead: each grid is compared against the next
# finer grid, coarsened back onto it by cell averaging,
#
#     e_N = || u_N - R(u_2N) ||
#
# If the scheme converges at order k then u_h = u + C hᵏ, so
# u_h - R(u_{h/2}) = C hᵏ (1 - 2⁻ᵏ): the self-error scales as hᵏ too and the
# EOC column still measures k (only the error constant differs from the
# exact-solution one).
#
# This is a deliberately separate pathway from `convergence_test`: it takes no
# `exact_solution`, returns `SelfConvergenceResult`s rather than
# `AnalysisResult`s, and labels its order column `Self-EOC`. The two error
# notions are not comparable and must not be mixed.
#
# With n grid sizes you get n-1 error rows: the finest grid appears only as the
# reference for the second-finest, never as a row of its own.

# --------------------------------------------------
# Build semi for a given grid size
# --------------------------------------------------

function make_semi(N)
    mesh = CartesianMesh(
        (N,),
        (0.0,),
        (1.0,),
        periodicity = (true,)   # for the hyperbolic part only
    )

    equations_hyperbolic = EulerPressureLess1D(gamma = 3.0)
    equations_elliptic   = PoissonBoltzmann(lambda = 1.0)

    solver = FVSolver(flux = FluxEnergyStable(0.0), ndims = 1)

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

    return SemidiscretizationHyperbolicElliptic(
        mesh,
        (equations_hyperbolic, equations_elliptic),
        manufactured_ic,
        solver;
        source_terms = source_terms_hyperbolic,
        source_terms_elliptic = nothing,
        boundary_conditions = boundary_conditions
    )
end

# --------------------------------------------------
# Run the self-referenced convergence test
# --------------------------------------------------
#
# Grid sizes must double exactly — 2:1 restriction is undefined otherwise, and
# `self_convergence_test` errors out rather than silently producing a wrong
# order.
#
# The solution should be smooth for EOC calculation, so `nolimiter` is the choice
# that exposes the scheme's formal second order; `minmod` clips the slope at the
# extrema of the sine and drags the observed order back towards 1.
#
# WATCH THE ARTIFACT FLOOR AT SHORT TIMES. Two O(h²) effects are present before
# the scheme has accumulated any error of its own:
#
#   * `manufactured_ic` is sampled at cell *centres*, so the initial data
#     differs from the true cell averages by O(h²);
#   * restricting φ (a point value, not a cell average) is O(h²) inexact.
#
# Running with tspan = (0.0, 0.0) isolates them: the table then reports a clean
# Self-EOC of exactly 2.000 for all three variables — that floor, not the
# scheme, is what is being measured. At t = 0.1 the accumulated scheme error is
# still the same size as this floor and the two partially cancel, which makes
# the Self-EOC column erratic (negative values, values above 2). By t = 0.5 the
# scheme error is ~20x the floor and the measurement is meaningful again.
#
# So: if the EOC column looks nonsensical, re-run with tspan = (0.0, 0.0) to
# see how big the floor is, and integrate long enough that the errors sit well
# above it.
self_convergence_test(
    make_semi,
    [40, 80, 160, 320],
    (0.0, 1.0),
    # IMEXIntegrator(FirstOrderThreeStagesIMEX());
    IMEXIntegrator(SecondOrderFiveStagesIMEX());
    limiter = nolimiter
)
