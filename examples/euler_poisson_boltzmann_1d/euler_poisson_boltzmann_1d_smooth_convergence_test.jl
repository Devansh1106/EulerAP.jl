using EulerAP

# --------------------------------------------------
# Reference-solution convergence test for a smooth sech density bump
# --------------------------------------------------
#
#     ρ(0, x) = 1 - 0.3 sech(2x),   u(0, x) = 0,   x ∈ [-10, 10],   T = 1
#
# `smooth_ic` (see its docstring): a shallow density depression on a uniform
# background, at rest. The perturbation is localised — sech(2·10) ≈ 4e-9 — so
# the data is periodic on [-10, 10] to well below any discretization error, and
# periodic BCs can be used for both the hyperbolic and the elliptic part
# without introducing a jump at the domain edges.
#
# There is no analytic solution for this state, so the N = 1280 run is used as
# the "exact" one: every measured grid is compared against that same reference,
# coarsened onto it by averaging 1280/N cells,
#
#     e_N = || u_N - R(u_1280) ||
#
# `ref_convergence_test` does the whole thing — the reference run, the per-grid
# runs, the coarsening factor (16, 8, 4, 2 here) and the table. It is a
# separate pathway from the other two: `convergence_test` needs an analytic
# `exact_solution`, and `self_convergence_test` compares each grid against the
# next finer grid of the sweep (2:1, consecutive) instead of against a fixed
# reference.

# --------------------------------------------------
# Build semi for a given grid size
# --------------------------------------------------

lambda = 1e0

function make_semi(N)
    mesh = CartesianMesh(
        (N,),
        (-10.0,),
        (10.0,),
        periodicity = (true,)   # for the hyperbolic part only
    )

    equations_hyperbolic = EulerPressureLess1D(gamma = 3.0)
    equations_elliptic   = PoissonBoltzmann(lambda = lambda)

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
        smooth_ic,
        solver;   # solver_elliptic defaults to NewtonRaphson() from NLS
        source_terms = source_terms_hyperbolic,
        source_terms_elliptic = nothing,   # built internally for this system
        boundary_conditions = boundary_conditions
    )
end

# --------------------------------------------------
# Run the reference-solution convergence test
# --------------------------------------------------
#
# Each measured grid must divide the reference exactly: the factor is
# `reference_grid_size / N` and whole reference cells have to tile one measured
# cell. It need not be a power of two, and the sweep need not double from row
# to row — the `Ref-EOC` column uses the actual ratio of consecutive rows.
#
# The solution stays smooth over T = 1 (the bump only starts to spread), so
# `nolimiter` is the choice that exposes the scheme's formal second order;
# `minmod` clips the slope at the extremum of the sech and drags the observed
# order back towards 1.
#
# WATCH THE ARTIFACT FLOOR AT SHORT TIMES. Two O(h²) effects are present before
# the scheme has accumulated any error of its own:
#
#   * `smooth_ic` is sampled at cell *centres*, so the initial data differs
#     from the true cell averages by O(h²);
#   * restricting φ (a point value, not a cell average) is O(h²) inexact — the
#     mean of the r reference centres in a measured cell of width H is
#     φ(x_j) + (H²/24)(1 - r⁻²) φ'', which a larger factor r does not shrink.
#
# Running with tspan = (0.0, 0.0) isolates them: the table then reports a clean
# Ref-EOC of 2 for all variables — that floor, not the scheme, is what is being
# measured. If the EOC column at T = 1 looks nonsensical, re-run with
# tspan = (0.0, 0.0) to see how big the floor is, and integrate long enough
# that the errors sit well above it.
#
# The coarse end is pre-asymptotic: the sech bump has a half-width of about 0.9
# on a domain of length 20, so N = 80 (Δx = 0.25) only just resolves it. The
# fine end has the reference floor: N = 640 is only one refinement from 1280,
# and its row carries some of the reference's own error.
#
# Measured at T = 1 with the settings below (Ref-EOC on the L2 column):
#
#            80 -> 160     160 -> 320     320 -> 640
#     ρ         1.06           2.31           2.57
#     ρu        2.07           1.94           2.02
#     φ         0.71           2.60           3.35
#
# i.e. second order once the bump is resolved. The whole sweep, reference
# included, takes about 20 s, so raising `reference_grid_size` is cheap if the
# last row is in doubt: at 2560 the 320 -> 640 entries come out as
# 1.95 / 1.73 / 2.61, which is how much of that row was the reference's error.

ref_convergence_test(
    make_semi,
    # [40, 80, 160, 320, 640, 1280, 2560],
    [100, 200, 400, 800, 1600],
    (0.0, 1.0),
    # IMEXIntegrator(FirstOrderThreeStagesIMEX());
    IMEXIntegrator(SecondOrderFiveStagesIMEX());
    reference_grid_size = 3200,
    limiter = nolimiter
)

# self_convergence_test(
#     make_semi,
#     [40, 80, 160, 320],
#     (0.0, 1.0),
#     # IMEXIntegrator(FirstOrderThreeStagesIMEX());
#     IMEXIntegrator(SecondOrderFiveStagesIMEX());
#     limiter = nolimiter
# )