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

lambda = 1.0   # dispersive regime (Debye length of order the domain size)

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
convergence_test(
    make_semi,
    [200, 400, 800, 1600],
    (0.0, t_L / 5),
    # IMEXIntegrator(FirstOrderThreeStagesIMEX());
    IMEXIntegrator(SecondOrderFiveStagesIMEX());
    exact_solution = exact_solution_soliton
)
