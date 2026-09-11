using EulerAP
using Printf   # for the final-time text dump at the bottom of this file

# --------------------------------------------------
# Smooth sech density bump, at rest
# --------------------------------------------------
#
#     ρ(0, x) = 1 - 0.3 sech(2x),   v(0, x) = 0,   x ∈ [-10, 10],   T = 1
#
# A single run of `smooth_ic` (see its docstring): a shallow, localised density
# depression on a uniform background, released from rest. The ambipolar field
# set up by the depression pulls the plasma inwards, so the bump fills in and
# launches a pair of outward-travelling waves.
#
# The bump is localised — sech(2·10) ≈ 4e-9 — so the data is periodic on
# [-10, 10] to well below any discretization error, and periodic BCs can be
# used for both the hyperbolic and the elliptic part without introducing a jump
# at the domain edges.
#
# For the convergence study on this same state, see
# `euler_poisson_boltzmann_1d_smooth_convergence_test.jl`.

# --------------------------------------------------
# Mesh
# --------------------------------------------------

mesh = CartesianMesh(
    (400,),
    (-10.0,),
    (10.0,),
    periodicity = (true,)   # For hyperbolic part only
)

lambda = 1e0
tspan = (0.0, 1.0)

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
    flux = FluxEnergyStable(0.0), #0.0 is dummy; it will be overwritten inside by calculating η at every time step
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
    smooth_ic,
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
# This solution stays smooth, so `nolimiter` is the choice that keeps the
# scheme's full second order; `minmod` would clip the slope at the extremum of
# the sech.
limiter = minmod

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

# The IMEX schemes pick their own step from a CFL condition, and this flow is
# slow (|v| stays below 0.05), so T = 1 is reached in a handful of steps — the
# callback intervals are 1 accordingly, or nothing would ever be reported.
callbacks = CallbackSet(
    AliveCallback(interval=1),
    PerformanceCallback(),
    SummaryCallback(),
    AnalysisCallback(interval=1)
)

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = "data_new"

mesh_str = join(mesh.cells_per_dimension, "x")

initial_filename =
    "euler_poisson_boltzmann_1d_smooth_$(mesh_str)_initial_second.h5"

solution_filename =
    "euler_poisson_boltzmann_1d_smooth_$(mesh_str)_$(lambda)_$(last(tspan))_second.h5"

# Same stem as the HDF5 file, so the two outputs of a run sit side by side.
profile_filename =
    "euler_poisson_boltzmann_1d_smooth_$(mesh_str)_$(lambda)_$(last(tspan))_second.txt"

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

# --------------------------------------------------
# Write the final-time profile as text
# --------------------------------------------------
#
# `save_solution` above already stores everything in HDF5; this writes the same
# final state as a plain whitespace-separated table so it can be eyeballed in a
# terminal or read by anything that takes columns (gnuplot, np.loadtxt, ...)
# without an HDF5 reader. Deliberately local to this example rather than added
# to `io/`: it is a convenience for inspecting this one smooth run.

"""
    write_solution_profile(sol, semi, filename)

Write the final-time solution as one row per cell, columns `x  rho  v  phi`:

    # t = 1.0
    #              x               rho                 v               phi
      -9.90000000e+00    1.00000000e+00    ...

Velocity is reported rather than momentum — `v = m / ρ`, the primitive variable
one actually looks at — and `x` is the cell centre, taken from
`coordinates` so it matches the mesh the solver used instead of being
re-derived here.
"""
# function write_solution_profile(sol, semi, filename)

#     mesh    = semi.mesh
#     nvars_h = EulerAP.nvariables(semi.equations)             # (ρ, m)
#     nvars_e = EulerAP.nvariables(semi.equations_elliptic)    # (φ,)
#     nc      = EulerAP.ncells(mesh)

#     # `solution_vector`/`solution_time` rather than `sol.u`/`sol.t` so this
#     # keeps working for either solution type `solve` may hand back (see their
#     # definitions in io/save_solution.jl).
#     u = EulerAP.solution_vector(sol)
#     t = EulerAP.solution_time(sol)

#     # Block layout: the hyperbolic block [ρ₁, m₁, ..., ρₙ, mₙ] comes first,
#     # the elliptic block [φ₁, ..., φₙ] follows it — same convention
#     # `save_solution` documents and unpacks.
#     phi_offset = nvars_h * nc

#     mkpath(dirname(filename))

#     open(filename, "w") do io
#         println(io, "# euler_poisson_boltzmann_1d_smooth")
#         println(io, "# t      = ", t)
#         println(io, "# lambda = ", lambda)
#         println(io, "# ncells = ", nc, "   dx = ", mesh.dx[1])
#         @printf(io, "#%15s %17s %17s %17s\n", "x", "rho", "v", "phi")

#         for cell in 1:nc
#             x   = EulerAP.coordinates(CartesianIndex(cell), mesh)[1]
#             rho = u[EulerAP.global_dof(cell, 1, nvars_h)]
#             m   = u[EulerAP.global_dof(cell, 2, nvars_h)]
#             phi = u[phi_offset + EulerAP.global_dof(cell, 1, nvars_e)]

#             @printf(io, "%16.8e %17.8e %17.8e %17.8e\n", x, rho, m / rho, phi)
#         end
#     end

#     println("Saved final-time profile to ", filename)
#     return nothing
# end

# write_solution_profile(
#     sol,
#     semi,
#     joinpath(OUTPUT_DIR, profile_filename)
# )
