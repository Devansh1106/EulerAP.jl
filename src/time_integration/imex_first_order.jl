# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

using TimerOutputs

# ============================================================================
# Stage 1: Explicit Correction (density predictor)
# ============================================================================
# ρ̂_i = ρ_i^n - Δt/Δx * [ρ̄_{i+1/2} * (u_i + u_{i+1})/2
#                         - ρ̄_{i-1/2} * (u_{i-1} + u_i)/2]
# ============================================================================

function perform_stage!(
    ::ExplicitCorrectionStage,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCacheFirstOrder,
    integrator::IMEXIntegrator,
    dt,
    t,
)

    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma
    nvars = nvariables(equations)

    # NOTE: EPB is 1D-only today (see the note on ImplicitPredictionStage
    # below), so this stage only ever runs for axis = 1 in practice.
    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # Seed with ρⁿ; interface flux differences are subtracted below — one
    # `explicit_density_flux` evaluation per interface instead of two.
    @inbounds for i in 1:nx
        cache.rho_hat[i] = cache.u[global_dof(i, 1, nvars)]
    end

    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx - 1)
        rho_l, vel_l = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(i), t)
        rho_r, vel_r = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(i + 1), t)
        rho_half = gamma_mean(rho_l, rho_r, gamma)

        f = explicit_density_flux(vel_l, vel_r, rho_half)
        cache.rho_hat[i]     -= (dt / dx) * f
        cache.rho_hat[i + 1] += (dt / dx) * f
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(nx), t)
        rho_r, vel_r = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(1), t)
        rho_half = gamma_mean(rho_l, rho_r, gamma)

        f = explicit_density_flux(vel_l, vel_r, rho_half)
        cache.rho_hat[nx] -= (dt / dx) * f
        cache.rho_hat[1]  += (dt / dx) * f
    else
        rho_gl, vel_gl = rho_vel_at(cache.u, cache.vel, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), t)
        rho_1,  vel_1  = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(1), t)
        rho_half = gamma_mean(rho_gl, rho_1, gamma)

        f_l = explicit_density_flux(vel_gl, vel_1, rho_half)
        cache.rho_hat[1] += (dt / dx) * f_l

        rho_nx, vel_nx = rho_vel_at(cache.u, cache.vel, semi, CartesianIndex(nx), t)
        rho_gr, vel_gr = rho_vel_at(cache.u, cache.vel, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), t)
        rho_half = gamma_mean(rho_nx, rho_gr, gamma)

        f_r = explicit_density_flux(vel_nx, vel_gr, rho_half)
        cache.rho_hat[nx] -= (dt / dx) * f_r
    end

    return nothing
end

# ============================================================================
# Stage 2: Implicit Prediction (solve elliptic equation)
# ============================================================================
# -α(x_{i-1} - 2x_i + x_{i+1}) + f(x_i) = ρ̂_i all at time step n+1
# where α = (λ² + ηΔt^2) / Δx² and f(x) = exp(x) for Poisson-Boltzmann
# NOTE: Currently 1D-only; 2D requires a 5-point stencil sparse solver.
# ============================================================================

function perform_stage!(
    ::ImplicitPredictionStage,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCacheFirstOrder,
    integrator::IMEXIntegrator,
    dt,
    t,
)

    λ = semi.equations_elliptic.lambda

    # assembly functions use these to add the
    # per-cell η dt² ρ̄ⁿ_{i±1/2} corrections to the constant-coefficient
    # Laplacian stencil.
    params = semi.cache_elliptic.newton_cache.params
    params.u = cache.u
    params.eta = cache.eta
    params.dt  = dt

    solve_newton!(
        cache.phi,
        cache.rho_hat,
        -λ^2,
        semi,
        t,
    )

    return nothing
end

# ============================================================================
# Stage 3: Implicit Correction (final ρ, ρu update)
# ============================================================================
# F^1_{face} = ρ̄ * (u_r + u_l)/2 - (x_r - x_l) * ηΔt/Δx
# F^2_{face} = u_l * max(F^1, 0) + u_r * min(F^1, 0)
# S^2_{face} = -ρ̄ * (x_r - x_l) / Δx
#
# Swept independently along each Cartesian direction.
# ============================================================================

function perform_stage!(
    ::ImplicitCorrectionStage,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCacheFirstOrder,
    integrator::IMEXIntegrator,
    dt,
    t,
)
    equations = semi.equations
    solver    = semi.solver

    mesh  = semi.mesh
    nvars = nvariables(equations)

    # NOTE: EPB is 1D-only today (see the note on ImplicitPredictionStage
    # above), so this stage only ever runs for axis = 1 in practice.
    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    eta      = cache.eta
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # Stage starts from uⁿ. `cache.u` is only ever READ during the loop
    # below (never written), so it can be read directly — no need to copy
    # it into a separate buffer first. `write_state` is where the interface
    # deltas actually accumulate; it must start as a copy of uⁿ since we
    # need cell i's ORIGINAL state still available when a later face touches
    # cell i again (face i-1/i and face i/i+1 both read cell i).
    read_state  = cache.u
    write_state = cache.u_buffer
    copyto!(write_state, read_state)

    state_phi(I) = (cell_state(read_state, I, semi, t),
                    _elliptic_var(cache.phi, semi, I, t))

    # One `solver.flux` evaluation per interface, applied to both neighbors:
    # `flux` with opposite sign (conservative), `source` with the SAME sign
    # onto both (nonconservative-like — matches the original
    # `(dt/2)*(right.source + left.source)` per-cell accumulation, since an
    # interior cell touches exactly two interfaces).
    function apply_interface!(cell_l, cell_r, u_l, phi_l, u_r, phi_r)
        contrib = solver.flux(u_l, u_r, phi_l, phi_r, 1, equations, dt, dx, eta)

        rho_l_idx, mom_l_idx = global_dof(cell_l, 1, nvars), global_dof(cell_l, 2, nvars)
        rho_r_idx, mom_r_idx = global_dof(cell_r, 1, nvars), global_dof(cell_r, 2, nvars)

        write_state[rho_l_idx] -= (dt / dx) * contrib.flux[1]
        write_state[rho_r_idx] += (dt / dx) * contrib.flux[1]
        write_state[mom_l_idx] -= (dt / dx) * contrib.flux[2]
        write_state[mom_r_idx] += (dt / dx) * contrib.flux[2]

        write_state[mom_l_idx] += (dt / 2) * contrib.source
        write_state[mom_r_idx] += (dt / 2) * contrib.source

        return nothing
    end

    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx - 1)
        u_l, phi_l = state_phi(CartesianIndex(i))
        u_r, phi_r = state_phi(CartesianIndex(i + 1))
        apply_interface!(i, i + 1, u_l, phi_l, u_r, phi_r)
    end

    # Boundary face(s)
    if periodic
        u_l, phi_l = state_phi(CartesianIndex(nx))
        u_r, phi_r = state_phi(CartesianIndex(1))
        apply_interface!(nx, 1, u_l, phi_l, u_r, phi_r)
    else
        Ig_l = neighbor_index(CartesianIndex(1), semi, 1, -1)
        u_gl, phi_gl = state_phi(Ig_l)
        u_1,  phi_1  = state_phi(CartesianIndex(1))
        contrib = solver.flux(u_gl, u_1, phi_gl, phi_1, 1, equations, dt, dx, eta)
        rho_1_idx, mom_1_idx = global_dof(1, 1, nvars), global_dof(1, 2, nvars)
        write_state[rho_1_idx] += (dt / dx) * contrib.flux[1]
        write_state[mom_1_idx] += (dt / dx) * contrib.flux[2]
        write_state[mom_1_idx] += (dt / 2) * contrib.source

        u_nx, phi_nx = state_phi(CartesianIndex(nx))
        Ig_r = neighbor_index(CartesianIndex(nx), semi, 1, 1)
        u_gr, phi_gr = state_phi(Ig_r)
        contrib = solver.flux(u_nx, u_gr, phi_nx, phi_gr, 1, equations, dt, dx, eta)
        rho_nx_idx, mom_nx_idx = global_dof(nx, 1, nvars), global_dof(nx, 2, nvars)
        write_state[rho_nx_idx] -= (dt / dx) * contrib.flux[1]
        write_state[mom_nx_idx] -= (dt / dx) * contrib.flux[2]
        write_state[mom_nx_idx] += (dt / 2) * contrib.source
    end

    # NDIMS == 1, so the single sweep above is already the final state —
    # advance directly from `write_state`.
    copyto!(cache.u, write_state)
    return nothing
end

# ============================================================================
# Main driver: solve_imex
# ============================================================================

"""
    solve_imex(semi::AbstractSemidiscretization,
               integrator::IMEXIntegrator,
               tspan,
               scheme::FirstOrderThreeStagesIMEX;
               dt,
               callbacks = CallbackSet())

Advance the semidiscretization using an IMEX time integrator.
"""

function solve_imex(semi::AbstractSemidiscretization,
                    integrator::IMEXIntegrator,
                    tspan,
                    scheme::FirstOrderThreeStagesIMEX;
                    dt,
                    # Accepted (but unused: this scheme is piecewise constant
                    # in space and reconstructs no slopes) so that a limiter
                    # choice can be passed to `solve`/`convergence_test`
                    # without knowing which IMEX scheme is in use.
                    limiter = nothing,
                    callbacks = CallbackSet())

    t = first(tspan)

    # Full initial state in block layout: [ρ₁, m₁, ρ₂, m₂, ..., ρₙ, mₙ, φ₁, φ₂, ..., φₙ]
    u = initial_condition(t, semi)

    nvars_hyper     = nvariables(semi.equations)
    nvars_elliptic  = nvariables(semi.equations_elliptic)

    nc = ncells(semi.mesh)

    n_hyper    = nvars_hyper * nc
    n_elliptic = nvars_elliptic * nc

    # Use views into the block-layout state for the IMEX solver:
    #   u_hyper = u[1:n_hyper]  = [ρ₁, m₁, ρ₂, m₂, ..., ρₙ, mₙ]
    #   phi     = u[n_hyper+1:end] = [φ₁, φ₂, ..., φₙ]
    u_hyper = @view u[1:n_hyper]
    phi     = @view u[n_hyper+1:end]

    # --------------------------------------------------
    # IMEX cache
    # --------------------------------------------------

    cache = IMEXCacheFirstOrder(
        u_hyper,
        phi,
    )

    iteration = 0

    # --------------------------------------------------
    # Callback infrastructure
    # --------------------------------------------------

    simulation = Simulation(
        semi,
        integrator,
        tspan,
        dt,
        zero(eltype(u)),
        zero(eltype(u)),
    )

    stats = CallbackStats(eltype(u))
    semi.cache.stats = stats

    # Initial mass
    stats.initial_mass = total_mass(cache.u, semi)

    context = CallbackContext(
        simulation,
        EulerAPSolution(u, t),
        stats,
    )

    initialize_callbacks!(callbacks, context)

    # --------------------------------------------------
    # Time stepping
    # --------------------------------------------------

    @timeit stats.timer "total_runtime" begin

        while t < last(tspan)
            # Velocity derived from uⁿ, cached for the rest of this
            # timestep (compute_dt_2!, ExplicitCorrectionStage!) instead of
            # each re-deriving it via division — see
            # update_primitive_variables!/rho_vel_at.
            update_primitive_variables!(cache, semi)

            # Compute diffusion coefficient from current density state
            # (must happen BEFORE compute_dt_2!, which uses cache.eta)
            compute_eta!(cache, semi, t)
            cfl_dt = compute_dt_2!(cache, semi, t)
            actual_dt = min(cfl_dt, last(tspan) - t)

            stats.mass_before = total_mass(cache.u, semi)

            # Stage 1: Explicit density correction (evaluated at current time t)
            @timeit stats.timer "ExplicitCorrectionStage" begin
                perform_stage!(
                    ExplicitCorrectionStage(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t,
                )
            end

            stats.mass_after_stage1 =
                sum(cache.rho_hat) * prod(semi.mesh.dx)

            stats.relative_mass_error_stage1 =
                abs(stats.mass_after_stage1 -
                    stats.mass_before) /
                abs(stats.mass_before)

            # Stage 2: Implicit elliptic solve (predicts φ at future time t + dt)
            @timeit stats.timer "ImplicitPredictionStage" begin
                perform_stage!(
                    ImplicitPredictionStage(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t + actual_dt,
                )
            end

            # Stage 3: Implicit correction (evaluated at current time t)
            @timeit stats.timer "ImplicitCorrectionStage" begin
                perform_stage!(
                    ImplicitCorrectionStage(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t,
                )
            end

            stats.mass_after_stage3 =
                total_mass(cache.u, semi)

            stats.relative_mass_error_stage3 =
                abs(stats.mass_after_stage3 -
                    stats.mass_before) /
                abs(stats.mass_before)

            stats.minimum_density =
                minimum_density(cache.u, semi)

            stats.maximum_velocity =
                maximum_velocity(cache.u, semi)

            t += actual_dt
            iteration += 1

            stats.iteration = iteration
            stats.time = t
            stats.dt = actual_dt

            context.solution = EulerAPSolution(u, t)

            perform_callbacks!(
                callbacks,
                context,
            )

            delete!(stats.timer.inner_timers, "ExplicitCorrectionStage")
            delete!(stats.timer.inner_timers, "ImplicitPredictionStage")
            delete!(stats.timer.inner_timers, "ImplicitCorrectionStage")
        end
    end
    finalize_callbacks!(callbacks, context)
    return EulerAPSolution(u, t)
end

end # @muladd