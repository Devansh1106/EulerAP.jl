# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

using TimerOutputs

@inline function _hyperbolic_state(u, I, semi, t)
    nvars = nvariables(semi.equations)
    state = cell_state(u, I, semi, t)
    return SVector{nvars}(ntuple(k -> @inbounds(state[k]), nvars))
end

"""
    update_primitive_variables!(cache::IMEXCache, semi)

Compute velocity `vel = m/ρ` at every interior cell from the current
conservative state `cache.u` and store it in `cache.vel`. Called once per
timestep, before any stage reads velocity, so `compute_dt_1!`,
`compute_dt_2!`, and `ExplicitCorrectionStage!` all read the same
precomputed values via `rho_vel_at` instead of each re-deriving `vel` from
`cache.u` independently (previously up to several times per cell per
timestep, across those three functions).

Only interior cells are cached: ghost/boundary states depend on `t` and the
boundary condition and are touched far less often, so they're computed on
the fly by `rho_vel_at` instead.
"""
function update_primitive_variables!(cache::IMEXCache, semi::AbstractSemidiscretization)
    mesh  = semi.mesh
    nvars = nvariables(semi.equations)
    nx    = size(mesh, 1)

    @inbounds for i in 1:nx
        rho = cache.u[global_dof(i, 1, nvars)]
        if rho <= 0
            error("""
                  Negative density entering update_primitive_variables!
                  cell = $i
                  rho  = $rho
                  """)
        end
        mom = cache.u[global_dof(i, 2, nvars)]
        cache.vel[i] = mom / rho
    end

    return nothing
end

"""
    rho_vel_at(cache::IMEXCache, semi, I, t)

Return `(rho, vel)` at cell/ghost index `I`. Interior cells (including a
`neighbor_index`-clamped `ExtrapolateBC`/`NeumannBC` ghost, which resolves
to a valid interior index) read `cache.u`/`cache.vel` directly — no
division. Genuine out-of-range ghost states (`DirichletBC`/`MixedBC`, or a
periodic wrap that `neighbor_index` has already resolved to a valid
interior index too) are computed on the fly since they depend on `t` and
aren't cached.
"""
@inline function rho_vel_at(cache::IMEXCache, semi::AbstractSemidiscretization,
                            I::CartesianIndex, t)
    nx = size(semi.mesh, 1)
    if 1 <= I[1] <= nx
        cell  = cell_index(I, semi)
        nvars = nvariables(semi.equations)
        return cache.u[global_dof(cell, 1, nvars)], cache.vel[cell]
    end

    s = _hyperbolic_state(cache.u, I, semi, t)
    rho = s[1]
    if rho <= 0
        error("""
              Negative density in rho_vel_at (ghost state)
              time = $t
              rho  = $rho
              """)
    end
    return rho, s[2] / rho
end

"""
    explicit_density_flux(rho_l, rho_r, vel_l, vel_r, gamma)

Density flux used in the explicit correction stage of the IMEX scheme,

    F̂ = ρ̄ (u_l + u_r)/2,

where ρ̄ is the γ-mean of the left and right densities.
"""
@inline function explicit_density_flux(rho_l,
                                       rho_r,
                                       vel_l,
                                       vel_r,
                                       gamma)

    # Central flux (unstable for advection — kept for reference)
    rho_half = gamma_mean(rho_l, rho_r, gamma)
    return rho_half * 0.5 * (vel_l + vel_r)
end

# ============================================================================
# Stage 1: Explicit Correction (density predictor)
# ============================================================================
# ρ̂_i = ρ_i^n - Δt/Δx * [ρ̄_{i+1/2} * (u_i + u_{i+1})/2
#                         - ρ̄_{i-1/2} * (u_{i-1} + u_i)/2]
# ============================================================================

function perform_stage!(
    ::ExplicitCorrectionStage,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCache,
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
        rho_l, vel_l = rho_vel_at(cache, semi, CartesianIndex(i), t)
        rho_r, vel_r = rho_vel_at(cache, semi, CartesianIndex(i + 1), t)
        f = explicit_density_flux(rho_l, rho_r, vel_l, vel_r, gamma)
        cache.rho_hat[i]     -= (dt / dx) * f
        cache.rho_hat[i + 1] += (dt / dx) * f
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = rho_vel_at(cache, semi, CartesianIndex(nx), t)
        rho_r, vel_r = rho_vel_at(cache, semi, CartesianIndex(1), t)
        f = explicit_density_flux(rho_l, rho_r, vel_l, vel_r, gamma)
        cache.rho_hat[nx] -= (dt / dx) * f
        cache.rho_hat[1]  += (dt / dx) * f
    else
        rho_gl, vel_gl = rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), t)
        rho_1,  vel_1  = rho_vel_at(cache, semi, CartesianIndex(1), t)
        f_l = explicit_density_flux(rho_gl, rho_1, vel_gl, vel_1, gamma)
        cache.rho_hat[1] += (dt / dx) * f_l

        rho_nx, vel_nx = rho_vel_at(cache, semi, CartesianIndex(nx), t)
        rho_gr, vel_gr = rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), t)
        f_r = explicit_density_flux(rho_nx, rho_gr, vel_nx, vel_gr, gamma)
        cache.rho_hat[nx] -= (dt / dx) * f_r
    end

    return nothing
end

# ============================================================================
# compute_eta!
# ============================================================================
# Compute the coefficient η as the maximum over all interfaces of
#
#     η = max_i  1.5 * ρ̄_{i+1/2} / ρ_i
#
# where ρ̄_{i+1/2} = gamma_mean(ρ_i, ρ_{i+1}, γ).
# ============================================================================

@inline function compute_eta!(cache::IMEXCache, semi::AbstractSemidiscretization, t)
    equations = semi.equations
    mesh      = semi.mesh
    gamma     = equations.gamma

    T = eltype(mesh.dx)
    eta_val = zero(T)

    @inbounds for I in eachcell(mesh)
        # Center density
        u_cc = _hyperbolic_state(cache.u, I, semi, t)
        rho_c = u_cc[1]

        # Right neighbor
        Ip1 = neighbor_index(I, semi, 1, 1)
        u_rr = _hyperbolic_state(cache.u, Ip1, semi, t)
        rho_r = u_rr[1]

        # Gamma-mean at right interface
        rho_half = gamma_mean(rho_c, rho_r, gamma)

        if rho_c > zero(T)
            eta_val = max(eta_val, 1.5 * rho_half / rho_c)
        else
            error("Density is negative: rho_i = $rho_c")
        end
    end

    cache.eta = eta_val
    return nothing
end

"""
    compute_dt_1!(cache, semi, t)

NOTE: not currently called anywhere in `solve_imex` (only `compute_dt_2!`
is); kept as an alternative CFL condition. Updated alongside
`compute_dt_2!`/`ExplicitCorrectionStage!` for consistency should it be
wired in later.
"""
@inline function compute_dt_1!(cache::IMEXCache, semi::AbstractSemidiscretization, t)
    mesh      = semi.mesh
    lambda    = semi.equations_elliptic.lambda
    T = eltype(mesh.dx)
    dx = mesh.dx[1]

    k_val = typemin(T)

    @inbounds for I in eachcell(mesh)

        # Center state — interior, so this reads cache.u/cache.vel directly
        # (no division); see update_primitive_variables!/rho_vel_at.
        rho_c, vel_c = rho_vel_at(cache, semi, I, t)

        # Potential at center
        phi_c = _elliptic_var(cache.phi, I, semi, t)

        y = 4*lambda^2 / (dx^2)
        denom = exp(phi_c) + y
        second_term = sqrt(rho_c / denom)
        k = abs(vel_c) + second_term

        if k > k_val
            k_val = k
        end
    end
    dt_val = 0.75 * dx / k_val
    return dt_val
end

@inline function compute_dt_2!(cache::IMEXCache, semi::AbstractSemidiscretization, t)
    mesh      = semi.mesh
    eta       = cache.eta

    T = eltype(mesh.dx)
    dx = mesh.dx[1]

    k_val = typemin(T)

    @inbounds for I in eachcell(mesh)
        cell = cell_index(I, semi)

        # Center state — interior, reads cache.u/cache.vel directly.
        rho_i, vel_i = rho_vel_at(cache, semi, I, t)

        # Potential at center
        phi_i = _elliptic_var(cache.phi, I, semi, t)

        # Right neighbor — interior for all but the last cell at a
        # non-periodic right boundary, where it falls back to an on-the-fly
        # ghost evaluation inside rho_vel_at.
        Ip1 = neighbor_index(I, semi, 1, 1)
        rho_r, vel_r = rho_vel_at(cache, semi, Ip1, t)

        # Potential at right neighbor
        phi_r = _elliptic_var(cache.phi, Ip1, semi, t)

        # Density check
        if rho_i < 1e-12 || rho_r < 1e-12
            error("""
                  Density below threshold in compute_dt!
                  time    = $t
                  cell    = $cell
                  rho_i   = $rho_i
                  rho_r   = $rho_r
                  eta     = $eta
                  """)
        end

        avg = (vel_i + vel_r)/2
        phi_diff = eta * abs(phi_r - phi_i)
        k = abs(avg) + phi_diff

        if k > k_val
            k_val = k
        end
    end
    dt_val = 0.1 * dx / k_val

    if dt_val < 1e-12
        error("""
              dt below threshold in compute_dt!
              time        = $t
              dt          = $dt_val
              eta         = $eta
              k_val       = $k_val
              """)
    end
    return dt_val
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
    cache::IMEXCache,
    integrator::IMEXIntegrator,
    dt,
    t,
)

    λ = semi.equations_elliptic.lambda

    # assembly functions use these to add the
    # per-cell η dt² ρ̄ⁿ_{i±1/2} corrections to the constant-coefficient
    # Laplacian stencil.
    params = semi.cache_elliptic.newton_cache.params
    params.rho = cache.u
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
    cache::IMEXCache,
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

    state_phi(I) = (_hyperbolic_state(read_state, I, semi, t),
                    _elliptic_var(cache.phi, I, semi, t))

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

    # NDIMS == 1, so the single sweep above is already the final state.
    # `cache.u_new` isn't used by this stage (it exists for a possible
    # future multi-axis ping-pong, mirroring the unused `dl/d/du` fields in
    # `EllipticCache`) — advance directly from `write_state`.
    copyto!(cache.u, write_state)
    return nothing
end

# ============================================================================
# Main driver: solve_imex
# ============================================================================

"""
    solve_imex(semi::AbstractSemidiscretization,
               integrator::IMEXIntegrator,
               tspan;
               dt,
               callbacks = CallbackSet())

Advance the semidiscretization using an IMEX time integrator.
"""

function solve_imex(semi::AbstractSemidiscretization,
                    integrator::IMEXIntegrator,
                    tspan;
                    dt,
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

    cache = IMEXCache(
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