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

    NDIMS = ndims(mesh)
    nvars = nvariables(equations)

    for axis in 1:NDIMS
        dx = mesh.dx[axis]
        @inbounds for I in eachcell(mesh)
            cell = cell_index(I, semi)

            # -----------------------------
            # Centre state
            # -----------------------------

            u_cc = _hyperbolic_state(cache.u,
                                     I,
                                     semi,
                                     t)

            rho_c = u_cc[1]
            vel_c = u_cc[2] / rho_c

            # -----------------------------
            # Right state
            # -----------------------------

            Ip1 = neighbor_index(I, semi, axis, 1)
            u_rr = _hyperbolic_state(cache.u,
                                     Ip1,
                                     semi,
                                     t)

            rho_r = u_rr[1]
            vel_r = u_rr[2] / rho_r

            # -----------------------------
            # Left state
            # -----------------------------

            Im1 = neighbor_index(I, semi, axis, -1)

            u_ll = _hyperbolic_state(cache.u,
                                     Im1,
                                     semi,
                                     t)

            rho_l = u_ll[1]
            vel_l = u_ll[2] / rho_l

            if rho_c <= 0 || rho_r <= 0 || rho_l <= 0
                error("""
                        Negative density entering ExplicitCorrectionStage

                        time = $t
                        cell = $cell

                        rho_i = $rho_c
                        rho_r = $rho_r
                        rho_l = $rho_l
                        """)
            end

            flux_rr = explicit_density_flux(rho_c,
                                            rho_r,
                                            vel_c,
                                            vel_r,
                                            gamma)

            flux_ll = explicit_density_flux(rho_l,
                                            rho_c,
                                            vel_l,
                                            vel_c,
                                            gamma)

            # -----------------------------
            # Finite-volume update
            # -----------------------------

            rho_idx = global_dof(cell, 1, nvars)

            cache.rho_hat[cell] = cache.u[rho_idx] - (dt / dx) * (flux_rr - flux_ll)
        end
    end
    return nothing
end

# ============================================================================
# compute_eta!
# ============================================================================
# Compute the coefficient η as the maximum over all interfaces of
#
#     η = max_i  1.5 * (ρ̄_{i+1/2})² / ρ_i
#
# where ρ̄_{i+1/2} = gamma_mean(ρ_i, ρ_{i+1}, γ).
# ============================================================================

@inline function compute_eta!(cache::IMEXCache, semi::AbstractSemidiscretization)
    equations = semi.equations
    mesh      = semi.mesh
    gamma     = equations.gamma

    T = eltype(mesh.dx)
    eta_val = zero(T)

    @inbounds for I in eachcell(mesh)
        # Center density
        u_cc = _hyperbolic_state(cache.u, I, semi, zero(T))
        rho_c = u_cc[1]

        # Right neighbor
        Ip1 = neighbor_index(I, semi, 1, 1)
        u_rr = _hyperbolic_state(cache.u, Ip1, semi, zero(T))
        rho_r = u_rr[1]

        # Gamma-mean at right interface
        rho_half = gamma_mean(rho_c, rho_r, gamma)

        # Local contribution: 1.5 * (ρ̄)² / ρ_i
        if rho_c > zero(T)
            # for now using w/o max TODO
            eta_val = max(eta_val, 1.5 * (rho_half^2) / rho_c)
            # eta_val = 1.5 * rho_half^2 / rho_c
        else
            error("Density is negative: rho_i = $rho_c")
        end
    end

    cache.eta = eta_val
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
    cache::IMEXCache,
    integrator::IMEXIntegrator,
    dt,
    t,
)

    # Compute diffusion coefficient from current density
    compute_eta!(cache, semi)

    λ = semi.equations_elliptic.lambda
    η = cache.eta

    solve_newton!(
        cache.phi,
        cache.rho_hat,
        -(λ^2 + η * dt^2),
        semi,
        t,
    )

    return nothing
end

# ============================================================================
# Stage 3: Implicit Correction (final ρ, ρu update)
# ============================================================================
# F^1_{face} = ρ̄ * (u_r + u_l)/2 - (x_r - x_l) * ηΔt/Δx
# F^2_{face} = u_l * max(F^1, 0) + u_r * max(-F^1, 0)
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
    NDIMS = ndims(mesh)
    nvars = nvariables(equations)

    # Stage starts from uⁿ
    copyto!(cache.u_new, cache.u)

    # TODO: DEBUG: set phi = 1 for all cells
    # fill!(cache.phi, 1)

    # Read/write buffers
    read_state  = cache.u_new
    write_state = cache.u_buffer

    for axis in 1:NDIMS
        dx = mesh.dx[axis]
        @inbounds for I in eachcell(mesh)

            cell = cell_index(I, semi)

            # --------------------------------------------------
            # Center state
            # --------------------------------------------------

            u_cc = _hyperbolic_state(
                read_state,
                I,
                semi,
                t,
            )

            phi_cc = _elliptic_var(
                cache.phi,
                I,
                semi,
                t,
            )

            # --------------------------------------------------
            # Right interface
            # --------------------------------------------------

            Ip1 = neighbor_index(I, semi, axis, 1)

            u_rr = _hyperbolic_state(
                read_state,
                Ip1,
                semi,
                t,
            )

            phi_rr = _elliptic_var(
                cache.phi,
                Ip1,
                semi,
                t,
            )

            right = solver.flux(
                u_cc,
                u_rr,
                phi_cc,
                phi_rr,
                axis,
                equations,
                dt,
                dx,
                cache.eta,
            )

            # --------------------------------------------------
            # Left interface
            # --------------------------------------------------

            Im1 = neighbor_index(I, semi, axis, -1)

            u_ll = _hyperbolic_state(
                read_state,
                Im1,
                semi,
                t,
            )

            phi_ll = _elliptic_var(
                cache.phi,
                Im1,
                semi,
                t,
            )

            left = solver.flux(
                u_ll,
                u_cc,
                phi_ll,
                phi_cc,
                axis,
                equations,
                dt,
                dx,
                cache.eta,
            )

            # --------------------------------------------------
            # Finite-volume update
            # --------------------------------------------------

            rho_idx = global_dof(cell, 1, nvars)
            mom_idx = global_dof(cell, 2, nvars)

            write_state[rho_idx] = read_state[rho_idx] - (dt / dx) * (right.flux[1] - left.flux[1])

            write_state[mom_idx] = read_state[mom_idx] - (dt / dx) * (right.flux[2] - left.flux[2])

            write_state[mom_idx] += (dt / 2) * (right.source + left.source)

            # if write_state[rho_idx] <= 0
            #     println("Negative density created")
            #     println("time = ", t)
            #     println("cell = ", cell)
            #     println("rho_old = ", read_state[rho_idx])
            #     println("F_left  = ", left.flux[1])
            #     println("F_right = ", right.flux[1])
            #     println("flux difference = ", right.flux[1] - left.flux[1])
            #     println("dt/dx = ", dt/dx)
            # end

        end
        # Swap read/write buffers for the next directional sweep
        read_state, write_state = write_state, read_state
    end

    # If the final sweep ended in u_buffer (odd number of dimensions),
    # copy it back into u_new.
    if read_state !== cache.u_new
        copyto!(cache.u_new, read_state)
    end

    # Advance the solution
    copyto!(cache.u, cache.u_new)
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

        while t < last(tspan) - eps(t)
            actual_dt = min(dt, last(tspan) - t)

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

            # u is already in block layout, so cache.u and cache.phi are views into u.
            # No re-interleave needed — callbacks see the correct state directly.
            context.solution = EulerAPSolution(u, t)

            perform_callbacks!(
                callbacks,
                context,
            )
        end
    end
    finalize_callbacks!(callbacks, context)
    return EulerAPSolution(u, t)
end

end # @muladd