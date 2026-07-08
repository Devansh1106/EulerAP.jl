# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

using TimerOutputs

@inline function _cell_var(u, I::CartesianIndex{NDIMS}, semi, t, var::Int) where {NDIMS}
    state = cell_state(u, I, semi, t)
    @inbounds return state[var]
end

@inline function _hyperbolic_state(u,
                                   I::CartesianIndex{NDIMS},
                                   semi,
                                   t) where {NDIMS}

    nvars = nvariables(semi.equations)

    values = ntuple(k -> _cell_var(u, I, semi, t, k), nvars)

    return SVector{nvars}(values)
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

    rho_half = gamma_mean(rho_l,
                          rho_r,
                          gamma)

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

            flux_rr = explicit_density_flux(
                rho_c,
                rho_r,
                vel_c,
                vel_r,
                gamma,
            )

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

            flux_ll = explicit_density_flux(
                rho_l,
                rho_c,
                vel_l,
                vel_c,
                gamma,
            )

            # -----------------------------
            # Finite-volume update
            # -----------------------------

            rho_idx = global_dof(cell,
                                 1,
                                 nvars)

            cache.rho_hat[cell] =
                cache.u[rho_idx] -
                (dt / dx) *
                (flux_rr - flux_ll)
        end
    end
    return nothing
end

# ============================================================================
# Stage 2: Implicit Prediction (solve elliptic equation)
# ============================================================================
# -α(x_{i-1} - 2x_i + x_{i+1}) + f(x_i) = ρ̂_i
# where α = (λ² + ηΔt) / Δx² and f(x) = exp(x) for Poisson-Boltzmann
# NOTE: Currently 1D-only; 2D requires a 5-point stencil sparse solver.
# ============================================================================


"""
    elliptic_residual!(F, x_elliptic, rho_hat, semi, dt, eta, t)

Residual for the nonlinear elliptic equation:
    F_i = -α(x_{i-1} - 2x_i + x_{i+1}) + f(x_i) - ρ̂_i
where f(x) = exp(x) for Poisson-Boltzmann.
"""
function elliptic_residual!(F, x_elliptic, rho_hat, semi, dt, eta, t)
    equations_elliptic = semi.equations_elliptic
    mesh = semi.mesh
    nx = size(mesh, 1)
    dx = mesh.dx[1]
    lambda = equations_elliptic.lambda
    alpha = (lambda^2 + eta * dt) / dx^2

    @inbounds for i in 1:nx
        # Ghost cell values for Laplacian stencil
        x_im1 = i > 1 ? x_elliptic[i - 1] : elliptic_ghost_value(x_elliptic, i - 1, nx, semi, t)
        x_i   = x_elliptic[i]
        x_ip1 = i < nx ? x_elliptic[i + 1] : elliptic_ghost_value(x_elliptic, i + 1, nx, semi, t)

        # Laplacian: x_{i-1} - 2x_i + x_{i+1}
        laplacian = x_im1 - 2 * x_i + x_ip1

        # Equation-specific point source
        point_src = elliptic_point_source(x_i, equations_elliptic)

        F[i] = -alpha * laplacian + point_src - rho_hat[i]
    end

    return nothing
end

function perform_stage!(
    ::ImplicitPredictionStage,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCache,
    integrator::IMEXIntegrator,
    dt,
    t,
)
    equations_elliptic = semi.equations_elliptic
    elliptic_cache = semi.cache_elliptic

    mesh = semi.mesh

    nx = size(mesh, 1)
    dx = mesh.dx[1]

    lambda = equations_elliptic.lambda
    eta = semi.solver.flux.eta

    alpha = (lambda^2 + eta * dt) / dx^2

    dl = elliptic_cache.dl
    d  = elliptic_cache.d
    du = elliptic_cache.du

    function res!(F, x, p)
        elliptic_residual!(
            F,
            x,
            cache.rho_hat,
            semi,
            dt,
            eta,
            t,
        )
        return nothing
    end

    function jacobian!(J, x, p)
        @inbounds for i in 1:nx
            point_deriv =
                elliptic_point_source_derivative(
                    x[i],
                    equations_elliptic,
                )

            d[i] = 2 * alpha + point_deriv

            if i > 1
                dl[i] = -alpha
            end

            if i < nx
                du[i] = -alpha
            end

        end
        copyto!(J.dl, 1, dl, 2, nx - 1)
        copyto!(J.d, d)
        copyto!(J.du, 1, du, 1, nx - 1)
        return nothing
    end

    jac_prototype = semi.cache_elliptic.jacobian

    nlf = NonlinearFunction(
        res!;
        jac = jacobian!,
        jac_prototype = jac_prototype,
    )

    prob = NonlinearProblem(
        nlf,
        copy(cache.phi),
        nothing,
    )

    sol = NonlinearSolve.solve(
        prob,
        NewtonRaphson();
        abstol = 1e-10,
        reltol = 1e-10,
    )
    copyto!(cache.phi, sol.u)
    return nothing
end

# ============================================================================
# Stage 3: Implicit Correction (final ρ, ρu update)
# ============================================================================
# F^1_{face} = ρ̄ * (u_r - u_l)/2 - (x_r - x_l) * ηΔt/Δx
# F^2_{face} = u_l * max(F^1, 0) + u_r * max(-F^1, 0)
# S^2_{face} = ρ̄ * (x_r - x_l) / Δx
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

    copyto!(cache.u_new, cache.u)

    for axis in 1:NDIMS
        dx = mesh.dx[axis]

        copyto!(cache.u_buffer, cache.u_new)
        @inbounds for I in eachcell(mesh)

            cell = cell_index(I, semi)

            # --------------------------------------------------
            # Center state
            # --------------------------------------------------

            u_cc = _hyperbolic_state(cache.u_new, I, semi, t)

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
                cache.u_new,
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
            )

            # --------------------------------------------------
            # Left interface
            # --------------------------------------------------

            Im1 = neighbor_index(I, semi, axis, -1)

            u_ll = _hyperbolic_state(
                cache.u_new,
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
            )

            # --------------------------------------------------
            # Finite-volume update
            # --------------------------------------------------

            rho_idx = global_dof(cell, 1, nvars)
            mom_idx = global_dof(cell, 2, nvars)

            cache.u_buffer[rho_idx] =
                cache.u_new[rho_idx] -
                (dt / dx) *
                (right.flux[1] - left.flux[1])

            cache.u_buffer[mom_idx] =
                cache.u_new[mom_idx] -
                (dt / dx) *
                (right.flux[2] - left.flux[2])

            cache.u_buffer[mom_idx] +=
                (dt / 2) *
                (right.source + left.source)

        end
        copyto!(cache.u_new, cache.u_buffer)
    end
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
function _deinterleave!(u_hyper, phi, u_interleaved,
                        nc, nvars_total, nvars_hyper)
    @inbounds for cell in 1:nc
        src_base = (cell - 1) * nvars_total
        dst_base = (cell - 1) * nvars_hyper
        u_hyper[dst_base + 1] = u_interleaved[src_base + 1]
        u_hyper[dst_base + 2] = u_interleaved[src_base + 2]
        phi[cell] = u_interleaved[src_base + 3]
    end
    return nothing
end

function _reinterleave!(u_interleaved, u_hyper, phi,
                        nc, nvars_total, nvars_hyper)
    @inbounds for cell in 1:nc
        src_base = (cell - 1) * nvars_hyper
        dst_base = (cell - 1) * nvars_total
        u_interleaved[dst_base + 1] = u_hyper[src_base + 1]
        u_interleaved[dst_base + 2] = u_hyper[src_base + 2]
        u_interleaved[dst_base + 3] = phi[cell]
    end
    return nothing
end

function solve_imex(semi::AbstractSemidiscretization,
                    integrator::IMEXIntegrator,
                    tspan;
                    dt,
                    callbacks = CallbackSet())

    t = first(tspan)

    # Full initial state in interleaved layout: [ρ₁, m₁, φ₁, ρ₂, m₂, φ₂, ...]
    u = initial_condition(t, semi)

    nvars_hyper     = nvariables(semi.equations)
    nvars_elliptic  = nvariables(semi.equations_elliptic)
    nvars_total     = nvars_hyper + nvars_elliptic  # = 3

    nc = ncells(semi.mesh)

    n_hyper    = nvars_hyper * nc
    n_elliptic = nvars_elliptic * nc

    # De-interleave into block layout for IMEX solver:
    #   u_hyper = [ρ₁, m₁, ρ₂, m₂, ..., ρ₁₀₀, m₁₀₀]
    #   phi     = [φ₁, φ₂, ..., φ₁₀₀]
    u_hyper = zeros(eltype(u), n_hyper)
    phi     = zeros(eltype(u), n_elliptic)

    _deinterleave!(u_hyper, phi, u, nc, nvars_total, nvars_hyper)

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

            t += actual_dt
            iteration += 1

            stats.iteration = iteration
            stats.time = t
            stats.dt = actual_dt

            # Re-interleave before callbacks so they see the correct state
            _reinterleave!(u, cache.u, cache.phi, nc, nvars_total, nvars_hyper)

            context.solution = EulerAPSolution(u, t)

            perform_callbacks!(
                callbacks,
                context,
            )
        end
    end
    finalize_callbacks!(callbacks, context)
    # Final re-interleave to ensure returned solution is correct
    _reinterleave!(u, cache.u, cache.phi, nc, nvars_total, nvars_hyper)
    return EulerAPSolution(u, t)
end

end # @muladd