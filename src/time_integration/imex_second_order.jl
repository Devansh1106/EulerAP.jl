# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

using TimerOutputs

function perform_stage!(::ExplicitCorrectionStage1,
                        semi::SemidiscretizationHyperbolicElliptic,
                        cache::IMEXCacheSecondOrder,
                        integrator::IMEXIntegrator,
                        dt,
                        t,
                        coeffs,)

    equations = semi.equations
    mesh      = semi.mesh

    nvars = nvariables(equations)

    # NOTE: EPB is 1D-only today (see the note on ImplicitPredictionStage
    # below), so this stage only ever runs for axis = 1 in practice.
    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    # periodic = semi.boundary_conditions.left isa PeriodicBC

    copyto!(cache.u_reconstructed, cache.u)

    reconstruct_slopes!(cache, semi, t)

    calculate_density_flux_diff_stage1(semi, cache, t, dt)
    calculate_momentum_flux_diff_stage1(semi, cache, t, dt)

    # Seed with ρⁿ; interface flux differences are subtracted below — one
    # `explicit_density_flux` evaluation per interface instead of two.
    @inbounds for i in 1:nx
        # ----- storing rho_n and m_n first -----
        # 1: density
        cache.rho_hat[i] = cache.u[global_dof(i, 1, nvars)]     
        cache.rho_exp[i] = cache.rho_hat[i]

        # 2: momentum (we do not need m_hat)
        cache.m_exp[i] = cache.u[global_dof(i, 2, nvars)]

        # adding the fluxes after storing rho_n and m_n
        # rho_exp and m_exp
        cache.rho_exp[i] -= (dt / dx) * coeffs.gamma_ars * (cache.explicit_density_flux_diff_stage1[i] + cache.semi_implicit_density_flux_diff_stage1[i])
    
        cache.m_exp[i] -= (dt / dx) * coeffs.gamma_ars * cache.momentum_flux_diff_stage1[i]
    end
    
    wrap_array!(cache.u_reconstructed, cache.rho_exp, cache.m_exp, semi)

    reconstruct_slopes!(cache, semi, t)

    # for u^2_E
    calculate_explicit_density_flux_diff_stage2(semi, cache)

    @inbounds for i in 1:nx
        # rho_hat and no m_hat (its same as m_n)
        # boundary treatment has been taken care of in flux difference calculation
        cache.rho_hat[i] -= (dt / dx) * coeffs.gamma_ars * cache.explicit_density_flux_diff_stage2[i]
    end
    return nothing
end


function perform_stage!(
    ::ImplicitPredictionStage1,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCacheSecondOrder,
    integrator::IMEXIntegrator,
    dt,
    t,
    coeffs,
)

    λ = semi.equations_elliptic.lambda

    # assembly functions use these to add the
    # per-cell η dt² ρ̄ⁿ_{i±1/2} corrections to the constant-coefficient
    # Laplacian stencil.
    params = semi.cache_elliptic.newton_cache.params
    params.u = cache.u
    params.eta = cache.eta * coeffs.gamma_ars
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

# ImplicitCorrectionStage for stage 1 of ARS222 (second order) is not required since the output won't be used anywhere in the stage 2

# -------------------------------------------------------------
# ------------------------- Stage 2 ---------------------------
# -------------------------------------------------------------

function perform_stage!(::ExplicitCorrectionStage2,
                        semi::SemidiscretizationHyperbolicElliptic,
                        cache::IMEXCacheSecondOrder,
                        integrator::IMEXIntegrator,
                        dt,
                        t,
                        coeffs,)

    equations = semi.equations
    mesh      = semi.mesh

    nvars = nvariables(equations)
    # gamma = equations.gamma
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # NOTE: EPB is 1D-only today (see the note on ImplicitPredictionStage
    # below), so this stage only ever runs for axis = 1 in practice.
    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # u_reconstructed already contains rho_exp and m_exp

    calculate_semi_implicit_density_flux_diff_stage2(semi, cache, t, dt)
    calculate_momentum_flux_diff_stage2(semi, cache, t, dt)

    # Seed with ρⁿ; interface flux differences are subtracted below — one
    # `explicit_density_flux` evaluation per interface instead of two.
    @inbounds for i in 1:nx
        # ----- storing rho_n and m_n first -----
        # 1: density
        cache.rho_hat[i] = cache.u[global_dof(i, 1, nvars)]    # re-using the stage 1 rho_hat since that is not required in this stage hence can be rewritten 
        cache.m_hat[i]   = cache.u[global_dof(i, 2, nvars)] 
        cache.rho_exp[i] = cache.rho_hat[i]

        # 2: momentum (we do not need m_hat)
        cache.m_exp[i] = cache.m_hat[i]

        # adding the fluxes after storing rho_n and m_n
        # rho_exp and m_exp for stage 2
        f = cache.explicit_density_flux_diff_stage2[i] + cache.semi_implicit_density_flux_diff_stage2[i]
        cache.rho_exp[i] -= (dt / dx) * coeffs.delta_ars * 
                            (cache.explicit_density_flux_diff_stage1[i] + cache.semi_implicit_density_flux_diff_stage1[i])
        cache.rho_exp[i] -= (dt / dx) * (1.0 - coeffs.delta_ars) * f
    
        cache.m_exp[i] -= (dt / dx) * coeffs.delta_ars * cache.momentum_flux_diff_stage1[i]
        cache.m_exp[i] -= (dt / dx) * (1.0 - coeffs.delta_ars) * cache.momentum_flux_diff_stage2[i]

        cache.rho_hat[i] -= (dt / dx) * (1.0 - coeffs.gamma_ars) * f
        cache.m_hat[i] -= (dt / dx) * (1.0 - coeffs.gamma_ars) * cache.momentum_flux_diff_stage2[i]
    end

    # putting rho^3_E and m^3_E in u_reconstructed
    wrap_array!(cache.u_reconstructed, cache.rho_exp, cache.m_exp, semi)
    reconstruct_slopes!(cache, semi, t)

    # fill cache.rho_hat directly so no need of loop after this.
    calculate_explicit_density_flux_diff_stage3(semi, cache, dt, coeffs.gamma_ars)
    return nothing
end

function perform_stage!(
    ::ImplicitPredictionStage2,
    semi::SemidiscretizationHyperbolicElliptic,
    cache::IMEXCacheSecondOrder,
    integrator::IMEXIntegrator,
    dt,
    t,
    coeffs,
)

    λ = semi.equations_elliptic.lambda

    # assembly functions use these to add the
    # per-cell η dt² ρ̄ⁿ_{i±1/2} corrections to the constant-coefficient
    # Laplacian stencil.
    params = semi.cache_elliptic.newton_cache.params
    params.u = cache.u
    params.eta = cache.eta * coeffs.gamma_ars
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

function perform_stage!(::ImplicitCorrectionStage2,
                        semi::SemidiscretizationHyperbolicElliptic,
                        cache::IMEXCacheSecondOrder,
                        integrator::IMEXIntegrator,
                        dt,
                        t,
                        coeffs,)

    equations = semi.equations
    mesh      = semi.mesh
    solver    = semi.solver
    orientation = 1 # since 1D currently
    eta = cache.eta

    nvars = nvariables(equations)
    periodic = semi.boundary_conditions.left isa PeriodicBC
    nx       = size(mesh, 1)
    dx       = mesh.dx[1]

    # calculate_semi_implicit_density_flux_diff_stage3(semi, cache, t)
    # wrap_array!(cache.u_reconstructed, cache.rho_exp, cache.m_exp, semi)

    # DEBUG: check positivity of the explicit predictor before reconstruction
    m_min, i_min = findmin(cache.rho_exp)
    if m_min < 0
        w = 5
        lo, hi = max(1, i_min - w), min(nx, i_min + w)
        println("="^100)
        println("DEBUG: rho_exp went negative: min = ", m_min, " at cell ", i_min,
                " | t = ", t, " | dt = ", dt, " | eta = ", cache.eta, " | dx = ", dx)
        println(rpad("cell", 6), rpad("x", 12), rpad("rho_exp", 16), rpad("rho_hat", 16),
                rpad("vel_exp", 14), rpad("phi", 16), rpad("eta*dt*dphi_drift", 20), "CFL_drift")
        @inbounds for i in lo:hi
            I     = CartesianIndex(i)
            x_i   = coordinates(I, semi.mesh)[1]
            rho_i = cache.rho_exp[i]
            vel_i = cache.m_exp[i] / rho_i
            phi_i = _elliptic_var(cache.phi, semi, I, t)
            # drift-velocity scale of the semi-implicit flux at the right face:
            # semi_implicit_density_flux ~ -eta * dt * rho_half * (phi_r - phi_l) / dx
            phi_ip = _elliptic_var(cache.phi, semi, CartesianIndex(min(i + 1, nx)), t)
            drift  = eta * dt * (phi_ip - phi_i) / dx          # effective drift velocity scale
            cfl_d  = abs(drift) * dt / dx                      # compare against O(1)
            println(rpad(i, 6), rpad(x_i, 12), rpad(rho_i, 16), rpad(cache.rho_hat[i], 16),
                    rpad(vel_i, 14), rpad(phi_i, 16), rpad(drift, 20), cfl_d)
        end
        error("rho_exp went negative: min = ", m_min, " at cell ", i_min)
    end

    # reconstruct_slopes!(cache, semi, t)

    @inbounds for i in 1:nx
        I = CartesianIndex(i)
        rho_idx = global_dof(I, 1, nvars)
        m_idx   = global_dof(I, 2, nvars)
        cache.u[rho_idx] = cache.rho_hat[i]
        cache.u[m_idx]   = cache.m_hat[i]
    end

    # directly fill cache.u[rho] with the final solution of density as 
    # there is no need to store the semi implicit density flux
    # u^3_E and reconstructed slopes are already there from last stages
    calculate_semi_implicit_density_flux_diff_stage3(semi, cache, t, dt, coeffs.gamma_ars)

    # calculate_momentum_flux_diff_stage3(semi, cache, t, dt, coeffs.gamma_ars)

    # One `solver.flux` evaluation per interface, applied to both neighbors
    # (same pattern as the first-order `ImplicitCorrectionStage`).
    function apply_interface_momentum!(cell_l, cell_r, u_l, phi_l, u_r, phi_r)
        contrib = solver.flux(u_l, u_r, phi_l, phi_r, orientation, equations, dt, dx, eta)

        m_idx_l = global_dof(cell_l, 2, nvars)
        m_idx_r = global_dof(cell_r, 2, nvars)

        g = (dt / dx) * coeffs.gamma_ars * contrib.flux[2]
        m = (dt / 2) * coeffs.gamma_ars * contrib.source

        cache.u[m_idx_l] -= g
        cache.u[m_idx_r] += g

        cache.u[m_idx_l] += m
        cache.u[m_idx_r] += m
        return nothing
    end

    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx-1)
        u_l   = reconstructed_conservative_state_at(cache, semi, CartesianIndex(i), :left)
        u_r   = reconstructed_conservative_state_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        apply_interface_momentum!(i, i + 1, u_l, phi_l, u_r, phi_r)
    end

    # Boundary face(s)
    if periodic
        u_l   = reconstructed_conservative_state_at(cache, semi, CartesianIndex(nx), :left)
        u_r   = reconstructed_conservative_state_at(cache, semi, CartesianIndex(1), :right)
        phi_l = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)
        apply_interface_momentum!(nx, 1, u_l, phi_l, u_r, phi_r)
    else
        # left boudnary
        Ig_l = neighbor_index(CartesianIndex(1), semi, 1, -1)
        u_gl = reconstructed_conservative_state_at(cache, semi, Ig_l, :left)
        phi_gl = _elliptic_var(cache.phi, semi, Ig_l, t)
        u_1    = reconstructed_conservative_state_at(cache, semi, CartesianIndex(1), :right)
        phi_1  = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        contrib = solver.flux(u_gl, u_1, phi_gl, phi_1, orientation, equations, dt, dx, eta)

        # f_1 = (dt / dx) * coeffs.gamma_ars * contrib.flux[1]
        g_1 = (dt / dx) * coeffs.gamma_ars * contrib.flux[2]
        m_1 = (dt / 2) * coeffs.gamma_ars * contrib.source

        m_1_idx = global_dof(1, 2, nvars)
        cache.u[m_1_idx] += g_1
        cache.u[m_1_idx] += m_1

        # right boundary
        Ig_r = neighbor_index(CartesianIndex(nx), semi, 1, 1)
        u_nx = reconstructed_conservative_state_at(cache, semi, CartesianIndex(nx), :left)
        phi_nx = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        u_gr   = reconstructed_conservative_state_at(cache, semi, Ig_r, :right)
        phi_gr = _elliptic_var(cache.phi, semi, Ig_r, t)

        contrib = solver.flux(u_nx, u_gr, phi_nx, phi_gr, orientation, equations, dt, dx, eta)

        g_2 = (dt / dx) * coeffs.gamma_ars * contrib.flux[2]
        m_2 = (dt / 2) * coeffs.gamma_ars * contrib.source


        m_nx_idx = global_dof(nx, 2, nvars)
        cache.u[m_nx_idx]    -= g_2
        cache.u[m_nx_idx]    += m_2 # source always gets added (- sign is taken care in the function itself)
    end
    return nothing
end

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
                    tspan,
                    scheme::SecondOrderFiveStagesIMEX;
                    dt,
                    callbacks = CallbackSet())

    t = first(tspan)
    gamma_ars = 1-sqrt(2)/2
    delta_ars = 1-1/(2*gamma_ars)
    coeffs = ARS222(gamma_ars, delta_ars)

    # Full initial state in block layout: [ρ₁, m₁, ρ₂, m₂, ..., ρₙ, mₙ, φ₁, φ₂, ..., φₙ]
    u = initial_condition(t, semi)

    nvars_hyper     = nvariables(semi.equations)
    # nvars_elliptic  = nvariables(semi.equations_elliptic)

    nc = ncells(semi.mesh)

    n_hyper    = nvars_hyper * nc
    # n_elliptic = nvars_elliptic * nc

    # Use views into the block-layout state for the IMEX solver:
    #   u_hyper = u[1:n_hyper]  = [ρ₁, m₁, ρ₂, m₂, ..., ρₙ, mₙ]
    #   phi     = u[n_hyper+1:end] = [φ₁, φ₂, ..., φₙ]
    u_hyper = @view u[1:n_hyper]
    phi     = @view u[n_hyper+1:end]

    # --------------------------------------------------
    # IMEX cache
    # --------------------------------------------------

    cache = IMEXCacheSecondOrder(
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

            # Flux-difference accumulators are reused across timesteps and
            # filled with +=/-= by the helpers below — reset them every step.
            reset_flux_diff_accumulators!(cache)

            stats.mass_before = total_mass(cache.u, semi)

            # Stage 1: Explicit density correction (evaluated at current time t)
            @timeit stats.timer "ExplicitCorrectionStage1" begin
                perform_stage!(
                    ExplicitCorrectionStage1(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t,
                    coeffs,
                )
            end

            stats.mass_after_stage1 =
                sum(cache.rho_hat) * prod(semi.mesh.dx)

            stats.relative_mass_error_stage1 =
                abs(stats.mass_after_stage1 -
                    stats.mass_before) /
                abs(stats.mass_before)

            # Stage 2: Implicit elliptic solve (predicts φ at future time t + dt)
            @timeit stats.timer "ImplicitPredictionStage1" begin
                perform_stage!(
                    ImplicitPredictionStage1(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t + actual_dt,
                    coeffs,
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

            stats.mass_before = total_mass(cache.u, semi)

            # Stage 1: Explicit density correction (evaluated at current time t)
            @timeit stats.timer "ExplicitCorrectionStage2" begin
                perform_stage!(
                    ExplicitCorrectionStage2(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t,
                    coeffs,
                )
            end

            stats.mass_after_stage1 =
                sum(cache.rho_hat) * prod(semi.mesh.dx)

            stats.relative_mass_error_stage1 =
                abs(stats.mass_after_stage1 -
                    stats.mass_before) /
                abs(stats.mass_before)

            # Stage 2: Implicit elliptic solve (predicts φ at future time t + dt)
            @timeit stats.timer "ImplicitPredictionStage2" begin
                perform_stage!(
                    ImplicitPredictionStage2(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t + actual_dt,
                    coeffs,
                )
            end

            # Stage 3: Implicit correction (evaluated at current time t)
            @timeit stats.timer "ImplicitCorrectionStage2" begin
                perform_stage!(
                    ImplicitCorrectionStage2(),
                    semi,
                    cache,
                    integrator,
                    actual_dt,
                    t,
                    coeffs,
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

            delete!(stats.timer.inner_timers, "ExplicitCorrectionStage1")
            delete!(stats.timer.inner_timers, "ImplicitPredictionStage1")

            delete!(stats.timer.inner_timers, "ExplicitCorrectionStage2")
            delete!(stats.timer.inner_timers, "ImplicitPredictionStage2")
            delete!(stats.timer.inner_timers, "ImplicitCorrectionStage2") 
        end
    end
    finalize_callbacks!(callbacks, context)
    return EulerAPSolution(u, t)
end

# ------------------------------------------------------------------- #
# ---------- Helper functions for storing flux differences----------- #
# ------------------------------------------------------------------- #

# Reset the per-timestep flux-difference accumulators. They are filled with
# +=/-= by the `calculate_*_flux_diff_*` helpers and reused across timesteps.
@inline function reset_flux_diff_accumulators!(cache::IMEXCacheSecondOrder)
    fill!(cache.explicit_density_flux_diff_stage1, zero(eltype(cache.u)))
    fill!(cache.semi_implicit_density_flux_diff_stage1, zero(eltype(cache.u)))
    fill!(cache.momentum_flux_diff_stage1, zero(eltype(cache.u)))
    fill!(cache.explicit_density_flux_diff_stage2, zero(eltype(cache.u)))
    fill!(cache.semi_implicit_density_flux_diff_stage2, zero(eltype(cache.u)))
    fill!(cache.momentum_flux_diff_stage2, zero(eltype(cache.u)))
    return nothing
end

@inline function calculate_density_flux_diff_stage1(semi::SemidiscretizationHyperbolicElliptic, cache, t::T, dt::T) where T

    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma
    eta   = cache.eta

    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # for density flux
    # Interior faces: nx - 1 of them (face: between cell i and cell i+1)
    @inbounds for i in 1:(nx - 1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)

        flux_exp      = explicit_density_flux(vel_l, vel_r, rho_half)
        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

        cache.explicit_density_flux_diff_stage1[i]     += flux_exp
        cache.explicit_density_flux_diff_stage1[i + 1] -= flux_exp

        cache.semi_implicit_density_flux_diff_stage1[i]     += flux_semi_imp
        cache.semi_implicit_density_flux_diff_stage1[i + 1] -= flux_semi_imp
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)
        rho_half = gamma_mean(rho_l, rho_r, gamma)

        flux_exp = explicit_density_flux(vel_l, vel_r, rho_half)
        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

        cache.explicit_density_flux_diff_stage1[nx] += flux_exp
        cache.explicit_density_flux_diff_stage1[1]  -= flux_exp

        cache.semi_implicit_density_flux_diff_stage1[nx] += flux_semi_imp
        cache.semi_implicit_density_flux_diff_stage1[1]  -= flux_semi_imp
    else
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_gl         = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), t)
        phi_1         = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)
        rho_half_l = gamma_mean(rho_gl, rho_1, gamma)

        flux_exp_l = explicit_density_flux(vel_gl, vel_1, rho_half_l)
        flux_semi_imp_l = semi_implicit_density_flux(phi_gl, phi_1, rho_half_l, eta, dt, dx)

        cache.explicit_density_flux_diff_stage1[1]      -= flux_exp_l
        cache.semi_implicit_density_flux_diff_stage1[1] -= flux_semi_imp_l

        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), :right)
        phi_nx         = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_gr         = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), t)
        rho_half_r = gamma_mean(rho_nx, rho_gr, gamma)

        flux_exp_r = explicit_density_flux(vel_nx, vel_gr, rho_half_r)
        flux_semi_imp_r = semi_implicit_density_flux(phi_nx, phi_gr, rho_half_r, eta, dt, dx)

        cache.explicit_density_flux_diff_stage1[nx]      += flux_exp_r
        cache.semi_implicit_density_flux_diff_stage1[nx] += flux_semi_imp_r
    end
end

# Face momentum flux: conservative upwind part G of the total face density
# flux, plus nonconservative source a = ½ ρ̄ (φ_r - φ_l). Shared by both
# stages; the φ-source is added with the SAME sign to both cells of a face
# (nonconservative) with a ½ factor, so that the final `(dt/dx)` scaling
# reproduces exactly the first-order `ImplicitCorrectionStage` treatment
# `(dt/2) * contrib.source`, where `contrib.source = -ρ̄ (φ_r - φ_l) / dx`.
@inline function face_momentum_flux(gamma, eta, dt, dx,
                                    rho_l, vel_l, rho_r, vel_r, phi_l, phi_r)
    rho_half = gamma_mean(rho_l, rho_r, gamma)

    F_rho = explicit_density_flux(vel_l, vel_r, rho_half) +
            semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

    G = vel_l * max(F_rho, zero(F_rho)) + vel_r * min(F_rho, zero(F_rho))
    a = -0.5 * rho_half * (phi_r - phi_l)
    return G, a
end

# The upwind split needs the *face* density flux, which cannot be recovered
# from the cell-wise divergence arrays (`*_flux_diff_*`) — so the momentum
# contributions are assembled per face here. The φ-source is added with the
# SAME sign to both cells of a face (nonconservative) with a ½ factor, so
# that the final `(dt/dx)` scaling reproduces exactly the first-order
# `ImplicitCorrectionStage` treatment `(dt/2) * contrib.source`, where
# `contrib.source = -ρ̄ (φ_r - φ_l) / dx`.
@inline function calculate_momentum_flux_diff_stage1(semi::SemidiscretizationHyperbolicElliptic, cache, t::T, dt::T) where T
    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma
    eta   = cache.eta

    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx-1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_l, vel_l, rho_r, vel_r, phi_l, phi_r)
        cache.momentum_flux_diff_stage1[i]     += G - a
        cache.momentum_flux_diff_stage1[i + 1] += -G - a
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_l, vel_l, rho_r, vel_r, phi_l, phi_r)
        cache.momentum_flux_diff_stage1[nx] += G - a
        cache.momentum_flux_diff_stage1[1]  += -G - a
    else
        # left boundary face (ghost, cell 1): cell 1 receives it as its left face
        Ig_l = neighbor_index(CartesianIndex(1), semi, 1, -1)
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, Ig_l, :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_gl         = _elliptic_var(cache.phi, semi, Ig_l, t)
        phi_1          = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_gl, vel_gl, rho_1, vel_1, phi_gl, phi_1)
        cache.momentum_flux_diff_stage1[1] += -G - a

        # right boundary face (cell nx, ghost): cell nx receives it as its right face
        Ig_r = neighbor_index(CartesianIndex(nx), semi, 1, 1)
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, Ig_r, :right)
        phi_nx         = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_gr         = _elliptic_var(cache.phi, semi, Ig_r, t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_nx, vel_nx, rho_gr, vel_gr, phi_nx, phi_gr)
        cache.momentum_flux_diff_stage1[nx] += G - a
    end
end


# stage 2 helper functions
@inline function calculate_explicit_density_flux_diff_stage2(semi::SemidiscretizationHyperbolicElliptic, cache)

    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma

    nx       = size(mesh, 1)
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # for density flux
    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx - 1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)

        flux_exp     = explicit_density_flux(vel_l, vel_r, rho_half)

        cache.explicit_density_flux_diff_stage2[i]     += flux_exp
        cache.explicit_density_flux_diff_stage2[i + 1] -= flux_exp
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)

        rho_half = gamma_mean(rho_l, rho_r, gamma)

        flux_exp = explicit_density_flux(vel_l, vel_r, rho_half)

        cache.explicit_density_flux_diff_stage2[nx] += flux_exp
        cache.explicit_density_flux_diff_stage2[1]  -= flux_exp
    else
        # left boundary
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)

        rho_half_l = gamma_mean(rho_gl, rho_1, gamma)

        flux_exp_l = explicit_density_flux(vel_gl, vel_1, rho_half_l)

        cache.explicit_density_flux_diff_stage2[1] -= flux_exp_l

        # right boundary
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), :right)

        rho_half_r = gamma_mean(rho_nx, rho_gr, gamma)

        flux_exp_r = explicit_density_flux(vel_nx, vel_gr, rho_half_r)

        cache.explicit_density_flux_diff_stage2[nx] += flux_exp_r
    end
end

@inline function calculate_momentum_flux_diff_stage2(semi::SemidiscretizationHyperbolicElliptic, cache, t::T, dt::T) where T
    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma
    eta   = cache.eta

    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx-1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_l, vel_l, rho_r, vel_r, phi_l, phi_r)
        cache.momentum_flux_diff_stage2[i]     += G - a
        cache.momentum_flux_diff_stage2[i + 1] += -G - a
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_l, vel_l, rho_r, vel_r, phi_l, phi_r)
        cache.momentum_flux_diff_stage2[nx] += G - a
        cache.momentum_flux_diff_stage2[1]  += -G - a
    else
        # left boundary face (ghost, cell 1): cell 1 receives it as its left face
        Ig_l = neighbor_index(CartesianIndex(1), semi, 1, -1)
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, Ig_l, :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_gl         = _elliptic_var(cache.phi, semi, Ig_l, t)
        phi_1          = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_gl, vel_gl, rho_1, vel_1, phi_gl, phi_1)
        cache.momentum_flux_diff_stage2[1] += -G - a

        # right boundary face (cell nx, ghost): cell nx receives it as its right face
        Ig_r = neighbor_index(CartesianIndex(nx), semi, 1, 1)
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, Ig_r, :right)
        phi_nx         = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_gr         = _elliptic_var(cache.phi, semi, Ig_r, t)

        G, a = face_momentum_flux(gamma, eta, dt, dx, rho_nx, vel_nx, rho_gr, vel_gr, phi_nx, phi_gr)
        cache.momentum_flux_diff_stage2[nx] += G - a
    end
end

@inline function calculate_semi_implicit_density_flux_diff_stage2(semi::SemidiscretizationHyperbolicElliptic, cache, t::T, dt::T) where T
    
    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma

    nx       = size(mesh, 1)
    periodic = semi.boundary_conditions.left isa PeriodicBC

    eta = cache.eta
    dx  = mesh.dx[1]

    # Interior faces: nx - 1 of them (face: between cell i and cell i+1)
    @inbounds for i in  1:(nx - 1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)

        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

        cache.semi_implicit_density_flux_diff_stage2[i]     += flux_semi_imp
        cache.semi_implicit_density_flux_diff_stage2[i + 1] -= flux_semi_imp
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        rho_half = gamma_mean(rho_l, rho_r, gamma)

        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

        cache.semi_implicit_density_flux_diff_stage2[1]  -= flux_semi_imp
        cache.semi_implicit_density_flux_diff_stage2[nx] += flux_semi_imp
    else
        # left boundary
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_gl        = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), t)
        phi_1        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        rho_half_l = gamma_mean(rho_gl, rho_1, gamma)

        flux_semi_imp_l = semi_implicit_density_flux(phi_gl, phi_1, rho_half_l, eta, dt, dx)

        
        cache.semi_implicit_density_flux_diff_stage2[1] -= flux_semi_imp_l

        # right boundary
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), :right)
        phi_nx        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_gr        = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), t)

        rho_half_r = gamma_mean(rho_nx, rho_gr, gamma)

        flux_semi_exp_r = semi_implicit_density_flux(phi_nx, phi_gr, rho_half_r, eta, dt, dx)
        
        cache.semi_implicit_density_flux_diff_stage2[nx] += flux_semi_exp_r
    end
end

# (ρ, v) primitive pair from a conservative (ρ, m) cell state, used so that
# slope-limiting/reconstruction happens on velocity rather than momentum
# (avoids amplifying independent ρ- and m-slope mismatches into large
# spurious velocities where ρ is small, e.g. near-vacuum tails).
@inline rho_vel(u) = (u[1], u[2] / u[1])

@inline function reconstruct_slopes!(cache,
                                     semi,
                                     t;
                                     limiter = minmod,)
    mesh      = semi.mesh
    equations = semi.equations

    dx    = mesh.dx[1]
    nx    = size(mesh, 1)
    nvars = nvariables(equations)

    periodic = semi.boundary_conditions.left isa PeriodicBC

    @inbounds begin

        # ----------------------------------------------------------
        # Interior cells
        # ----------------------------------------------------------
        for i in 2:(nx - 1)

            I = CartesianIndex(i)

            u_l = cell_state(
                cache.u_reconstructed,
                CartesianIndex(i - 1),
                semi,
                t,
            )

            u_c = cell_state(
                cache.u_reconstructed,
                I,
                semi,
                t,
            )

            u_r = cell_state(
                cache.u_reconstructed,
                CartesianIndex(i + 1),
                semi,
                t,
            )

            rho_l, vel_l = rho_vel(u_l)
            rho_c, vel_c = rho_vel(u_c)
            rho_r, vel_r = rho_vel(u_r)

            rho_idx = global_dof(I, 1, nvars)
            vel_idx = global_dof(I, 2, nvars)   # same slot; now a velocity slope

            cache.slopes[rho_idx] = limiter((rho_c - rho_l) / dx, (rho_r - rho_c) / dx)
            cache.slopes[vel_idx] = limiter((vel_c - vel_l) / dx, (vel_r - vel_c) / dx)
        end


        # ==========================================================
        # Left boundary: cell i = 1
        # ==========================================================

        I = CartesianIndex(1)

        u_c = cell_state(
            cache.u_reconstructed,
            I,
            semi,
            t,
        )

        u_r = cell_state(
            cache.u_reconstructed,
            CartesianIndex(2),
            semi,
            t,
        )

        if periodic

            # Periodic neighbour of cell 1
            u_l = cell_state(
                cache.u_reconstructed,
                CartesianIndex(nx),
                semi,
                t,
            )

        else

            # Ghost cell supplied by the BC machinery
            Ighost = neighbor_index(
                I,
                semi,
                1,
                -1,
            )

            u_l = cell_state(
                cache.u_reconstructed,
                Ighost,
                semi,
                t,
            )
        end

        rho_l, vel_l = rho_vel(u_l)
        rho_c, vel_c = rho_vel(u_c)
        rho_r, vel_r = rho_vel(u_r)

        rho_idx = global_dof(I, 1, nvars)
        vel_idx = global_dof(I, 2, nvars)

        cache.slopes[rho_idx] = limiter((rho_c - rho_l) / dx, (rho_r - rho_c) / dx)
        cache.slopes[vel_idx] = limiter((vel_c - vel_l) / dx, (vel_r - vel_c) / dx)


        # ==========================================================
        # Right boundary: cell i = nx
        # ==========================================================

        I = CartesianIndex(nx)

        u_l = cell_state(
            cache.u_reconstructed,
            CartesianIndex(nx - 1),
            semi,
            t,
        )

        u_c = cell_state(
            cache.u_reconstructed,
            I,
            semi,
            t,
        )

        if periodic

            # Periodic neighbour of cell nx
            u_r = cell_state(
                cache.u_reconstructed,
                CartesianIndex(1),
                semi,
                t,
            )

        else

            # Ghost cell supplied by the BC machinery
            Ighost = neighbor_index(
                I,
                semi,
                1,
                1,
            )

            u_r = cell_state(
                cache.u_reconstructed,
                Ighost,
                semi,
                t,
            )
        end

        rho_l, vel_l = rho_vel(u_l)
        rho_c, vel_c = rho_vel(u_c)
        rho_r, vel_r = rho_vel(u_r)

        rho_idx = global_dof(I, 1, nvars)
        vel_idx = global_dof(I, 2, nvars)

        cache.slopes[rho_idx] = limiter((rho_c - rho_l) / dx, (rho_r - rho_c) / dx)
        cache.slopes[vel_idx] = limiter((vel_c - vel_l) / dx, (vel_r - vel_c) / dx)
    end

    return nothing
end



# helper function for explicit part involving rho^3_E and u^3_E
@inline function calculate_explicit_density_flux_diff_stage3(semi::SemidiscretizationHyperbolicElliptic, cache, dt::T, gamma_ars::T) where T 
    
    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma

    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    # for density flux
    # Interior faces: nx - 1 of them, between cell i and cell i+1
    @inbounds for i in 1:(nx-1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)

        flux_exp     = explicit_density_flux(vel_l, vel_r, rho_half)
        f            = (dt / dx) * gamma_ars * flux_exp
        cache.rho_hat[i]     -= f
        cache.rho_hat[i + 1] += f
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)

        rho_half = gamma_mean(rho_l, rho_r, gamma)

        flux_exp = explicit_density_flux(vel_l, vel_r, rho_half)
        f = (dt / dx) * gamma_ars * flux_exp
        cache.rho_hat[nx] -= f
        cache.rho_hat[1]  += f
    else
        # left boundary
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)

        rho_half_l = gamma_mean(rho_gl, rho_1, gamma)

        flux_exp_l = explicit_density_flux(vel_gl, vel_1, rho_half_l)
        f = (dt / dx) * gamma_ars * flux_exp_l

        cache.rho_hat[1] += f

        # right boundary
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), :right)

        rho_half_r = gamma_mean(rho_nx, rho_gr, gamma)

        flux_exp_r = explicit_density_flux(vel_nx, vel_gr, rho_half_r)
        f = (dt / dx) * gamma_ars * flux_exp_r

        cache.rho_hat[nx] -= f
    end
end

# helper function for explicit part involving phi^3_E
@inline function calculate_semi_implicit_density_flux_diff_stage3(semi::SemidiscretizationHyperbolicElliptic, cache, t::T, dt::T, gamma_ars::T) where T    
    equations = semi.equations
    mesh      = semi.mesh

    gamma = equations.gamma
    nvars = nvariables(equations)

    nx       = size(mesh, 1)
    dx       = mesh.dx[1]
    periodic = semi.boundary_conditions.left isa PeriodicBC

    eta = cache.eta
    dx  = mesh.dx[1]
    # Interior faces: nx - 1 of them (face: between cell i and cell i+1)
    @inbounds for i in  1:(nx - 1)
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(i + 1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(i), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(i + 1), t)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)
        rho_idx_l    = global_dof(i, 1, nvars)
        rho_idx_r    = global_dof(i + 1, 1, nvars)

        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)
        f = (dt / dx) * gamma_ars * flux_semi_imp

        cache.u[rho_idx_l] -= f
        cache.u[rho_idx_r] += f
    end

    # Boundary face(s)
    if periodic
        rho_l, vel_l = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_r, vel_r = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_l        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_r        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        rho_half     = gamma_mean(rho_l, rho_r, gamma)
        rho_idx_l    = global_dof(nx, 1, nvars)
        rho_idx_r    = global_dof(1, 1, nvars)

        flux_semi_imp = semi_implicit_density_flux(phi_l, phi_r, rho_half, eta, dt, dx)

        f = (dt / dx) * gamma_ars * flux_semi_imp

        cache.u[rho_idx_r] += f
        cache.u[rho_idx_l] -= f
    else
        # left boundary
        rho_gl, vel_gl = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), :left)
        rho_1,  vel_1  = reconstructed_rho_vel_at(cache, semi, CartesianIndex(1), :right)
        phi_gl        = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(1), semi, 1, -1), t)
        phi_1        = _elliptic_var(cache.phi, semi, CartesianIndex(1), t)

        rho_half_l = gamma_mean(rho_gl, rho_1, gamma)
        rho_idx_1  = global_dof(1, 1, nvars)

        flux_semi_imp_l = semi_implicit_density_flux(phi_gl, phi_1, rho_half_l, eta, dt, dx)
        f = (dt / dx) * gamma_ars * flux_semi_imp_l

        cache.u[rho_idx_1] += f

        # right boundary
        rho_nx, vel_nx = reconstructed_rho_vel_at(cache, semi, CartesianIndex(nx), :left)
        rho_gr, vel_gr = reconstructed_rho_vel_at(cache, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), :right)
        phi_nx        = _elliptic_var(cache.phi, semi, CartesianIndex(nx), t)
        phi_gr        = _elliptic_var(cache.phi, semi, neighbor_index(CartesianIndex(nx), semi, 1, 1), t)

        rho_half_r = gamma_mean(rho_nx, rho_gr, gamma)

        flux_semi_imp_r = semi_implicit_density_flux(phi_nx, phi_gr, rho_half_r, eta, dt, dx)
        rho_idx_nx  = global_dof(nx, 1, nvars)
        f = (dt / dx) * gamma_ars * flux_semi_imp_r

        cache.u[rho_idx_nx] -= f
    end
end

end # @muladd