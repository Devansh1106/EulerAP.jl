# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

abstract type AbstractTimeIntegrator end

# ======================================
# ----------- Implicit Euler ----------- 
# ======================================
"""
    ImplicitEulerCustom()

Backward Euler time integrator.

The nonlinear system

    uⁿ⁺¹ - uⁿ - Δt F(uⁿ⁺¹) = 0

is solved with Newton iterations.
"""
struct ImplicitEulerCustom <: AbstractTimeIntegrator end

"""
    solve(semi,
          tspan,
          ::ImplicitEulerCustom;
          dt = minimum_cell_size(semi.mesh.dx),
          abstol = 1e-8,
          reltol = 1e-8,
          callback=CallbackSet())

Advance the semidiscretization using a custom
Backward Euler time integrator.
"""
function solve(semi,
               tspan,
               integrator::ImplicitEulerCustom;
               dt = minimum_cell_size(semi.mesh),
               abstol = 1e-8,
               reltol = 1e-8,
               callbacks=CallbackSet())

    return solve_implicit_euler(semi,
                                integrator,
                                tspan;
                                dt = dt,
                                abstol = abstol,
                                reltol = reltol,
                                callbacks=callbacks)
end

# TODO: add a solve() here for IMEXIntegrator

# Used in Callbacks
@inline integrator(context::CallbackContext) = context.simulation.integrator

# ======================================
# ----------- IMEXIntegrator ----------- 
# ======================================

abstract type AbstractIMEXStage end
abstract type AbstractIMEXScheme end

# Stages can be defined here
struct ExplicitCorrectionStage <: AbstractIMEXStage end
struct ImplicitCorrectionStage <: AbstractIMEXStage end
struct ImplicitPredictionStage <: AbstractIMEXStage end

# Schemes can be defined here
struct FirstOrderThreeStagesIMEX <: AbstractIMEXScheme end
struct SecondOrderFiveStagesIMEX <: AbstractIMEXScheme end

"""
    stages(scheme)

Return the ordered tuple of stages executed by an IMEX scheme.
"""
function stages end

@inline stages(::FirstOrderThreeStagesIMEX) = (
    ExplicitCorrectionStage(),
    ImplicitPredictionStage(),
    ImplicitCorrectionStage(),
)

# Second-order (ARS222) stages — numbered per RK stage
struct ExplicitCorrectionStage1 <: AbstractIMEXStage end
struct ImplicitPredictionStage1 <: AbstractIMEXStage end
struct ExplicitCorrectionStage2 <: AbstractIMEXStage end
struct ImplicitPredictionStage2 <: AbstractIMEXStage end
struct ImplicitCorrectionStage2 <: AbstractIMEXStage end

@inline stages(::SecondOrderFiveStagesIMEX) = (
    ExplicitCorrectionStage1(),
    ImplicitPredictionStage1(),
    ExplicitCorrectionStage2(),
    ImplicitPredictionStage2(),
    ImplicitCorrectionStage2(),
)

"""
    IMEXIntegrator{S} <: AbstractTimeIntegrator

IMEX time integrator carrying the scheme (e.g., `FirstOrderThreeStagesIMEX`).
"""
struct IMEXIntegrator{S <: AbstractIMEXScheme} <: AbstractTimeIntegrator
    scheme::S
end

struct ARS222{T}
    gamma_ars::T
    delta_ars::T
end

"""
    IMEXCacheFirstOrder

Cache storing the intermediate solution states required by an IMEX first order time integration method.

The cache owns only algorithmic states. Solver-specific workspaces
(e.g. Newton residuals, Jacobians) are owned by the `EllipticCache`(@ref).
"""
mutable struct IMEXCacheFirstOrder{TU, TW, TP, TR, TE}
    u::TU               # Current solution u^n
    u_buffer::TW        # Hyperbolic work buffer
    rho_hat::TR         # intermediate states
    vel::TR             # part of primitive variable storing
    phi::TP             # Elliptic solution ϕ^{n+1}
    eta::TE             # η: recomputed each timestep
end

function IMEXCacheFirstOrder(u0::TU, phi0::TP) where {TU, TP}
    T = eltype(u0)
    IMEXCacheFirstOrder(u0,
                        similar(u0),            # u_buffer
                        similar(phi0),          # rho_hat (since rho_hat is 1 scalar per cell same as phi0)
                        similar(phi0),          # vel
                        phi0,
                        zero(T),)               # eta
end

"""
    IMEXCacheSecondOrder

Cache storing the intermediate solution states required by an IMEX second order time integration method.

The cache owns only algorithmic states. Solver-specific workspaces
(e.g. Newton residuals, Jacobians) are owned by the `EllipticCache`(@ref).
"""
mutable struct IMEXCacheSecondOrder{TU, TW, TP, TR, TE, TL}
    u::TU           # current solution u^n
    u_reconstructed::TW   # reconstructed state used for slope-limited interfaces
    rho_hat::TR     # intermediate states
    m_hat::TR       # intermediate states
    rho_exp::TR     # intermediate states
    m_exp::TR       # intermediate states
    vel::TR         # part of primitive variable storing
    phi::TP         # Elliptic solution ϕ^{n+1}
    eta::TE         # η: recomputed each timestep
    explicit_density_flux_diff_stage1::TR           # storing explicit density flux differences for re-use
    semi_implicit_density_flux_diff_stage1::TR      # storing semi implicit density flux differences for re-use
    momentum_flux_diff_stage1::TR

    explicit_density_flux_diff_stage2::TR           # storing explicit density flux differences for re-use
    semi_implicit_density_flux_diff_stage2::TR      # storing semi implicit density flux differences for re-use
    momentum_flux_diff_stage2::TR
    slopes::TW
    limiter::TL     # slope limiter; single source of truth for interior *and*
                    # ghost-cell slopes (see `reconstructed_ghost_rho_vel`)
end

function IMEXCacheSecondOrder(u0::TU, phi0::TP; limiter = minmod) where {TU, TP}
    T = eltype(u0)
    IMEXCacheSecondOrder(u0,
                        similar(u0),            # u_reconstructed
                        similar(phi0),          # rho_hat (since rho_hat is 1 scalar per cell same as phi0)
                        similar(phi0),          # m_hat
                        similar(phi0),          # rho_exp
                        similar(phi0),          # m_exp
                        similar(phi0),          # vel
                        phi0,
                        zero(T),                # eta
                        similar(phi0),          # explicit_density_flux_diff_stage1
                        similar(phi0),          # semi_implicit_density_flux_diff_stage1
                        similar(phi0),          # momentum_flux_diff_stage1
                        similar(phi0),          # explicit_density_flux_diff_stage2
                        similar(phi0),          # semi_implicit_density_flux_diff_stage2
                        similar(phi0),          # momentum_flux_diff_stage2
                        similar(u0),            # slopes
                        limiter,)
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

@inline function compute_eta!(cache::Union{IMEXCacheFirstOrder, IMEXCacheSecondOrder}, semi::AbstractSemidiscretization, t)
    equations = semi.equations
    mesh      = semi.mesh
    gamma     = equations.gamma

    T = eltype(mesh.dx)
    eta_val = zero(T)

    @inbounds for I in eachcell(mesh)
        # Center density
        u_cc = cell_state(cache.u, I, semi, t)
        rho_c = u_cc[1]

        # Right neighbor
        Ip1 = neighbor_index(I, semi, 1, 1)
        u_rr = cell_state(cache.u, Ip1, semi, t)
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
@inline function compute_dt_1!(cache::IMEXCacheFirstOrder, semi::AbstractSemidiscretization, t)
    mesh      = semi.mesh
    lambda    = semi.equations_elliptic.lambda
    T = eltype(mesh.dx)
    dx = mesh.dx[1]

    k_val = typemin(T)

    @inbounds for I in eachcell(mesh)

        # Center state — interior, so this reads cache.u/cache.vel directly
        # (no division); see update_primitive_variables!/rho_vel_at.
        rho_c, vel_c = rho_vel_at(cache.u, cache.vel, semi, I, t)

        # Potential at center
        phi_c = _elliptic_var(cache.phi, semi, I, t)

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

@inline function compute_dt_2!(cache::Union{IMEXCacheFirstOrder, IMEXCacheSecondOrder}, semi::AbstractSemidiscretization, t)
    mesh      = semi.mesh
    eta       = cache.eta

    T = eltype(mesh.dx)
    dx = mesh.dx[1]

    k_val = typemin(T)

    @inbounds for I in eachcell(mesh)
        cell = cell_index(I, semi)

        # Center state — interior, reads cache.u/cache.vel directly.
        rho_i, vel_i = rho_vel_at(cache.u, cache.vel, semi, I, t)

        # Potential at center
        phi_i = _elliptic_var(cache.phi, semi, I, t)

        # Right neighbor — interior for all but the last cell at a
        # non-periodic right boundary, where it falls back to an on-the-fly
        # ghost evaluation inside rho_vel_at.
        Ip1 = neighbor_index(I, semi, 1, 1)
        rho_r, vel_r = rho_vel_at(cache.u, cache.vel, semi, Ip1, t)

        # Potential at right neighbor
        phi_r = _elliptic_var(cache.phi, semi, Ip1, t)

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

"""
    update_primitive_variables!(cache::Union{IMEXCacheFirstOrder, IMEXCacheSecondOrder}, semi)

Compute velocity `vel = m/ρ` at every interior cell from the current
conservative state `cache.u` and store it in `cache.vel`. Called once per
timestep, before any stage reads velocity, so `compute_dt_1!`,
`compute_dt_2!`, and `ExplicitCorrectionStage!` all read the same
precomputed values via `rho_vel_at` instead of each re-deriving `vel` from
`cache.u` independently.

Only interior cells are cached: ghost/boundary states may depend on `t` and the
boundary condition and are touched far less often, so they're computed on
the fly by `rho_vel_at` instead.
"""
function update_primitive_variables!(cache::Union{IMEXCacheFirstOrder, IMEXCacheSecondOrder},
                                     semi::AbstractSemidiscretization)
    mesh  = semi.mesh
    nvars = nvariables(semi.equations)
    nx    = size(mesh, 1)

    @inbounds for i in 1:nx
        # Fetch indices once to avoid redundant function calls
        idx_rho = global_dof(i, 1, nvars)
        idx_mom = global_dof(i, 2, nvars)
        
        rho = cache.u[idx_rho]
        
        # Extract error path to a non-allocating helper function
        if rho <= 0
            throw_negative_density_error(i, rho)
        end
        
        mom = cache.u[idx_mom]
        cache.vel[i] = mom / rho
    end

    # Helper function
    @noinline function throw_negative_density_error(i, rho)
        error("Negative density entering update_primitive_variables!\ncell = $i\nrho  = $rho")
    end

    return nothing
end

"""
    rho_vel_at(u, vel, semi, I, t)

Return `(rho, vel)` at cell/ghost index `I`. Interior cells (including a
`neighbor_index`-clamped `ExtrapolateBC`/`NeumannBC` ghost, which resolves
to a valid interior index) read `u`/`vel` directly — no division. Genuine
out-of-range ghost states (`DirichletBC`/`MixedBC`, or a periodic wrap that
`neighbor_index` has already resolved to a valid interior index too) are
computed on the fly since they may depend on `t` and aren't cached.
"""
@inline function rho_vel_at(u, vel, semi::AbstractSemidiscretization,
                            I::CartesianIndex, t)
    nx = size(semi.mesh, 1)
    # This branch is needed in order to use cached velocity and return it directly instead of 
    # performing divisions for all cells (division are very slow operations).
    if 1 <= I[1] <= nx
        cell  = cell_index(I, semi)
        nvars = nvariables(semi.equations)
        return SVector{2}(u[global_dof(cell, 1, nvars)], vel[cell])
    end

    # Only being called for boundary cells (due to their possible dependence on `t`)
    s = cell_state(u, I, semi, t)
    rho = s[1]
    if rho <= 0
        error("""
              Negative density in rho_vel_at (ghost state)
              time = $t
              rho  = $rho
              """)
    end
    return SVector{2}(rho, s[2] / rho)
end

@inline function rho_vel_at(u, semi::AbstractSemidiscretization,
                            I::CartesianIndex, t)
    s = cell_state(u, I, semi, t)
    rho = s[1]
    if rho <= 0
        error("Negative density in rho_vel_at: rho = $rho")
    end
    return SVector{2}(rho, s[2] / rho)
end

"""
    reconstructed_ghost_rho_vel(cache, semi, I, side, t)

Reconstructed `(rho, vel)` at the domain-facing edge of the ghost cell `I`
(`I` outside `1:size(semi.mesh, 1)`), using **two** ghost cells: `I` itself
and the next one further out (`I - 1`/`I + 1`, whichever continues away from
the domain), plus the adjacent interior cell. This mirrors exactly how an
interior cell's own slope is built from its two neighbors, so the ghost gets
a properly limited slope instead of being treated as piecewise constant;
the limiter is `cache.limiter`, the same one `reconstruct_slopes!` uses for
interior cells. `cell_state` resolves each of the three stencil points through the
same `apply_bc` machinery regardless of how far outside the domain they are,
so this works uniformly for `ExtrapolateBC`, `DirichletBC`, `NeumannBC` and
`MixedBC` (never call this for `PeriodicBC`; `apply_bc` errors for it, and
periodic ghosts should be resolved to their wrapped interior index before
reaching here, as `reconstruct_slopes!` already does).
"""
@inline function reconstructed_ghost_rho_vel(cache, semi, I, side, t)
    dx = semi.mesh.dx[1]

    Il = CartesianIndex(I[1] - 1)
    Ir = CartesianIndex(I[1] + 1)

    u_l = cell_state(cache.u_reconstructed, Il, semi, t)
    u_c = cell_state(cache.u_reconstructed, I,  semi, t)
    u_r = cell_state(cache.u_reconstructed, Ir, semi, t)

    rho_l, vel_l = u_l[1], u_l[2] / u_l[1]
    rho_c, vel_c = u_c[1], u_c[2] / u_c[1]
    rho_r, vel_r = u_r[1], u_r[2] / u_r[1]

    limiter = cache.limiter

    slope_rho = limiter((rho_c - rho_l) / dx, (rho_r - rho_c) / dx)
    slope_vel = limiter((vel_c - vel_l) / dx, (vel_r - vel_c) / dx)

    sgn = side === :left ? 1.0 : -1.0

    rho = rho_c + sgn * 0.5 * dx * slope_rho
    vel = vel_c + sgn * 0.5 * dx * slope_vel

    return rho, vel
end

@inline function reconstructed_rho_vel_at(cache,
                                          semi,
                                          I,
                                          side,
                                          t,)
    equations = semi.equations
    dx        = semi.mesh.dx[1]
    nvars     = nvariables(equations)

    # Ghost cell: reconstruct its own slope from two ghost layers (see
    # `reconstructed_ghost_rho_vel`) instead of returning the piecewise
    # constant BC value.
    if !(1 <= I[1] <= size(semi.mesh, 1))
        return reconstructed_ghost_rho_vel(cache, semi, I, side, t)
    end

    rho_idx = global_dof(I, 1, nvars)
    vel_idx = global_dof(I, 2, nvars)   # cache.slopes here is now a velocity slope

    sgn = side === :left ? 1.0 : -1.0

    rho_c = cache.u_reconstructed[rho_idx]
    m_c   = cache.u_reconstructed[vel_idx]
    vel_c = m_c / rho_c

    rho = rho_c + sgn * 0.5 * dx * cache.slopes[rho_idx]
    vel = vel_c + sgn * 0.5 * dx * cache.slopes[vel_idx]

    return rho, vel
end


# Reconstructed *conservative* state (ρ, m) at the given side of cell I.
# Unlike `reconstructed_rho_vel_at` (which returns (ρ, vel)), this returns the
# state layout expected by `solver.flux`.
@inline function reconstructed_conservative_state_at(cache, semi, I, side, t)
    nvars = nvariables(semi.equations)
    dx    = semi.mesh.dx[1]

    # Ghost cell: reconstruct its own slope from two ghost layers (see
    # `reconstructed_ghost_rho_vel`) instead of returning the piecewise
    # constant BC value.
    if !(1 <= I[1] <= size(semi.mesh, 1))
        rho, vel = reconstructed_ghost_rho_vel(cache, semi, I, side, t)
        return SVector{2}(rho, rho * vel)
    end

    rho_idx = global_dof(I, 1, nvars)
    vel_idx = global_dof(I, 2, nvars)   # cache.slopes here is now a velocity slope

    sgn = side === :left ? 1.0 : -1.0

    rho_c = cache.u_reconstructed[rho_idx]
    m_c   = cache.u_reconstructed[vel_idx]
    vel_c = m_c / rho_c

    rho = rho_c + sgn * 0.5 * dx * cache.slopes[rho_idx]
    vel = vel_c + sgn * 0.5 * dx * cache.slopes[vel_idx]

    return SVector{2}(rho, rho * vel)
end

# Fill a preallocated u (hyperbolic block) from rho and m arrays
function wrap_array!(u, rho::T, m::T, semi::SemidiscretizationHyperbolicElliptic) where T
    nvars = nvariables(semi.equations)
    @inbounds for i in eachindex(rho)
        u[global_dof(i, 1, nvars)] = rho[i]
        u[global_dof(i, 2, nvars)] = m[i]
    end
    return nothing
end

"""
    solve(semi, tspan, integrator::IMEXIntegrator;
          dt, callbacks=CallbackSet())

Advance the semidiscretization using the IMEX solver.
"""
function solve(semi,
               tspan,
               integrator::IMEXIntegrator;
            #    dt = 0.0,
               dt = minimum_cell_size(semi.mesh),
               # Accepted (but unused: the IMEX schemes pick their own step
               # from a CFL condition every iteration) so that generic
               # callers like `convergence_test`, written against
               # `ImplicitEulerCustom`'s `solve`, also work with an
               # `IMEXIntegrator`.
               abstol = nothing,
               reltol = nothing,
               callbacks=CallbackSet())

    return solve_imex(semi,
                      integrator,
                      tspan,
                      integrator.scheme;
                      dt = dt,
                      callbacks=callbacks)
end
end # @muladd