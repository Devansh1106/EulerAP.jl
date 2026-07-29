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

"""
    IMEXIntegrator{S} <: AbstractTimeIntegrator

IMEX time integrator carrying the scheme (e.g., `FirstOrderThreeStagesIMEX`).
The elliptic solve uses a hand-coded Thomas algorithm via NonlinearSolve.jl.
"""
struct IMEXIntegrator{S <: AbstractIMEXScheme} <: AbstractTimeIntegrator
    scheme::S
end

"""
    IMEXCache

Cache storing the intermediate solution states required by an IMEX
time integration method.

The cache owns only algorithmic states. Solver-specific workspaces
(e.g. Newton residuals, Jacobians, Krylov vectors) are owned by the
`EllipticCache`(@ref).
"""
mutable struct IMEXCache{TU,TP, TR, TW, TE}
    # Current solution u^n
    u::TU

    rho_hat::TR

    # Corrected solution u^{n+1}
    u_new::TW

    # Hyperbolic work buffer
    u_buffer::TW

    # Elliptic solution ϕ^{n+1}
    phi::TP

    # Diffusion coefficient η, recomputed each timestep
    eta::TE
end

function IMEXCache(u0, phi0)
    # since rho_hat is 1 scalar per cell same as phi0
    rho_hat = similar(phi0)
    T = eltype(u0)
    n = length(u0)
    IMEXCache(u0,
              rho_hat,
              Vector{T}(undef, n),
              Vector{T}(undef, n),
              phi0,
              zero(T))
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
               callbacks=CallbackSet())

    return solve_imex(semi,
                      integrator,
                      tspan;
                      dt = dt,
                      callbacks=callbacks)
end
end # @muladd