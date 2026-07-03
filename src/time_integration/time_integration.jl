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

# struct depatches on type of the scheme
stages(::FirstOrderThreeStagesIMEX) = (
    ExplicitCorrectionStage(),
    ImplicitPredictionStage(),
    ImplicitCorrectionStage(),
)

struct IMEXIntegrator{S <: AbstractIMEXScheme} <: AbstractTimeIntegrator
    scheme::S
end



end # @muladd