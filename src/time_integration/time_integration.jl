# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

abstract type AbstractTimeIntegrator end

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

# Used in Callbacks
@inline integrator(context::CallbackContext) = context.simulation.integrator

end # @muladd