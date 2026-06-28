# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# ============================================================================
# Abstract callback interface
# ============================================================================

"""
    AbstractCallback

Abstract supertype for all callbacks.
"""
abstract type AbstractCallback end

"""
    CallbackContext

Object passed to every callback.

It contains

- the simulation setup,
- the current numerical solution,
- the runtime statistics.
"""
mutable struct CallbackContext{SimulationType,Solution,Stats}

    simulation::SimulationType

    solution::Solution

    stats::Stats

end

@inline mesh(context::CallbackContext) = semi(context).mesh

@inline equations(context::CallbackContext) = semi(context).equations

"""
    CallbackSet(callbacks...)

Container storing all callbacks executed during a simulation.
"""
struct CallbackSet{Callbacks<:Tuple}
    callbacks::Callbacks
end

CallbackSet(callbacks...) = CallbackSet(callbacks)


# ============================================================================
# Simulation setup
# ============================================================================

"""
    Simulation

Container storing everything that defines a simulation.

This object is immutable throughout the simulation and contains the
semidiscretization together with the time integration parameters.
"""
struct Simulation{Semi,Integrator,T}

    semi::Semi

    integrator::Integrator

    tspan::Tuple{T,T}

    dt::T

    abstol::T

    reltol::T

end


# ============================================================================
# Runtime statistics
# ============================================================================

"""
    CallbackStats

Stores simulation state together with runtime statistics.

The time integrator updates this object during the simulation while
callbacks only read from it.
"""
mutable struct CallbackStats{T}

    # ------------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------------

    iteration::Int

    time::T

    dt::T

    # ------------------------------------------------------------------------
    # Timings
    # ------------------------------------------------------------------------

    rhs_time::T

    jacobian_time::T

    linear_solver_time::T

    nonlinear_solver_time::T

    total_runtime::T

    # ------------------------------------------------------------------------
    # Function call counters
    # ------------------------------------------------------------------------

    rhs_calls::Int

    jacobian_calls::Int

    # ------------------------------------------------------------------------
    # Solver statistics
    # ------------------------------------------------------------------------

    nonlinear_iterations::Int

    linear_iterations::Int

    nonlinear_solves::Int

    linear_solves::Int

    # ------------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------------

    allocations::Int

    bytes_allocated::Int

end


"""
    CallbackStats(T)

Construct zero-initialized callback statistics.
"""
function CallbackStats(::Type{T}) where {T}

    CallbackStats(

        0,
        zero(T),
        zero(T),

        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),

        0,
        0,

        0,
        0,

        0,
        0,

        0,
        0
    )
end


"""
    reset!(stats)

Reset callback statistics.
"""
function reset!(stats::CallbackStats)

    stats.iteration = 0
    stats.time = zero(stats.time)
    stats.dt = zero(stats.dt)

    stats.rhs_time = zero(stats.rhs_time)
    stats.jacobian_time = zero(stats.jacobian_time)
    stats.linear_solver_time = zero(stats.linear_solver_time)
    stats.nonlinear_solver_time = zero(stats.nonlinear_solver_time)
    stats.total_runtime = zero(stats.total_runtime)

    stats.rhs_calls = 0
    stats.jacobian_calls = 0

    stats.nonlinear_iterations = 0
    stats.linear_iterations = 0

    stats.nonlinear_solves = 0
    stats.linear_solves = 0

    stats.allocations = 0
    stats.bytes_allocated = 0

    return nothing
end


# ============================================================================
# Callback context
# ============================================================================
@inline function print_summary_line(name, value)
    println(rpad(name, 22), ": ", value)
end

# ============================================================================
# Generic callback interface
# ============================================================================

"""
    initialize!(callback, context)

Initialize a callback before time integration begins.
"""
initialize!(::AbstractCallback,
            ::CallbackContext) = nothing


"""
    perform!(callback, context)

Execute a callback during time integration.
"""
perform!(::AbstractCallback,
         ::CallbackContext) = nothing


"""
    finalize!(callback, context)

Finalize a callback after time integration.
"""
finalize!(::AbstractCallback, ::CallbackContext) = nothing

# ============================================================================
# Callback execution
# ============================================================================

"""
    initialize_callbacks!(callbacks, context)

Initialize all callbacks before time integration.
"""
function initialize_callbacks!(callbacks::CallbackSet,
                               context::CallbackContext)

    for callback in callbacks.callbacks
        initialize!(callback, context)
    end

    return nothing
end


"""
    perform_callbacks!(callbacks, context)

Execute all callbacks.
"""
function perform_callbacks!(callbacks::CallbackSet,
                            context::CallbackContext)

    for callback in callbacks.callbacks
        perform!(callback, context)
    end

    return nothing
end


"""
    finalize_callbacks!(callbacks, context)

Finalize all callbacks after time integration.
"""
function finalize_callbacks!(callbacks::CallbackSet,
                             context::CallbackContext)

    for callback in callbacks.callbacks
        finalize!(callback, context)
    end

    return nothing
end

end # @muladd