# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

using TimerOutputs

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
mutable struct CallbackContext{SimulationType, Solution, Stats}
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
struct Simulation{Semi, Integrator, T}
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

Timing and memory allocation data are tracked via a `TimerOutput`,
which automatically records wall-clock time, number of allocations,
and total bytes allocated for each labeled section.
"""
mutable struct CallbackStats{T}

    # ------------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------------

    iteration::Int
    time::T
    dt::T

    # ------------------------------------------------------------------------
    # Timing & memory (TimerOutput)
    # ------------------------------------------------------------------------

    timer::TimerOutput

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

    # -- Conservation --
    initial_mass::T

    mass_before::T
    mass_after_stage1::T
    mass_after_stage3::T

    relative_mass_error_stage1::T
    relative_mass_error_stage3::T

    minimum_density::T
    maximum_velocity::T
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

        TimerOutput(),

        0,
        0,

        0,
        0,

        0,
        0,

        zero(T),   # initial_mass

        zero(T),   # mass_before
        zero(T),   # mass_after_stage1
        zero(T),   # mass_after_stage3

        zero(T),   # rel error stage1
        zero(T),   # rel error stage3

        typemax(T),# minimum density
        zero(T),   # maximum velocity
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

    reset_timer!(stats.timer)

    stats.rhs_calls = 0
    stats.jacobian_calls = 0

    stats.nonlinear_iterations = 0
    stats.linear_iterations = 0

    stats.nonlinear_solves = 0
    stats.linear_solves = 0

    # ------------------------------------------------------------------------
    # Conservation diagnostics
    # ------------------------------------------------------------------------

    T = typeof(stats.time)

    stats.initial_mass = zero(T)

    stats.mass_before = zero(T)
    stats.mass_after_stage1 = zero(T)
    stats.mass_after_stage3 = zero(T)

    stats.relative_mass_error_stage1 = zero(T)
    stats.relative_mass_error_stage3 = zero(T)

    stats.minimum_density = typemax(T)
    stats.maximum_velocity = zero(T)

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