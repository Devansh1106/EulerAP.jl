# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PerformanceCallback(; interval = 100)

Print accumulated runtime statistics.
"""
struct PerformanceCallback <: AbstractCallback
    interval::Int
end

PerformanceCallback(; interval = 100) = PerformanceCallback(interval)


function perform!(callback::PerformanceCallback,
                  context::CallbackContext;
                  force = false)

    stats = context.stats

    if !force && stats.iteration % callback.interval != 0
        return nothing
    end

    # println()
    # println("====================== Performance =========================")

    # TimerOutputs table (time + allocations + ncalls)
    show(stdout, stats.timer; allocations = true, compact = false)

    # println("============================================================")

    return nothing
end

function finalize!(callback::PerformanceCallback,
                   context::CallbackContext)

    perform!(callback, context; force = true)
    return nothing
end

end # @muladd