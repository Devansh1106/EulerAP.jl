# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AliveCallback(; interval = 100)

Print simulation progress every `interval` timesteps.
"""
struct AliveCallback <: AbstractCallback
    interval::Int
end

AliveCallback(; interval = 100) = AliveCallback(interval)


function perform!(callback::AliveCallback,
                  context::CallbackContext;
                  force = false)

    stats = context.stats

    if !force && stats.iteration % callback.interval != 0
        return nothing
    end

    println()
    println("======================= Alive ==============================")

    print_summary_line("Iteration", stats.iteration)
    print_summary_line("Time", stats.time)
    print_summary_line("Time step", stats.dt)

    println("============================================================")

    return nothing
end

function finalize!(callback::AliveCallback,
                   context::CallbackContext)

    perform!(callback, context; force = true)
    return nothing
end

end # @muladd
