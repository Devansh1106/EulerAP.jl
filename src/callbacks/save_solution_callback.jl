# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SaveSolutionCallback(; interval = 100,
                           directory = "output",
                           prefix = "solution")

Save the numerical solution every `interval` timesteps.
"""
struct SaveSolutionCallback <: AbstractCallback

    interval::Int

    directory::String

    prefix::String

end


function SaveSolutionCallback(;
    interval = 100,
    directory = "output",
    prefix = "solution")

    return SaveSolutionCallback(
        interval,
        directory,
        prefix
    )
end


function perform!(callback::SaveSolutionCallback,
                  context::CallbackContext)

    stats = context.stats

    if stats.iteration % callback.interval != 0
        return nothing
    end

    mkpath(callback.directory)
    
    filename = joinpath(
        callback.directory,
        string(
            callback.prefix,
            "_",
            lpad(stats.iteration, 6, '0'),
            ".h5"
        )
    )

    save_solution(
        context.solution,
        context.simulation.semi,
        filename
    )

    println("[Output] Saved solution to ", filename)

    return nothing
end

end # @muladd