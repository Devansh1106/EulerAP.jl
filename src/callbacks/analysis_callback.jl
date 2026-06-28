# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AnalysisCallback(; interval=100,
                       exact_solution=nothing)

Compute and print error norms every `interval` timesteps.
"""
struct AnalysisCallback{F} <: AbstractCallback

    interval::Int

    exact_solution::Union{Nothing,F}

end


"""
    AnalysisCallback(; interval=100,
                       exact_solution=nothing)

Construct an analysis callback.
"""
function AnalysisCallback(; interval = 100,
                            exact_solution = nothing)

    return AnalysisCallback{typeof(exact_solution)}(
        interval,
        exact_solution
    )
end


function perform!(callback::AnalysisCallback,
                  context::CallbackContext)

    stats = context.stats

    if stats.iteration % callback.interval != 0
        return nothing
    end

    callback.exact_solution === nothing && return nothing

    simulation = context.simulation

    result = compute_errors(
        context.solution,
        simulation.semi;
        exact_solution = callback.exact_solution
    )

    println()
    println("================ Analysis ================")

    for (variable, norms) in enumerate(result.norms)

        println("Variable ", variable)

        print_summary_line("L¹", norms.L1)
        print_summary_line("L²", norms.L2)
        print_summary_line("L∞", norms.Linf)

        println()

    end

    println("==========================================")

    return nothing
end

end # @muladd