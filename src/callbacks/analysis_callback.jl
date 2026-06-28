# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AnalysisCallback(; exact_solution, interval = typemax(Int))

Compute error norms against an exact solution.

By default, the analysis is performed only once at the final time.
"""
struct AnalysisCallback{F} <: AbstractCallback
    interval::Int
    exact_solution::F
end

function AnalysisCallback(;
                          exact_solution,
                          interval = typemax(Int))

    return AnalysisCallback(interval,
                            exact_solution)
end


function perform!(callback::AnalysisCallback,
                  context::CallbackContext;
                  force = false)

    stats = context.stats

    if !force && stats.iteration % callback.interval != 0
        return nothing
    end

    # callback.exact_solution === nothing && return nothing

    simulation = context.simulation

    result = compute_errors(
        context.solution,
        simulation.semi;
        exact_solution = callback.exact_solution
    )

    println()
    println("======================== Analysis ==========================")

    for (variable, norms) in enumerate(result.norms)

        println("Variable ", variable)

        print_summary_line("L¹", norms.L1)
        print_summary_line("L²", norms.L2)
        print_summary_line("L∞", norms.Linf)

        println()

    end

    println("============================================================")


    return nothing
end

function finalize!(callback::AnalysisCallback,
                   context::CallbackContext)

    perform!(callback, context; force = true)
    return nothing
end

end # @muladd
