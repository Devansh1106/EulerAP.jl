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


PerformanceCallback(; interval = 100) =
    PerformanceCallback(interval)


function perform!(callback::PerformanceCallback,
                  context::CallbackContext)

    stats = context.stats

    if stats.iteration % callback.interval != 0
        return nothing
    end

    println()
    println("============== Performance ==============")

    print_summary_line("RHS calls",
                       stats.rhs_calls)

    print_summary_line("Jacobian calls",
                       stats.jacobian_calls)

    print_summary_line("Newton iterations",
                       stats.nonlinear_iterations)

    print_summary_line("Linear iterations",
                       stats.linear_iterations)

    print_summary_line("RHS time (s)",
                       stats.rhs_time)

    print_summary_line("Jacobian time (s)",
                       stats.jacobian_time)

    print_summary_line("Linear solve (s)",
                       stats.linear_solver_time)

    print_summary_line("Newton solve (s)",
                       stats.nonlinear_solver_time)

    print_summary_line("Total runtime (s)",
                       stats.total_runtime)

    println("=========================================")

    return nothing
end

end # @muladd