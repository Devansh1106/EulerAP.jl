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

    println()
    println("====================== Performance =========================")

    # --------------------------------------------------
    # Counts
    # --------------------------------------------------

    print_summary_line("RHS calls",
                       stats.rhs_calls)

    print_summary_line("Jacobian calls",
                       stats.jacobian_calls)

    print_summary_line("Total Time Steps",
                       stats.nonlinear_solves)

    print_summary_line("Newton iterations",
                       stats.nonlinear_iterations)

    print_summary_line("Linear iterations",
                       stats.linear_iterations)

    println()

    # --------------------------------------------------
    # Timings
    # --------------------------------------------------

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

    println()

    # --------------------------------------------------
    # Derived statistics
    # --------------------------------------------------

    avg_rhs_time =
        stats.rhs_calls == 0 ? 0.0 :
        stats.rhs_time / stats.rhs_calls

    avg_jacobian_time =
        stats.jacobian_calls == 0 ? 0.0 :
        stats.jacobian_time / stats.jacobian_calls

    avg_newton_time =
        stats.nonlinear_solves == 0 ? 0.0 :
        stats.nonlinear_solver_time / stats.nonlinear_solves

    avg_newton_iterations =
        stats.nonlinear_solves == 0 ? 0.0 :
        stats.nonlinear_iterations / stats.nonlinear_solves

    avg_rhs_per_newton =
        stats.nonlinear_solves == 0 ? 0.0 :
        stats.rhs_calls / stats.nonlinear_solves

    avg_jacobian_per_newton =
        stats.nonlinear_solves == 0 ? 0.0 :
        stats.jacobian_calls / stats.nonlinear_solves

    print_summary_line("Avg RHS time (s)",
                       avg_rhs_time)

    print_summary_line("Avg Jacobian Assembly time (s)",
                       avg_jacobian_time)

    print_summary_line("Avg Newton time (s)",
                       avg_newton_time)

    print_summary_line("Avg Newton iterations",
                       avg_newton_iterations)

    print_summary_line("Avg RHS/Newton calls",
                       avg_rhs_per_newton)

    print_summary_line("Avg Jacobians/Newton calls",
                       avg_jacobian_per_newton)

    println("============================================================")

    return nothing
end

function finalize!(callback::PerformanceCallback,
                   context::CallbackContext)

    perform!(callback, context; force = true)
    return nothing
end

end # @muladd
