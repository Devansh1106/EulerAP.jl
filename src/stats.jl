"""
    print_run_stats(label, stats, nsteps_done, model=nothing; gamma=nothing)

Print a compact summary of wall time, allocations, and GC time for a run.
"""
function print_run_stats(label, 
                         stats::RunStats, 
                         nsteps_done::Int, 
                         model::Union{RelaxationParams, Nothing}=nothing; 
                         gamma::Union{Float64, Nothing}=nothing)

    if nsteps_done == 0
        println(label, " stats: no steps completed")
        return
    end

    step_times   = view(stats.step_times, 1:nsteps_done)
    step_bytes   = view(stats.step_bytes, 1:nsteps_done)
    step_gctimes = view(stats.step_gctimes, 1:nsteps_done)

    avg_step_time  = sum(step_times) / nsteps_done
    avg_step_bytes = sum(step_bytes) / nsteps_done
    avg_step_gc    = sum(step_gctimes) / nsteps_done

    println(label, " stats:")

    if model !== nothing
        println("  grid resolution = ", model.size)
        println("  eps = ", model.eps)
    end
    if gamma !== nothing
        println("  gamma = ", gamma)
    end

    n_threads = get(ENV, "MKL_NUM_THREADS", string(Threads.nthreads()))

    println(
        "  total threads = ", n_threads)

    println(
        "  total wall time = ",
        round(stats.total_time; digits = 6),
        " s"
    )

    println(
        "  total allocations = ",
        round(stats.total_bytes / 2^20; digits = 3),
        " MiB"
    )

    println(
        "  total GC time = ",
        round(stats.total_gctime; digits = 6),
        " s"
    )

    println(
        "  steps completed = ",
        nsteps_done
    )

    println(
        "  first step time = ",
        round(step_times[1]; digits = 6),
        " s"
    )

    println(
        "  first step allocations = ",
        round(step_bytes[1] / 2^20; digits = 3),
        " MiB"
    )

    println(
        "  first step GC time = ",
        round(step_gctimes[1]; digits = 6),
        " s"
    )

    println(
        "  avg step time = ",
        round(avg_step_time; digits = 6),
        " s"
    )

    println(
        "  max step time = ",
        round(maximum(step_times); digits = 6),
        " s"
    )

    println(
        "  avg step allocations = ",
        round(avg_step_bytes / 2^20; digits = 3),
        " MiB"
    )

    println(
        "  max step allocations = ",
        round(maximum(step_bytes) / 2^20; digits = 3),
        " MiB"
    )

    println(
        "  avg step GC time = ",
        round(avg_step_gc; digits = 6),
        " s"
    )

    println(
        "  max step GC time = ",
        round(maximum(step_gctimes); digits = 6),
        " s"
    )
end