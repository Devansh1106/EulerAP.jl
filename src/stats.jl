function print_run_stats(label, stats::RunStats, nsteps_done::Int)

    if nsteps_done == 0
        println(label, " stats: no steps completed")
        return
    end

    step_times = view(stats.step_times, 1:nsteps_done)
    step_bytes = view(stats.step_bytes, 1:nsteps_done)
    step_gctimes = view(stats.step_gctimes, 1:nsteps_done)

    avg_step_time = sum(step_times) / nsteps_done
    avg_step_bytes = sum(step_bytes) / nsteps_done
    avg_step_gc = sum(step_gctimes) / nsteps_done

    println(label, " stats:")

    println("  total GC time = ",
        round(stats.total_gctime; digits = 6), " s")

    println("  steps completed = ", nsteps_done)

    println("  avg step time = ",
        round(avg_step_time; digits = 6), " s")

    println("  max step time = ",
        round(maximum(step_times); digits = 6), " s")

    println("  avg step allocations = ",
        round(avg_step_bytes / 2^20; digits = 3), " MiB")
end