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

"""
    save_solution_h5(sol::Union{sol1D, sol2D}, p::RelaxationParams; output_dir="data")

Write the solution fields (coordinates, u_init, u_final) and metadata to an HDF5 file.

The file is named `sol(dims)D_(meshsize)_(eps).h5` and placed in `output_dir`.
For 1D:      `sol1D_(Nx)_(eps).h5`
For 2D:      `sol2D_(Nx)x(Ny)_(eps).h5`
"""
function save_solution_h5(sol::Union{sol1D, sol2D}, p::RelaxationParams; output_dir="data", suffix="", t_final=nothing)
    ndims_ = ndims(p)

    # Build file name
    if ndims_ == 1
        meshsize = "$(p.size[1])"
    else
        meshsize = "$(p.size[1])x$(p.size[2])"
    end
    if t_final !== nothing && suffix == ""
        fname = "sol$(ndims_)D_$(meshsize)_$(p.eps)_t=$(t_final).h5"
    else
        fname = "sol$(ndims_)D_$(meshsize)_$(p.eps)$(suffix).h5"
    end
    fpath = joinpath(output_dir, fname)

    mkpath(output_dir)

    h5open(fpath, "w") do h5f
        # Write coordinates
        if sol isa sol1D
            write(h5f, "x", collect(sol.x))
        else
            write(h5f, "x", collect(sol.x))
            write(h5f, "y", collect(sol.y))
        end

        # Write solution vectors
        write(h5f, "u_init",  collect(sol.u_init))
        write(h5f, "u_final", collect(sol.u_final))

        # Write metadata
        write(h5f, "ncells", sol._ncells)
        write(h5f, "eps",    p.eps)
        write(h5f, "size",   collect(p.size))
        if t_final !== nothing
            write(h5f, "t_final", t_final)
        end
    end

    println("Solution saved to ", fpath)
    return fpath
end
