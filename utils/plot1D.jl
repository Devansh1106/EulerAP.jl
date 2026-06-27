#!/usr/bin/env julia
"""
    plot1D.jl

Read one or more 1D HDF5 solution files (new format with top-level `eps`
and `ncells` scalars) and plot density & velocity profiles on the same figure.

The first file is treated as the **initial condition** (plotted with a dashed
black line).  Subsequent files are **final solutions** and are distinguished
by line style/colour.

Legend logic:
  - Identify which parameters vary across the *final* files.
  - Preference for legend labels: eps > mesh > t (final time).
  - Common parameters go into the plot title.

Output is saved as:
    plots_new/compare_<basename1>_<basename2>_....png

Usage:
    julia --project=. utils/plot1D.jl [--output <file.png>] <initial.h5> [final1.h5 final2.h5 ...]

Examples:
    # Initial condition + two final solutions with different epsilon
    julia --project=. utils/plot1D.jl initial.h5 sol_eps1.0.h5 sol_eps0.1.h5

    # Initial condition + two final solutions with different mesh sizes
    julia --project=. utils/plot1D.jl initial.h5 sol_N100.h5 sol_N200.h5

    # Specify custom output path
    julia --project=. utils/plot1D.jl --output myplot.png initial.h5 sol_eps1.0.h5

    # Plot only the initial condition (no final files)
    julia --project=. utils/plot1D.jl initial.h5
"""
# using HDF5
# using Plots

# Line styles / colors to cycle through
const LINE_STYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]
const LINE_COLORS = [:black, :red, :blue, :green, :orange, :purple, :brown,
                     :pink, :olive, :cyan, :magenta, :navy]

function read_solution_1d(filepath::String)
    data = Dict{String, Any}()

    h5open(filepath, "r") do f
        # ---- convenience top-level attributes ----
        eps_str = read(f, "eps")
        data["eps"] = eps_str

        mesh_str = read(f, "ncells")
        data["ncells"] = mesh_str

        # ---- mesh sub-group ----
        cells = read(f, "mesh/cells_per_dimension")
        x_min = read(f, "mesh/coordinates_min")[1]
        x_max = read(f, "mesh/coordinates_max")[1]
        dx    = read(f, "mesh/dx")[1]

        N = cells[1]

        # Cell-centred coordinates
        x = [x_min + (i - 0.5) * dx for i in 1:N]
        data["x"] = x
        data["N"] = N

        # ---- equation parameters ----
        gamma = read(f, "equations/gamma")
        data["gamma"] = gamma

        # ---- metadata ----
        t = read(f, "metadata/time")
        data["t"] = t

        # ---- solution ----
        u = read(f, "solution/u")           # shape (nvars, ndofs)
        rho = u[1, :]
        mx  = u[2, :]
        ux  = mx ./ rho

        data["rho"] = rho
        data["ux"]  = ux
    end

    return data
end

function main()
    # Parse optional --output flag
    output_file = nothing
    input_files = String[]
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--output" || ARGS[i] == "-o"
            i += 1
            if i > length(ARGS)
                println(stderr, "Error: --output requires a filename argument")
                exit(1)
            end
            output_file = ARGS[i]
        else
            push!(input_files, ARGS[i])
        end
        i += 1
    end

    if length(input_files) < 1
        println(stderr, "Usage: julia --project=. utils/plot1D.jl [--output <file.png>] <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(input_files)

    # First file is the initial condition
    init = read_solution_1d(input_files[1])

    # Remaining files are final solutions
    final_files = [read_solution_1d(input_files[i]) for i in 2:nfiles]
    nfinal = length(final_files)

    # ------------------------------------------------------------------
    # Determine what varies across final files -> legend vs title
    # ------------------------------------------------------------------
    if nfinal > 0
        epsilons = [f["eps"]   for f in final_files]
        mesh_strs = [f["ncells"] for f in final_files]
        ts       = [f["t"]       for f in final_files]
        gammas   = [f["gamma"]   for f in final_files]

        eps_unique   = unique(epsilons)
        mesh_unique  = unique(mesh_strs)
        t_unique     = unique(ts)
        gamma_unique = unique(gammas)

        # Decide which parameter goes in the legend (priority: eps > mesh > t)
        if length(eps_unique) > 1
            legend_labels = ["ε = $(e)" for e in epsilons]
            legend_param  = :eps
        elseif length(mesh_unique) > 1
            legend_labels = ["N = $(m)" for m in mesh_strs]
            legend_param  = :mesh
        elseif length(t_unique) > 1
            legend_labels = ["t = $(t)" for t in ts]
            legend_param  = :t
        else
            legend_labels = ["final $(i)" for i in 1:nfinal]
            legend_param  = :none
        end

        # Build title from common (non-varying) parameters across final files
        title_parts = String[]
        if length(gamma_unique) == 1
            push!(title_parts, "γ = $(gamma_unique[1])")
        end
        if legend_param != :eps && length(eps_unique) == 1
            push!(title_parts, "ε = $(eps_unique[1])")
        end
        if legend_param != :mesh && length(mesh_unique) == 1
            push!(title_parts, "N = $(mesh_unique[1])")
        end
        if legend_param != :t && length(t_unique) == 1
            push!(title_parts, "t = $(t_unique[1])")
        end
        plot_title = join(title_parts, ", ")
    else
        # Only an initial condition file provided
        legend_labels = String[]
        legend_param  = :none
        title_parts = String[]
        push!(title_parts, "γ = $(init["gamma"])")
        push!(title_parts, "ε = $(init["eps"])")
        push!(title_parts, "N = $(init["ncells"])")
        plot_title = join(title_parts, ", ")
    end

    # ------------------------------------------------------------------
    # Build output path
    # ------------------------------------------------------------------
    if output_file !== nothing
        out_path = output_file
        mkpath(dirname(out_path))
    else
        bases = [splitext(basename(f))[1] for f in input_files]
        out_path = joinpath("plots_new", "compare_$(join(bases, "_")).png")
        mkpath("plots_new")
    end

    # ------------------------------------------------------------------
    # Density subplot
    # ------------------------------------------------------------------
    p1 = plot(xlabel = "x", ylabel = "ρ", title = "Density")

    # Initial condition from first file (dashed black)
    plot!(p1, init["x"], init["rho"],
          lw = 2, ls = :dash, color = :black,
          label = "Initial")

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        plot!(p1, f["x"], f["rho"],
              lw = 2, ls = ls, color = lc,
              label = legend_labels[i])
    end

    # ------------------------------------------------------------------
    # Velocity subplot
    # ------------------------------------------------------------------
    p2 = plot(xlabel = "x", ylabel = "u_x", title = "Velocity")

    # Initial condition from first file (dashed black)
    plot!(p2, init["x"], init["ux"],
          lw = 2, ls = :dash, color = :black,
          label = "Initial")

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        plot!(p2, f["x"], f["ux"],
              lw = 2, ls = ls, color = lc,
              label = legend_labels[i])
    end

    fig = plot(p1, p2, layout = (1, 2), size = (1400, 550),
               plot_title = plot_title)

    savefig(fig, out_path)
    println("Saved comparison plot to $out_path")
end

main()