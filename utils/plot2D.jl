#!/usr/bin/env julia
"""
    plot2D.jl

Read one or more 2D HDF5 solution files (new format with top-level `eps`
and `ncells` scalars) and plot density, velocity_x, and velocity_y as
2D heatmaps.

The first file is treated as the **initial condition** (shown in the first
column).  Subsequent files are **final solutions** and are shown in the
remaining columns.

Layout:
    Row 1: Density (ρ)        — heatmap
    Row 2: Velocity X (u_x)   — heatmap
    Row 3: Velocity Y (u_y)   — heatmap
    Col 1: Initial state      — labeled "Initial"
    Col 2..N: Final states    — labeled by the varying parameter (ε / mesh / t)

Legend / title logic (identical to plot1D.jl):
  - Identify which parameters vary across the *final* files.
  - Preference for legend labels: eps > mesh > t (final time).
  - Common parameters go into the plot title.

Output is saved as:
    plots_new/compare_<basename1>_<basename2>_....png

Usage:
    julia --project=. utils/plot2D.jl [--output <file.png>] <initial.h5> [final1.h5 final2.h5 ...]

Examples:
    julia --project=. utils/plot2D.jl initial.h5 sol_eps1.0.h5 sol_eps0.1.h5
    julia --project=. utils/plot2D.jl --output myplot.png initial.h5 sol_final.h5
"""
# using HDF5
# using Plots

function read_solution_2d(filepath::String)
    data = Dict{String, Any}()

    h5open(filepath, "r") do f
        # ---- convenience top-level attributes ----
        data["eps"]    = read(f, "eps")
        data["ncells"] = read(f, "ncells")

        # ---- mesh sub-group ----
        cells = read(f, "mesh/cells_per_dimension")
        x_min = read(f, "mesh/coordinates_min")
        x_max = read(f, "mesh/coordinates_max")
        dx    = read(f, "mesh/dx")

        Nx, Ny = cells

        # Cell-centred coordinates
        x = [x_min[1] + (i - 0.5) * dx[1] for i in 1:Nx]
        y = [x_min[2] + (j - 0.5) * dx[2] for j in 1:Ny]
        data["x"]  = x
        data["y"]  = y
        data["Nx"] = Nx
        data["Ny"] = Ny

        # ---- equation parameters ----
        data["gamma"] = read(f, "equations/gamma")

        # ---- metadata ----
        data["t"] = read(f, "metadata/time")

        # ---- solution ----
        # u has shape (nvars, ndofs), ndofs = Nx * Ny
        # Flat index: idx = i + (j-1)*Nx  (column-major, i fastest)
        u_full = read(f, "solution/u")    # (nvars, Nx*Ny)
        rho_flat = u_full[1, :]
        mx_flat  = u_full[2, :]
        my_flat  = u_full[3, :]

        # Reshape to (Nx, Ny) — column-major means it maps to (i, j) correctly.
        rho = reshape(rho_flat, Nx, Ny)
        mx  = reshape(mx_flat,  Nx, Ny)
        my  = reshape(my_flat,  Nx, Ny)

        ux = mx ./ rho
        uy = my ./ rho

        # Transpose for heatmap display (x horizontal, y vertical)
        data["rho_2d"] = rho'
        data["ux_2d"]  = ux'
        data["uy_2d"]  = uy'
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
        println(stderr, "Usage: julia --project=. utils/plot2D.jl [--output <file.png>] <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(input_files)

    # First file is the initial condition
    init = read_solution_2d(input_files[1])

    # Remaining files are final solutions
    final_files = [read_solution_2d(input_files[i]) for i in 2:nfiles]
    nfinal = length(final_files)

    # ------------------------------------------------------------------
    # Determine what varies across final files -> legend vs title
    # ------------------------------------------------------------------
    if nfinal > 0
        epsilons  = [f["eps"]    for f in final_files]
        mesh_strs = [f["ncells"] for f in final_files]
        ts        = [f["t"]      for f in final_files]
        gammas    = [f["gamma"]  for f in final_files]

        eps_unique   = unique(epsilons)
        mesh_unique  = unique(mesh_strs)
        t_unique     = unique(ts)
        gamma_unique = unique(gammas)

        # Decide which parameter goes in the label (priority: eps > mesh > t)
        if length(eps_unique) > 1
            col_labels = ["ε = $(e)" for e in epsilons]
            label_param = :eps
        elseif length(mesh_unique) > 1
            col_labels = ["N = $(m)" for m in mesh_strs]
            label_param = :mesh
        elseif length(t_unique) > 1
            col_labels = ["t = $(t)" for t in ts]
            label_param = :t
        else
            col_labels = ["final $(i)" for i in 1:nfinal]
            label_param = :none
        end

        # Build title from common (non-varying) parameters across final files
        title_parts = String[]
        if length(gamma_unique) == 1
            push!(title_parts, "γ = $(gamma_unique[1])")
        end
        if label_param != :eps && length(eps_unique) == 1
            push!(title_parts, "ε = $(eps_unique[1])")
        end
        if label_param != :mesh && length(mesh_unique) == 1
            push!(title_parts, "N = $(mesh_unique[1])")
        end
        if label_param != :t && length(t_unique) == 1
            push!(title_parts, "t = $(t_unique[1])")
        end
        plot_title = join(title_parts, ", ")
    else
        # Only an initial condition file provided
        col_labels = String[]
        label_param = :none
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
    # Build figure with heatmap subplots
    # Layout: 3 rows (ρ, u_x, u_y) × (1 + nfinal) columns
    # ------------------------------------------------------------------
    total_cols = 1 + nfinal
    total_rows = 3

    # Shared axis limits for side-by-side comparability
    if nfinal > 0
        rho_min = min(minimum(init["rho_2d"]), minimum(minimum(f["rho_2d"]) for f in final_files))
        rho_max = max(maximum(init["rho_2d"]), maximum(maximum(f["rho_2d"]) for f in final_files))
        ux_min  = min(minimum(init["ux_2d"]),  minimum(minimum(f["ux_2d"])  for f in final_files))
        ux_max  = max(maximum(init["ux_2d"]),  maximum(maximum(f["ux_2d"])  for f in final_files))
        uy_min  = min(minimum(init["uy_2d"]),  minimum(minimum(f["uy_2d"])  for f in final_files))
        uy_max  = max(maximum(init["uy_2d"]),  maximum(maximum(f["uy_2d"])  for f in final_files))
    else
        rho_min = minimum(init["rho_2d"])
        rho_max = maximum(init["rho_2d"])
        ux_min  = minimum(init["ux_2d"])
        ux_max  = maximum(init["ux_2d"])
        uy_min  = minimum(init["uy_2d"])
        uy_max  = maximum(init["uy_2d"])
    end

    fig = plot(layout = (total_rows, total_cols),
               size = (400 * total_cols, 400 * total_rows),
               plot_title = plot_title,
               left_margin = 10Plots.mm)

    # Subplot index helper (row-major order in Plots.jl layout)
    sp(row, col) = (row - 1) * total_cols + col

    # Row 1: Density (ρ)
    heatmap!(fig, init["x"], init["y"], init["rho_2d"],
             subplot = sp(1, 1),
             xlabel = "x", ylabel = "ρ", title = "Initial",
             aspect_ratio = :equal,
             clims = (rho_min, rho_max),
             framestyle = :box)
    for (ci, f) in enumerate(final_files)
        heatmap!(fig, f["x"], f["y"], f["rho_2d"],
                 subplot = sp(1, ci + 1),
                 xlabel = "x",
                 title = col_labels[ci],
                 aspect_ratio = :equal,
                 clims = (rho_min, rho_max),
                 framestyle = :box)
    end

    # Row 2: Velocity X (u_x)
    heatmap!(fig, init["x"], init["y"], init["ux_2d"],
             subplot = sp(2, 1),
             xlabel = "x", ylabel = "u_x", title = "Initial",
             aspect_ratio = :equal,
             clims = (ux_min, ux_max),
             framestyle = :box)
    for (ci, f) in enumerate(final_files)
        heatmap!(fig, f["x"], f["y"], f["ux_2d"],
                 subplot = sp(2, ci + 1),
                 xlabel = "x",
                 title = col_labels[ci],
                 aspect_ratio = :equal,
                 clims = (ux_min, ux_max),
                 framestyle = :box)
    end

    # Row 3: Velocity Y (u_y)
    heatmap!(fig, init["x"], init["y"], init["uy_2d"],
             subplot = sp(3, 1),
             xlabel = "x", ylabel = "u_y", title = "Initial",
             aspect_ratio = :equal,
             clims = (uy_min, uy_max),
             framestyle = :box)
    for (ci, f) in enumerate(final_files)
        heatmap!(fig, f["x"], f["y"], f["uy_2d"],
                 subplot = sp(3, ci + 1),
                 xlabel = "x",
                 title = col_labels[ci],
                 aspect_ratio = :equal,
                 clims = (uy_min, uy_max),
                 framestyle = :box)
    end

    savefig(fig, out_path)
    println("Saved 2D comparison plot to $out_path")
end

main()