#!/usr/bin/env julia
"""
    plot2D.jl

Read one or more 2D HDF5 solution files (new format with top-level `eps`
and `ncells` scalars) and plot density, velocity_x, velocity_y, and electric
potential (if present) as 2D heatmaps.

The first file is treated as the **initial condition** (shown in the first
column).  Subsequent files are **final solutions** and are shown in the
remaining columns.

Layout:
    Row 1: Density (ρ)        — heatmap
    Row 2: Velocity X (u_x)   — heatmap
    Row 3: Velocity Y (u_y)   — heatmap
    Row 4: Electric Potential (φ) — heatmap (if nvars ≥ 4)
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

        nvars = read(f, "metadata/nvariables")
        data["nvars"] = nvars

        # ---- solution ----
        # u has shape (nvars, ndofs), ndofs = Nx * Ny
        # Flat index: idx = i + (j-1)*Nx  (column-major, i fastest)
        u_full = read(f, "solution/u")    # (nvars, Nx*Ny)
        rho_flat = u_full[1, :]
        mx_flat  = u_full[2, :]

        # Reshape to (Nx, Ny) — column-major means it maps to (i, j) correctly.
        rho = reshape(rho_flat, Nx, Ny)
        mx  = reshape(mx_flat,  Nx, Ny)

        ux = mx ./ rho

        # Transpose for heatmap display (x horizontal, y vertical)
        data["rho_2d"] = rho'
        data["ux_2d"]  = ux'

        # 2D hyperbolic system has 3 variables (ρ, m_x, m_y)
        if nvars >= 3
            my_flat = u_full[3, :]
            my = reshape(my_flat, Nx, Ny)
            uy = my ./ rho
            data["uy_2d"] = uy'
        end

        # EPB systems may have 4+ variables (ρ, m_x, m_y, φ)
        if nvars >= 4
            phi_flat = u_full[4, :]
            phi = reshape(phi_flat, Nx, Ny)
            data["phi_2d"] = phi'
        end
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

    # Determine number of variables (assume consistent across files)
    nvars = init["nvars"]

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
    # Layout: rows (ρ, u_x, u_y, φ) × (1 + nfinal) columns
    # ------------------------------------------------------------------
    total_cols = 1 + nfinal

    # Determine number of rows based on available variables
    total_rows = 2  # Default: ρ and u only
    if nvars >= 3
        total_rows = 3  # Add u_y
    end
    if nvars >= 4
        total_rows = 4  # Add φ
    end

    # Shared axis limits for side-by-side comparability
    # Collect all relevant data for limit computation
    rho_arrays = [init["rho_2d"]]
    ux_arrays = [init["ux_2d"]]
    uy_arrays = haskey(init, "uy_2d") ? [init["uy_2d"]] : []
    phi_arrays = haskey(init, "phi_2d") ? [init["phi_2d"]] : []

    for f in final_files
        push!(rho_arrays, f["rho_2d"])
        push!(ux_arrays, f["ux_2d"])
        if haskey(f, "uy_2d")
            push!(uy_arrays, f["uy_2d"])
        end
        if haskey(f, "phi_2d")
            push!(phi_arrays, f["phi_2d"])
        end
    end

    # Compute limits safely
    rho_min = minimum(minimum(r) for r in rho_arrays)
    rho_max = maximum(maximum(r) for r in rho_arrays)
    ux_min = minimum(minimum(u) for u in ux_arrays)
    ux_max = maximum(maximum(u) for u in ux_arrays)

    uy_min = isempty(uy_arrays) ? 0.0 : minimum(minimum(u) for u in uy_arrays)
    uy_max = isempty(uy_arrays) ? 0.0 : maximum(maximum(u) for u in uy_arrays)

    phi_min = isempty(phi_arrays) ? 0.0 : minimum(minimum(p) for p in phi_arrays)
    phi_max = isempty(phi_arrays) ? 0.0 : maximum(maximum(p) for p in phi_arrays)

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
             xlabel = "x", ylabel = "uₓ", title = "Initial",
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

    # Row 3: Velocity Y (u_y) - for 2D systems
    if total_rows >= 3
        heatmap!(fig, init["x"], init["y"], init["uy_2d"],
                 subplot = sp(3, 1),
                 xlabel = "x", ylabel = "uᵧ", title = "Initial",
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
    end

    # Row 4: Electric Potential (φ) - for EPB systems
    if total_rows >= 4
        heatmap!(fig, init["x"], init["y"], init["phi_2d"],
                 subplot = sp(4, 1),
                 xlabel = "x", ylabel = "φ", title = "Initial",
                 aspect_ratio = :equal,
                 clims = (phi_min, phi_max),
                 framestyle = :box)
        for (ci, f) in enumerate(final_files)
            heatmap!(fig, f["x"], f["y"], f["phi_2d"],
                     subplot = sp(4, ci + 1),
                     xlabel = "x",
                     title = col_labels[ci],
                     aspect_ratio = :equal,
                     clims = (phi_min, phi_max),
                     framestyle = :box)
        end
    end

    savefig(fig, out_path)
    println("Saved 2D comparison plot to $out_path")
end

main()