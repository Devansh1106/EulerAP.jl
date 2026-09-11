#!/usr/bin/env julia
"""
    plot1D.jl

Read one or more 1D HDF5 solution files (new format with top-level `eps`
and `ncells` scalars) and plot density, velocity, and electric potential
profiles on the same figure.

The first file is treated as the **initial condition** (plotted with a solid
black line).  Subsequent files are **final solutions**, distinguished first by
line style and colour and only then by markers: the first final is dotted, the
second dashed, and from the third on a marker is added on top of the remaining
styles, since by then the line styles alone no longer separate the curves.

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
using Printf

# Marker types / colors to cycle through
const MARKER_TYPES = [:circle, :diamond, :square, :x, :cross, :plus,
                      :hexagon, :pentagon, :dtriangle, :utriangle]
const LINE_COLORS = [:black, :red, :blue, :green, :orange, :purple, :brown,
                     :pink, :olive, :cyan, :magenta, :navy]

# Line styles for the *final* solutions. The initial condition takes :solid, so
# these start at the next distinct style — dotted, then dashed.
const FINAL_LINE_STYLES = [:dot, :dash, :dashdot, :dashdotdot]

# How many final curves are drawn by line style alone, before markers kick in.
const N_UNMARKED_FINALS = 2

# Roughly how many markers to put on a marked curve, regardless of mesh size.
const N_MARKERS = 20

# Line width of a final curve. `:dot` renders as round dots whose diameter *is*
# the line width, so the dotted curve gets its own (larger) value — at the
# shared width its dots are too small to read against the solid initial line.
const FINAL_LINE_WIDTH = 2
const DOTTED_LINE_WIDTH = 5

"""
    plot_final!(p, x, y, i, label)

Draw the `i`-th final solution onto subplot `p`.

The initial condition is a solid line, so the finals continue the sequence of
line styles: `i = 1` is dotted, `i = 2` is dashed, and both are drawn without
markers — the style alone tells them apart. From `i = 3` on the remaining
styles are reused with a marker added.

The markers are a *separate* subsampled scatter series rather than a `marker`
attribute on the line, because Plots.jl has no marker-thinning attribute
(`markevery` is matplotlib's and is silently ignored here), and a marker on
every cell is an unreadable blob on a fine mesh. The scatter carries no legend
entry — the line style and colour already identify the curve.
"""
function plot_final!(p, x, y, i, label)
    ls = FINAL_LINE_STYLES[(i - 1) % length(FINAL_LINE_STYLES) + 1]
    lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]

    lw = ls === :dot ? DOTTED_LINE_WIDTH : FINAL_LINE_WIDTH

    plot!(p, x, y, lw = lw, ls = ls, color = lc, label = label)

    if i > N_UNMARKED_FINALS
        mk  = MARKER_TYPES[(i - N_UNMARKED_FINALS - 1) % length(MARKER_TYPES) + 1]
        idx = 1:max(1, length(x) ÷ N_MARKERS):length(x)
        scatter!(p, x[idx], y[idx],
                 marker = mk, markersize = 3, color = lc,
                 markerstrokecolor = lc, markerstrokewidth = 1,
                 label = "")
    end

    return p
end

"""
    read_solution_1d(filepath::String) -> Dict

Read a 1D HDF5 solution file and return a dictionary containing all available
variables. Supports 2 variables (hyperbolic: ρ, m) or 3 variables
(Euler-Poisson-Boltzmann: ρ, m, φ).
"""
function read_solution_1d(filepath::String)
    data = Dict{String, Any}()

    h5open(filepath, "r") do f
        # ---- convenience top-level attributes ----
        if haskey(f, "eps")
            eps_str = read(f, "eps")
        else
            eps_str = read(f, "equations/lambda")
        end
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

        nvars = read(f, "metadata/nvariables")
        data["nvars"] = nvars

        # ---- solution ----
        # u has shape (nvars, ndofs) in interleaved layout
        u = read(f, "solution/u")
        rho = u[1, :]
        mx  = u[2, :]

        data["rho"] = rho
        data["mx"] = mx
        data["ux"] = mx ./ rho

        # Electric potential (EPB systems have 3+ variables)
        if nvars >= 3
            phi = u[3, :]
            data["phi"] = phi
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
        println(stderr, "Usage: julia --project=. utils/plot1D.jl [--output <file.png>] <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(input_files)

    # First file is the initial condition
    init = read_solution_1d(input_files[1])

    # Remaining files are final solutions
    final_files = [read_solution_1d(input_files[i]) for i in 2:nfiles]
    nfinal = length(final_files)

    # Determine number of variables (assume consistent across files)
    nvars = init["nvars"]

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
            legend_labels = ["ε = $(@sprintf("%.6f", e))" for e in epsilons]
            legend_param  = :eps
        elseif length(mesh_unique) > 1
            legend_labels = ["N = $(m)" for m in mesh_strs]
            legend_param  = :mesh
        elseif length(t_unique) > 1
            legend_labels = ["t = $(@sprintf("%.6f", t))" for t in ts]
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

    # Initial condition from first file (solid black)
    plot!(p1, init["x"], init["rho"],
          lw = 2, ls = :solid, color = :black,
          label = "Initial")

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        plot_final!(p1, f["x"], f["rho"], i, legend_labels[i])
    end

    # ------------------------------------------------------------------
    # Velocity subplot
    # ------------------------------------------------------------------
    p2 = plot(xlabel = "x", ylabel = "uₓ", title = "Velocity")

    # Initial condition from first file (solid black)
    plot!(p2, init["x"], init["ux"],
          lw = 2, ls = :solid, color = :black,
          label = "Initial")

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        plot_final!(p2, f["x"], f["ux"], i, legend_labels[i])
    end

    # ------------------------------------------------------------------
    # Momentum subplot (debug)
    # ------------------------------------------------------------------
    # p_mom = plot(xlabel = "x", ylabel = "mₓ", title = "Momentum")
    #
    # # Initial condition from first file (solid black)
    # plot!(p_mom, init["x"], init["mx"],
    #       lw = 2, ls = :solid, color = :black,
    #       label = "Initial")
    #
    # # Final states from remaining files
    # for (i, f) in enumerate(final_files)
    #     plot_final!(p_mom, f["x"], f["mx"], i, legend_labels[i])
    # end

    # ------------------------------------------------------------------
    # Electric potential subplot (if 3+ variables)
    # ------------------------------------------------------------------
    # nrows = nvars >= 3 ? 4 : 3
    nrows = nvars >= 3 ? 3 : 2
    # plots = [p1, p2, p_mom]
    plots = [p1, p2]

    if nvars >= 3
        p3 = plot(xlabel = "x", ylabel = "φ", title = "Electric Potential")

        # Initial condition from first file (solid black)
        plot!(p3, init["x"], init["phi"],
              lw = 2, ls = :solid, color = :black,
              label = "Initial")

        # Final states from remaining files
        for (i, f) in enumerate(final_files)
            if haskey(f, "phi")
                plot_final!(p3, f["x"], f["phi"], i, legend_labels[i])
            end
        end

        push!(plots, p3)
    end

    fig = plot(plots..., layout = (1, nrows), size = (500 * nrows, 550),
               plot_title = plot_title)

    savefig(fig, out_path)
    println("Saved comparison plot to $out_path")
end

main()