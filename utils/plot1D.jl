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
  - `--labels "a,b,c"` overrides the result, one entry per input file in the
    order given, the first file included. An empty entry keeps that curve's
    automatic label, so `--labels ",1st order,2nd order"` renames only the
    finals. Needed whenever the files differ by something the automatic logic
    cannot see — comparing a first- against a second-order run at the same
    eps/mesh/t, for instance, where every automatic label would come out as
    "final i".

Output is saved as:
    plots_new/compare_<basename1>_<basename2>_....png

Zooming:
  - `--zoom` restricts every subplot to the most prominent feature (the peak of
    the density, located automatically), which is usually the only place the
    curves still separate on a fine mesh.
  - `--xlim a,b` does the same with an explicit window.
  - `--ylim a,b` sets the y range by hand, for a close-up of the apex itself.
  With `--zoom`/`--xlim` alone the y-axis is rescaled to what is visible; Plots.jl
  does not do that on its own, and without it a zoomed plot keeps the full-domain
  y-range and the feature flattens out.

Usage:
    julia --project=. utils/plot1D.jl [--output <file.png>] [--zoom | --xlim a,b] [--ylim a,b] [--labels "a,b,..."] <initial.h5> [final1.h5 final2.h5 ...]

Examples:
    # Initial condition + two final solutions with different epsilon
    julia --project=. utils/plot1D.jl initial.h5 sol_eps1.0.h5 sol_eps0.1.h5

    # Initial condition + two final solutions with different mesh sizes
    julia --project=. utils/plot1D.jl initial.h5 sol_N100.h5 sol_N200.h5

    # Specify custom output path
    julia --project=. utils/plot1D.jl --output myplot.png initial.h5 sol_eps1.0.h5

    # Plot only the initial condition (no final files)
    julia --project=. utils/plot1D.jl initial.h5

    # Zoom onto the peak, where a fine-mesh comparison actually differs
    julia --project=. utils/plot1D.jl --zoom initial.h5 sol.h5

    # Explicit zoom window
    julia --project=. utils/plot1D.jl --xlim 20,30 initial.h5 sol.h5

    # Close-up of the apex of a peak, where the curves separate by a few percent
    julia --project=. utils/plot1D.jl --xlim 23,27 --ylim 1.75,1.95 initial.h5 sol.h5

    # Explicit legend entries: reference curve plus two schemes on one mesh
    julia --project=. utils/plot1D.jl --labels "reference N=1000,1st order N=100,2nd order N=100" \
        ref.h5 first.h5 second.h5
"""
# using HDF5
# using Plots
using Printf

# Marker types to cycle through
const MARKER_TYPES = [:circle, :diamond, :square, :x, :cross, :plus,
                      :hexagon, :pentagon, :dtriangle, :utriangle]

# Colour reserved for the initial condition.
const INITIAL_LINE_COLOR = :black

# Default legend entry for the first file. Only a default: the first file is not
# always an initial condition — it is just as often a fine-mesh reference that a
# coarse run is being compared against — so `--labels` can rename it.
const INITIAL_LABEL = "Initial"

# Colours for the *final* solutions. Black is deliberately absent: it belongs to
# the initial condition, and including it made the first final curve come out in
# the same colour as the initial, leaving line style as the only thing telling
# the two apart. Same reasoning as `FINAL_LINE_STYLES` excluding `:solid`.
const FINAL_LINE_COLORS = [:red, :blue, :green, :orange, :purple, :brown,
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

# `--zoom` window width, as a multiple of the feature's half-prominence width.
# 1.5 frames the peak with a little background on each side. Larger values pull
# in more flat background and shrink the feature; for a close-up of the apex
# itself — where a fine-mesh first/second-order comparison separates by a few
# percent — `--xlim`/`--ylim` are the right tools, since no automatic rule knows
# how tight you want it.
const PEAK_WINDOW_FACTOR = 1.5

"""
    window_ylims(series, xlo, xhi; pad = 0.05) -> (ylo, yhi) or nothing

y-range spanned by `series` — a vector of `(x, y)` pairs — within `[xlo, xhi]`,
padded by `pad` of that span.

Plots.jl does **not** rescale the y-axis when `xlims` is narrowed, so a zoom
without this keeps the full-domain y-range and the feature being zoomed into
collapses to a flat line. Returns `nothing` if no sample falls in the window,
in which case the caller should leave the axis alone.
"""
function window_ylims(series, xlo, xhi; pad = 0.05)
    ylo, yhi = Inf, -Inf
    for (x, y) in series
        for k in eachindex(x)
            if xlo <= x[k] <= xhi && isfinite(y[k])
                ylo = min(ylo, y[k])
                yhi = max(yhi, y[k])
            end
        end
    end
    isfinite(ylo) || return nothing
    span = yhi - ylo
    span == 0 && (span = max(abs(yhi), one(yhi)))
    return (ylo - pad * span, yhi + pad * span)
end

"""
    peak_window(series; width_factor = PEAK_WINDOW_FACTOR) -> (xlo, xhi) or nothing

Window around the most prominent feature across `series`.

The background level is taken as the mean of each curve, which is right for a
localised feature on a flat background (a soliton, a bump) — the case `--zoom`
is for. The feature's width is measured at half its maximum deviation from that
background, and the window is `width_factor` times that width, centred on the
extremum. The union over all series is returned, so a curve whose peak has
shifted is still inside the frame.
"""
function peak_window(series; width_factor = PEAK_WINDOW_FACTOR)
    xlo, xhi = Inf, -Inf
    xmin, xmax = Inf, -Inf
    for (x, y) in series
        length(y) < 3 && continue
        xmin = min(xmin, minimum(x))
        xmax = max(xmax, maximum(x))

        base = sum(y) / length(y)
        dev  = abs.(y .- base)
        m, k = findmax(dev)
        m == 0 && continue

        # Nearest sample on each side where the deviation has fallen to half.
        l = k
        while l > 1 && dev[l] > 0.5 * m
            l -= 1
        end
        r = k
        while r < length(dev) && dev[r] > 0.5 * m
            r += 1
        end

        half = max(x[r] - x[l], eps(float(one(eltype(x)))))
        xlo  = min(xlo, x[k] - width_factor * half / 2)
        xhi  = max(xhi, x[k] + width_factor * half / 2)
    end
    isfinite(xlo) || return nothing
    # Clamp to the data: an overhanging window is just dead margin.
    return (max(xlo, xmin), min(xhi, xmax))
end

"""
    apply_window!(p, series, xlim; ylim = nothing)

Restrict subplot `p` to `xlim` and rescale its y-axis to the data visible there.
An explicit `ylim` overrides that rescaling — use it to close in on the apex of
a peak, where the automatic range (which spans the whole visible curve) is still
too coarse to separate the curves. A `nothing` `xlim` with no `ylim` leaves the
subplot untouched.
"""
function apply_window!(p, series, xlim; ylim = nothing)
    xlim === nothing && ylim === nothing && return p
    xlim === nothing || xlims!(p, xlim)
    if ylim !== nothing
        ylims!(p, ylim)
    elseif xlim !== nothing
        yl = window_ylims(series, xlim[1], xlim[2])
        yl === nothing || ylims!(p, yl)
    end
    return p
end

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
    lc = FINAL_LINE_COLORS[(i - 1) % length(FINAL_LINE_COLORS) + 1]

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
    xlim        = nothing    # explicit window from --xlim
    ylim        = nothing    # explicit y range from --ylim
    auto_zoom   = false      # --zoom: locate the window from the data
    labels      = nothing    # explicit legend entries from --labels
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--output" || ARGS[i] == "-o"
            i += 1
            if i > length(ARGS)
                println(stderr, "Error: --output requires a filename argument")
                exit(1)
            end
            output_file = ARGS[i]
        elseif ARGS[i] == "--zoom"
            auto_zoom = true
        elseif ARGS[i] == "--labels"
            i += 1
            if i > length(ARGS)
                println(stderr, "Error: --labels requires a comma-separated list, e.g. --labels \"1st order,2nd order\"")
                exit(1)
            end
            labels = strip.(split(ARGS[i], ','))
        elseif ARGS[i] == "--xlim" || ARGS[i] == "--ylim"
            flag = ARGS[i]
            i += 1
            if i > length(ARGS)
                println(stderr, "Error: $flag requires an argument of the form a,b")
                exit(1)
            end
            parts = split(ARGS[i], ',')
            if length(parts) != 2
                println(stderr, "Error: $flag expects exactly two comma-separated numbers, got \"$(ARGS[i])\"")
                exit(1)
            end
            lo = tryparse(Float64, strip(parts[1]))
            hi = tryparse(Float64, strip(parts[2]))
            if lo === nothing || hi === nothing || !(lo < hi)
                println(stderr, "Error: $flag needs two numbers with a < b, got \"$(ARGS[i])\"")
                exit(1)
            end
            if flag == "--xlim"
                xlim = (lo, hi)
            else
                ylim = (lo, hi)
            end
        else
            push!(input_files, ARGS[i])
        end
        i += 1
    end

    if length(input_files) < 1
        println(stderr, "Usage: julia --project=. utils/plot1D.jl [--output <file.png>] [--zoom | --xlim a,b] <file1.h5> [file2.h5 ...]")
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
    # Apply --labels over the automatic legend entries
    # ------------------------------------------------------------------
    # One entry per input file, first file included, in the order given. An
    # empty entry falls through to the automatic label, so a leading comma
    # renames only the finals.
    init_label = INITIAL_LABEL

    if labels !== nothing
        if length(labels) > nfiles
            println(stderr, "Warning: --labels has $(length(labels)) entries for $(nfiles) file(s); ignoring the extras")
        end
        if !isempty(labels) && !isempty(labels[1])
            init_label = labels[1]
        end
        for k in 2:min(length(labels), nfiles)
            isempty(labels[k]) || (legend_labels[k - 1] = labels[k])
        end
    end

    # ------------------------------------------------------------------
    # Build output path
    # ------------------------------------------------------------------
    if output_file !== nothing
        out_path = output_file
        mkpath(dirname(out_path))
    else
        bases = [splitext(basename(f))[1] for f in input_files]
        # Suffix a zoomed plot so it sits beside the full-domain one instead of
        # replacing it — both are usually wanted together.
        suffix = (auto_zoom || xlim !== nothing || ylim !== nothing) ? "_zoom" : ""
        out_path = joinpath("plots_new", "compare_$(join(bases, "_"))$(suffix).png")
        mkpath("plots_new")
    end

    # ------------------------------------------------------------------
    # Density subplot
    # ------------------------------------------------------------------
    p1 = plot(xlabel = "x", ylabel = "ρ", title = "Density")

    # Every curve is also kept as a plain (x, y) pair: the zoom needs the data
    # back to rescale the y-axis, and Plots.jl subplots are not worth digging
    # into for that.
    rho_series = [(init["x"], init["rho"])]

    # Initial condition from first file (solid black)
    plot!(p1, init["x"], init["rho"],
          lw = 2, ls = :solid, color = INITIAL_LINE_COLOR,
          label = init_label)

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        plot_final!(p1, f["x"], f["rho"], i, legend_labels[i])
        push!(rho_series, (f["x"], f["rho"]))
    end

    # ------------------------------------------------------------------
    # Velocity subplot
    # ------------------------------------------------------------------
    p2 = plot(xlabel = "x", ylabel = "uₓ", title = "Velocity")

    ux_series = [(init["x"], init["ux"])]

    # Initial condition from first file (solid black)
    plot!(p2, init["x"], init["ux"],
          lw = 2, ls = :solid, color = INITIAL_LINE_COLOR,
          label = init_label)

    # Final states from remaining files
    for (i, f) in enumerate(final_files)
        plot_final!(p2, f["x"], f["ux"], i, legend_labels[i])
        push!(ux_series, (f["x"], f["ux"]))
    end

    # ------------------------------------------------------------------
    # Momentum subplot (debug)
    # ------------------------------------------------------------------
    # p_mom = plot(xlabel = "x", ylabel = "mₓ", title = "Momentum")
    #
    # # Initial condition from first file (solid black)
    # plot!(p_mom, init["x"], init["mx"],
    #       lw = 2, ls = :solid, color = INITIAL_LINE_COLOR,
    #       label = init_label)
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
    all_series = [rho_series, ux_series]

    if nvars >= 3
        p3 = plot(xlabel = "x", ylabel = "φ", title = "Electric Potential")

        phi_series = [(init["x"], init["phi"])]

        # Initial condition from first file (solid black)
        plot!(p3, init["x"], init["phi"],
              lw = 2, ls = :solid, color = INITIAL_LINE_COLOR,
              label = init_label)

        # Final states from remaining files
        for (i, f) in enumerate(final_files)
            if haskey(f, "phi")
                plot_final!(p3, f["x"], f["phi"], i, legend_labels[i])
                push!(phi_series, (f["x"], f["phi"]))
            end
        end

        push!(plots, p3)
        push!(all_series, phi_series)
    end

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    # The window is located from the *density* and then applied to every
    # subplot, so the three panels stay on a common x-axis and can be read
    # against each other. An explicit --xlim always wins over --zoom.
    if xlim === nothing && auto_zoom
        xlim = peak_window(rho_series)
        if xlim === nothing
            println(stderr, "Warning: --zoom found no feature to zoom into; plotting the full domain")
        end
    end

    if xlim !== nothing || ylim !== nothing
        for (p, series) in zip(plots, all_series)
            apply_window!(p, series, xlim; ylim = ylim)
        end
        xlim === nothing || @printf("Zoom window: x in [%.6g, %.6g]\n", xlim[1], xlim[2])
        ylim === nothing || @printf("Zoom window: y in [%.6g, %.6g]\n", ylim[1], ylim[2])
    end

    fig = plot(plots..., layout = (1, nrows), size = (500 * nrows, 550),
               plot_title = plot_title)

    savefig(fig, out_path)
    println("Saved comparison plot to $out_path")
end

main()