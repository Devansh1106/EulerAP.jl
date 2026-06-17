#!/usr/bin/env julia
"""
    plot1D_any.jl

Read an arbitrary number of 1D HDF5 solution files and plot all density &
velocity profiles on the same side-by-side figure.
The initial state is taken from the first file only, while final states are
plotted for every file. Each curve is distinguished by line style and colour.

By default the legend shows ε values. For files that have a `t_final` attribute
saved in the HDF5 (e.g., sol1D_100_1.0e-4_t=1.4.h5), you can also display the
final time in the legend by swapping which label is used in the plotting loops
(see commented alternative below).

Usage:
    julia --project=. utils/plot1D_any.jl <file1.h5> [file2.h5 ...]
"""
# using HDF5
# using Plots

# Line styles to cycle through
const LINE_STYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]
const LINE_COLORS = [:black, :red, :blue, :green, :orange, :purple, :brown,
                     :pink, :olive, :cyan, :magenta, :navy]

function read_final_h5_1d(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]       = read(h5f, "x")
        data["u_final"] = read(h5f, "u_final")
        data["ncells"]  = read(h5f, "ncells")
        data["eps"]     = read(h5f, "eps")
        # t_final is present only for files saved with t_final keyword
        if haskey(h5f, "t_final")
            data["t_final"] = read(h5f, "t_final")
        end
    end

    ncells = data["ncells"]
    x      = data["x"]

    rho = data["u_final"][1:ncells]
    mx  = data["u_final"][ncells + 1:2 * ncells]
    ux  = mx ./ rho

    # Primary label: ε value (always available)
    eps_str = "ε = $(data["eps"])"
    # Alternative label: final time (if saved in the file)
    t_str = haskey(data, "t_final") ? "t = $(data["t_final"])" : ""

    return x, rho, ux, eps_str, t_str
end

function read_first_h5_1d(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]      = read(h5f, "x")
        data["u_init"] = read(h5f, "u_init")
        data["ncells"] = read(h5f, "ncells")
        data["size"]   = Tuple(read(h5f, "size"))
        data["eps"]    = read(h5f, "eps")
    end

    ncells = data["ncells"]
    x      = data["x"]

    rho = data["u_init"][1:ncells]
    mx  = data["u_init"][ncells + 1:2 * ncells]
    ux  = mx ./ rho

    return x, rho, ux, data["size"][1], data["eps"]
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia --project=. utils/plot1D_any.jl <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(ARGS)

    # Read initial state from the FIRST file only
    x_init, rho_init, ux_init, mesh, eps_val = read_first_h5_1d(ARGS[1])

    # Read final states from ALL files
    finals = [read_final_h5_1d(ARGS[i]) for i in 1:nfiles]

    # Build output path from all basenames
    bases = [splitext(basename(ARGS[i]))[1] for i in 1:nfiles]
    out_path = joinpath("plots", "compare_all_$(join(bases, "_")).png")
    mkpath("plots")

    # Density subplot
    p1 = plot(xlabel="x", ylabel="ρ", title="Density")
    # Initial from first file (dashed black)
    plot!(p1, x_init, rho_init, lw=2, ls=:dash, color=:black, label="Initial")

    # Finals from all files
    # -- Default: use ε-based labels --
    for (i, (x, rho, _, eps_str, _)) in enumerate(finals)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        plot!(p1, x, rho, lw=2, ls=ls, color=lc, label="$(eps_str) (final)")
    end

    # -- Alternative: use t-based labels (comment out the loop above, uncomment below)
    # for (i, (x, rho, _, _, t_str)) in enumerate(finals)
    #     ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
    #     lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
    #     plot!(p1, x, rho, lw=2, ls=ls, color=lc, label="$(t_str) (final)")
    # end

    # Velocity subplot
    p2 = plot(xlabel="x", ylabel="u_x", title="Velocity")
    # Initial from first file (dashed black)
    plot!(p2, x_init, ux_init, lw=2, ls=:dash, color=:black, label="Initial")

    # Finals from all files
    # -- Default: use ε-based labels --
    for (i, (x, _, ux, eps_str, _)) in enumerate(finals)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        plot!(p2, x, ux, lw=2, ls=ls, color=lc, label="$(eps_str) (final)")
    end

    # -- Alternative: use t-based labels (comment out the loop above, uncomment below)
    # for (i, (x, _, ux, _, t_str)) in enumerate(finals)
    #     ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
    #     lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
    #     plot!(p2, x, ux, lw=2, ls=ls, color=lc, label="$(t_str) (final)")
    # end

    mesh_str = nfiles == 1 ? "N = $mesh" : "N = $mesh (first)"

    fig = plot(p1, p2, layout=(1, 2), size=(1400, 550),
               plot_title="For $mesh_str")

    savefig(fig, out_path)
    println("Saved comparison plot to $out_path")
end

main()