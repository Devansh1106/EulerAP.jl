#!/usr/bin/env julia
"""
    plot2D_cross_section.jl

Read an arbitrary number of 2D HDF5 solution files and plot cross-sections
along the x-direction at the y-midpoint. The initial state is taken from the
first file only, while final states are plotted for every file. Each curve is
distinguished by line style and colour.

Layout:
    Left:   Density (ρ) along x at mid-y
    Right:  Velocity X (u_x) along x at mid-y

Usage:
    julia --project=. utils/plot2D_cross_section.jl <file1.h5> [file2.h5 ...]
"""
# using HDF5
# using Plots

# Line styles to cycle through
const LINE_STYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]
const LINE_COLORS = [:black, :red, :blue, :green, :orange, :purple, :brown,
                     :pink, :olive, :cyan, :magenta, :navy]

function read_final_cross_section(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]       = read(h5f, "x")
        data["y"]       = read(h5f, "y")
        data["u_final"] = read(h5f, "u_final")
        data["ncells"]  = read(h5f, "ncells")
        data["eps"]     = read(h5f, "eps")
        data["size"]    = Tuple(read(h5f, "size"))
        if haskey(h5f, "t_final")
            data["t_final"] = read(h5f, "t_final")
        end
    end

    Nx, Ny = data["size"]
    ncells = data["ncells"]
    x      = data["x"]
    y      = data["y"]

    mid_y = Ny ÷ 2 + 1   # middle y-index (1-based)
    slice_indices = (mid_y - 1) * Nx + 1 : mid_y * Nx

    rho_slice = data["u_final"][slice_indices]
    mx_slice  = data["u_final"][ncells .+ slice_indices]
    ux_slice  = mx_slice ./ rho_slice

    eps_str = "ε = $(data["eps"])"
    t_str = haskey(data, "t_final") ? "t = $(data["t_final"])" : ""

    return x, rho_slice, ux_slice, eps_str, t_str
end

function read_first_cross_section(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]      = read(h5f, "x")
        data["y"]      = read(h5f, "y")
        data["u_init"] = read(h5f, "u_init")
        data["ncells"] = read(h5f, "ncells")
        data["size"]   = Tuple(read(h5f, "size"))
        data["eps"]    = read(h5f, "eps")
    end

    Nx, Ny = data["size"]
    ncells = data["ncells"]
    x      = data["x"]

    mid_y = Ny ÷ 2 + 1
    slice_indices = (mid_y - 1) * Nx + 1 : mid_y * Nx

    rho_slice = data["u_init"][slice_indices]
    mx_slice  = data["u_init"][ncells .+ slice_indices]
    ux_slice  = mx_slice ./ rho_slice

    return x, rho_slice, ux_slice, data["size"]
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia --project=. utils/plot2D_cross_section.jl <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(ARGS)

    # Read initial state from the FIRST file only
    x_init, rho_init, ux_init, mesh = read_first_cross_section(ARGS[1])

    # Read final states from ALL files
    finals = [read_final_cross_section(ARGS[i]) for i in 1:nfiles]

    # Build output path from all basenames
    bases = [splitext(basename(ARGS[i]))[1] for i in 1:nfiles]
    out_path = joinpath("plots", "cross_section_$(join(bases, "_")).png")
    mkpath("plots")

    # Density subplot
    p1 = plot(xlabel="x", ylabel="ρ", title="Density at mid-y")
    plot!(p1, x_init, rho_init, lw=2, ls=:dash, color=:black, label="Initial")
    for (i, (x, rho, _, eps_str, t_str)) in enumerate(finals)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        label = t_str != "" ? t_str : eps_str
        plot!(p1, x, rho, lw=2, ls=ls, color=lc, label="$(label) (final)")
    end

    # Velocity subplot
    p2 = plot(xlabel="x", ylabel="u_x", title="Velocity X at mid-y")
    plot!(p2, x_init, ux_init, lw=2, ls=:dash, color=:black, label="Initial")
    for (i, (x, _, ux, eps_str, t_str)) in enumerate(finals)
        ls = LINE_STYLES[(i - 1) % length(LINE_STYLES) + 1]
        lc = LINE_COLORS[(i - 1) % length(LINE_COLORS) + 1]
        label = t_str != "" ? t_str : eps_str
        plot!(p2, x, ux, lw=2, ls=ls, color=lc, label="$(label) (final)")
    end

    mesh_str = "$(mesh[1])x$(mesh[2])"
    mesh_label = nfiles == 1 ? "N = $mesh_str" : "N = $mesh_str (first)"

    fig = plot(p1, p2, layout=(1, 2), size=(1400, 550),
               plot_title="For $mesh_label")

    savefig(fig, out_path)
    println("Saved cross-section plot to $out_path")
end

main()