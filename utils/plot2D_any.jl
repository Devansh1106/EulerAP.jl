#!/usr/bin/env julia
"""
    plot2D_any.jl

Read an arbitrary number of 2D HDF5 solution files and plot density, u_x, and
u_y as heatmaps in a single figure. The initial state is taken from the first
file only and shown in the first column, while final states from every file
fill the remaining columns.

Layout:
    Row 1: Density (ρ)
    Row 2: Velocity X (u_x)
    Row 3: Velocity Y (u_y)
    Col 1: Initial state (from first file, labeled "Initial")
    Col 2..N: Final states from each file (labeled by t or ε)

Usage:
    julia --project=. utils/plot2D_any.jl <file1.h5> [file2.h5 ...]
"""
# using HDF5
# using Plots

function read_final_h5_2d(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]       = read(h5f, "x")
        data["y"]       = read(h5f, "y")
        data["u_final"] = read(h5f, "u_final")
        data["ncells"]  = read(h5f, "ncells")
        data["eps"]     = read(h5f, "eps")
        if haskey(h5f, "t_final")
            data["t_final"] = read(h5f, "t_final")
        end
    end

    ncells = data["ncells"]
    x      = data["x"]
    y      = data["y"]
    Nx, Ny = length(x), length(y)

    rho = data["u_final"][1:ncells]
    mx  = data["u_final"][ncells + 1:2 * ncells]
    my  = data["u_final"][2 * ncells + 1:3 * ncells]

    eps_str = "ε = $(data["eps"])"
    t_str = haskey(data, "t_final") ? "t = $(data["t_final"])" : ""

    return x, y, reshape(rho, Nx, Ny)', reshape(mx ./ rho, Nx, Ny)', reshape(my ./ rho, Nx, Ny)', eps_str, t_str
end

function read_first_h5_2d(filepath::String)
    data = Dict{String, Any}()
    h5open(filepath, "r") do h5f
        data["x"]      = read(h5f, "x")
        data["y"]      = read(h5f, "y")
        data["u_init"] = read(h5f, "u_init")
        data["ncells"] = read(h5f, "ncells")
        data["size"]   = Tuple(read(h5f, "size"))
    end

    ncells = data["ncells"]
    x      = data["x"]
    y      = data["y"]
    Nx, Ny = length(x), length(y)

    rho = data["u_init"][1:ncells]
    mx  = data["u_init"][ncells + 1:2 * ncells]
    my  = data["u_init"][2 * ncells + 1:3 * ncells]

    return x, y, reshape(rho, Nx, Ny)', reshape(mx ./ rho, Nx, Ny)', reshape(my ./ rho, Nx, Ny)', data["size"]
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia --project=. utils/plot2D_any.jl <file1.h5> [file2.h5 ...]")
        exit(1)
    end

    nfiles = length(ARGS)

    # Read initial state from the FIRST file only
    x_i, y_i, rho_i, ux_i, uy_i, mesh = read_first_h5_2d(ARGS[1])

    # Read final states from ALL files
    finals = [read_final_h5_2d(ARGS[i]) for i in 1:nfiles]

    # Build output path
    bases = [splitext(basename(ARGS[i]))[1] for i in 1:nfiles]
    out_path = joinpath("plots", "compare_all_$(join(bases, "_")).png")
    mkpath("plots")

    ncols = 1 + nfiles  # initial column + one per file
    nrows = 3           # ρ, u_x, u_y

    # Build labels for each column
    col_labels = ["Initial"]
    for (x, y, rho, ux, uy, eps_str, t_str) in finals
        # push!(col_labels, t_str)
        push!(col_labels, eps_str)
    end

    mesh_str = "$(mesh[1])x$(mesh[2])"
    plot_title = nfiles == 1 ? "N = $mesh_str" : "N = $mesh_str (first)"

    # Build all subplots
    plots = []
    for row in 1:nrows
        for col in 1:ncols
            if col == 1
                # Initial column
                if row == 1
                    p = heatmap(x_i, y_i, rho_i, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                elseif row == 2
                    p = heatmap(x_i, y_i, ux_i, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                else
                    p = heatmap(x_i, y_i, uy_i, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                end
            else
                # Final from file (col-1)
                x, y, rho, ux, uy, _, _ = finals[col - 1]
                if row == 1
                    p = heatmap(x, y, rho, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                elseif row == 2
                    p = heatmap(x, y, ux, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                else
                    p = heatmap(x, y, uy, aspect_ratio=:equal,
                                title=col_labels[col], xlabel="x", ylabel="y",
                                framestyle=:box)
                end
            end
            push!(plots, p)
        end
    end

    fig = plot(plots..., layout=(nrows, ncols), size=(350 * ncols, 350 * nrows),
               plot_title=plot_title)

    savefig(fig, out_path)
    println("Saved comparison plot to $out_path")
end

main()