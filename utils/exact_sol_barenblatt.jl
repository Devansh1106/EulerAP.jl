#!/usr/bin/env julia
"""
    exact_sol_barenblatt.jl

Generate HDF5 files containing the exact Barenblatt solution (density ρ and
momentum mx) at one or more specified times. The output files use the same
HDF5 layout as solver output in `data_new/` (with `mesh/`, `equations/`,
`metadata/` and `solution/` groups) so they can be plotted with
`utils/plot1D.jl`.

Usage:
    julia --project=. utils/exact_sol_barenblatt.jl <N> <gamma> <t1> [t2 ...]

Arguments:
    N         Number of grid cells (integer)
    gamma     Adiabatic exponent (e.g., 3.0, 2.0, 1.4)
    t1 t2 ... One or more times to evaluate the exact solution at

Example:
    julia --project=. utils/exact_sol_barenblatt.jl 100 3.0 1.4 1.6 1.8 2.0
    # Creates: data_new/exact_100_gamma=3.0_t=1.4.h5
    #          data_new/exact_100_gamma=3.0_t=1.6.h5
    #          data_new/exact_100_gamma=3.0_t=1.8.h5
    #          data_new/exact_100_gamma=3.0_t=2.0.h5

    julia --project=. utils/plot1D.jl data_new/exact_100_gamma=3.0_t=1.4.h5
"""
# using HDF5
# using Plots

const RHO_FLOOR = 1e-10

"""
    barenblatt(x, t, Γ, γ)

Evaluate the Barenblatt (porous-medium / gas dynamics) density profile.

    β = 1/(γ + 1)
    ξ = x / t^β
    ρ = t^{-β} · max(Γ - (γ-1)/(2γ(γ+1)) · ξ², 0)^{1/(γ-1)}
"""
function barenblatt(x::Real, t::Real, Γ::Real, γ::Real)
    t_eff = Float64(t)
    β = 1.0 / (γ + 1.0)
    ξ = x / (t_eff^β)

    factor = (γ - 1.0) / (2.0 * γ * (γ + 1.0))
    bracket_value = Γ - factor * (ξ^2)

    positive_part = max(bracket_value, 0.0)
    ρ = t_eff^(-β) * (positive_part^(1.0 / (γ - 1.0)))
    return ρ
end

"""
    exact_momentum(x, t, Γ, γ)

Compute exact momentum mx = ρ·u for the Barenblatt solution.
The velocity field is u = β · x / t inside the support and zero outside.
"""
function exact_momentum(x::Real, t::Real, Γ::Real, γ::Real)
    ρ = barenblatt(x, t, Γ, γ)
    β = 1.0 / (γ + 1.0)

    if ρ > 0.0
        u  = β * x / t
        mx = ρ * u
    else
        mx = 0.0
    end
    return mx
end

function main()
    if length(ARGS) < 3
        println(stderr, "Usage: julia --project=. utils/exact_sol_barenblatt.jl <N> <gamma> <t1> [t2 ...]")
        println(stderr, "  N       Number of grid cells (e.g., 100)")
        println(stderr, "  gamma   Adiabatic exponent (e.g., 3.0)")
        println(stderr, "  t1 ...  Times to evaluate exact solution at")
        exit(1)
    end

    N     = parse(Int, ARGS[1])
    gamma = parse(Float64, ARGS[2])
    times = [parse(Float64, ARGS[i]) for i in 3:length(ARGS)]

    # Domain: [-6, 6] matching the Barenblatt problem
    x_min, x_max = -6.0, 6.0
    dx = (x_max - x_min) / N
    x = range(x_min + dx / 2, x_max - dx / 2; length = N)

    mkpath("data_new")

    for t in times
        u = zeros(2, N)  # shape (nvars=2, ndofs=N) — same as solver output
        rho_view = @view u[1, :]
        mx_view  = @view u[2, :]

        for i in 1:N
            ρ_val = barenblatt(x[i], t, 1.0, gamma)
            ρ_val = max(ρ_val, RHO_FLOOR)
            rho_view[i] = ρ_val
            mx_view[i]  = exact_momentum(x[i], t, 1.0, gamma)
        end

        fname = "exact_$(N)_gamma=$(gamma)_t=$(t).h5"
        fpath = joinpath("data_new", fname)

        h5open(fpath, "w") do file
            # --------------------------------------------------
            # Convenience top-level scalars
            # --------------------------------------------------
            file["eps"] = "exact sol"
            file["ncells"] = string(N)

            # --------------------------------------------------
            # Mesh
            # --------------------------------------------------
            mesh_group = create_group(file, "mesh")
            mesh_group["cells_per_dimension"] = [N]
            mesh_group["coordinates_min"]     = [x_min]
            mesh_group["coordinates_max"]     = [x_max]
            mesh_group["dx"]                  = [dx]
            mesh_group["periodicity"]         = [0]   # false (extrapolate BC)

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------
            metadata_group = create_group(file, "metadata")
            metadata_group["time"]        = t
            metadata_group["ndims"]       = 1
            metadata_group["nvariables"]  = 2

            # --------------------------------------------------
            # Equation parameters
            # --------------------------------------------------
            equations_group = create_group(file, "equations")
            equations_group["gamma"]   = gamma
            equations_group["epsilon"] = "exact sol"

            # --------------------------------------------------
            # Solution
            # --------------------------------------------------
            solution_group = create_group(file, "solution")
            solution_group["u"] = u
        end

        println("Exact solution saved to ", fpath)
    end
end

main()