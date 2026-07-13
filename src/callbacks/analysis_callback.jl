# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AnalysisCallback(; exact_solution, interval = typemax(Int))

Compute error norms against an exact solution.

By default, the analysis is performed only once at the final time.
"""
struct AnalysisCallback{F} <: AbstractCallback
    interval::Int
    exact_solution::F
end

function AnalysisCallback(;
                          exact_solution = nothing,
                          interval = typemax(Int))

    return AnalysisCallback(interval,
                            exact_solution)
end


function perform!(callback::AnalysisCallback,
                  context::CallbackContext;
                  force = false)

    stats = context.stats

    if !force && stats.iteration % callback.interval != 0
        return nothing
    end

    println()
    println("======================== Analysis ==========================")

    # ------------------------------------------------------------
    # Error norms (only if an exact solution is available)
    # ------------------------------------------------------------

    if callback.exact_solution !== nothing

        simulation = context.simulation

        result = compute_errors(
            context.solution,
            simulation.semi;
            exact_solution = callback.exact_solution,
        )

        for (variable, norms) in enumerate(result.norms)

            println("Variable ", variable)

            print_summary_line("L¹", norms.L1)
            print_summary_line("L²", norms.L2)
            print_summary_line("L∞", norms.Linf)

            println()
        end

    else

        println("No exact solution provided.")
        println("Skipping error norm computation.")
        println()

    end

    # ------------------------------------------------------------
    # Solution diagnostics
    # ------------------------------------------------------------

    println("------------------------------------------------------------")
    println("Solution diagnostics")
    println("------------------------------------------------------------")

    print_summary_line(
        "Initial mass",
        stats.initial_mass,
    )

    print_summary_line(
        "Mass before step",
        stats.mass_before,
    )

    print_summary_line(
        "Mass after Stage 1",
        stats.mass_after_stage1,
    )

    print_summary_line(
        "Mass after Stage 3",
        stats.mass_after_stage3,
    )

    print_summary_line(
        "Rel. mass error S1",
        stats.relative_mass_error_stage1,
    )

    print_summary_line(
        "Rel. mass error S3",
        stats.relative_mass_error_stage3,
    )

    print_summary_line(
        "Minimum density",
        stats.minimum_density,
    )

    print_summary_line(
        "Maximum velocity",
        stats.maximum_velocity,
    )

    println("============================================================")
    return nothing
end

function finalize!(callback::AnalysisCallback,
                   context::CallbackContext)

    perform!(callback, context; force = true)
    return nothing
end

@inline function total_mass(u, semi)

    mesh = semi.mesh
    nvars = nvariables(semi.equations)

    mass = zero(eltype(u))

    @inbounds for I in eachcell(mesh)

        cell = cell_index(I, semi)

        rho_idx = global_dof(cell, 1, nvars)

        mass += u[rho_idx]
    end
    return mass * prod(mesh.dx)
end

function minimum_density(u, semi)

    nvars = nvariables(semi.equations)

    rho_min = typemax(eltype(u))

    @inbounds for I in eachcell(semi.mesh)

        cell = cell_index(I, semi)

        rho_idx = global_dof(cell,1,nvars)

        rho_min = min(rho_min,u[rho_idx])
    end
    return rho_min
end

function maximum_velocity(u, semi)

    nvars = nvariables(semi.equations)

    vmax = zero(eltype(u))

    @inbounds for I in eachcell(semi.mesh)

        cell = cell_index(I,semi)

        rho_idx = global_dof(cell,1,nvars)
        mom_idx = global_dof(cell,2,nvars)

        rho = u[rho_idx]

        if rho > zero(rho)

            vmax = max(vmax,
                       abs(u[mom_idx]/rho))
        end
    end
    return vmax
end
end # @muladd
