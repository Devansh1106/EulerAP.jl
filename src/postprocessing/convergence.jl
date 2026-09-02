# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    experimental_order(error_old, error_new)

Compute the experimental order of convergence.
"""
@inline function experimental_order(error_old,
                                    error_new)

    return log2(error_old / error_new)
end


"""
    convergence_table(cells,
                      results;
                      variable = 1,
                      eoc_label = "EOC",
                      title = nothing)

Print a convergence table.

`eoc_label` names the order column and `title` adds a line above the header —
both exist so the self-referenced pathway can reuse this printer while
remaining visibly distinct in the output (see `self_convergence_table`). The
defaults reproduce the exact-solution table unchanged.
"""
function convergence_table(cells,
                           results;
                           variable = 1,
                           eoc_label = "EOC",
                           title = nothing)

    println("---------------------------------------------------------------------")
    if title !== nothing
        println(title)
        println("---------------------------------------------------------------------")
    end
    @printf "  Cells      L1 Error        L2 Error       Linf Error   %9s\n" eoc_label
    println("---------------------------------------------------------------------")

    previous = nothing

    for (i, N) in enumerate(cells)

        norms = results[i].norms[variable]

        if previous === nothing
            @printf "%-8d  %14.6e  %14.6e  %14.6e  %10s\n" N norms.L1 norms.L2 norms.Linf "-"
        else
            eoc = experimental_order(previous.L2, norms.L2)
            @printf "%-8d  %14.6e  %14.6e  %14.6e  %10.3f\n" N norms.L1 norms.L2 norms.Linf eoc
        end

        previous = norms

    end

    println("---------------------------------------------------------------------")

    return nothing
end

"""
    convergence_table(cells, results, names)

Print one convergence table per variable, labelled with `names` (e.g. from
`variable_names(semi)`). This is the standard way convergence is reported in
the literature (see e.g. Tables 1-3 of Degond, Liu, Savelief & Vignal,
J. Sci. Comput. 2012): each conserved variable gets its own error/EOC
column, since different variables (here e.g. density vs. potential) need not
converge at the same observed rate.
"""
function convergence_table(cells,
                           results,
                           names)

    for (variable, name) in enumerate(names)
        println()
        println("Variable: ", name)
        convergence_table(cells, results; variable = variable)
    end

    return nothing
end

"""
    convergence_test(semi_builder, grid_sizes, tspan, integrator;
                     exact_solution, dt = nothing,
                     abstol = 1e-8, reltol = 1e-8, limiter = minmod)

Run a convergence test by solving at multiple grid resolutions.

- `semi_builder`: a function `N -> semidiscretization` that creates a problem for grid size N
- `grid_sizes`: array of grid sizes (e.g., [100, 200, 400]). Successive entries
  must differ by a factor of two — `experimental_order` assumes that refinement
  ratio.
- `tspan`, `integrator`: passed to `solve()`
- `exact_solution`: passed to `compute_errors()`, called as `exact_solution(x, t, semi)`
  regardless of `semi`'s type — see `compute_errors`
- `limiter`: forwarded to `solve()`; use [`nolimiter`](@ref) to measure the
  formal order of the second-order scheme on a smooth solution, since the
  default [`minmod`](@ref) clips slopes at smooth extrema and drags the
  observed order back towards 1. Ignored by integrators that reconstruct
  nothing (`ImplicitEulerCustom`, `FirstOrderThreeStagesIMEX`).

Prints one convergence table per variable (see `convergence_table`), labelled
via `variable_names(semi)`.
"""
function convergence_test(semi_builder, grid_sizes, tspan, integrator;
                          exact_solution, dt = nothing,
                          abstol = 1e-8, reltol = 1e-8,
                          limiter = minmod)

    results = AnalysisResult[]
    cells = Int[]

    semi = nothing
    for N in grid_sizes
        semi = semi_builder(N)
        dt_actual = dt === nothing ? minimum(semi.mesh.dx) : dt

        sol = solve(semi, tspan, integrator;
                    dt = dt_actual, abstol = abstol, reltol = reltol,
                    limiter = limiter)

        result = compute_errors(sol, semi; exact_solution = exact_solution)
        push!(results, result)
        push!(cells, ndofs(semi.mesh))
    end

    convergence_table(cells, results, variable_names(semi))
    return results
end

end # @muladd
