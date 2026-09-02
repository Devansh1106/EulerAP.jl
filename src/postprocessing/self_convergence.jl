# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# ============================================================================
# Self-referenced (Cauchy) convergence
# ============================================================================
#
# An alternative to `convergence_test`/`compute_errors` for test cases with no
# analytic solution. Instead of comparing the numerical solution against an
# exact one, each grid is compared against the *next finer grid*, coarsened
# back onto it by cell averaging:
#
#     e_N = || u_N - R(u_{2N}) ||
#
# where R is the 2:1 restriction operator. If the scheme converges at order k,
# u_h = u + C hᵏ, then
#
#     u_h - R(u_{h/2}) = C hᵏ (1 - 2⁻ᵏ) + …
#
# so e_N still scales as hᵏ: the constant changes, the observed order does not.
# `experimental_order`'s log2 is exact here rather than an assumption, since
# self-referencing structurally requires factor-2 refinement.
#
# Everything is deliberately kept separate from the exact-solution pathway —
# separate entry point (`self_convergence_test`, which takes no
# `exact_solution`), separate result type (`SelfConvergenceResult`) and a
# separate table label ("Self-EOC") — so the two error notions cannot be mixed
# up at a call site or in a saved log.
# ============================================================================

"""
    SelfConvergenceResult

Self-referenced error norms for one grid pair, one `ErrorNorms` per variable.

Deliberately a distinct type from [`AnalysisResult`](@ref): a self-referenced
error measures the difference between two numerical solutions, not the distance
to an exact solution, and the two must not be printed by the same table or
compared against each other.

`cells_coarse`/`cells_fine` record which pair produced it (`cells_fine` is
always `2^ndims * cells_coarse`).
"""
struct SelfConvergenceResult{T}
    norms::Vector{ErrorNorms{T}}
    cells_coarse::Int
    cells_fine::Int
end

# ----------------------------------------------------------------------------
# Restriction (fine -> coarse), generic in the number of dimensions
# ----------------------------------------------------------------------------

# The 2^NDIMS fine cells covering coarse cell `J`: fine indices 2J-1 and 2J in
# every dimension. Written dimension-agnostically so 2D works unchanged.
@inline function fine_children(J::CartesianIndex{NDIMS}) where {NDIMS}
    base = CartesianIndex(ntuple(d -> 2 * J[d] - 1, NDIMS))
    return (base + off for off in CartesianIndices(ntuple(_ -> 0:1, NDIMS)))
end

# Check that `mesh_fine` is exactly `mesh_coarse` refined by two in every
# dimension, on the same domain — the only case 2:1 restriction is defined for.
function check_refinement(mesh_coarse::AbstractMesh{NDIMS},
                          mesh_fine::AbstractMesh{NDIMS}) where {NDIMS}

    for d in 1:NDIMS
        if size(mesh_fine, d) != 2 * size(mesh_coarse, d)
            error("self-referenced convergence needs each grid to be exactly " *
                  "twice the previous one in every dimension, but dimension $d " *
                  "has $(size(mesh_coarse, d)) coarse vs $(size(mesh_fine, d)) " *
                  "fine cells (expected $(2 * size(mesh_coarse, d))). " *
                  "Use grid sizes like [40, 80, 160, 320].")
        end
    end

    if mesh_coarse.coordinates_min != mesh_fine.coordinates_min ||
       mesh_coarse.coordinates_max != mesh_fine.coordinates_max
        error("self-referenced convergence needs both grids on the same " *
              "domain, got [$(mesh_coarse.coordinates_min), " *
              "$(mesh_coarse.coordinates_max)] and " *
              "[$(mesh_fine.coordinates_min), $(mesh_fine.coordinates_max)].")
    end

    return nothing
end

# Restrict one interleaved block of `nvars` variables, i.e.
# [v₁(1), …, v_nvars(1), v₁(2), …] , writing into `dest` at `offset_dest` and
# reading from `src` at `offset_src`. Each coarse value is the plain mean of
# its 2^NDIMS children.
function restrict_block!(dest, src,
                         nvars,
                         mesh_coarse, mesh_fine,
                         offset_dest, offset_src)

    nchildren = 2^ndims(mesh_coarse)

    for J in eachcell(mesh_coarse)

        cell_coarse = cell_index(J, mesh_coarse)

        for v in 1:nvars

            acc = zero(eltype(dest))

            for I in fine_children(J)
                cell_fine = cell_index(I, mesh_fine)
                acc += src[offset_src + global_dof(cell_fine, v, nvars)]
            end

            dest[offset_dest + global_dof(cell_coarse, v, nvars)] = acc / nchildren
        end
    end

    return dest
end

"""
    restrict_solution(u_fine, semi_fine, semi_coarse)

Coarsen the fine-grid state `u_fine` onto the coarse grid of `semi_coarse` by
averaging each group of `2^ndims` fine cells, returning a vector in the coarse
grid's own layout.

Exact for the finite-volume variables (ρ, m): they are cell averages, and the
average of the two half-cell averages *is* the coarse cell average, to machine
precision.

Not exact for the potential φ, which the elliptic solver treats as a point
value at the cell centre (its Laplacian is the 3-point finite difference
`c/Δx²`): averaging the two fine centres gives `φ(x_j) + (Δx²/8) φ'' + O(Δx⁴)`.
That O(Δx²) restriction artifact sits at the same order as the true error of a
second-order scheme, so it perturbs the error *constant* for φ without changing
the observed order. It would, however, cap a measurement at order 2 for any
higher-order scheme.
"""
function restrict_solution(u_fine,
                           semi_fine::SemidiscretizationHyperbolic,
                           semi_coarse::SemidiscretizationHyperbolic)

    check_refinement(semi_coarse.mesh, semi_fine.mesh)

    nvars = nvariables(semi_coarse.equations)

    u_coarse = zeros(eltype(u_fine), nvars * ndofs(semi_coarse.mesh))

    restrict_block!(u_coarse, u_fine, nvars,
                    semi_coarse.mesh, semi_fine.mesh, 0, 0)

    return u_coarse
end

function restrict_solution(u_fine,
                           semi_fine::SemidiscretizationHyperbolicElliptic,
                           semi_coarse::SemidiscretizationHyperbolicElliptic)

    check_refinement(semi_coarse.mesh, semi_fine.mesh)

    nvars_hyper    = nvariables(semi_coarse.equations)
    nvars_elliptic = nvariables(semi_coarse.equations_elliptic)

    nc_coarse = ndofs(semi_coarse.mesh)
    nc_fine   = ndofs(semi_fine.mesh)

    n_hyper_coarse = nvars_hyper * nc_coarse
    n_hyper_fine   = nvars_hyper * nc_fine

    u_coarse = zeros(eltype(u_fine),
                     n_hyper_coarse + nvars_elliptic * nc_coarse)

    # Hyperbolic block: u[1:n_hyper] = [ρ₁, m₁, ρ₂, m₂, …]
    restrict_block!(u_coarse, u_fine, nvars_hyper,
                    semi_coarse.mesh, semi_fine.mesh, 0, 0)

    # Elliptic block: u[n_hyper+1:end] = [φ₁, φ₂, …]
    restrict_block!(u_coarse, u_fine, nvars_elliptic,
                    semi_coarse.mesh, semi_fine.mesh,
                    n_hyper_coarse, n_hyper_fine)

    return u_coarse
end

# ----------------------------------------------------------------------------
# Error between two states living on the same grid
# ----------------------------------------------------------------------------

# Full per-cell state, in the same variable order `variable_names(semi)`
# reports. Mirrors what `compute_errors` extracts, for the two semi types.
@inline function self_cell_state(u, I, semi::SemidiscretizationHyperbolic)
    return extract_cell_state(u, I, semi)
end

@inline function self_cell_state(u, I, semi::SemidiscretizationHyperbolicElliptic)

    nvars_hyper    = nvariables(semi.equations)
    nvars_elliptic = nvariables(semi.equations_elliptic)

    n_hyper = nvars_hyper * ndofs(semi.mesh)
    cell    = cell_index(I, semi)

    hyper = SVector{nvars_hyper}(
        ntuple(v -> u[global_dof(cell, v, nvars_hyper)], nvars_hyper)
    )

    elliptic = SVector{nvars_elliptic}(
        ntuple(v -> u[n_hyper + global_dof(cell, v, nvars_elliptic)],
               nvars_elliptic)
    )

    return vcat(hyper, elliptic)
end

"""
    errors_between(u_a, u_b, semi)

Discrete L¹, L² and L∞ norms of `u_a - u_b`, for two states on the *same* grid
`semi`, one `ErrorNorms` per variable. Uses exactly the same accumulator and
`finish` machinery as [`compute_errors`](@ref), so the norm definitions are
identical to the exact-solution pathway.
"""
function errors_between(u_a, u_b, semi::AbstractSemidiscretization)

    mesh = semi.mesh

    nvars = length(variable_names(semi))

    T = eltype(u_a)

    cell_volume = prod(mesh.dx)

    accumulators = Vector{ErrorAccumulator{T}}(undef, nvars)

    @inbounds for v in 1:nvars
        accumulators[v] = ErrorAccumulator(T)
    end

    for I in eachcell(mesh)

        a = self_cell_state(u_a, I, semi)
        b = self_cell_state(u_b, I, semi)

        @inbounds for v in 1:nvars
            accumulate!(accumulators[v], a[v] - b[v])
        end
    end

    norms = Vector{ErrorNorms{T}}(undef, nvars)

    @inbounds for v in 1:nvars
        norms[v] = finish(accumulators[v], cell_volume)
    end

    return norms
end

"""
    compute_errors_self(solution_coarse, solution_fine, semi_coarse, semi_fine)

Self-referenced error of the coarse solution: the coarse grid's own solution
compared against the fine solution coarsened onto it by cell averaging (see
[`restrict_solution`](@ref)). Returns a [`SelfConvergenceResult`](@ref).

No exact solution is involved — this is the counterpart of
[`compute_errors`](@ref) for test cases that have none. Both solutions must be
at the same time; that is checked.
"""
function compute_errors_self(solution_coarse,
                             solution_fine,
                             semi_coarse::AbstractSemidiscretization,
                             semi_fine::AbstractSemidiscretization)

    t_coarse = solution_time(solution_coarse)
    t_fine   = solution_time(solution_fine)

    if !isapprox(t_coarse, t_fine; rtol = 1e-10, atol = 1e-12)
        error("self-referenced errors compare two solutions at the same time, " *
              "but the coarse solution is at t = $t_coarse and the fine one " *
              "at t = $t_fine.")
    end

    u_coarse = solution_vector(solution_coarse)

    u_fine_restricted = restrict_solution(solution_vector(solution_fine),
                                          semi_fine,
                                          semi_coarse)

    norms = errors_between(u_coarse, u_fine_restricted, semi_coarse)

    return SelfConvergenceResult(norms,
                                 ndofs(semi_coarse.mesh),
                                 ndofs(semi_fine.mesh))
end

# ----------------------------------------------------------------------------
# Table + driver
# ----------------------------------------------------------------------------

"""
    self_convergence_table(cells, results, names)

Print one self-referenced convergence table per variable, labelled with
`names` (e.g. from `variable_names(semi)`).

The EOC column is labelled `Self-EOC` and each row is the coarse grid of one
(N, 2N) pair, so a printed table can never be mistaken for the exact-solution
table produced by [`convergence_table`](@ref).
"""
function self_convergence_table(cells, results, names)

    for (variable, name) in enumerate(names)
        println()
        println("Variable: ", name)
        convergence_table(cells, results;
                          variable = variable,
                          eoc_label = "Self-EOC",
                          title = "Self-referenced convergence " *
                                  "(each row: N vs. cell-averaged 2N solution)")
    end

    return nothing
end

"""
    self_convergence_test(semi_builder, grid_sizes, tspan, integrator;
                          dt = nothing, abstol = 1e-8, reltol = 1e-8,
                          limiter = minmod)

Run a *self-referenced* convergence test: solve at every resolution in
`grid_sizes` and compare each grid against the next finer one, coarsened back
onto it by cell averaging. Use this when the test case has no analytic
solution; use [`convergence_test`](@ref) when it does.

Note the deliberately different signature: there is no `exact_solution`
keyword, which is mandatory for [`convergence_test`](@ref). The two pathways
cannot be called interchangeably.

- `semi_builder`: a function `N -> semidiscretization`, as for `convergence_test`
- `grid_sizes`: resolutions in increasing order, each exactly twice the
  previous one in every dimension (e.g. `[40, 80, 160, 320]`); anything else is
  an error, since 2:1 restriction is undefined for it
- `tspan`, `integrator`, `dt`, `abstol`, `reltol`, `limiter`: passed to `solve()`

With `n` grid sizes this produces `n - 1` error rows (the finest grid appears
only as the reference for the second-finest) and `n - 2` EOC values. Only two
solutions are ever held in memory at once.

Returns the `Vector{SelfConvergenceResult}`, and prints one table per variable
via [`self_convergence_table`](@ref).
"""
function self_convergence_test(semi_builder, grid_sizes, tspan, integrator;
                               dt = nothing,
                               abstol = 1e-8, reltol = 1e-8,
                               limiter = minmod)

    if length(grid_sizes) < 2
        error("self-referenced convergence needs at least two grid sizes " *
              "(each grid is compared against the next finer one), got " *
              "$(length(grid_sizes)).")
    end

    results = SelfConvergenceResult[]
    cells   = Int[]

    # Only the previous (coarser) grid is kept, so memory stays at two grids
    # regardless of how many resolutions are swept.
    semi_prev = nothing
    sol_prev  = nothing

    semi = nothing

    for N in grid_sizes

        semi = semi_builder(N)

        dt_actual = dt === nothing ? minimum(semi.mesh.dx) : dt

        sol = solve(semi, tspan, integrator;
                    dt = dt_actual, abstol = abstol, reltol = reltol,
                    limiter = limiter)

        if semi_prev !== nothing
            push!(results, compute_errors_self(sol_prev, sol, semi_prev, semi))
            push!(cells, ndofs(semi_prev.mesh))
        end

        semi_prev = semi
        sol_prev  = sol
    end

    self_convergence_table(cells, results, variable_names(semi))

    return results
end

end # @muladd
