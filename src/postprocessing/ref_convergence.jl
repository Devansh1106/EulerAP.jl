# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# ============================================================================
# Reference-solution convergence
# ============================================================================
#
# The third convergence pathway, for test cases with no analytic solution where
# a single well-resolved run is to serve as the "exact" one. Every grid in the
# sweep is compared against that *same* reference solution, coarsened onto it
# by cell averaging:
#
#     e_m = || u_m - R_{M/m}(u_M) ||
#
# with M the reference resolution, m the measured one and R_r the r:1
# restriction that averages r cells (r^NDIMS in NDIMS dimensions) of the
# reference into one cell of the measured grid. The factor is not fixed: it is
# M/m, computed per grid, so [80, 160, 320, 640] against M = 1280 averages 16,
# 8, 4 and 2 reference cells per coarse cell respectively.
#
# How this differs from the two existing pathways:
#
#   * `convergence_test` needs an analytic `exact_solution(x, t, semi)`;
#   * `self_convergence_test` needs no reference run, but compares each grid
#     against the *next finer grid of the sweep* — structurally 2:1, and its
#     rows are correlated because each solution appears twice, once as the
#     measured grid and once as the reference;
#   * `ref_convergence_test` (here) pays for one extra, finest run and in
#     exchange measures every row against the same fixed yardstick, with the
#     finest measured grid a row of its own rather than being spent as the
#     reference.
#
# The measured error has a floor at the reference's own error: as m approaches
# M the difference stops being the coarse grid's error, so keep M at least a
# refinement or two beyond the finest measured grid and read the last row with
# that in mind.
#
# Kept in its own file with its own result type (`RefConvergenceResult`), its
# own restriction (`restrict_to_reference_grid`) and its own table label
# ("Ref-EOC") so that none of the three error notions can be mixed up at a call
# site or in a saved log. In particular, the 2:1-only restriction of
# `self_convergence.jl` is untouched by the general factor used here.
# ============================================================================

"""
    RefConvergenceResult

Error norms of one grid measured against a fixed fine reference solution, one
`ErrorNorms` per variable.

Deliberately a distinct type from [`AnalysisResult`](@ref) (distance to an
analytic solution) and from [`SelfConvergenceResult`](@ref) (difference from
the next finer grid of the sweep): the three are not comparable and must not be
printed by the same table.

`cells` is the measured grid, `cells_reference` the reference it was compared
against, and `factors` the per-dimension number of reference cells averaged
into one measured cell.
"""
struct RefConvergenceResult{T, NDIMS}
    norms::Vector{ErrorNorms{T}}
    cells::Int
    cells_reference::Int
    factors::NTuple{NDIMS, Int}
end

# ----------------------------------------------------------------------------
# Coarsening factor
# ----------------------------------------------------------------------------

"""
    reference_factors(mesh, mesh_reference)

How many reference cells are averaged into one cell of `mesh`, per dimension:
`M_d / m_d` for `m_d = size(mesh, d)` and `M_d = size(mesh_reference, d)`.

Errors unless both meshes span the same domain and every `M_d` is an exact
integer multiple of the corresponding `m_d` — cell averaging is only defined
when each measured cell is tiled exactly by whole reference cells. The factor
need not be a power of two, and need not be the same in every dimension.
"""
function reference_factors(mesh::AbstractMesh{NDIMS},
                           mesh_reference::AbstractMesh{NDIMS}) where {NDIMS}

    if mesh.coordinates_min != mesh_reference.coordinates_min ||
       mesh.coordinates_max != mesh_reference.coordinates_max
        error("reference convergence needs the measured grid and the " *
              "reference on the same domain, got [$(mesh.coordinates_min), " *
              "$(mesh.coordinates_max)] and " *
              "[$(mesh_reference.coordinates_min), " *
              "$(mesh_reference.coordinates_max)].")
    end

    return ntuple(NDIMS) do d

        m = size(mesh, d)
        M = size(mesh_reference, d)

        factor, remainder = divrem(M, m)

        if remainder != 0 || factor < 1
            error("reference convergence averages M/m reference cells into " *
                  "each measured cell, so the reference must be an integer " *
                  "multiple of every measured grid, but dimension $d has " *
                  "m = $m against M = $M. Use grid sizes that divide the " *
                  "reference exactly, e.g. [80, 160, 320, 640] against 1280.")
        end

        factor
    end
end

# ----------------------------------------------------------------------------
# Restriction (reference -> measured grid), generic in the number of dimensions
# ----------------------------------------------------------------------------

# The `prod(factors)` reference cells covering cell `J` of the measured grid:
# in dimension d, reference indices r_d*(J_d - 1) + 1 … r_d*J_d. Written
# dimension-agnostically so 2D works unchanged.
@inline function reference_children(J::CartesianIndex{NDIMS},
                                    factors::NTuple{NDIMS, Int}) where {NDIMS}
    base = CartesianIndex(ntuple(d -> factors[d] * (J[d] - 1) + 1, NDIMS))
    return (base + off
            for off in CartesianIndices(ntuple(d -> 0:(factors[d] - 1), NDIMS)))
end

# Coarsen one interleaved block of `nvars` variables, i.e.
# [v₁(1), …, v_nvars(1), v₁(2), …], writing into `dest` at `offset_dest` and
# reading from `src` at `offset_src`. Each coarse value is the plain mean of
# the `prod(factors)` reference cells it covers.
function coarsen_block!(dest, src,
                        nvars,
                        mesh, mesh_reference,
                        offset_dest, offset_src,
                        factors)

    ncells_averaged = prod(factors)

    for J in eachcell(mesh)

        cell = cell_index(J, mesh)

        for v in 1:nvars

            acc = zero(eltype(dest))

            for I in reference_children(J, factors)
                cell_reference = cell_index(I, mesh_reference)
                acc += src[offset_src + global_dof(cell_reference, v, nvars)]
            end

            dest[offset_dest + global_dof(cell, v, nvars)] = acc / ncells_averaged
        end
    end

    return dest
end

"""
    restrict_to_reference_grid(u_reference, semi_reference, semi)

Coarsen the reference state `u_reference` onto the grid of `semi` by averaging
the `M/m` reference cells covering each of its cells (`(M/m)^NDIMS` in NDIMS
dimensions), returning a vector in that grid's own layout.

The factor comes from the two meshes via [`reference_factors`](@ref), so one
reference run can be coarsened onto every grid of a sweep directly, with no
chain of intermediate grids.

Exact for the finite-volume variables (ρ, m): they are cell averages, and the
mean of the sub-cell averages *is* the coarse cell average, to machine
precision, for any factor.

Not exact for the potential φ, which the elliptic solver treats as a point
value at the cell centre (its Laplacian is the 3-point finite difference
`c/Δx²`). Averaging the `r` reference centres spanning a measured cell of width
`H` gives `φ(x_j) + (H²/24)(1 - r⁻²) φ'' + O(H⁴)` — `H²/32` at `r = 2`, tending
to `H²/24` for large `r`. That O(H²) artifact sits at the same order as the
true error of a second-order scheme, so it perturbs the error *constant* for φ
without changing the observed order, and a larger factor does not shrink it. It
would, however, cap a measurement at order 2 for any higher-order scheme.
"""
function restrict_to_reference_grid(u_reference,
                                    semi_reference::SemidiscretizationHyperbolic,
                                    semi::SemidiscretizationHyperbolic)

    factors = reference_factors(semi.mesh, semi_reference.mesh)

    nvars = nvariables(semi.equations)

    u = zeros(eltype(u_reference), nvars * ndofs(semi.mesh))

    coarsen_block!(u, u_reference, nvars,
                   semi.mesh, semi_reference.mesh, 0, 0, factors)

    return u
end

function restrict_to_reference_grid(u_reference,
                                    semi_reference::SemidiscretizationHyperbolicElliptic,
                                    semi::SemidiscretizationHyperbolicElliptic)

    factors = reference_factors(semi.mesh, semi_reference.mesh)

    nvars_hyper    = nvariables(semi.equations)
    nvars_elliptic = nvariables(semi.equations_elliptic)

    nc           = ndofs(semi.mesh)
    nc_reference = ndofs(semi_reference.mesh)

    n_hyper           = nvars_hyper * nc
    n_hyper_reference = nvars_hyper * nc_reference

    u = zeros(eltype(u_reference), n_hyper + nvars_elliptic * nc)

    # Hyperbolic block: u[1:n_hyper] = [ρ₁, m₁, ρ₂, m₂, …]
    coarsen_block!(u, u_reference, nvars_hyper,
                   semi.mesh, semi_reference.mesh, 0, 0, factors)

    # Elliptic block: u[n_hyper+1:end] = [φ₁, φ₂, …]
    coarsen_block!(u, u_reference, nvars_elliptic,
                   semi.mesh, semi_reference.mesh,
                   n_hyper, n_hyper_reference, factors)

    return u
end

# ----------------------------------------------------------------------------
# Error against the reference
# ----------------------------------------------------------------------------

"""
    compute_errors_reference(solution, solution_reference, semi, semi_reference)

Error of `solution` against the reference solution coarsened onto its grid (see
[`restrict_to_reference_grid`](@ref)). Returns a [`RefConvergenceResult`](@ref).

No exact solution is involved — this is the fixed-reference counterpart of
[`compute_errors`](@ref). Both solutions must be at the same time; that is
checked. The norms themselves come from [`errors_between`](@ref), the same
accumulator machinery the other two pathways use, so the norm definitions are
identical across all three.
"""
function compute_errors_reference(solution,
                                  solution_reference,
                                  semi::AbstractSemidiscretization,
                                  semi_reference::AbstractSemidiscretization)

    t           = solution_time(solution)
    t_reference = solution_time(solution_reference)

    if !isapprox(t, t_reference; rtol = 1e-10, atol = 1e-12)
        error("reference errors compare two solutions at the same time, but " *
              "the measured solution is at t = $t and the reference at " *
              "t = $t_reference.")
    end

    u_reference_restricted =
        restrict_to_reference_grid(solution_vector(solution_reference),
                                   semi_reference,
                                   semi)

    norms = errors_between(solution_vector(solution),
                           u_reference_restricted,
                           semi)

    return RefConvergenceResult(norms,
                                ndofs(semi.mesh),
                                ndofs(semi_reference.mesh),
                                reference_factors(semi.mesh,
                                                  semi_reference.mesh))
end

# ----------------------------------------------------------------------------
# Table + driver
# ----------------------------------------------------------------------------

"""
    ref_convergence_table(results, names; cells_reference = nothing)

Print one reference-solution convergence table per variable, labelled with
`names` (e.g. from `variable_names(semi)`).

Written separately from [`convergence_table`](@ref) for two reasons: the order
column is labelled `Ref-EOC` and the header names the reference, so a printed
table can never be mistaken for either of the other two pathways'; and the
order is computed from the *actual* ratio of consecutive rows,

    k = log(e_prev / e) / log(m / m_prev),

rather than assuming a factor of two. A sweep here only has to divide the
reference, so it need not double from row to row — for one that does, this
reduces to `experimental_order`'s log2 exactly. The extra `Factor` column
records how many reference cells went into one cell of that row.
"""
function ref_convergence_table(results, names; cells_reference = nothing)

    for (variable, name) in enumerate(names)

        println()
        println("Variable: ", name)
        println("---------------------------------------------------------------------------")
        if cells_reference !== nothing
            println("Convergence against the $(cells_reference)-cell reference solution")
            println("---------------------------------------------------------------------------")
        end
        @printf "  Cells   Factor      L1 Error        L2 Error       Linf Error    Ref-EOC\n"
        println("---------------------------------------------------------------------------")

        previous = nothing

        for result in results

            norms  = result.norms[variable]
            factor = prod(result.factors)

            if previous === nothing
                @printf "%-8d %-7d  %14.6e  %14.6e  %14.6e  %9s\n" result.cells factor norms.L1 norms.L2 norms.Linf "-"
            else
                norms_prev, cells_prev = previous
                eoc = log(norms_prev.L2 / norms.L2) /
                      log(result.cells / cells_prev)
                @printf "%-8d %-7d  %14.6e  %14.6e  %14.6e  %9.3f\n" result.cells factor norms.L1 norms.L2 norms.Linf eoc
            end

            previous = (norms, result.cells)
        end

        println("---------------------------------------------------------------------------")
    end

    return nothing
end

"""
    ref_convergence_test(semi_builder, grid_sizes, tspan, integrator;
                         reference_grid_size, dt = nothing,
                         abstol = 1e-8, reltol = 1e-8, limiter = minmod)

Run a convergence test against a fixed fine *reference solution*: solve once at
`reference_grid_size`, then solve at every resolution in `grid_sizes` and
compare each against the reference, coarsened onto that grid by averaging
`reference_grid_size / N` cells per cell. Use this when the test case has no
analytic solution and one well-resolved run is to stand in for it; use
[`convergence_test`](@ref) when an analytic solution exists, and
[`self_convergence_test`](@ref) to avoid the extra reference run at the price
of comparing consecutive grids of the sweep against each other.

Note the deliberately different signature: `reference_grid_size` is mandatory
and there is no `exact_solution` keyword, so none of the three pathways can be
called interchangeably.

- `semi_builder`: a function `N -> semidiscretization`, as for the other two
- `grid_sizes`: the resolutions to measure, in increasing order. Each one must
  divide `reference_grid_size` exactly — that ratio is the number of reference
  cells averaged per cell — but they need not be powers of two and need not
  double from row to row, since the `Ref-EOC` column uses the actual ratio of
  consecutive rows
- `reference_grid_size`: the resolution of the reference run. Keep it at least
  a refinement or two beyond `maximum(grid_sizes)`: the measured error cannot
  drop below the reference's own error, and rows close to the reference read
  too high or too low as a result
- `tspan`, `integrator`, `dt`, `abstol`, `reltol`, `limiter`: passed to `solve()`

With `n` grid sizes this produces `n` error rows and `n - 1` EOC values, from
`n + 1` solver runs. Only the reference solution and the current grid's are
held at once.

Returns the `Vector{RefConvergenceResult}`, and prints one table per variable
via [`ref_convergence_table`](@ref).
"""
function ref_convergence_test(semi_builder, grid_sizes, tspan, integrator;
                              reference_grid_size,
                              dt = nothing,
                              abstol = 1e-8, reltol = 1e-8,
                              limiter = minmod)

    if isempty(grid_sizes)
        error("reference convergence needs at least one grid size to measure.")
    end

    if any(>=(reference_grid_size), grid_sizes)
        error("every measured grid must be coarser than the reference, but " *
              "got grid sizes $grid_sizes against a reference of " *
              "$reference_grid_size cells per dimension.")
    end

    # The reference run: solved once, up front, and kept for the whole sweep.
    semi_reference = semi_builder(reference_grid_size)

    dt_reference = dt === nothing ? minimum(semi_reference.mesh.dx) : dt

    sol_reference = solve(semi_reference, tspan, integrator;
                          dt = dt_reference, abstol = abstol, reltol = reltol,
                          limiter = limiter)

    results = RefConvergenceResult[]

    for N in grid_sizes

        semi = semi_builder(N)

        dt_actual = dt === nothing ? minimum(semi.mesh.dx) : dt

        sol = solve(semi, tspan, integrator;
                    dt = dt_actual, abstol = abstol, reltol = reltol,
                    limiter = limiter)

        push!(results, compute_errors_reference(sol, sol_reference,
                                                semi, semi_reference))
    end

    ref_convergence_table(results, variable_names(semi_reference);
                          cells_reference = ndofs(semi_reference.mesh))

    return results
end

end # @muladd
