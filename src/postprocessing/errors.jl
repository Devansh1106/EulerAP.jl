# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    compute_errors(solution,
                   semi;
                   exact_solution)

Compute the discrete L¹, L² and L∞ error norms for every conserved
variable.
"""
function compute_errors(solution,
                        semi::AbstractSemidiscretization;
                        exact_solution)

    mesh = semi.mesh
    equations = semi.equations

    nvars = nvariables(equations)

    T = eltype(solution.u)

    cell_volume = prod(mesh.dx)

    accumulators = Vector{ErrorAccumulator{T}}(undef, nvars)

    @inbounds for v in 1:nvars
        accumulators[v] = ErrorAccumulator(T)
    end

    for I in eachcell(mesh)

        x = coordinates(I, mesh)

        numerical = extract_cell_state(
            solution.u,
            I,
            semi
        )

        exact = exact_solution(
            x,
            solution.t,
            equations
        )

        @inbounds for v in 1:nvars
            accumulate!(
                accumulators[v],
                numerical[v] - exact[v]
            )
        end

    end

    norms = Vector{ErrorNorms{T}}(undef, nvars)

    @inbounds for v in 1:nvars
        norms[v] = finish(
            accumulators[v],
            cell_volume
        )
    end

    return AnalysisResult(norms)
end

function compute_errors(solution,
                        semi::SemidiscretizationHyperbolicElliptic;
                        exact_solution)

    mesh = semi.mesh
    equations_hyperbolic = semi.equations
    equations_elliptic   = semi.equations_elliptic

    nvars_hyper    = nvariables(equations_hyperbolic)
    nvars_elliptic = nvariables(equations_elliptic)
    nvars_total    = nvars_hyper + nvars_elliptic

    nc = ndofs(mesh)
    n_hyper = nvars_hyper * nc

    u = solution_vector(solution)

    T = eltype(u)

    cell_volume = prod(mesh.dx)

    accumulators = Vector{ErrorAccumulator{T}}(undef, nvars_total)

    @inbounds for v in 1:nvars_total
        accumulators[v] = ErrorAccumulator(T)
    end

    for I in eachcell(mesh)

        x = coordinates(I, mesh)

        cell = cell_index(I, semi)

        # Extract hyperbolic part from block layout: u[1:n_hyper] = [ρ₁, m₁, ρ₂, m₂, ...]
        numerical_hyper = SVector{nvars_hyper}(
            ntuple(v -> u[global_dof(cell, v, nvars_hyper)], nvars_hyper)
        )

        # Extract elliptic part from block layout: u[n_hyper+1:end] = [φ₁, φ₂, ...]
        numerical_elliptic = SVector{nvars_elliptic}(
            ntuple(v -> u[n_hyper + (cell - 1) * nvars_elliptic + v], nvars_elliptic)
        )

        # Full numerical state
        numerical = vcat(numerical_hyper, numerical_elliptic)

        exact = exact_solution(
            x,
            solution.t,
            semi,
        )

        @inbounds for v in 1:nvars_total
            accumulate!(
                accumulators[v],
                numerical[v] - exact[v]
            )
        end

    end

    norms = Vector{ErrorNorms{T}}(undef, nvars_total)

    @inbounds for v in 1:nvars_total
        norms[v] = finish(
            accumulators[v],
            cell_volume
        )
    end

    return AnalysisResult(norms)
end

end # @muladd