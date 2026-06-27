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

Compute discrete error norms for every conserved variable.
"""
function compute_errors(solution,
                        semi;
                        exact_solution)

    mesh = semi.mesh
    equations = semi.equations

    nvars = nvariables(equations)

    T = eltype(solution.u)

    cell_volume = prod(mesh.dx)

    accumulators = [
        ErrorAccumulator(T)
        for _ in 1:nvars
    ]

    for I in eachcell(mesh)

        x = coordinates(I, mesh)

        numerical =
            extract_cell_state(
                solution.u,
                I,
                semi
            )

        exact =
            exact_solution(
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

    errors = Vector{ErrorNorms{T}}(undef, nvars)

    @inbounds for v in 1:nvars
        errors[v] = finish(
            accumulators[v],
            cell_volume
        )
    end

    return AnalysisResult(errors)
end

end # @muladd