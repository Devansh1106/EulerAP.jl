# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ErrorNorms

Discrete error norms for one conserved variable.
"""
struct ErrorNorms{T}
    L1::T
    L2::T
    Linf::T
end


"""
    ErrorAccumulator

Accumulator used to compute error norms in a single pass over the mesh.
"""
mutable struct ErrorAccumulator{T}
    L1::T
    L2::T
    Linf::T
end


"""
    AnalysisResult

Stores error norms for every conserved variable.

errors[v] contains the norms corresponding to conserved variable `v`.
"""
struct AnalysisResult{T}
    norms::Vector{ErrorNorms{T}}
end

end # @muladd