# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FVSolver{NDIMS, TFlux} <:
       AbstractSolver

    flux::TFlux
end

function FVSolver(; flux, ndims::Int)

    return FVSolver{ndims, typeof(flux)}(
        flux
    )
end


"""
    stencil_size(semi::AbstractSemidiscretization)

Returns size of the local stencil for Finite Volume 1D solver.
"""
@inline stencil_size(semi::AbstractSemidiscretization) = 2 * ndims(semi.mesh) + 1

end # @muladd