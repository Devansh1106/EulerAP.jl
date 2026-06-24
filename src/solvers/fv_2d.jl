# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    stencil_size(semi::AbstractSemidiscretization)

Returns size of the local stencil for Finite Volume 1D solver.
"""
@inline stencil_size(semi::AbstractSemidiscretization) = 2 * ndims(semi.mesh) + 1

@inline function stencil_indices(I::CartesianIndex{2},
                                 semi::AbstractSemidiscretization)

    center = cell_index(I, semi)

    # (I, semi, axis, sign); `x`: axis = 1
    left   = neighbor_index(I, semi, 1, -1)
    right  = neighbor_index(I, semi, 1,  1)

    # `y`: axis = 2
    bottom = neighbor_index(I, semi, 2, -1)
    top    = neighbor_index(I, semi, 2,  1)

    return (center, left, right, bottom, top)
end

function neighbor_index(I::CartesianIndex{2},
                        semi::AbstractSemidiscretization,
                        axis::Int,
                        size::Int)
    
end

end # @muladd