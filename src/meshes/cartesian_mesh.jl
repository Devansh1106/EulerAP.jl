# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    CartesianMesh{NDIMS} <: AbstractMesh{NDIMS}

A cartesian mesh.

Different number of cells per dimension are possible.
"""
struct CartesianMesh{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    cells_per_dimension::NTuple{NDIMS, Int}
    coordinates_min    ::NTuple{NDIMS, RealT}   # min coordinate of each dimension
    coordinates_max    ::NTuple{NDIMS, RealT}   # max coordinate of each dimension
    dx                 ::NTuple{NDIMS, RealT}
    periodicity        ::NTuple{NDIMS, Bool}
end

"""
    CartesianMesh(cells_per_dimension,
                  coordinates_min,
                  coordinates_max;
                  RealT = Float64
                  periodicity = ntuple(_ -> false, length(cells_per_dimension)))

Creates a CartesianMesh that represents a uncurved structured mesh with a rectangular domain, of the given size that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTuple{NDIMS, Int}`: the number of cells in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: an `NTuple{NDIMS, Bool}` deciding for each dimension if the boundaries 
                 in this dimension are periodic.
"""
function CartesianMesh(cells_per_dimension,
                       coordinates_min,
                       coordinates_max;
                       RealT = Float64,
                       periodicity = ntuple(_ -> false, length(cells_per_dimension)))

    NDIMS = length(cells_per_dimension)

    dx = ntuple(d -> 
               (coordinates_max[d] - coordinates_min[d])/cells_per_dimension[d],
               NDIMS)

    CartesianMesh{NDIMS, RealT}(
        Tuple(cells_per_dimension),
        Tuple(coordinates_min),
        Tuple(coordinates_max),
        dx,
        Tuple(periodicity)
    )
end

# Check if mesh is periodic
@inline isperiodic(mesh::CartesianMesh) = all(mesh.periodicity)
@inline isperiodic(mesh::CartesianMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::CartesianMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.size(mesh::CartesianMesh) = mesh.cells_per_dimension
@inline Base.size(mesh::CartesianMesh, i) = mesh.cells_per_dimension[i]

@inline ncells(mesh::CartesianMesh) = prod(size(mesh))

@inline eachcell(mesh::CartesianMesh{1}) = CartesianIndices((size(mesh, 1),))

@inline eachcell(mesh::CartesianMesh{2}) = CartesianIndices(size(mesh))

@inline cell_index(I::CartesianIndex,
                   semi::AbstractSemidiscretization) = cell_index(I, semi.mesh)

@inline cell_index(I::CartesianIndex{1}, mesh::CartesianMesh{1}) = I[1]

@inline function cell_index(I::CartesianIndex{2}, mesh::CartesianMesh{2})
    i, j = Tuple(I)
    return i + (j - 1) * size(mesh, 1)
end

@inline mesh(context::CallbackContext) = semi(context).mesh

"""
    minimum_cell_size(mesh)

Return the smallest mesh spacing.
"""
@inline minimum_cell_size(mesh::CartesianMesh{NDIMS}) where {NDIMS} =
    minimum(mesh.dx)

# ============================================================================
# Display
# ============================================================================

@inline function Base.show(io::IO, mesh::CartesianMesh{NDIMS}) where {NDIMS}
    print(io,
          NDIMS,
          "D Cartesian mesh (",
          prod(size(mesh)),
          " cells)")

end

end # @muladd