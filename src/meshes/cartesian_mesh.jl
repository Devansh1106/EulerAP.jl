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
    periodicity        ::NTuple{NDIMS, Bool}
    coordinates_min    ::NTuple{NDIMS, RealT}   # min coordinate of each dimension
    coordinates_max    ::NTuple{NDIMS, RealT}   # max coordinate of each dimension
    dx                 ::NTuple{NDIMS, RealT}
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
isperiodic(mesh::CartesianMesh) = all(mesh.periodicity)
isperiodic(mesh::CartesianMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::CartesianMesh{NDIMS}) where {NDIMS} = NDIMS
Base.size(mesh::CartesianMesh) = mesh.cells_per_dimension
Base.size(mesh::CartesianMesh, i) = mesh.cells_per_dimension[i]

function Base.show(io::IO, mesh::CartesianMesh)
    print(io, "CartesianMesh{", ndims(mesh), ", ", eltype(mesh), "}")
    return nothing
end


end # @muladd