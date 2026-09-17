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

# --------------------------------------------------
# Implicit solver types for IMEX / fully implicit
# --------------------------------------------------

"""
    EllipticSolver <: AbstractSolver

Direct tridiagonal Newton solver for cell-centered implicit equations
(e.g., the elliptic prediction step in the IMEX scheme).
Specialized for single-variable tridiagonal systems in 1D.
"""
struct EllipticSolver <: AbstractSolver end


"""
    stencil_size(semi::AbstractSemidiscretization)

Returns size of the local stencil for Finite Volume 1D solver.
"""
@inline stencil_size(semi::AbstractSemidiscretization) = 2 * ndims(semi.mesh) + 1

# Used on Callbacks
@inline solver(context::CallbackContext) = semi(context).solver

@inline function _wrap_index(i::Int, nx::Int)
    return mod1(i, nx)
end

@inline function local_state(x, local_cell::Int, equations)

    nvars = nvariables(equations)
    first = (local_cell - 1) * nvars + 1
    last  = local_cell * nvars

    return SVector{nvars}(@view x[first:last])
end

# A periodic ghost is not a BC-evaluated state: it *is* an interior cell, seen
# through the wrap. `_wrap_index` (`mod1`) resolves a ghost at any depth, so the
# second-order scheme's two ghost layers come out as u[0] = u[nx] and
# u[-1] = u[nx - 1] on the left, and u[nx + 1] = u[1], u[nx + 2] = u[2] on the
# right. `mod1` is the identity on `1:nx`, so wrapping every dimension leaves
# the in-range ones untouched and only folds the out-of-range one back in.
@inline function apply_bc(bc::PeriodicBC{NDIMS},
                          u,
                          I::CartesianIndex{NDIMS},
                          semi,
                          t) where {NDIMS}

    wrapped = CartesianIndex(ntuple(d -> _wrap_index(I[d], size(semi.mesh, d)),
                                    NDIMS))

    return extract_cell_state(u, wrapped, semi)
end

# ============================================================================
# Display
# ============================================================================

@inline Base.show(io::IO, ::FVSolver) = print(io, "Finite Volume")

@inline Base.show(io::IO, ::FluxRusanov) = print(io, "Rusanov")

@inline Base.show(io::IO, flux::FluxEnergyStable) = print(io, "Energy Stable (eta=$(flux.eta))")


end # @muladd