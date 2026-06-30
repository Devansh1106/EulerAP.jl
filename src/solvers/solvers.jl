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

@inline function apply_bc(bc::PeriodicBC,
                          u,
                          I,
                          semi,
                          t)

    error("PeriodicBC should never reach apply_bc")
end

# ============================================================================
# Display
# ============================================================================

@inline Base.show(io::IO, ::FVSolver) = print(io, "Finite Volume")

@inline Base.show(io::IO, ::FluxRusanov) = print(io, "Rusanov")

@inline Base.show(io::IO, flux::FluxEnergyStable) = print(io, "Energy Stable (eta=$(flux.eta))")


end # @muladd