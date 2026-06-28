# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# abstract supertype of specific semidiscretizations such as
# - SemidiscretizationHyperbolic for hyperbolic conservation laws
abstract type AbstractSemidiscretization end

"""
    AbstractEquations{NDIMS, NVARS}

An abstract supertype of specific equations such as the compressible Euler equations.
The type parameters encode the number of spatial dimensions (`NDIMS`) and the
number of primary variables (`NVARS`) of the physics model.
"""
abstract type AbstractEquations{NDIMS, NVARS} end

"""
    AbstractMesh{NDIMS}

An abstract supertype of specific mesh types such as `TreeMesh` or `StructuredMesh`.
The type parameters encode the number of spatial dimensions (`NDIMS`).
"""
abstract type AbstractMesh{NDIMS} end

abstract type AbstractSolver end

# ============================================================================
# Timing utilities
# ============================================================================

"""
    start_timer()

Return the current wall-clock time in nanoseconds.

Use together with [`elapsed_time`](@ref) for lightweight performance measurements.
"""
@inline start_timer() = time_ns()

"""
    elapsed_time(start_time)

Return the elapsed wall-clock time (in seconds) since `start_time`,
which must have been obtained using [`start_timer`](@ref).
"""
@inline elapsed_time(start_time::UInt64) = (time_ns() - start_time) * 1.0e-9

end # @muladd