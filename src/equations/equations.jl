# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

"""
    varnames(equations)

Human-readable name for each variable in `equations`, in the same order as
its state vector. Falls back to generic "Variable i" labels for any
`equations` type without a more specific method (added below, next to each
concrete equations type).
"""
@inline varnames(equations::AbstractEquations) = ntuple(v -> "Variable $v", nvariables(equations))

# ============================================================================
# Combined hyperbolic-elliptic equations
# ============================================================================

abstract type AbstractEulerPoissonBoltzmannEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end

# ============================================================================
# Hyperbolic equations
# ============================================================================

abstract type AbstractHyperbolicEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end

# ----------------------------------------------------------------------------
# Relaxation Euler system with pressure law p = ρ^γ
# ----------------------------------------------------------------------------

abstract type AbstractRelaxationEulerEquations{NDIMS, NVARS} <:
              AbstractHyperbolicEquations{NDIMS, NVARS} end

include("relaxation_euler/relaxation_euler_1d.jl")
include("relaxation_euler/relaxation_euler_2d.jl")

@inline Base.show(io::IO, ::RelaxationEulerEquations1D) = print(io, "Relaxation Euler equations (1D)")
@inline Base.show(io::IO, ::RelaxationEulerEquations2D) = print(io, "Relaxation Euler equations (2D)")

@inline varnames(::RelaxationEulerEquations1D) = ("Density", "Momentum")
@inline varnames(::RelaxationEulerEquations2D) = ("Density", "x-Momentum", "y-Momentum")


# ----------------------------------------------------------------------------
# Euler-Pressure-Less
# ----------------------------------------------------------------------------

abstract type AbstractEulerPressureLessEquations{NDIMS, NVARS} <:
              AbstractHyperbolicEquations{NDIMS, NVARS} end

include("euler_poisson_boltzmann/euler_pressure_less_1d.jl")
include("euler_poisson_boltzmann/euler_pressure_less_2d.jl")

@inline Base.show(io::IO, ::EulerPressureLess1D) = print(io, "Euler-Pressure-Less equations (1D)")
@inline Base.show(io::IO, ::EulerPressureLess2D) = print(io, "Euler-Pressure-Less equations (2D)")

@inline varnames(::EulerPressureLess1D) = ("Density", "Momentum")
@inline varnames(::EulerPressureLess2D) = ("Density", "x-Momentum", "y-Momentum")


# ============================================================================
# Elliptic equations
# ============================================================================

abstract type AbstractEllipticEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end

abstract type AbstractPoissonBoltzmannEquations{NDIMS, NVARS} <:
              AbstractEllipticEquations{NDIMS, NVARS} end

include("euler_poisson_boltzmann/euler_poisson_boltzmann.jl")

@inline Base.show(io::IO, ::PoissonBoltzmann) = print(io, "Poisson-Boltzmann equation (1D)")

@inline varnames(::PoissonBoltzmann) = ("Potential",)

end # @muladd