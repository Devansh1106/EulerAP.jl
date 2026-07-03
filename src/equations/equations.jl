# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

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


# ----------------------------------------------------------------------------
# Euler-Pressure-Less
# ----------------------------------------------------------------------------

abstract type AbstractEulerPressureLessEquations{NDIMS, NVARS} <:
              AbstractHyperbolicEquations{NDIMS, NVARS} end

include("euler_poisson_boltzmann/euler_pressure_less_1d.jl")
include("euler_poisson_boltzmann/euler_pressure_less_2d.jl")

@inline Base.show(io::IO, ::EulerPoissonBoltzmann1D) = print(io, "Euler-Pressure-Less equations (1D)")
@inline Base.show(io::IO, ::EulerPoissonBoltzmann2D) = print(io, "Euler-Pressure-Less equations (2D)")


# ============================================================================
# Elliptic equations
# ============================================================================

abstract type AbstractEllipticEquations{NDIMS, NVARS} <:
              AbstractEquations{NDIMS, NVARS} end

include("euler_poisson_boltzmann/poisson_boltzmann.jl")
# include("elliptic/poisson_boltzmann_2d.jl")

@inline Base.show(io::IO, ::PoissonBoltzmannEquations1D) = print(io, "Poisson-Boltzmann equation (1D)")
@inline Base.show(io::IO, ::PoissonBoltzmannEquations2D) = print(io, "Poisson-Boltzmann equation (2D)")

end # @muladd