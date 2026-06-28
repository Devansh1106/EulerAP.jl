# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

# ============================================================================
# Relaxation Euler system with pressure law = ρ^γ
# ============================================================================
abstract type AbstractRelaxationEulerEquations{NDIMS, NVARS} <: 
              AbstractEquations{NDIMS, NVARS} end

include("relaxation_euler_1d.jl")
include("relaxation_euler_2d.jl")

@inline Base.show(io::IO, ::RelaxationEulerEquations1D) = print(io, "Relaxation Euler equations (1D)")

# Euler Poisson Boltzmann
abstract type AbstractEulerPoissonBoltzmann{NDIMS, NVARS} <: 
    AbstractEquations{NDIMS, NVARS} end
    
include("euler_poisson_boltzmann_pressure_less_1d.jl")
include("euler_poisson_boltzmann_pressure_less_2d.jl")
    
@inline Base.show(io::IO, ::RelaxationEulerEquations2D) = print(io, "Relaxation Euler equations (2D)")



end # @muladd