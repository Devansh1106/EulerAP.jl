# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains code that is common in both 1D and 2D system i.e. the Poisson-Boltzmann equation
# and some precoded initial conditions.

struct PoissonBoltzmann{RealT <: Real} <: 
       AbstractPoissonBoltzmannEquations{1, 1}
    lambda::RealT        
end

# outer constructor for matching the type of values using promote()
function PoissonBoltzmann(; lambda)
    λ, = promote(lambda)
    return PoissonBoltzmann{typeof(λ)}(λ)
end

# --------------------------------------------------
# Local elliptic residual contributions
# --------------------------------------------------
# These functions define the equation-specific part of the elliptic
# prediction step in the IMEX scheme. The full residual is:
#
#   F_i = -α(x_{i-1} - 2x_i + x_{i+1}) + exp(x_i) - ρ̂_i
#
# where α = (λ² + ηΔt)/Δx² is handled by the IMEX solver.
# This function provides the exp(x_i) term and its Jacobian contribution.

"""
    elliptic_point_source(x_i)

Return the equation-specific nonlinear term for the elliptic equation.
For the Poisson-Boltzmann equation, this is exp(x).
"""
@inline function elliptic_point_source(x_i, equations::AbstractPoissonBoltzmannEquations)
    return exp(x_i)
end

"""
    elliptic_point_source_derivative(x_i)

Return the derivative of the equation-specific nonlinear term.
For the Poisson-Boltzmann equation, this is exp(x).
"""
@inline function elliptic_point_source_derivative(x_i, equations::AbstractPoissonBoltzmannEquations)
    return exp(x_i)
end

@inline initial_laplacian_coefficient(equations::AbstractEllipticEquations) = -equations.lambda^2


# TODO: Need to change EulerPressureLess1D to a common type for Hyper+elliptic system. This is a temporary setup.

end # @muladd
