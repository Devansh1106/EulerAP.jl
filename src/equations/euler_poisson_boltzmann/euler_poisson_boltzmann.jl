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

# TODO: Need to change EulerPressureLess1D to a common type for Hyper+elliptic system. This is a temporary setup.
"""
    initial_condition_riemann_epb(x, t, semi)

Specialized method for the coupled hyperbolic-elliptic system.
Returns the full state vector (ρ, ρu, φ) with φ = log(ρ) so that
exp(φ) = ρ satisfies the Poisson-Boltzmann equation at t=0.
"""
@inline function initial_condition_riemann_epb(x, t, equations::EulerPressureLess1D)
    if x[1] < 0.5
        rho = 2.0
    else
        rho = 1.0
    end
    phi = log(rho)
    return SVector{3}(rho, 0.0, 0.0)
end

end # @muladd
