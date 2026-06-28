# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    RelaxationEulerEquations2D(gamma, epsilon)

The isentropic Euler equations in two space dimensions with polytropic pressure law
``p(\rho) = \rho^\gamma`` augmented with stiff friction term and diffusive scaling.
"""
struct RelaxationEulerEquations2D{RealT <: Real} <:
       AbstractRelaxationEulerEquations{2, 3}
    gamma::RealT
    epsilon::RealT      # Scaling parameter
    inv_epsilon::RealT  # = inv(epsilon); preferring fast multiplication instead of slow division
end

# outer constructor for matching the type of values using promote()
function RelaxationEulerEquations2D(; gamma, epsilon)
    γ, ϵ, inv_epsilon = promote(gamma, epsilon, inv(epsilon))
    return RelaxationEulerEquations2D{typeof(γ)}(γ, ϵ, inv_epsilon)
end

end # @muladd