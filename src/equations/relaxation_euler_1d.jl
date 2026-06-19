# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    RelaxationEulerEquations1D(gamma, epsilon)

The isentropic Euler equations
```math
\begin{aligned}
\partial_t\rho + \partial_x m &= 0, \\
\partial_t m + \partial_x\left(\frac{m^2}{\rho}+\frac{\rho^\gamma}{\varepsilon}\right) &= -\frac{m}{\varepsilon}
\end{aligned}
```
in one space dimension with polytropic pressure law ``p(\rho) = \rho^\gamma`` augmented with stiff friction term and diffusive scaling. Here ``\rho`` is the density, ``m`` is the velocity and ``p`` is the pressure (as a function of density).
"""
struct RelaxationEulerEquations1D{RealT <: Real} <: 
       AbstractRelaxationEulerEquations{1, 3}
    gamma::RealT        
    epsilon::RealT      # Scaling parameter
    inv_epsilon::RealT  # = inv(epsilon); preferring fast multiplication instead of slow division

    # inner constructor for matching the type of values using promote()
    function RelaxationEulerEquations1D(gamma, epsilon)
        γ, ϵ, inv_epsilon = promote(gamma, epsilon, inv(epsilon))
        return new{typeof(γ)}(γ, ϵ, inv_epsilon)
    end
end

# cons: Conservative variable (rho, rho_u)
# prim: Primitive variable (rho, u)
# code related to this is not needed yets

"""
    initial_condition_riemann(x, t, equations::RelaxationEulerEquations1D)

A Riemann type initial data.
"""
function initial_condition_riemann(x, t, equations::RelaxationEulerEquations1D)
    RealT = eltype(x)
    rho_l = RealT(1.0)
    rho_r = RealT(0.5)
    rho = x[1] <= RealT(0.5) ? rho_l : rho_r
    return SVector(rho, zero(RealT))    
end


end # @muladd