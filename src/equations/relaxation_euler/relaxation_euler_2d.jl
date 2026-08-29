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

# cons: Conservative variable (rho, m1, m2)
# prim: Primitive variable (rho, u1, u2)

# orientation = 1 for the x-axis
#               2 for the y-axis
@inline function flux(u::SVector{3}, orientation,
                      equations::RelaxationEulerEquations2D)
    rho = u[1]
    m1  = u[2]
    m2  = u[3]
    p   = rho^equations.gamma

    if orientation == 1
        return SVector(m1,
                       m1^2 / rho + p * equations.inv_epsilon,
                       m1 * m2 / rho)
    else # orientation == 2
        return SVector(m2,
                       m1 * m2 / rho,
                       m2^2 / rho + p * equations.inv_epsilon)
    end
end

@inline function max_abs_speed(u::SVector{3}, orientation,
                               equations::RelaxationEulerEquations2D)

    rho = u[1]
    rho <= 0 && error("Negative density: rho = $rho")

    normal_momentum = u[1 + orientation] # u[2] for orientation=1, u[3] for orientation=2
    velocity = normal_momentum / rho
    alpha = sqrt(equations.gamma * rho^(equations.gamma - 1)) # sqrt(γ * ρ^(γ-1))

    c = sqrt(equations.inv_epsilon)
    return abs(velocity) + alpha * c
end

# --------------------------------------------------
# Source term
# --------------------------------------------------

@inline function source_terms(u, equations::RelaxationEulerEquations2D)
    # ρ_t += 0
    # m1_t += -m1/ε (stiff friction)
    # m2_t += -m2/ε (stiff friction)
    return SVector(zero(u[1]), -u[2] * equations.inv_epsilon, -u[3] * equations.inv_epsilon)
end

# --------------------------------------------------
# Initial condition: Riemann (2D)
# --------------------------------------------------

"""
    initial_condition_riemann(x, t, equations::RelaxationEulerEquations2D)

A 2D Riemann initial data: circular discontinuity at r = 0.5.
"""
function initial_condition_riemann(x, t, equations::RelaxationEulerEquations2D)
    RealT = eltype(x)
    r = sqrt(x[1]^2 + x[2]^2)
    rho = r < RealT(0.5) ? RealT(1.0) : RealT(0.125)
    return SVector(rho, zero(RealT), zero(RealT))
end

end # @muladd
