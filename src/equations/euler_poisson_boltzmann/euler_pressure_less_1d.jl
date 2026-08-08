# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct EulerPressureLess1D{RealT <: Real} <: 
       AbstractEulerPressureLessEquations{1, 2}
    gamma::RealT        
end

# outer constructor for matching the type of values using promote()
function EulerPressureLess1D(; gamma)
    γ, = promote(gamma)
    return EulerPressureLess1D{typeof(γ)}(γ)
end

@inline function flux(u::SVector{2}, orientation, 
                      equations::EulerPressureLess1D)

    rho = u[1]
    m   = u[2]

    v = m / rho

    return SVector{2}(m, m*v)
end

# --------------------------------------------------
# Source term
# --------------------------------------------------

@inline function source_terms_hyperbolic(u, equations::EulerPressureLess1D)
    # ρ_t += 0
    # m_t += 0 (no source term for pressure-less Euler)
    return SVector{2}(zero(u[1]), zero(u[1]))
end

"""
    initial_condition_riemann_epb(x, t, equations::EulerPressureLess1D)

Specialized method for the coupled hyperbolic-elliptic system.
Returns the full hyperbolic state vector (ρ, ρu). Initial value for ϕ
is calculated in the [`initial_condition`](@ref) in [`semidiscretization_hyperbolic_elliptic`](@ref). 
"""
@inline function initial_condition_riemann_epb(x, t, equations::EulerPressureLess1D)
    nr = 0.5
    if x[1] < 0.0
        rho = 1.0
    else
        rho = nr
    end
    return SVector{2}(rho, 0.0)
end

"""
    initial_condition_five_branch(x, t, equations::EulerPressureLess1D)

Five-branch test problem initial condition.
- Density: Gaussian profile ρ(0,x) = (1/π) * exp(-(x-π)²)
- Velocity: Sinusoidal profile u(0,x) = sin³(x)
Domain: [0, 2π]
"""
@inline function initial_condition_five_branch(x, t, equations::EulerPressureLess1D)
    RealT = eltype(x)
    rho = (one(RealT) / π) * exp(-(x[1] - π)^2)
    u = sin(x[1])^3
    return SVector{2}(rho, rho * u)
end

@inline function initial_condition_seven_branch(x, t, equations::EulerPressureLess1D)
    RealT = eltype(x)
    rho = one(RealT) / π * exp(-(x[1] - π)^2)
    u = sin(2*x[1]) * cos(x[1])
    return SVector{2}(rho, rho * u)
end

@inline function initial_condition_shock_tube(x, t, equation::EulerPressureLess1D)
    RealT = eltype(x)
    rho = one(RealT)
    if x[1] < 0.0
        u = one(RealT)
    else
        u = -one(RealT)
    end
    return SVector{2}(rho, rho*u)
end

@inline function initial_condition_plasma_expansion(x, t, equations::EulerPressureLess1D)
    RealT = eltype(x)
    rho = 0.5 - (one(RealT) / π) * atan(π * x[1])
    return SVector{2}(rho, zero(rho))
end


end # @muladd
