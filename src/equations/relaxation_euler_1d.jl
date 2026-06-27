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
       AbstractRelaxationEulerEquations{1, 2}
    gamma::RealT        
    epsilon::RealT      # Scaling parameter
    inv_epsilon::RealT  # = inv(epsilon); preferring fast multiplication instead of slow division
end

# outer constructor for matching the type of values using promote()
function RelaxationEulerEquations1D(; gamma, epsilon)
    γ, ϵ, inv_epsilon = promote(gamma, epsilon, inv(epsilon))
    return RelaxationEulerEquations1D{typeof(γ)}(γ, ϵ, inv_epsilon)
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

# orientation = 1 for the x-axis
#               2 for the y-axis
# TODO: add a similar 2D version in 2D file
@inline function flux(u::SVector{2}, orientation, 
                      equations::RelaxationEulerEquations1D)

    rho = u[1]
    m   = u[2]

    v = m / rho
    p = rho^equations.gamma

    return SVector(m, 
                   m*v + p*equations.inv_epsilon)
end

@inline function max_abs_speed(u::SVector{2}, orientation, 
                               equations::RelaxationEulerEquations1D)

    @assert orientation == 1

    rho = u[1]
    rho <= 0 && error("Negative density: rho = $rho")

    normal_momentum = u[1 + orientation] # essentially u[2] for 1D
    velocity = normal_momentum / rho
    alpha = sqrt(equations.gamma * rho^(equations.gamma - 1)) # sqrt(γ * ρ^(γ-1))

    c = sqrt(equations.inv_epsilon)
    return abs(velocity) + alpha * c
end

# --------------------------------------------------
# Source term
# --------------------------------------------------

@inline function source_terms(u, equations::RelaxationEulerEquations1D)
    # ρ_t += 0
    # m_t += -m/ε (stiff friction)
    return SVector(zero(u[1]), -u[2] * equations.inv_epsilon)
end

# --------------------------------------------------
# Initial condition: Smoothed single box
# --------------------------------------------------

"""
    initial_condition_box(x, t, equations::RelaxationEulerEquations1D)

A single smoothed box initial condition:
- Density: smoothed top-hat between a = -2 and b = 2
- Velocity: derived from gradient of the smoothing
- Parameters: Δ = 0.1, γ = 3.0, RHO_FLOOR = 1e-10
"""
function initial_condition_single_box(x, t, equations::RelaxationEulerEquations1D)
    RealT = eltype(x)
    DELTA = RealT(0.1)
    RHO_FLOOR = RealT(1e-10)
    a = RealT(-2.0)
    b = RealT(2.0)
    gamma = equations.gamma

    heaviside_smooth(x) = RealT(0.5) * (one(RealT) + tanh(x / DELTA))

    X = x[1] - a
    Y = x[1] - b
    ha = heaviside_smooth(X)
    hb = heaviside_smooth(Y)
    ρ = max(RHO_FLOOR, ha - hb)

    u = -tanh(X / DELTA) * tanh(X / DELTA)
    u += tanh(Y / DELTA) * tanh(Y / DELTA)
    u = u / (RealT(2) * DELTA)
    u = u * (-gamma) * ρ^(gamma - RealT(2))
    mx = ρ * u
    return SVector(ρ, mx)
end

# --------------------------------------------------
# Initial condition: Smoothed double box
# --------------------------------------------------

"""
    initial_condition_double_box(x, t, equations::RelaxationEulerEquations1D)

Two smoothed boxes: one between a=-2, b=-1 and another between c=1, d=2.
Parameters: Δ = 0.005, γ = 3.0, RHO_FLOOR = 1e-8
"""
function initial_condition_double_box(x, t, equations::RelaxationEulerEquations1D)
    RealT = eltype(x)
    DELTA = RealT(0.005)
    RHO_FLOOR = RealT(1e-8)
    a = RealT(-2.0)
    b = RealT(-1.0)
    c = RealT(1.0)
    d = RealT(2.0)
    gamma = equations.gamma

    heaviside_smooth(x) = RealT(0.5) * (one(RealT) + tanh(x / DELTA))

    X  = x[1] - a
    Y  = x[1] - b
    _X = x[1] - c
    _Y = x[1] - d

    ha = heaviside_smooth(X)
    hb = heaviside_smooth(Y)
    hc = heaviside_smooth(_X)
    hd = heaviside_smooth(_Y)

    if a <= x[1] <= b
        ρ = max(RHO_FLOOR, ha - hb)
    elseif c <= x[1] <= d
        ρ = max(RHO_FLOOR, hc - hd)
    else
        ρ = RHO_FLOOR
    end

    uab = -tanh(X / DELTA) * tanh(X / DELTA)
    uab += tanh(Y / DELTA) * tanh(Y / DELTA)
    uab = uab / (RealT(2) * DELTA)
    uab = uab * (-gamma) * ρ^(gamma - RealT(2))

    ucd = -tanh(_X / DELTA) * tanh(_X / DELTA)
    ucd += tanh(_Y / DELTA) * tanh(_Y / DELTA)
    ucd = ucd / (RealT(2) * DELTA)
    ucd = ucd * (-gamma) * ρ^(gamma - RealT(2))

    if a <= x[1] <= b
        u = uab
    elseif c <= x[1] <= d
        u = ucd
    else
        u = RealT(0)
    end
    mx = ρ * u
    return SVector(ρ, mx)
end

# --------------------------------------------------
# Initial condition: Sinusoidal
# --------------------------------------------------

"""
    initial_condition_sinosidal(x, t, equations::RelaxationEulerEquations1D)

A smooth sinusoidal perturbation:
    ρ = 1 + 0.2 * sin(8π * x)
    u = -0.2π * sin(8π * x)
"""
function initial_condition_sinosidal(x, t, equations::RelaxationEulerEquations1D)
    RealT = eltype(x)
    rho = one(RealT) + RealT(0.2) * sin(RealT(8) * π * x[1])
    u   = -RealT(0.2) * π * sin(RealT(8) * π * x[1])
    return SVector(rho, rho * u)
end

# --------------------------------------------------
# Initial condition: Sinusoidal Riemann
# --------------------------------------------------

"""
    initial_condition_sinosidal_riemann(x, t, equations::RelaxationEulerEquations1D)

Mixed Riemann + sinusoidal initial data.
    x ∈ [-5, -1): ρ = 2.0
    x ∈ [-1,  1): ρ = 0.5 * (3 + sin(3πx/2))
    x ∈ [ 1,  5]: ρ = 1.0
"""
function initial_condition_sinosidal_riemann(x, t, equations::RelaxationEulerEquations1D)
    RealT = eltype(x)
    if -5 <= x[1] < -1
        rho = RealT(2.0)
    elseif -1 <= x[1] < 1
        rho = RealT(0.5) * (RealT(3) + sin(RealT(3) * π * x[1] / RealT(2)))
    elseif 1 <= x[1] <= 5
        rho = RealT(1.0)
    else
        throw(DomainError(x[1], "x must be in the range [-5, 5]"))
    end
    return SVector(rho, zero(RealT))
end

# --------------------------------------------------
# Initial condition: Barenblatt (exact solution)
# --------------------------------------------------

"""
    initial_condition_barenblatt(x, t, equations::RelaxationEulerEquations1D)

Barenblatt exact solution for the porous medium equation at the given time t.
Uses Γ = 1.0.
"""
function initial_condition_barenblatt(x, t, equations::RelaxationEulerEquations1D)
    t == 0.0 && throw(ArgumentError(
        "Barenblatt initial condition is singular at t = 0. " *
        "Use a positive time (e.g., t = 0.001)."))

    RealT = eltype(x)
    RHO_FLOOR = RealT(1e-10)
    gamma = equations.gamma
    Γ = RealT(1.0)

    t_eff = Float64(t)
    β = 1.0 / (gamma + 1.0)
    ξ = x[1] / (t_eff^β)
    factor = (gamma - 1.0) / (2.0 * gamma * (gamma + 1.0))
    bracket = Γ - factor * (ξ^2)
    positive = max(bracket, zero(bracket))
    ρ = t_eff^(-β) * (positive^(1.0 / (gamma - 1.0)))
    ρ = max(ρ, RHO_FLOOR)

    if ρ > RHO_FLOOR
        u = β * x[1] / t_eff
        mx = ρ * u
    else
        mx = RealT(0)
    end
    return SVector(ρ, mx)
end


end # @muladd
