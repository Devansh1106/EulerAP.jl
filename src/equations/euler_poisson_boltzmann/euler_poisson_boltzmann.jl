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
# where α = (λ² + ηΔt^2)/Δx² is handled by the IMEX solver.
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

@inline initial_laplacian_coefficient(equations::AbstractEllipticEquations) = -(equations.lambda^2)


# TODO: Need to change EulerPressureLess1D to a common type for Hyper+elliptic system. This is a temporary setup.



# -------------------------------------------------------------------------
#                              Soliton Test Case
# -------------------------------------------------------------------------
# soliton_profile.jl — precompute the Sagdeev-potential soliton profile

sagdeev_rhs(phi, u0) = (1 + 2 * phi/u0^2)^(-0.5) - exp(-phi)

"""
Shoot the ODE φ'' = sagdeev_rhs(φ, u0) from φ(0)=0, φ'(0)=η,
find the peak (φ'=0), and truncate the domain at L = 2*x_peak
so the profile is (numerically) periodic on [0, L].
Returns (L, xs, phis) on a fine grid of spacing dx.
"""
function solve_soliton_profile(u0, eta; n=100000)
    xmax = 50.0
    dx = xmax / n
    xs   = zeros(n)
    phis = zeros(n)
    phi, dphi = 0.0, eta
    imin, phimin = 1, phi
    for i in 1:n
        xs[i] = (i - 1) * dx
        phis[i] = phi
        if phi < phimin
            phimin, imin = phi, i
        end
        # RK4 step for [phi, dphi]
        deriv(p, dp) = (dp, sagdeev_rhs(p, u0))
        k1 = deriv(phi, dphi)
        k2 = deriv(phi + dx/2*k1[1], dphi + dx/2*k1[2])
        k3 = deriv(phi + dx/2*k2[1], dphi + dx/2*k2[2])
        k4 = deriv(phi + dx*k3[1],   dphi + dx*k3[2])
        phi  += dx/6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
        dphi += dx/6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
    end
    L = 2 * xs[imin]
    iL = 2 * imin - 1
    if iL > n
        iL = n
        L = xs[iL]
    end
    return L, xs[1:iL], phis[1:iL]
end

"""
Given the phi-profile, build (rho, u_lab) arrays via (5.2) and the
lab-frame velocity formula u_lab = u0 + u0/n_s(x) (n0 = 1).
"""
function soliton_density_velocity(phis, u0)
    ns = @. (1 + 2phis/u0^2)^(-0.5)
    u_lab = @. u0 + u0 / ns
    return ns, u_lab
end

# Periodic linear interpolation onto the fine (xs, ys) table
function interp_periodic(x, L, xs, ys)
    xm = mod(x, L)
    dx = xs[2] - xs[1]
    i = clamp(Int(floor(xm / dx)) + 1, 1, length(xs) - 1)
    t = (xm - xs[i]) / dx
    return (1 - t) * ys[i] + t * ys[i + 1]
end

function make_initial_condition_soliton(u0, eta)
    L, xs, phis = solve_soliton_profile(u0, eta)
    ns, u_lab = soliton_density_velocity(phis, u0)
    ic = @inline function (x, t, equations)
        RealT = eltype(x)
        rho = interp_periodic(x[1], L, xs, ns)
        u   = interp_periodic(x[1], L, xs, u_lab)
        return SVector(RealT(rho), RealT(rho * u))
    end
    return ic, L
end

end # @muladd
