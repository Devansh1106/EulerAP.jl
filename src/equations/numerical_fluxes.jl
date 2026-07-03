# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains general numerical fluxes that are not specific to certain equations

# Add more flux types as they are added into the code
struct FluxRusanov end

"""
    FluxEnergyStable(eta)

Energy-stable numerical flux for the relaxation Euler system.

# Arguments
- `eta`: Weight for the `eta * delta_t` diffusion term. Must be passed by the user.
"""
struct FluxEnergyStable{RealT}
    eta::RealT
end

"""
    (u_ll, u_rr, orientation, equations, dt, dx=nothing)

Rusanov numerical flux for a face normal to the `orientation`-th axis.
`dt` and `dx` are accepted for interface compatibility with `FluxEnergyStable` but ignored.
"""
@inline function (flux_::FluxRusanov)(u_ll, u_rr, orientation, equations::AbstractEquations, dt, dx=nothing)

    # Defined in speific equations/ files
    flux_ll = flux(u_ll,
                   orientation,
                   equations)

    flux_rr = flux(u_rr,
                   orientation,
                   equations)

    λ = max(max_abs_speed(u_ll, orientation, equations),
            max_abs_speed(u_rr, orientation, equations))

    return 0.5 * (flux_ll + flux_rr - λ * (u_rr - u_ll))
end


"""
    (u_ll, u_rr, orientation, equations, dt, dx)

Energy-stable numerical flux for the relaxation Euler system, normal to the
`orientation`-th axis. The states are `SVector`s with layout `(rho, m_1, ..., m_NDIMS)`.
`dt` is the time-step size used to compute the diffusion coefficient `eta * dt`.
`dx` is accepted for interface compatibility with other flux types but ignored.
"""
@inline function (flux_::FluxEnergyStable)(u_ll, u_rr, orientation, equations::AbstractEquations, dt, dx=nothing)
    # Extract equation parameters
    gamma = equations.gamma
    eps   = equations.epsilon
    eta_diff_t = flux_.eta * dt

    # Left / right states
    rho_l = u_ll[1]
    rho_r = u_rr[1]
    P_l = rho_l^gamma
    P_r = rho_r^gamma

    # Standard arithmetic mean {{ϱ}}
    avg = 0.5 * (rho_l + rho_r)

    # Auxiliary variables
    f = (rho_r - rho_l) / (rho_r + rho_l)
    ν = f * f

    if ν < 1e-8
        # Taylor expansion
        c1 = (gamma - 2.0) / 3.0
        c2 = -(gamma + 1.0) * (gamma - 2.0) * (gamma - 3.0) / 45.0
        c3 = (gamma + 1.0) * (gamma - 2.0) * (gamma - 3.0) * (2.0 * gamma * (gamma - 2.0) - 9.0) / 945.0
        rho_half = avg * (1.0 + ν * (c1 + ν * (c2 + ν * c3)))
    else
        denom = rho_r^(gamma - 1.0) - rho_l^(gamma - 1.0)
        rho_half = (gamma - 1.0) / gamma * (rho_r^gamma - rho_l^gamma) / denom
    end

    # Normal velocities
    vel_l = u_ll[1 + orientation] / rho_l
    vel_r = u_rr[1 + orientation] / rho_r

    # Density flux
    F_rho = rho_half * 0.5 * (vel_l + vel_r) - (eta_diff_t / eps) * (P_r - P_l) - ((rho_r - rho_l) / eps)

    # Upwind splitting
    Fp = max(F_rho, 0.0)
    Fm = max(-F_rho, 0.0)

    # Assemble flux vector (dimension-agnostic)
    N = length(u_ll)
    components = ntuple(N) do k
        if k == 1
            return F_rho
        elseif k == 1 + orientation
            # Normal momentum flux
            return Fp * vel_l + Fm * vel_r - (u_rr[k] - u_ll[k]) + (P_r - P_l) / eps
        else
            # Transverse momentum flux
            v_l = u_ll[k] / rho_l
            v_r = u_rr[k] / rho_r
            return Fp * v_l + Fm * v_r - (u_rr[k] - u_ll[k])
        end
    end

    return SVector{N}(components)
end

# --------------------------------------------------
# Energy-stable flux for Euler-Poisson-Boltzmann
# --------------------------------------------------
# State vector layout: (rho, m_1, ..., m_NDIMS, Phi)
# where Phi is the electric potential (last component).
#
# Density flux:
#   F^1_{i+1/2} = rho_half * (vel_r - vel_l) / 2
#               - (Phi_r - Phi_l) * eta * dt / dx
#
# Normal momentum flux:
#   F^2_{i+1/2} = vel_l * (F^1)^+ + vel_r * (F^1)^-
#
# where (·)^+ = max(·, 0), (·)^- = max(-·, 0), vel = m/rho.
@inline function (flux_::FluxEnergyStable)(u_ll, u_rr, orientation,
                                           equations::AbstractEulerPoissonBoltzmannEquations,
                                           dt, dx)

    gamma = equations.gamma
    eta_diff_t = flux_.eta * dt

    # Left / right density
    rho_l = u_ll[1]
    rho_r = u_rr[1]

    # Normal velocity
    vel_l = u_ll[1 + orientation] / rho_l
    vel_r = u_rr[1 + orientation] / rho_r

    # Electric potential (last component of state)
    Phi_l = u_ll[end]
    Phi_r = u_rr[end]

    # --------------------------------------------------
    # rho_half = (gamma - 1)/gamma * (rho_r^gamma - rho_l^gamma)
    #           / (rho_r^(gamma-1) - rho_l^(gamma-1))
    # --------------------------------------------------
    P_l = rho_l^gamma
    P_r = rho_r^gamma
    avg = 0.5 * (rho_l + rho_r)

    f = (rho_r - rho_l) / (rho_r + rho_l)
    ν = f * f

    if ν < 1e-8
        # Taylor expansion to avoid near-zero denominator
        c1 = (gamma - 2.0) / 3.0
        c2 = -(gamma + 1.0) * (gamma - 2.0) * (gamma - 3.0) / 45.0
        c3 = (gamma + 1.0) * (gamma - 2.0) * (gamma - 3.0) * (2.0 * gamma * (gamma - 2.0) - 9.0) / 945.0
        rho_half = avg * (1.0 + ν * (c1 + ν * (c2 + ν * c3)))
    else
        denom = rho_r^(gamma - 1.0) - rho_l^(gamma - 1.0)
        rho_half = (gamma - 1.0) / gamma * (rho_r^gamma - rho_l^gamma) / denom
    end

    # --------------------------------------------------
    # First flux (density)
    # --------------------------------------------------
    F_rho = rho_half * 0.5 * (vel_r - vel_l) - (Phi_r - Phi_l) * eta_diff_t / dx

    # --------------------------------------------------
    # Upwind splitting for momentum flux
    # --------------------------------------------------
    Fp = max(F_rho, 0.0)
    Fm = max(-F_rho, 0.0)

    # --------------------------------------------------
    # Assemble full flux vector
    # --------------------------------------------------
    # The Phi component (last component) is not advected by the hyperbolic flux;
    # we pass through the state value so the FV divergence produces zero net change
    # for Phi from the explicit hyperbolic step. Phi is updated by the implicit
    # solver separately.
    N = length(u_ll)
    components = ntuple(N) do k
        if k == 1
            return F_rho
        elseif k == 1 + orientation
            # Normal momentum flux
            return Fp * vel_l + Fm * vel_r
        else
            # Transverse momentum or Phi: pass-through (no explicit hyperbolic update)
            return u_ll[k]
        end
    end

    return SVector{N}(components)
end


end # @muladd