using StaticArrays

"""
    _state_flux(u, orientation, eps, gamma)

Compute the analytical physical flux vector of the relaxation system in a specific spatial direction.

This function evaluates the directional physical flux components required by the Riemann 
solver (e.g., Rusanov). It operates in a dimension-agnostic manner by leveraging the 
compile-time length `N` of the static state vector `u` and aligning the calculations 
with the specified coordinate axis.

# Mathematical Formulation
For a given coordinate axis aligned with `orientation` (where \$n\$ is the normal axis 
and \$t\$ represents any transverse axes), the flux vector corresponding to the state 
\$\\mathbf{u} = (\\rho, m_n, m_t)^T\$ is given by:

    f(\\mathbf{u}) = [ m_n,  m_n * v_n + p/eps,  m_t * v_n ]^T

where:
- \$\\rho\$ is the fluid density.
- \$m_n\$ is the momentum component normal to the interface.
- \$v_n = m_n / \\rho\$ is the normal fluid velocity.
- \$p = \\rho^\\gamma\$ is the polytropic pressure law.
- \\epsilon (`eps`) is the scaling/tuning parameter of the relaxation system.

# Arguments
- `u::SVector{N, T}`: The state vector with layout `(rho, m_1, ..., m_NDIMS)`.
- `orientation::Int`: The spatial axis index along which to calculate the flux 
  (e.g., 1 for the x-axis, 2 for the y-axis).
- `eps::Float64`: The relaxation parameter (\$1/\\epsilon\$ scales the pressure term).
- `gamma::Float64`: The adiabatic exponent used in the pressure law (\$p = \\rho^\\gamma\$).

# Returns
- `NTuple{N, T}`: A compile-time unrolled tuple containing the evaluated flux components 
  ordered identically to the state vector layout.
"""
# orientation = 1 for the x-axis
#               2 for the y-axis
@inline function _state_flux(u::SVector{N, T}, 
                             orientation::Int,
                             eps::Float64, 
                             gamma::Float64) where {N, T}
    rho             = u[1]
    normal_momentum = u[1 + orientation]
    normal_velocity = normal_momentum / rho
    # Match the pressure law p = rho^gamma used in rusanov_flux
    p = rho^gamma 

    return ntuple(N) do k
        if k == 1
            return normal_momentum
        elseif k == 1 + orientation
            return normal_momentum * normal_velocity + (p / eps)
        else
            return u[k] * normal_velocity
        end
    end
end

"""
    rusanov_flux(u_l, u_r, orientation, eps; gamma=1.4)

Rusanov numerical flux for a face normal to the `orientation`-th axis.
The states are `SVector`s with layout `(rho, m_1, ..., m_NDIMS)`.
"""
@inline function rusanov_flux(u_l::SVector{N, TL}, 
                              u_r::SVector{N, TR}, 
                              orientation::Int, 
                              eps; 
                              gamma::Float64) where {N, TL, TR}
    rho_l = u_l[1]
    rho_r = u_r[1]

    if rho_l <= 0.0 || rho_r <= 0.0
        error("Negative or zero density: rho_l=$rho_l, rho_r=$rho_r")
    end

    normal_l = u_l[1 + orientation]
    normal_r = u_r[1 + orientation]

    vel_l = normal_l / rho_l
    vel_r = normal_r / rho_r

    alpha_l = sqrt(gamma * rho_l^(gamma - 1))
    alpha_r = sqrt(gamma * rho_r^(gamma - 1))

    c = sqrt(1 / eps)
    lambda = max(abs(vel_l) + alpha_l * c, abs(vel_r) + alpha_r * c)

    flux_l = _state_flux(u_l, orientation, eps, gamma)
    flux_r = _state_flux(u_r, orientation, eps, gamma)

    return SVector{N}(ntuple(k -> 
                             0.5 * (flux_l[k] + flux_r[k]) - 0.5 * lambda * (u_r[k] - u_l[k]), 
                             N))
end

@inline function rusanov_flux(u_l, 
                              u_r, 
                              orientation::Int, 
                              eps; 
                              gamma::Float64 = 1.4)

    return rusanov_flux(SVector(u_l), SVector(u_r), orientation, eps; gamma = gamma)
end

# =============================================================================
# Energy-Stable Flux
# =============================================================================
"""
    energy_stable_flux(u_l, u_r, orientation, eps; gamma=1.4, eta_diff_t=0.0)

Energy-stable numerical flux for the relaxation Euler system, normal to the
`orientation`-th axis. The states are `SVector`s with layout `(rho, m_1, ..., m_NDIMS)`.

# Mathematical Formulation

Density component uses a gamma-mean interface density:

    ρ_{1/2} = (γ-1)/γ · (P_r^γ - P_l^γ) / (P_r^{γ-1} - P_l^{γ-1})

with P = ρ^γ, and the flux is:

    F^ρ_{1/2} = ρ_{1/2} · (u_l + u_r)/2  -  (η·Δt/ε + 1) · (P_r - P_l)

Normal momentum flux is upwinded via sign-split of F^ρ:

    F^{ρu}_{1/2} = (F^ρ)^+ · u_l  +  (F^ρ)^- · u_r  -  ((ρu)_r - (ρu)_l)  +  (P_r - P_l)/ε

Transverse momentum flux follows the same upwind pattern without the pressure term:

    F^{ρv}_{1/2} = (F^ρ)^+ · v_l  +  (F^ρ)^- · v_r  -  ((ρv)_r - (ρv)_l)

# Keyword Arguments
- `gamma::Float64`: Adiabatic exponent (default 1.4).
- `eta_diff_t::Float64`: Numerical diffusion coefficient η·Δt in the density flux
  pressure-diffusion term (default 0.0).
"""
@inline function energy_stable_flux(u_l::SVector{N, TL}, 
                                     u_r::SVector{N, TR}, 
                                     orientation::Int, 
                                     eps; 
                                     gamma::Float64,
                                     eta_diff_t::Float64) where {N, TL, TR}
    # --- Left / right state extraction ---
    rho_l = u_l[1]
    rho_r = u_r[1]

    P_l = rho_l^gamma
    P_r = rho_r^gamma
    γ = gamma

    # --- Gamma-mean interface density ---
    #   ρ_{1/2} = (γ-1)/γ · (P_r^γ - P_l^γ) / (P_r^{γ-1} - P_l^{γ-1})
    #   Falls back to arithmetic mean when pressures are nearly equal.
    # denom = P_r^(gamma - 1.0) - P_l^(gamma - 1.0)
    # if abs(denom) > 1e-15 * max(P_l^(gamma - 1.0), P_r^(gamma - 1.0), 1e-15)
    #     rho_half = (gamma - 1.0) / gamma * (P_r^gamma - P_l^gamma) / denom
    # else
    #     @warn "Falling back to arithmetic mean (denom is too small in γ-mean)"
    #     rho_half = 0.5 * (rho_l + rho_r)
    # end

    # Standard arithmetic mean {{ϱ}}
    avg = 0.5 * (rho_l + rho_r)
    
    # Auxiliary variables from Eq. (A.2)
    f = (rho_r - rho_l) / (rho_r + rho_l)
    ν = f * f
    
    if ν < 1e-8
        @warn "Falling to Taylor expansion"
        # Pre-calculate constant polynomial coefficients based on γ
        # term1: (γ - 2) / 3
        c1 = (γ - 2.0) / 3.0
        
        # term2: - (γ + 1)*(γ - 2)*(γ - 3) / 45
        c2 = - (γ + 1.0) * (γ - 2.0) * (γ - 3.0) / 45.0
        
        # term3: (γ + 1)*(γ - 2)*(γ - 3)*(2γ*(γ - 2) - 9) / 945
        c3 = (γ + 1.0) * (γ - 2.0) * (γ - 3.0) * (2.0 * γ * (γ - 2.0) - 9.0) / 945.0
        
        # Efficient polynomial evaluation using Horner's method: 1 + ν*(c1 + ν*(c2 + ν*c3))
        rho_half = avg * (1.0 + ν * (c1 + ν * (c2 + ν * c3)))
    else
        denom = rho_r^(gamma - 1.0) - rho_l^(gamma - 1.0)
        rho_half = (gamma - 1.0) / gamma * (rho_r^gamma - rho_l^gamma) / denom
    end



    # --- Normal velocities ---
    vel_l = u_l[1 + orientation] / rho_l
    vel_r = u_r[1 + orientation] / rho_r

    # --- Density flux ---
    #   F^ρ = ρ_{1/2} · (u_l + u_r)/2  -  (η·Δt/ε + 1) · (P_r - P_l)
    # F_rho = rho_half * 0.5 * (vel_l + vel_r) - (eta_diff_t / eps + 1.0) * (P_r - P_l)
    F_rho = rho_half * 0.5 * (vel_l + vel_r) - (eta_diff_t / eps) * (P_r - P_l) - ((rho_r - rho_l) / eps)

    # --- Upwind splitting ---
    Fp = max(F_rho, 0.0)
    Fm = max(-F_rho, 0.0)

    # --- Assemble flux vector (dimension-agnostic) ---
    components = ntuple(N) do k
        if k == 1
            return F_rho
        elseif k == 1 + orientation
            # Normal momentum flux
            return Fp * vel_l + Fm * vel_r - (u_r[k] - u_l[k]) + (P_r - P_l) / eps
        else
            # Transverse momentum flux
            v_l = u_l[k] / rho_l
            v_r = u_r[k] / rho_r
            return Fp * v_l + Fm * v_r - (u_r[k] - u_l[k])
        end
    end

    return SVector{N}(components)
end

@inline function energy_stable_flux(u_l, 
                                    u_r, 
                                    orientation::Int, 
                                    eps; 
                                    gamma::Float64,
                                    eta_diff_t::Float64)

    return energy_stable_flux(SVector(u_l), SVector(u_r), orientation, eps;
                              gamma = gamma, eta_diff_t = eta_diff_t)
end

# =============================================================================
# Flux Resolution
# =============================================================================
# Case A: User passes a built-in shortcut name (e.g., :rusanov or "rusanov")
"""
    resolve_flux(flux_name; gamma, eta=nothing)

Resolve a flux specification to a `FluxPair`. Supported inputs are the symbol
`:rusanov`, the string `"rusanov"`, or a two-function tuple.

The returned `FluxPair` has a mutable `dt` field (`Ref{Float64}`) that the solver
updates at each time step. For `:energy_stable`, the flux computes
`eta_diff_t = eta * dt` at call time.
"""
function resolve_flux(flux_name::Symbol; gamma = nothing, eta = nothing)
    if flux_name === :rusanov
        if gamma === nothing
            error(":rusanov requires `gamma` keyword argument")
        end
        return FluxPair((u_l, u_r, orientation, eps) -> 
                        rusanov_flux(u_l, 
                                     u_r, 
                                     orientation, 
                                     eps; 
                                     gamma = gamma),
                                     Ref(0.0))

    elseif flux_name === :energy_stable
        if gamma === nothing
            error(":energy_stable requires `gamma` keyword argument")
        end
        if eta === nothing
            error(":energy_stable requires `eta` keyword argument")
        end
        dt_ref = Ref(0.0)
        return FluxPair((u_l, u_r, orientation, eps) -> 
                        energy_stable_flux(u_l, 
                                           u_r, 
                                           orientation, 
                                           eps; 
                                           gamma = gamma,
                                           eta_diff_t = eta * dt_ref[]),
                                           dt_ref)

    else
        error("Unknown flux name ':$flux_name'. Supported built-ins are :rusanov, :energy_stable")
    end
end

# Support strings by converting them to a symbol
resolve_flux(flux_name::AbstractString; kwargs...) = 
             resolve_flux(Symbol(flux_name); kwargs...)

# Case B: User passes a raw 2-tuple of custom functions (e.g., (my_fx, my_fy))
"""
    resolve_flux((flux_x, flux_y))

Wrap a pair of callables as a `FluxPair`. These are the user defined flux functions hence dependence in γ and eta_diff_t etc. must be defined by the user in their functions.
"""
function resolve_flux(flux_tuple::Tuple{F1, F2}) where {F1, F2}

    return FluxPair((u_l, u_r, orientation, eps) -> begin
        if orientation == 1
            return flux_tuple[1](u_l, u_r, eps)
        else
            return flux_tuple[2](u_l, u_r, eps)
        end
    end,
    Ref(0.0))
end

# Allow calling a FluxPair directly: fp(u_l, u_r, orientation, eps)
@inline (fp::FluxPair)(u_l, u_r, orientation, eps) = fp.flux(u_l, u_r, orientation, eps)
