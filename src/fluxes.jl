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
                              gamma::Float64 = 1.4) where {N, TL, TR}
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

# Input resolution using Multiple Dispatch
# Case A: User passes a built-in shortcut name (e.g., :rusanov or "rusanov")
"""
    resolve_flux(flux_name; gamma=1.4)

Resolve a flux specification to a `FluxPair`. Supported inputs are the symbol
`:rusanov`, the string `"rusanov"`, a two-function tuple, or an existing
`FluxPair`.
"""
function resolve_flux(flux_name::Symbol; gamma::Float64 = 1.4)
    if flux_name === :rusanov
        return FluxPair((u_l, u_r, orientation, eps) -> 
                        rusanov_flux(u_l, 
                                     u_r, 
                                     orientation, 
                                     eps; 
                                     gamma = gamma))

    # elseif flux_name === :lax_friedrichs
    #     return FluxPair((u_l, u_r, orientation, eps) -> 
    #                     lax_friedrichs_flux(u_l, 
    #                                         u_r, 
    #                                         orientation, 
    #                                         eps; 
    #                                         gamma = gamma))

    else
        error("Unknown flux name ':$flux_name'. Supported built-ins are :rusanov, :lax_friedrichs")
    end
end

# Support strings by converting them to a symbol
resolve_flux(flux_name::AbstractString; gamma::Float64 = 1.4) = 
             resolve_flux(Symbol(flux_name); gamma = gamma)

# Case B: User passes a raw 2-tuple of custom functions (e.g., (my_fx, my_fy))
"""
    resolve_flux((flux_x, flux_y); gamma=1.4)

Wrap a pair of callables as a `FluxPair`.
"""
function resolve_flux(flux_tuple::Tuple{F1, F2}; 
                      gamma::Float64 = 1.4) where {F1, F2}

    return FluxPair((u_l, u_r, orientation, eps) -> begin
        if orientation == 1
            return flux_tuple[1](u_l, u_r, eps)
        else
            return flux_tuple[2](u_l, u_r, eps)
        end
    end)
end

# Case C: Already a FluxPair - return as-is
# called in operators.jl in implicit_part!
resolve_flux(flux_pair::FluxPair) = flux_pair