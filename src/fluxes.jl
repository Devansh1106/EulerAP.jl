@inline function rusanov_flux_x(
    rho_l, mx_l, my_l,
    rho_r, mx_r, my_r,
    eps)

    ux_l = mx_l / rho_l
    ux_r = mx_r / rho_r
    # rho_face = _safe_face_density(rho_l, rho_r)

    # ux_l = mx_l / rho_face
    # ux_r = mx_r / rho_face

    fx_l_1 = mx_l
    fx_l_2 = mx_l * ux_l + rho_l / eps
    fx_l_3 = my_l * ux_l

    fx_r_1 = mx_r
    fx_r_2 = mx_r * ux_r + rho_r / eps
    fx_r_3 = my_r * ux_r

    c = sqrt(1 / eps)

    alpha = max(abs(ux_l) + c, abs(ux_r) + c)
    # alpha = _smooth_max(_smooth_abs(ux_l, FLUX_REG) + c, _smooth_abs(ux_r, FLUX_REG) + c, FLUX_REG)


    f1 = 0.5 * (fx_l_1 + fx_r_1) - 0.5 * alpha * (rho_r - rho_l)
    f2 = 0.5 * (fx_l_2 + fx_r_2) - 0.5 * alpha * (mx_r - mx_l)
    f3 = 0.5 * (fx_l_3 + fx_r_3) - 0.5 * alpha * (my_r - my_l)

    return f1, f2, f3
end

@inline function rusanov_flux_y(
    rho_l, mx_l, my_l,
    rho_r, mx_r, my_r,
    eps)

    uy_l = my_l / rho_l
    uy_r = my_r / rho_r
    # rho_face = _safe_face_density(rho_l, rho_r)

    # uy_l = my_l / rho_face
    # uy_r = my_r / rho_face

    fy_l_1 = my_l
    fy_l_2 = mx_l * uy_l
    fy_l_3 = my_l * uy_l + rho_l / eps

    fy_r_1 = my_r
    fy_r_2 = mx_r * uy_r
    fy_r_3 = my_r * uy_r + rho_r / eps

    c = sqrt(1 / eps)

    alpha = max(abs(uy_l) + c, abs(uy_r) + c)
    # alpha = _smooth_max(_smooth_abs(uy_l, FLUX_REG) + c, _smooth_abs(uy_r, FLUX_REG) + c, FLUX_REG)

    f1 = 0.5 * (fy_l_1 + fy_r_1) - 0.5 * alpha * (rho_r - rho_l)
    f2 = 0.5 * (fy_l_2 + fy_r_2) - 0.5 * alpha * (mx_r - mx_l)
    f3 = 0.5 * (fy_l_3 + fy_r_3) - 0.5 * alpha * (my_r - my_l)

    return f1, f2, f3
end

# Register built-in fluxes in a type-stable dictionary
const BUILTIN_FLUXES = Dict{Symbol, FluxPair}(
    :rusanov => FluxPair(rusanov_flux_x, rusanov_flux_y)
    # Easily add future additions here: :lax_friedrichs => FluxPair(...)
)

# Clean input resolution using Multiple Dispatch
# Case A: User passes a built-in shortcut name (e.g., :rusanov or "rusanov")
function resolve_flux(flux_name::Symbol)
    if haskey(BUILTIN_FLUXES, flux_name)
        return BUILTIN_FLUXES[flux_name]
    else
        error("Unknown flux name ':$flux_name'. Supported built-ins are: $(keys(BUILTIN_FLUXES))")
    end
end

# Support strings transparently by converting them to a symbol
resolve_flux(flux_name::AbstractString) = resolve_flux(Symbol(flux_name))

# Case B: User passes a raw 2-tuple of custom functions (e.g., (my_fx, my_fy))
function resolve_flux(flux_tuple::Tuple{Any, Any})
    return FluxPair(flux_tuple[1], flux_tuple[2])
end

# Case C: Already a FluxPair - return as-is
# called in operators.jl in implicit_part!
resolve_flux(flux_pair::FluxPair) = flux_pair