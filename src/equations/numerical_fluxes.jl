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
    (u_ll, u_rr, orientation, equations::AbstractEquations)

Rusanov numerical flux for a face normal to the `orientation`-th axis.
"""
@inline function (FluxRusanov::FluxRusanov)(u_ll, u_rr, orientation, equations::AbstractEquations)

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


end # @muladd