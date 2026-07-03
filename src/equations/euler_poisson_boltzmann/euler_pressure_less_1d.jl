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
    γ = promote(gamma)
    return EulerPressureLess1D{typeof{γ}}(γ)
end

@inline function flux(u::SVector{3}, orientation, 
                      equations::EulerPressureLess1D)

    rho = u[1]
    m   = u[2]

    v = m / rho

    return SVector(m, m*v)
end

# --------------------------------------------------
# Source term
# --------------------------------------------------

@inline function source_terms(u, equations::EulerPressureLess1D)
    # ρ_t += 0
    # m_t += 
    return SVector(zero(u[1]), -u[2] * equations.inv_epsilon)
end

end # @muladd