# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

# ============================================================================
# Slope limiters
# ============================================================================
#
# Every limiter is called as `limiter(a, b)` on the two *one-sided slopes* of
# the cell, already divided by Δx (see `reconstruct_slopes!`):
#
#     a = (Uᵢ - Uᵢ₋₁) / Δx        b = (Uᵢ₊₁ - Uᵢ) / Δx
#
# so the central difference is available as `(a + b) / 2`:
#
#     (a + b) / 2 = (Uᵢ₊₁ - Uᵢ₋₁) / (2Δx)
#
# and returns the limited slope of the cell. Limiters that need a parameter
# (θ, ε) are callable structs, so they can be built in an example file and
# handed to `solve`/`convergence_test` as `limiter = ...` like any function.
# ============================================================================

"""
    minmod(a, b)
    minmod(a, b, c)

Minmod limiter: the argument of smallest magnitude if all arguments share a
sign, zero otherwise. TVD, and the default limiter of the second-order IMEX
scheme.
"""
@inline function minmod(a, b)
    if a * b <= 0
        return zero(a)
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

# Three-argument minmod, by nesting: if any pair disagrees in sign the inner
# call returns zero and `minmod(0, c)` keeps it zero, which is exactly the
# "all three must share a sign" definition.
@inline minmod(a, b, c) = minmod(minmod(a, b), c)


"""
    nolimiter(a, b)

Unlimited **central-difference** slope, i.e. no limiting at all:

    (a + b) / 2 = (Uᵢ₊₁ - Uᵢ₋₁) / (2Δx)

Pass this instead of [`minmod`](@ref) when measuring the experimental order of
convergence on a *smooth* solution: `minmod` clips the slope at smooth extrema
and pulls the observed order back towards 1, hiding the formal second order of
the scheme. It is not TVD — it oscillates near discontinuities and can lose
positivity of ρ — so use it only for smooth convergence tests.
"""
@inline nolimiter(a, b) = 0.5 * (a + b)   # == (Uᵢ₊₁ - Uᵢ₋₁) / (2Δx)


"""
    minmod_theta(a, b, theta)

Generalized (Kurganov–Tadmor) minmod slope, `theta ∈ [1, 2]`:

    MM( θ (Uᵢ - Uᵢ₋₁)/Δx , (Uᵢ₊₁ - Uᵢ₋₁)/(2Δx) , θ (Uᵢ₊₁ - Uᵢ)/Δx )

θ scales only the two one-sided differences; the central difference is left
bare so the limiter stays *consistent*: on smooth data all three arguments are
≈ uₓ, the bare central term is the smallest in magnitude for θ ≥ 1, and minmod
returns it unchanged — recovering [`nolimiter`](@ref)'s second-order slope.
θ then controls only how much steepening is tolerated near a discontinuity:
θ = 1 is the most diffusive (classical minmod), θ = 2 the most compressive
still-TVD choice (the "monotonized central" limiter).

Scaling the central term by θ as well would return `θ uₓ` on smooth data, i.e.
inflate every slope by θ and destroy consistency.

See [`MinmodTheta`](@ref) for the callable form used as a `limiter` argument.
"""
@inline function minmod_theta(a, b, theta)

    central = 0.5 * (a + b)   # == (Uᵢ₊₁ - Uᵢ₋₁) / (2Δx)

    return minmod(theta * a, central, theta * b)
end


"""
    MinmodTheta(theta = 2.0)

Callable [`minmod_theta`](@ref) limiter carrying its θ, e.g.

    convergence_test(...; limiter = MinmodTheta(1.5))
"""
struct MinmodTheta{T}
    theta::T
end

MinmodTheta() = MinmodTheta(2.0)

@inline function (limiter::MinmodTheta)(a, b)
    return minmod_theta(a, b, limiter.theta)
end


"""
    cweno(a, b, epsilon)

CWENO-type nonlinear average of the two one-sided slopes,

    (φ(a) a + φ(b) b) / (φ(a) + φ(b)),    φ(s) = (ε + s²)⁻²

The weights are largest where the slope is smallest, so the smoother of the
two one-sided slopes dominates: near a discontinuity the steep side is
suppressed (without the hard clipping of `minmod`), while on smooth data
`a ≈ b` gives equal weights and hence the central difference, so second-order
accuracy is retained.

Evaluated in the equivalent reciprocal-free form

    (a (ε + b²)² + b (ε + a²)²) / ((ε + b²)² + (ε + a²)²)

obtained by multiplying numerator and denominator by (ε + a²)²(ε + b²)². This
is algebraically identical but avoids forming φ itself, which is O(ε⁻²) = O(10¹²)
for the default ε = 10⁻⁶ and small slopes.

See [`CWENO`](@ref) for the callable form used as a `limiter` argument.
"""
@inline function cweno(a, b, epsilon)

    # φ(a) = inv(A²), φ(b) = inv(B²) — never formed explicitly, see docstring.
    A = epsilon + a * a
    B = epsilon + b * b

    A2 = A * A
    B2 = B * B

    return (a * B2 + b * A2) / (B2 + A2)
end


"""
    CWENO(epsilon = 1e-6)

Callable [`cweno`](@ref) limiter carrying its ε, e.g.

    convergence_test(...; limiter = CWENO())
"""
struct CWENO{T}
    epsilon::T
end

CWENO() = CWENO(1e-6)

@inline function (limiter::CWENO)(a, b)
    return cweno(a, b, limiter.epsilon)
end

end # @muladd
