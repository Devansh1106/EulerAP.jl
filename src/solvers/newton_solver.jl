# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

"""
    fill_face_densities_averaged!(rh, u, semi, gamma, t)
    fill_face_densities_reconstructed!(rh, cache, semi, gamma, t)

Fill `rh[1:nx+1]` with the γ-mean density at every interface: `rh[k]` is the
face to the LEFT of cell `k` (so `rh[nx+1]` is the face to the right of cell
`nx`). One `gamma_mean` evaluation per face — each is shared by cell `k - 1`'s
`alpha_ip1` and cell `k`'s `alpha_im1`.

These densities are the `ρ̄ⁿ_{i±1/2}` of the `η dt² ρ̄ⁿ_{i±1/2}` correction that
`update_correction_coefficients!` folds into the Laplacian stencil. That
correction is the implicit half of the semi-implicit density flux, so each
face's density here **must be the same ρ̄ that the flux assembly uses at that
same face**. Only then does the elliptic equation the Newton solve inverts
hold exactly for the ρ the flux actually produces, i.e.

    -λ² Δφⁿ⁺¹ + e^{φⁿ⁺¹} = ρⁿ⁺¹        (to Newton tolerance)

which is the asymptotic-preserving property the semi-implicit coupling is
there for. Measured on the smooth sech test at nx = 160, the residual of that
identity after a full step is ~3e-15 with matching ρ̄ and ~1e-7 when the two
disagree.

Hence one function per scheme, differing only in where the face density comes
from. They are separate names rather than two methods of one name because they
take the same number of arguments and the second-order IMEX cache type is not
yet defined this early in the include order, so neither arity nor dispatch can
tell them apart.

  * `..._reconstructed!` (second order). The flux assembly reconstructs both
    sides of the face before taking the γ-mean (see
    `calculate_semi_implicit_density_flux_diff_stage2`/`_stage3`), so this
    calls the same `reconstructed_rho_vel_at` to do it: cell `i` contributes
    its right edge — `:left` of the face — and cell `i + 1` its left edge,
    `:right`. Reading the cell *averages* here instead left the elliptic
    coefficients out of step with that reconstruction. Both are second-order
    approximations of the face density, so the mismatch is only O(Δx²) —
    `≈ Δx² ρ''/4`, since the averaged pair overshoots the face value by
    `Δx² ρ''/8` and the reconstructed pair undershoots it by the same — and
    the measured EOC on the smooth test is unchanged. What it buys is the
    exact identity above rather than a nearby one.

  * `..._averaged!` (first order, and the initial-condition solve). Nothing is
    reconstructed anywhere, so the face density is the γ-mean of the two cell
    averages, exactly as `calculate_explicit_density_flux_diff!` takes it.

The ghost cells at the two ends are addressed differently in the two,
deliberately. The reconstructed form indexes ghosts as `0` and `nx + 1`
directly, as `reconstruct_slopes!` and the second-order flux assembly do,
because `neighbor_index` clamps back into the domain for
`NeumannBC`/`ExtrapolateBC` and would pair an interior cell's slope with a
ghost face. The averaged form keeps the `neighbor_index` lookup, matching the
first-order flux assembly, for which the clamp *is* the ghost state.
"""
@inline function fill_face_densities_averaged!(rh, u, semi, gamma, t)

    nx = ncells(semi.mesh)

    rho_at(I) = cell_state(u, I, semi, t)[1]

    @inbounds for i in 1:(nx - 1)
        rh[i + 1] = gamma_mean(rho_at(CartesianIndex(i)),
                               rho_at(CartesianIndex(i + 1)), gamma)
    end

    rh[1] = gamma_mean(rho_at(neighbor_index(CartesianIndex(1), semi, 1, -1)),
                       rho_at(CartesianIndex(1)), gamma)
    rh[nx + 1] = gamma_mean(rho_at(CartesianIndex(nx)),
                            rho_at(neighbor_index(CartesianIndex(nx), semi, 1, 1)), gamma)

    return nothing
end

@inline function fill_face_densities_reconstructed!(rh, cache, semi, gamma, t)

    nx = ncells(semi.mesh)

    # Right edge of cell I (the value on the left side of I's right face).
    rho_l_at(I) = reconstructed_rho_vel_at(cache, semi, I, :left, t)[1]
    # Left edge of cell I (the value on the right side of I's left face).
    rho_r_at(I) = reconstructed_rho_vel_at(cache, semi, I, :right, t)[1]

    @inbounds for i in 1:(nx - 1)
        rh[i + 1] = gamma_mean(rho_l_at(CartesianIndex(i)),
                               rho_r_at(CartesianIndex(i + 1)), gamma)
    end

    rh[1] = gamma_mean(rho_l_at(CartesianIndex(0)),
                       rho_r_at(CartesianIndex(1)), gamma)
    rh[nx + 1] = gamma_mean(rho_l_at(CartesianIndex(nx)),
                            rho_r_at(CartesianIndex(nx + 1)), gamma)

    return nothing
end

"""
    update_correction_coefficients!(cache, semi, params)

Precompute the per-cell Laplacian coefficients

    alpha_im1[i] = alpha - eta_dt2 * rh_left(i)  / dx^2
    alpha_ip1[i] = alpha - eta_dt2 * rh_right(i) / dx^2

used by both `assemble_nonlinear_residual!` and `assemble_nonlinear_jacobian!`.

These depend only on `params.u`, `params.eta`, `params.dt`, and
`params.laplacian_coeff` — none of which change across the Newton
iterations of a single elliptic solve (only `phi` does). Calling this once
per `solve_newton!`, rather than recomputing `gamma_mean` and the density
ghost lookups on every residual/Jacobian assembly, removes that
Newton-iteration-count multiplier entirely. It also evaluates each
interface's `gamma_mean` exactly once (previously twice: once as the right
neighbor of cell i, once as the left neighbor of cell i+1).

Note `alpha_i` is not stored: `alpha_i = -2*alpha + eta_dt2*(rh_l+rh_r)/dx^2
= -(alpha_im1 + alpha_ip1)` always, so callers derive it inline.

The face densities `rh` come from `fill_face_densities_reconstructed!` when
the second-order scheme has supplied its cache in `params.reconstruction_cache`
and from `fill_face_densities_averaged!` otherwise; see those for why each
must match the flux assembly of its scheme.
"""
function update_correction_coefficients!(cache, semi, params::NewtonParameters)
    mesh  = semi.mesh
    nx    = ncells(mesh)
    dx    = mesh.dx[1]
    alpha = params.laplacian_coeff / dx^2

    eta_dt2 = params.eta * params.dt^2

    if eta_dt2 == 0
        fill!(cache.alpha_im1, alpha)
        fill!(cache.alpha_ip1, alpha)
        return nothing
    end

    u        = params.u
    gamma    = semi.equations.gamma
    t        = params.t

    # rh[k] = gamma-mean density at the face to the LEFT of cell k, k=1..nx;
    # rh[nx+1] = face to the right of cell nx. One evaluation per interface.
    rh = Vector{eltype(cache.alpha_im1)}(undef, nx + 1)

    reconstruction_cache = params.reconstruction_cache

    if reconstruction_cache === nothing
        fill_face_densities_averaged!(rh, u, semi, gamma, t)
    else
        fill_face_densities_reconstructed!(rh, reconstruction_cache, semi, gamma, t)
    end

    @inbounds for i in 1:nx
        cache.alpha_im1[i] = alpha - eta_dt2 * rh[i] / dx^2
        cache.alpha_ip1[i] = alpha - eta_dt2 * rh[i + 1] / dx^2
    end

    return nothing
end

"""
    assemble_nonlinear_residual!(F, phi, params)

Assemble the nonlinear residual

    c Δφ + f(φ) - rhs = 0

where

    c = params.laplacian_coeff

Reads the Newton-invariant `alpha_im1`/`alpha_ip1` coefficients from
`semi.cache_elliptic`; see `update_correction_coefficients!`.
"""
function assemble_nonlinear_residual!(
    F,
    phi,
    params::NewtonParameters,
    semi,
)

    mesh = semi.mesh
    equations = semi.equations_elliptic
    cache = semi.cache_elliptic

    rhs = params.rhs
    t = params.t

    @inbounds for I in eachcell(mesh)

        cell = cell_index(I, semi)

        Im1 = CartesianIndex(I[1] - 1)
        Ip1 = CartesianIndex(I[1] + 1)

        phi_i = phi[cell]

        phi_l = _elliptic_var(
            phi,
            semi,
            Im1,
            t,
        )

        phi_r = _elliptic_var(
            phi,
            semi,
            Ip1,
            t,
        )

        alpha_im1 = cache.alpha_im1[cell]
        alpha_ip1 = cache.alpha_ip1[cell]
        alpha_i   = -(alpha_im1 + alpha_ip1)

        F[cell] = alpha_im1 * phi_l +
                  alpha_i * phi_i +
                  alpha_ip1 * phi_r +
                  elliptic_point_source(
                      phi_i,
                      equations,
                  ) - rhs[cell]
    end
    return nothing
end

"""
    assemble_nonlinear_jacobian!(J, phi, params)

Assemble the Jacobian corresponding to `assemble_nonlinear_residual!`.

`J` is expected to be a `SparseMatrixCSC` whose sparsity pattern already
includes the periodic "corner" entries `(1, nx)` and `(nx, 1)` (see
`create_elliptic_cache`). We write directly into `J` via `setindex!`
rather than going through a plain tridiagonal Thomas-algorithm buffer,
since `LinearAlgebra.Tridiagonal` cannot represent those corner entries
and silently assembling only the interior bands there produced a Jacobian
that was inconsistent with the (correctly periodic) residual — this was
the source of the Newton solver degrading/diverging as N grew, since the
missing coupling terms scale like `alpha = coeff/dx^2 ~ O(N^2)`.
"""
function assemble_nonlinear_jacobian!(J,
                                      phi,
                                      params::NewtonParameters,
                                      semi)

    mesh = semi.mesh
    equations = semi.equations_elliptic
    cache = semi.cache_elliptic

    nx = ncells(mesh)

    # Both sides are required to be PeriodicBC together (enforced by
    # `check_periodicity_mesh_boundary_conditions`), so checking one side
    # is sufficient. NOTE: this is the *elliptic* field's periodicity
    # (boundary_conditions_elliptic), distinct from the hyperbolic
    # boundary_conditions used for the ρ-ghost lookups inside
    # `update_correction_coefficients!` — the two can differ per example.
    periodic = semi.boundary_conditions_elliptic.left isa PeriodicBC
    left_bc  = semi.boundary_conditions_elliptic.left
    right_bc = semi.boundary_conditions_elliptic.right

    fill!(J.nzval, zero(eltype(J)))

    @inbounds for i in 1:nx
        alpha_im1 = cache.alpha_im1[i]
        alpha_ip1 = cache.alpha_ip1[i]
        alpha_i   = -(alpha_im1 + alpha_ip1)

        diag = alpha_i + elliptic_point_source_derivative(phi[i], equations)

        if i > 1
            J[i, i - 1] = alpha_im1
        elseif periodic && nx > 2
            J[i, nx] = alpha_im1
        elseif !(left_bc isa DirichletBC)
            # Neumann/Extrapolate ghost = phi[1] + const  =>  d(ghost)/d(phi[1]) = 1
            diag += alpha_im1
        end

        if i < nx
            J[i, i + 1] = alpha_ip1
        elseif periodic && nx > 2
            J[i, 1] = alpha_ip1
        elseif !(right_bc isa DirichletBC)
            diag += alpha_ip1
        end

        J[i, i] = diag
    end

    return nothing
end

function solve_newton!(phi,
                       rhs,
                       laplacian_coeff,
                       semi,
                       t)

    cache = semi.cache_elliptic
    newton_cache = cache.newton_cache

    newton_cache.params.rhs = rhs
    newton_cache.params.laplacian_coeff = laplacian_coeff
    newton_cache.params.t = t

    update_correction_coefficients!(cache, semi, newton_cache.params)

    reinit!(
        newton_cache.nonlinear_cache,
        phi;
        p = newton_cache.params,
    )

    sol = solve!(
        newton_cache.nonlinear_cache,
    )
    copyto!(phi, sol.u)
    return nothing
end

end # @muladd