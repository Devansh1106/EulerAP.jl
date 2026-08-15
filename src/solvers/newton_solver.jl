# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

"""
    update_correction_coefficients!(cache, semi, params)

Precompute the per-cell Laplacian coefficients

    alpha_im1[i] = alpha - eta_dt2 * rh_left(i)  / dx^2
    alpha_ip1[i] = alpha - eta_dt2 * rh_right(i) / dx^2

used by both `assemble_nonlinear_residual!` and `assemble_nonlinear_jacobian!`.

These depend only on `params.rho`, `params.eta`, `params.dt`, and
`params.laplacian_coeff` — none of which change across the Newton
iterations of a single elliptic solve (only `phi` does). Calling this once
per `solve_newton!`, rather than recomputing `gamma_mean` and the density
ghost lookups on every residual/Jacobian assembly, removes that
Newton-iteration-count multiplier entirely. It also evaluates each
interface's `gamma_mean` exactly once (previously twice: once as the right
neighbor of cell i, once as the left neighbor of cell i+1).

Note `alpha_i` is not stored: `alpha_i = -2*alpha + eta_dt2*(rh_l+rh_r)/dx^2
= -(alpha_im1 + alpha_ip1)` always, so callers derive it inline.
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

    rho      = params.rho
    gamma    = semi.equations.gamma
    t        = params.t
    periodic = semi.boundary_conditions.left isa PeriodicBC

    rho_at(I) = _hyperbolic_ghost_state(rho, I, semi, t)[1]

    # rh[k] = gamma-mean density at the face to the LEFT of cell k, k=1..nx;
    # rh[nx+1] = face to the right of cell nx. One evaluation per interface.
    rh = Vector{eltype(cache.alpha_im1)}(undef, nx + 1)

    @inbounds for i in 1:(nx - 1)
        rh[i + 1] = gamma_mean(rho_at(CartesianIndex(i)), rho_at(CartesianIndex(i + 1)), gamma)
    end

    if periodic
        wrap = gamma_mean(rho_at(CartesianIndex(nx)), rho_at(CartesianIndex(1)), gamma)
        rh[1] = wrap
        rh[nx + 1] = wrap
    else
        rh[1] = gamma_mean(rho_at(neighbor_index(CartesianIndex(1), semi, 1, -1)),
                           rho_at(CartesianIndex(1)), gamma)
        rh[nx + 1] = gamma_mean(rho_at(CartesianIndex(nx)),
                                rho_at(neighbor_index(CartesianIndex(nx), semi, 1, 1)), gamma)
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
            Im1,
            semi,
            t,
        )

        phi_r = _elliptic_var(
            phi,
            Ip1,
            semi,
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