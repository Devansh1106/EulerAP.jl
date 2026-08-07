# # By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# # Since these FMAs can increase the performance of many numerical algorithms,
# # we need to opt-in explicitly.
# @muladd begin
# #! format: noindent

# """
#     assemble_nonlinear_residual!(F, phi, params)

# Assemble the nonlinear residual

#     c Δφ + f(φ) = rhs

# where

#     c = params.laplacian_coeff
# """
# function assemble_nonlinear_residual!(
#     F,
#     phi,
#     params::NewtonParameters,
#     semi,
# )

#     mesh = semi.mesh
#     equations = semi.equations_elliptic

#     rhs = params.rhs
#     coeff = params.laplacian_coeff
#     t = params.t

#     dx = mesh.dx[1]

#     alpha = coeff / dx^2

#     @inbounds for I in eachcell(mesh)

#         cell = cell_index(I, semi)

#         Im1 = CartesianIndex(I[1] - 1)
#         Ip1 = CartesianIndex(I[1] + 1)

#         phi_i = phi[cell]

#         phi_l = _elliptic_var(
#             phi,
#             Im1,
#             semi,
#             t,
#         )

#         phi_r = _elliptic_var(
#             phi,
#             Ip1,
#             semi,
#             t,
#         )

#         F[cell] = alpha *
#                   (-2 * phi_i + phi_l + phi_r) +
#                   elliptic_point_source(
#                       phi_i,
#                       equations,
#                   ) - rhs[cell]
#     end
#     return nothing
# end

# """
#     assemble_nonlinear_jacobian!(J, phi, params)

# Assemble the Jacobian corresponding to
# `assemble_nonlinear_residual!`.
# """
# function assemble_nonlinear_jacobian!(J,
#                                       phi,
#                                       params::NewtonParameters,
#                                       semi)

#     cache = semi.cache_elliptic

#     mesh = semi.mesh
#     equations = semi.equations_elliptic

#     coeff = params.laplacian_coeff

#     dx = mesh.dx[1]

#     alpha = coeff / dx^2

#     # Access Thomas algorithm arrays from elliptic cache
#     dl = cache.dl
#     d  = cache.d
#     du = cache.du

#     nx = ncells(mesh)
#     @inbounds begin
#         for i in 1:nx
#             d[i] = -2 * alpha +
#                    elliptic_point_source_derivative(
#                        phi[i],
#                        equations,
#                    )
#         end

#         for i in 2:nx
#             dl[i] = alpha
#         end

#         for i in 1:nx-1
#             du[i] = alpha
#         end
#     end
#     copyto!(J.d, d)
#     copyto!(J.dl, 1, dl, 2, nx - 1)
#     copyto!(J.du, 1, du, 1, nx - 1)
#     return nothing
# end

# function solve_newton!(phi,
#                        rhs,
#                        laplacian_coeff,
#                        semi,
#                        t)

#     cache = semi.cache_elliptic
#     newton_cache = cache.newton_cache

#     newton_cache.params.rhs = rhs
#     newton_cache.params.laplacian_coeff = laplacian_coeff
#     newton_cache.params.t = t

#     reinit!(
#         newton_cache.nonlinear_cache,
#         phi;
#         p = newton_cache.params,
#     )

#     sol = solve!(
#         newton_cache.nonlinear_cache,
#     )
#     copyto!(phi, sol.u)
#     return nothing
# end

# end # @muladd


# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

"""
    assemble_nonlinear_residual!(F, phi, params)

Assemble the nonlinear residual

    c Δφ + f(φ) - rhs = 0

where

    c = params.laplacian_coeff
"""
function assemble_nonlinear_residual!(
    F,
    phi,
    params::NewtonParameters,
    semi,
)

    mesh = semi.mesh
    equations = semi.equations_elliptic

    rhs = params.rhs
    coeff = params.laplacian_coeff
    t = params.t

    dx = mesh.dx[1]

    alpha = coeff / dx^2

    # IMEX per-cell correction: η dt² ρ̄ⁿ_{i±1/2} / dx². When `eta_dt2 == 0`
    # (e.g. the initial-condition solve, which never sets `params.eta`/`dt`),
    # the assembly reduces to the plain constant-coefficient Laplacian and we
    # skip the density lookups entirely (avoiding `gamma_mean(0, 0, γ) = NaN`).
    eta_dt2 = params.eta * params.dt^2
    rho = params.rho
    gamma = semi.equations.gamma
    apply_correction = eta_dt2 != 0

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

        if apply_correction
            rho_i = _hyperbolic_ghost_state(rho, I, semi, t)[1]
            rho_l = _hyperbolic_ghost_state(rho, Im1, semi, t)[1]
            rho_r = _hyperbolic_ghost_state(rho, Ip1, semi, t)[1]
            rh_l = gamma_mean(rho_l, rho_i, gamma)
            rh_r = gamma_mean(rho_i, rho_r, gamma)

            alpha_im1 = alpha - eta_dt2 * rh_l / dx^2
            alpha_ip1 = alpha - eta_dt2 * rh_r / dx^2
            alpha_i   = -2 * alpha + eta_dt2 * (rh_l + rh_r) / dx^2
        else
            alpha_im1 = alpha
            alpha_ip1 = alpha
            alpha_i   = -2 * alpha
        end

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

    coeff = params.laplacian_coeff
    t = params.t

    dx = mesh.dx[1]

    alpha = coeff / dx^2

    # IMEX per-cell correction: η dt² ρ̄ⁿ_{i±1/2} / dx². When `eta_dt2 == 0`
    # (e.g. the initial-condition solve, which never sets `params.eta`/`dt`),
    # the assembly reduces to the plain constant-coefficient Laplacian and we
    # skip the density lookups entirely (avoiding `gamma_mean(0, 0, γ) = NaN`).
    eta_dt2 = params.eta * params.dt^2
    rho = params.rho
    gamma = semi.equations.gamma
    apply_correction = eta_dt2 != 0

    nx = ncells(mesh)

    # Both sides are required to be PeriodicBC together (enforced by
    # `check_periodicity_mesh_boundary_conditions`), so checking one side
    # is sufficient.
    periodic = semi.boundary_conditions_elliptic.left isa PeriodicBC
    left_bc  = semi.boundary_conditions_elliptic.left
    right_bc = semi.boundary_conditions_elliptic.right

    fill!(J.nzval, zero(eltype(J)))
    
    @inbounds for i in 1:nx
        if apply_correction
            I = CartesianIndex(i)
            Im1 = CartesianIndex(i - 1)
            Ip1 = CartesianIndex(i + 1)

            rho_i = _hyperbolic_ghost_state(rho, I, semi, t)[1]
            rho_l = _hyperbolic_ghost_state(rho, Im1, semi, t)[1]
            rho_r = _hyperbolic_ghost_state(rho, Ip1, semi, t)[1]
            rh_l = gamma_mean(rho_l, rho_i, gamma)
            rh_r = gamma_mean(rho_i, rho_r, gamma)

            alpha_im1 = alpha - eta_dt2 * rh_l / dx^2
            alpha_ip1 = alpha - eta_dt2 * rh_r / dx^2
            alpha_i   = -2 * alpha + eta_dt2 * (rh_l + rh_r) / dx^2
        else
            alpha_im1 = alpha
            alpha_ip1 = alpha
            alpha_i   = -2 * alpha
        end

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