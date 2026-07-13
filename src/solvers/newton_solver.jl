# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

"""
    assemble_nonlinear_residual!(F, phi, params)

Assemble the nonlinear residual

    c Δφ + f(φ) = rhs

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

        F[cell] = alpha *
                  (-2 * phi_i + phi_l + phi_r) +
                  elliptic_point_source(
                      phi_i,
                      equations,
                  ) - rhs[cell]
    end
    return nothing
end

"""
    assemble_nonlinear_jacobian!(J, phi, params)

Assemble the Jacobian corresponding to
`assemble_nonlinear_residual!`.
"""
function assemble_nonlinear_jacobian!(J,
                                      phi,
                                      params::NewtonParameters,
                                      semi)

    cache = semi.cache_elliptic

    mesh = semi.mesh
    equations = semi.equations_elliptic

    coeff = params.laplacian_coeff

    dx = mesh.dx[1]

    alpha = coeff / dx^2

    # Access Thomas algorithm arrays from elliptic cache
    dl = cache.dl
    d  = cache.d
    du = cache.du

    nx = ncells(mesh)
    @inbounds begin
        for i in 1:nx
            d[i] = -2 * alpha +
                   elliptic_point_source_derivative(
                       phi[i],
                       equations,
                   )
        end

        for i in 2:nx
            dl[i] = alpha
        end

        for i in 1:nx-1
            du[i] = alpha
        end
    end
    copyto!(J.d, d)
    copyto!(J.dl, 1, dl, 2, nx - 1)
    copyto!(J.du, 1, du, 1, nx - 1)
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