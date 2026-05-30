function build_problem(;
    ic_func,
    nx = 32,        # optional parameters when nothing is provided by the user
    ny = 32,
    eps = 0.05,
    xmin = -1.0,
    xmax = 1.0,
    ymin = -1.0,
    ymax = 1.0,
    left_bc = :periodic,
    right_bc = :periodic,
    bottom_bc = :periodic,
    top_bc = :periodic,
    bc_funcs = nothing,
    flux = :rusanov
)
    x = range(xmin, xmax; length = nx + 1)[1:end-1]
    y = range(ymin, ymax; length = ny + 1)[1:end-1]

    # Create boundary condition configuration
    bc_config = BCConfig(
        left = left_bc,
        right = right_bc,
        bottom = bottom_bc,
        top = top_bc,
        bc_funcs = bc_funcs
    )

    p = RelaxationParams(
        eps,
        nx,
        ny,
        (xmax - xmin) / nx,
        (ymax - ymin) / ny,
        xmin,
        xmax,
        ymin,
        ymax,
        bc_config
    )
    ncells = nx * ny
    u0 = zeros(3 * ncells)

    for j in 1:ny
        for i in 1:nx
            idx = cell_index(i, j, p)
            rho0, mx0, my0 = ic_func(x[i], y[j])

            u0[idx]              = rho0
            u0[ncells + idx]     = mx0
            u0[2 * ncells + idx] = my0
        end
    end
    return u0, x, y, p, build_jacobian_cache(p; flux = flux)
end