"""
    build_problem(; kwargs...)

Construct the initial state, grid coordinates, model parameters, and Jacobian
cache.
"""
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

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Generate arrays representing the cell centers
    x = range(xmin + dx/2, xmax - dx/2; length = nx)
    y = range(ymin + dy/2, ymax - dy/2; length = ny)

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