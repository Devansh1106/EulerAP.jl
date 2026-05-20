function initial_condition(x, y)

    r2 = x^2 + y^2

    rho0 = 1 - 0.25 * exp(2 * (1 - r2))

    ux0 = y * exp(1 - r2)
    uy0 = -x * exp(1 - r2)

    return rho0, rho0 * ux0, rho0 * uy0
end

function build_problem(;
    nx = 32,        # optional parameters when nothing is provided by the user
    ny = 32,
    eps = 0.05,
    xmin = -1.0,
    xmax = 1.0,
    ymin = -1.0,
    ymax = 1.0
)

    x = range(xmin, xmax; length = nx + 1)[1:end-1]
    y = range(ymin, ymax; length = ny + 1)[1:end-1]

    p = RelaxationParams(
        eps,
        nx,
        ny,
        (xmax - xmin) / nx,
        (ymax - ymin) / ny,
        xmin,
        xmax,
        ymin,
        ymax
    )

    ncells = nx * ny

    u0 = zeros(3 * ncells)

    for j in 1:ny
        for i in 1:nx

            idx = cell_index(i, j, p)

            rho0, mx0, my0 = initial_condition(x[i], y[j])

            u0[idx] = rho0
            u0[ncells + idx] = mx0
            u0[2 * ncells + idx] = my0
        end
    end

    return u0, x, y, p, build_jacobian_prototype(p)
end