using Test
using EulerAP
using LinearAlgebra

@testset "Temporal Convergence (first-order)" begin
    function smooth_ic(x, y, t)
        rho = 1.0 + 0.1 * sin(2π * x) * sin(2π * y)
        mx = 0.05 * cos(2π * x)
        my = -0.05 * sin(2π * y)
        return rho, mx, my
    end

    size = (16, 16)
    dt_values = [0.04, 0.02, 0.01]
    t_final = 0.04
    reference_dt = 0.005
    errors = Float64[]

    u0, coords, p, cache = build_problem(ic_func = smooth_ic, size = size, left_bc = :periodic, right_bc = :periodic, bottom_bc = :periodic, top_bc = :periodic, tspan = (0.0, t_final))

    u_ref, stats_ref, nsteps_ref = solve_backward_euler(u0, p, (0.0, t_final), cache; dt = reference_dt, tol = 1e-8)

    for dt in dt_values
        u_final, stats, nsteps = solve_backward_euler(u0, p, (0.0, t_final), cache; dt = dt, tol = 1e-8)

        # Compare to a finer-step reference on the same grid so the test measures
        # backward Euler's first-order time accuracy rather than spatial refinement.
        err = norm(u_final .- u_ref) / sqrt(length(u0))
        push!(errors, err)
    end

    # compute observed order between successive grid refinements
    orders = Float64[]
    for i in 1:(length(errors)-1)
        r = errors[i] / errors[i+1]
        order = log(r)/log(2)
        push!(orders, order)
    end

    @test isempty(orders) || (sum(orders) / length(orders)) > 0.8
end

@testset "Spatial convergence (first-order)" begin
    function smooth_ic(x, y, t)
        rho = 1.0 + 0.1 * sin(2π * x) * sin(2π * y)
        mx = 0.05 * cos(2π * x)
        my = -0.05 * sin(2π * y)
        return rho, mx, my
    end

    function restrict_2x2(u_fine, p_fine, p_coarse)
        nc_fine = prod(p_fine.size)
        nc_coarse = prod(p_coarse.size)
        u_coarse = zeros(3 * nc_coarse)

        for jc in 1:p_coarse.size[2]
            for ic in 1:p_coarse.size[1]
                i_f = 2 * ic - 1
                j_f = 2 * jc - 1

                fine_cells = (
                    EulerAP.cell_index(i_f, j_f, p_fine),
                    EulerAP.cell_index(i_f + 1, j_f, p_fine),
                    EulerAP.cell_index(i_f, j_f + 1, p_fine),
                    EulerAP.cell_index(i_f + 1, j_f + 1, p_fine),
                )

                c = EulerAP.cell_index(ic, jc, p_coarse)
                for var in 0:2
                    sum_val = 0.0
                    for cell in fine_cells
                        sum_val += u_fine[var * nc_fine + cell]
                    end
                    u_coarse[var * nc_coarse + c] = sum_val / 4.0
                end
            end
        end

        return u_coarse
    end

    grid_sizes = [8, 16, 32]
    t_final = 0.01
    errors = Float64[]

    for nx in grid_sizes
        ny = nx
        nx_f = 2 * nx
        ny_f = 2 * ny

        u_c, _, _, p_c, cache_c = build_problem(ic_func = smooth_ic, size = (nx, ny), left_bc = :periodic, right_bc = :periodic, bottom_bc = :periodic, top_bc = :periodic, tspan = (0.0, t_final))
        u_f, _, _, p_f, cache_f = build_problem(ic_func = smooth_ic, size = (nx_f, ny_f), left_bc = :periodic, right_bc = :periodic, bottom_bc = :periodic, top_bc = :periodic, tspan = (0.0, t_final))

        dt_c = 0.05 * minimum(p_c.dx)^2
        dt_f = 0.05 * minimum(p_f.dx)^2

        u_c_final, stats_c, nsteps_c = solve_backward_euler(u_c, p_c, (0.0, t_final), cache_c; dt = dt_c, tol = 1e-8)
        u_f_final, stats_f, nsteps_f = solve_backward_euler(u_f, p_f, (0.0, t_final), cache_f; dt = dt_f, tol = 1e-8)

        u_f_restricted = restrict_2x2(u_f_final, p_f, p_c)
        err = norm(u_c_final .- u_f_restricted) / sqrt(length(u_c_final))
        push!(errors, err)
    end

    orders = Float64[]
    for i in 1:(length(errors)-1)
        push!(orders, log(errors[i] / errors[i+1]) / log(2))
    end

    @test isempty(orders) || (sum(orders) / length(orders)) > 0.8
end
