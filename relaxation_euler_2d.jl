using ADTypes
using LinearAlgebra
using NonlinearSolve
using Plots
using SparseArrays
using SparseMatrixColorings # for sparse AutoDiff to work (otherwise it will switch to dense AD)
using LinearSolve
# using KLU

linsolve = KLUFactorization()


# 2D relaxation Euler-like system on [0,1]x[0,1] with periodic BC:
#   rho_t + (mx)_x + (my)_y = 0
#   mx_t + (mx^2/rho + rho/eps)_x + (mx*my/rho)_y = -mx/eps
#   my_t + (mx*my/rho)_x + (my^2/rho + rho/eps)_y = -my/eps

struct RelaxationParams
    eps::Float64
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
end

# convert 2D (i,j) cell index to 1D index (flattening in column-major order)
@inline function cell_index(i, j, p::RelaxationParams)
    return (j - 1) * p.nx + i
end

# Rusanov flux for the 2D relaxation Euler system in the x direction
@inline function rusanov_flux_x(rho_l, mx_l, my_l, rho_r, mx_r, my_r, eps)
    ux_l = mx_l / rho_l
    ux_r = mx_r / rho_r

    fx_l_1 = mx_l
    fx_l_2 = mx_l * ux_l + rho_l / eps
    fx_l_3 = my_l * ux_l

    fx_r_1 = mx_r
    fx_r_2 = mx_r * ux_r + rho_r / eps
    fx_r_3 = my_r * ux_r

    c = sqrt(1 / eps)
    alpha = max(abs(ux_l) + c, abs(ux_r) + c)

    f1 = 0.5 * (fx_l_1 + fx_r_1) - 0.5 * alpha * (rho_r - rho_l)
    f2 = 0.5 * (fx_l_2 + fx_r_2) - 0.5 * alpha * (mx_r - mx_l)
    f3 = 0.5 * (fx_l_3 + fx_r_3) - 0.5 * alpha * (my_r - my_l)
    return f1, f2, f3
end

# Rusanov flux for the 2D relaxation Euler system in the y direction
@inline function rusanov_flux_y(rho_l, mx_l, my_l, rho_r, mx_r, my_r, eps)
    uy_l = my_l / rho_l
    uy_r = my_r / rho_r

    fy_l_1 = my_l
    fy_l_2 = mx_l * uy_l
    fy_l_3 = my_l * uy_l + rho_l / eps

    fy_r_1 = my_r
    fy_r_2 = mx_r * uy_r
    fy_r_3 = my_r * uy_r + rho_r / eps

    c = sqrt(1 / eps)
    alpha = max(abs(uy_l) + c, abs(uy_r) + c)

    f1 = 0.5 * (fy_l_1 + fy_r_1) - 0.5 * alpha * (rho_r - rho_l)
    f2 = 0.5 * (fy_l_2 + fy_r_2) - 0.5 * alpha * (mx_r - mx_l)
    f3 = 0.5 * (fy_l_3 + fy_r_3) - 0.5 * alpha * (my_r - my_l)
    return f1, f2, f3
end

# Implicit part for the 2D relaxation Euler system (all fluxes + stiff source terms)
function implicit_part!(du, u, p::RelaxationParams, t)
    ncells = p.nx * p.ny
    rho = @view u[1:ncells]
    mx = @view u[ncells + 1:2 * ncells]
    my = @view u[2 * ncells + 1:3 * ncells]

    drho = @view du[1:ncells]
    dmx = @view du[ncells + 1:2 * ncells]
    dmy = @view du[2 * ncells + 1:3 * ncells]

    fill!(drho, 0.0)
    fill!(dmx, 0.0)
    fill!(dmy, 0.0)

    for j in 1:p.ny
        for i in 1:p.nx
            # periodic bc indexing 
            i_right = i == p.nx ? 1 : i + 1
            l = cell_index(i, j, p)
            r = cell_index(i_right, j, p)

            f1, f2, f3 = rusanov_flux_x(rho[l], mx[l], my[l], rho[r], mx[r], my[r], p.eps)
            drho[l] -= f1 / p.dx
            dmx[l] -= f2 / p.dx
            dmy[l] -= f3 / p.dx
            drho[r] += f1 / p.dx
            dmx[r] += f2 / p.dx
            dmy[r] += f3 / p.dx
        end
    end

    for j in 1:p.ny
        j_top = j == p.ny ? 1 : j + 1
        for i in 1:p.nx
            b = cell_index(i, j, p)
            t = cell_index(i, j_top, p)

            f1, f2, f3 = rusanov_flux_y(rho[b], mx[b], my[b], rho[t], mx[t], my[t], p.eps)
            drho[b] -= f1 / p.dy
            dmx[b] -= f2 / p.dy
            dmy[b] -= f3 / p.dy
            drho[t] += f1 / p.dy
            dmx[t] += f2 / p.dy
            dmy[t] += f3 / p.dy
        end
    end

    return nothing
end

function build_jacobian_prototype(p::RelaxationParams)
    ncells = p.nx * p.ny
    # notice that jac_prototype is sparse already 
    jac_prototype = spzeros(Float64, 3 * ncells, 3 * ncells)

    for j in 1:p.ny
        for i in 1:p.nx
            # periodic bc indexing for neighbors
            cell = cell_index(i, j, p)
            left = cell_index(i == 1 ? p.nx : i - 1, j, p)
            right = cell_index(i == p.nx ? 1 : i + 1, j, p)
            bottom = cell_index(i, j == 1 ? p.ny : j - 1, p)
            top = cell_index(i, j == p.ny ? 1 : j + 1, p)

            for row_var in 0:2 # loop over rho, mx, my (3 variables) in rows
                row = row_var * ncells + cell # since u = [ρ, mx, my] stacked
                for neighbor in (cell, left, right, bottom, top)
                    for col_var in 0:2  # loop over rho, mx, my (3 variables) in cols
                        jac_prototype[row, col_var * ncells + neighbor] = 1.0
                    end
                end
            end
        end
    end

    return jac_prototype
end

mutable struct ImplicitStepData
    model::RelaxationParams
    dt::Float64
    t::Float64
    u_prev::Vector{Float64}
    rhs_cache::AbstractVector
end

# for stats collection during the solve
mutable struct RunStats
    total_time::Float64
    total_bytes::Int
    total_gctime::Float64
    step_times::Vector{Float64}
    step_bytes::Vector{Int}
    step_gctimes::Vector{Float64}
end

function RunStats(nsteps::Int)
    return RunStats(0.0, 0, 0.0, zeros(Float64, nsteps), zeros(Int, nsteps), zeros(Float64, nsteps))
end

function backward_euler_residual!(res, u, p::ImplicitStepData)
    if !(eltype(p.rhs_cache) === eltype(u) && length(p.rhs_cache) == length(u))
        p.rhs_cache = similar(u)
    end
    implicit_part!(p.rhs_cache, u, p.model, p.t)
    @. res = u - p.u_prev - p.dt * p.rhs_cache
    return nothing
end

function print_run_stats(label, stats::RunStats, nsteps_done::Int)
    if nsteps_done == 0
        println(label, " stats: no steps completed")
        return
    end

    step_times = view(stats.step_times, 1:nsteps_done)
    step_bytes = view(stats.step_bytes, 1:nsteps_done)
    step_gctimes = view(stats.step_gctimes, 1:nsteps_done)

    avg_step_time = sum(step_times) / nsteps_done
    avg_step_bytes = sum(step_bytes) / nsteps_done
    avg_step_gc = sum(step_gctimes) / nsteps_done

    println(label, " stats:")
    println("  total wall time = ", round(stats.total_time; digits = 6), " s")
    println("  total allocations = ", round(stats.total_bytes / 2^20; digits = 3), " MiB")
    println("  total GC time = ", round(stats.total_gctime; digits = 6), " s")
    println("  steps completed = ", nsteps_done)
    println("  first step time = ", round(step_times[1]; digits = 6), " s")
    println("  first step allocations = ", round(step_bytes[1] / 2^20; digits = 3), " MiB")
    println("  avg step time = ", round(avg_step_time; digits = 6), " s")
    println("  max step time = ", round(maximum(step_times); digits = 6), " s")
    println("  avg step allocations = ", round(avg_step_bytes / 2^20; digits = 3), " MiB")
    println("  max step allocations = ", round(maximum(step_bytes) / 2^20; digits = 3), " MiB")
    println("  avg step GC time = ", round(avg_step_gc; digits = 6), " s")
end

function initial_condition(x, y)
    # rho0 = 1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)
    # ux0 = 0.1 * cos(2 * pi * x)
    # uy0 = 0.1 * cos(2 * pi * y)
    r2 = x^2 + y^2
    rho0 = 1 - 0.25 * exp(2*(1-r2))
    ux0 = y * exp(1-r2)
    uy0 = -x * exp(1-r2)
    return rho0, rho0 * ux0, rho0 * uy0
end

function build_problem(; nx=32, ny=32, eps=0.05, tspan=(0.0, 0.05), xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)
    x = range(xmin, xmax; length=nx + 1)[1:end-1]
    y = range(ymin, ymax; length=ny + 1)[1:end-1]
    p = RelaxationParams(eps, nx, ny, (xmax - xmin) / nx, (ymax - ymin) / ny, xmin, xmax, ymin, ymax)

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

# dt is here in this function
function solve_backward_euler(u0, p::RelaxationParams, tspan, jac_prototype; dt=5.0e-2)

    nls_function = NonlinearFunction(backward_euler_residual!; jac_prototype = jac_prototype)
    nls_algorithm = NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 4), concrete_jac = true, linsolve = linsolve)

    u = copy(u0)
    step_data = ImplicitStepData(p, dt, tspan[1], copy(u0), similar(u0))
    nsteps_target = ceil(Int, (tspan[2] - tspan[1]) / dt)
    stats = RunStats(nsteps_target)

    t = tspan[1]
    nsteps_done = 0
    nonlinear_problem = NonlinearProblem(nls_function, copy(u), step_data)

    cache = init(
        nonlinear_problem,
        nls_algorithm;
        abstol = 1e-10,
        reltol = 1e-10
    )

    total_timed = @timed while t < tspan[2] - 10 * eps(tspan[2] + 1.0)
        dt_step = min(dt, tspan[2] - t)
        step_data.dt = dt_step
        step_data.t = t + dt_step
        step_data.u_prev .= u

        nsteps_done += 1
        step_timed = @timed begin
            cache.u .= u
            nonlinear_solution = solve!(cache)
            u .= nonlinear_solution.u
        end
        stats.step_times[nsteps_done] = step_timed.time
        stats.step_bytes[nsteps_done] = step_timed.bytes
        stats.step_gctimes[nsteps_done] = step_timed.gctime
        t += dt_step
        end
        println("Saved rho_final_2d.png, ux_final_2d.png, and uy_final_2d.png in the working directory.")
    stats.total_gctime = total_timed.gctime

    return u, stats, nsteps_done
end

tspan = (0.0, 0.05)
u0, x, y, p, jac_prototype = build_problem(; nx=32*2, ny=32*2, eps=0.05, tspan) # domain info can also be provided here
u_final, solve_stats, nsteps_done = solve_backward_euler(u0, p, tspan, jac_prototype; dt=p.dx)

ncells = p.nx * p.ny
rho_final = @view u_final[1:ncells]
mx_final = @view u_final[ncells + 1:2 * ncells]
my_final = @view u_final[2 * ncells + 1:3 * ncells]

ux_final = mx_final ./ rho_final
uy_final = my_final ./ rho_final

rho_grid = reshape(rho_final, p.nx, p.ny)
ux_grid = reshape(ux_final, p.nx, p.ny)
uy_grid = reshape(uy_final, p.nx, p.ny)

println("Solved 2D relaxation Euler system on [", p.xmin, ",", p.xmax, "]x[", p.ymin, ",", p.ymax, "] with eps = ", p.eps)
println("Final time = ", tspan[2])
println("Mean density = ", sum(rho_final) / ncells)
println("Mean ux = ", sum(ux_final) / ncells)
println("Mean uy = ", sum(uy_final) / ncells)

print_run_stats("Solve", solve_stats, nsteps_done)

rho_plot = heatmap(x, y, permutedims(rho_grid); xlabel="x", ylabel="y", title="Final density rho", aspect_ratio=:equal)
savefig(rho_plot, "rho_final_2d.png")

ux_plot = heatmap(x, y, permutedims(ux_grid); xlabel="x", ylabel="y", title="Final velocity ux", aspect_ratio=:equal)
savefig(ux_plot, "ux_final_2d.png")

uy_plot = heatmap(x, y, permutedims(uy_grid); xlabel="x", ylabel="y", title="Final velocity uy", aspect_ratio=:equal)
savefig(uy_plot, "uy_final_2d.png")

println("Saved rho_final_2d.png, ux_final_2d.png, and uy_final_2d.png in the working directory.")
