using ClimaTimeSteppers
using LinearAlgebra
using Plots
using SparseArrays

const CTS = ClimaTimeSteppers

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

# Explicit tendency for the 2D relaxation Euler system (non-stiff fluxes)
function explicit_tendency!(du, u, p::RelaxationParams, t)
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
            t_idx = cell_index(i, j_top, p)

            f1, f2, f3 = rusanov_flux_y(rho[b], mx[b], my[b], rho[t_idx], mx[t_idx], my[t_idx], p.eps)
            drho[b] -= f1 / p.dy
            dmx[b] -= f2 / p.dy
            dmy[b] -= f3 / p.dy
            drho[t_idx] += f1 / p.dy
            dmx[t_idx] += f2 / p.dy
            dmy[t_idx] += f3 / p.dy
        end
    end

    return nothing
end

# Implicit tendency for the 2D relaxation Euler system (stiff source terms)
function implicit_tendency!(du, u, p::RelaxationParams, t)
    ncells = p.nx * p.ny
    rho = @view u[1:ncells]
    mx = @view u[ncells + 1:2 * ncells]
    my = @view u[2 * ncells + 1:3 * ncells]

    drho = @view du[1:ncells]
    dmx = @view du[ncells + 1:2 * ncells]
    dmy = @view du[2 * ncells + 1:3 * ncells]

    fill!(drho, 0.0)
    @. dmx = -mx / p.eps
    @. dmy = -my / p.eps

    return nothing
end

function wfact!(w, u, p::RelaxationParams, dtgamma, t)
    ncells = p.nx * p.ny
    if typeof(w) <: SparseMatrixCSC
        fill!(w.nzval, 0.0)
    elseif hasproperty(w, :diag)
        fill!(w.diag, 0.0)
    end

    for i in 1:ncells
        w[i, i] = -1.0
    end

    stiff_diag = -(1.0 + dtgamma / p.eps)
    for i in (ncells + 1):(3 * ncells)
        w[i, i] = stiff_diag
    end

    return nothing
end

function initial_condition(x, y)
    rho0 = 1.0 + 0.2 * sin(2 * pi * x) * sin(2 * pi * y)
    ux0 = 0.1 * cos(2 * pi * x)
    uy0 = 0.1 * cos(2 * pi * y)
    return rho0, rho0 * ux0, rho0 * uy0
end

function build_problem(; nx=64, ny=64, eps=0.1, tspan=(0.0, 0.2))
    x = range(0.0, 1.0; length=nx + 1)[1:end-1]
    y = range(0.0, 1.0; length=ny + 1)[1:end-1]
    p = RelaxationParams(eps, nx, ny, 1.0 / nx, 1.0 / ny)

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

    jac_prototype = Diagonal(ones(3 * ncells))
    imp = CTS.ODEFunction(implicit_tendency!; jac_prototype=jac_prototype, Wfact=wfact!)
    f = CTS.ClimaODEFunction(; T_exp! = explicit_tendency!, T_imp! = imp)
    prob = CTS.ODEProblem(f, u0, tspan, p)

    return prob, x, y, p
end

prob, x, y, p = build_problem()
alg = CTS.IMEXAlgorithm(
    CTS.ARS343(),
    CTS.NewtonsMethod(; max_iters=4, update_j=CTS.UpdateEvery(CTS.NewTimeStep)),
)
sol = CTS.solve(prob, alg; dt=1.5e-3, saveat=[prob.tspan[2]])

u_final = sol.u[end]
ncells = p.nx * p.ny
rho_final = @view u_final[1:ncells]
mx_final = @view u_final[ncells + 1:2 * ncells]
my_final = @view u_final[2 * ncells + 1:3 * ncells]

ux_final = mx_final ./ rho_final
uy_final = my_final ./ rho_final

rho_grid = reshape(rho_final, p.nx, p.ny)
ux_grid = reshape(ux_final, p.nx, p.ny)
uy_grid = reshape(uy_final, p.nx, p.ny)

println("Solved 2D relaxation Euler system on [0,1]x[0,1] with eps = ", p.eps)
println("Final time = ", prob.tspan[2])
println("Mean density = ", sum(rho_final) / ncells)
println("Mean ux = ", sum(ux_final) / ncells)
println("Mean uy = ", sum(uy_final) / ncells)

rho_plot = heatmap(x, y, permutedims(rho_grid); xlabel="x", ylabel="y", title="Final density rho", aspect_ratio=:equal)
savefig(rho_plot, "rho_final_2d.png")

ux_plot = heatmap(x, y, permutedims(ux_grid); xlabel="x", ylabel="y", title="Final velocity ux", aspect_ratio=:equal)
savefig(ux_plot, "ux_final_2d.png")

uy_plot = heatmap(x, y, permutedims(uy_grid); xlabel="x", ylabel="y", title="Final velocity uy", aspect_ratio=:equal)
savefig(uy_plot, "uy_final_2d.png")

println("Saved rho_final_2d.png, ux_final_2d.png, and uy_final_2d.png in the working directory.")
