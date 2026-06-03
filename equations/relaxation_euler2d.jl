using EulerAP
using Plots

tspan = (0.0, 8.0)
tol = 1e-8

# Boundary condition choices: :periodic, :dirichlet, :neumann
# Set per-side BCs here 
# left_bc   = :neumann
# right_bc  = :neumann
# bottom_bc = :neumann
# top_bc    = :neumann
# left_bc   = :periodic
# right_bc  = :periodic
# bottom_bc = :periodic
# top_bc    = :periodic
left_bc   = :dirichlet
right_bc  = :dirichlet
bottom_bc = :dirichlet
top_bc    = :dirichlet

# homogeneous_dirichlet(x, y, t) = (0.0, 0.0, 0.0)
# homogeneous_dirichlet_left(x, y, t) = (1.0, 0.0, 0.0)
# homogeneous_dirichlet_right(x, y, t) = (0.025, 0.0, 0.0)

# bc_funcs = Dict(
#     :left => homogeneous_neumann,
#     :right => homogeneous_neumann,
#     :bottom => homogeneous_neumann,
#     :top => homogeneous_neumann
# )

# Neumann BC callbacks: return normal derivatives (d_rho/dn, d_mx/dn, d_my/dn)
# left_neumann(x, y, t)   = (0.0, 0.0, 0.0)
# right_neumann(x, y, t)  = (0.0, 0.0, 0.0)
# bottom_neumann(x, y, t) = (0.0, 0.0, 0.0)
# top_neumann(x, y, t)    = (0.0, 0.0, 0.0)

# bc_funcs = Dict(
#     :left => left_neumann,
#     :right => right_neumann,
#     :bottom => bottom_neumann,
#     :top => top_neumann
# )

# Initial condition (smooth elliptical profile)
# Domain: x in [-10, 10], y in [-10, 10]
# Final time suggestions from the image: 4, 8, 16
const a = 5 / 2
const b = 2 / 5
const c = 1 / 10

function initial_condition_smooth(x, y)
    profile = sqrt(a * x^2 + b * y^2)
    rho0 = 1 + 0.25 * (1 - tanh((profile - 1) / c))
    return rho0, 0.0, 0.0
end

# Initial condition (Riemann / shock-tube style)
# Domain: x in [0, 1.6], y in [0, 0.1]
# Initial density: rho_L for x <= x_m, rho_R for x > x_m
# Velocity: zero everywhere
const rho_L = 1.0
# Set rho_R to one of {0.5, 0.1, 0.025} for different test cases
const rho_R = 0.025
const x_m = 0.8

function initial_condition_riemann(x, y)
    rho0 = x <= x_m ? rho_L : rho_R
    return rho0, 0.0, 0.0
end

function bc_match_ic(x, y, t)
    return initial_condition_smooth(x,y)
end

bc_funcs = Dict(
    :left => bc_match_ic,
    :right => bc_match_ic,
    :bottom => bc_match_ic,
    :top => bc_match_ic
)

# Radial initial condition
# function initial_condition(x, y)
#     r2 = x^2 + y^2

#     rho0 = 1 - 0.1 * exp(2 * (1 - r2)) # initially it was 0.25 instead of 0.1, giving -ve density
#     ux0 = y * exp(1 - r2)
#     uy0 = -x * exp(1 - r2)
#     return rho0, rho0 * ux0, rho0 * uy0
# end

u0, x, y, p, jac_cache = build_problem(
    nx = 512,
    ny = 512,
    eps = 0.05,
    left_bc = left_bc,
    right_bc = right_bc,
    bottom_bc = bottom_bc,
    top_bc = top_bc,
    bc_funcs = bc_funcs,
    ic_func = initial_condition_smooth,
    flux = :rusanov,
    xmin = -10.0,
    xmax = 10.0,
    ymin = -10.0,
    ymax = 10.0
    # xmin = 0.0,
    # xmax = 1.6,
    # ymin = 0.0,
    # ymax = 0.1
)

u_final, solve_stats, nsteps_done =
    solve_backward_euler(
        u0,
        p,
        tspan,
        jac_cache;
        dt = p.dx,
        tol = tol,
        flux = :rusanov
    )

ncells = p.nx * p.ny

rho_final = @view u_final[1:ncells]
mx_final  = @view u_final[ncells + 1:2 * ncells]
my_final  = @view u_final[2 * ncells + 1:3 * ncells]

ux_final = mx_final ./ rho_final
uy_final = my_final ./ rho_final

rho_grid = reshape(rho_final, p.nx, p.ny)
ux_grid  = reshape(ux_final, p.nx, p.ny)
uy_grid  = reshape(uy_final, p.nx, p.ny)

println("Solved 2D relaxation Euler system (MKL)")
println("Final time = ", tspan[2])

println("Mean density = ", sum(rho_final) / ncells)
println("Mean ux = ", sum(ux_final) / ncells)
println("Mean uy = ", sum(uy_final) / ncells)

print_run_stats("Solve", solve_stats, nsteps_done, p)
n_threads = get(ENV, "MKL_NUM_THREADS", string(Threads.nthreads()))

rho_plot = heatmap(
    x,
    y,
    permutedims(rho_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final density rho",
    aspect_ratio = :equal
)

savefig(rho_plot, "plots/rho_final_2d_$(p.nx)_$(n_threads).png")

# # Extract a 1D slice along the middle of the y-axis
# mid_y_index = div(p.ny, 2)
# rho_1d = rho_grid[:, mid_y_index]

# # Create a standard line plot
# rho_1d_plot = plot(
#     x, 
#     rho_1d;
#     xlabel = "x",
#     ylabel = "Density (rho)",
#     title = "1D Density Profile at t = $(tspan[2])",
#     linewidth = 2,
#     legend = false
# )

# savefig(rho_1d_plot, "plots/rho_1d_profile_$(p.nx).png")
# println("Saved rho_1d_profile.png")

ux_plot = heatmap(
    x,
    y,
    permutedims(ux_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final velocity ux",
    aspect_ratio = :equal
)

savefig(ux_plot, "plots/ux_final_2d_$(p.nx)_$(n_threads).png")

uy_plot = heatmap(
    x,
    y,
    permutedims(uy_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final velocity uy",
    aspect_ratio = :equal
)

savefig(uy_plot, "plots/uy_final_2d_$(p.nx)_$(n_threads).png")

println("Saved rho_final_2d.png")
println("Saved ux_final_2d.png")
println("Saved uy_final_2d.png")
