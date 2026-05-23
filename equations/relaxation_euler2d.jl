using EulerAP
using Plots

tspan = (0.0, 0.5)
tol = 1e-8

# Boundary condition choices: :periodic, :dirichlet, :neumann
# Set per-side BCs here (can change before running)
left_bc   = :periodic
right_bc  = :periodic
bottom_bc = :periodic
top_bc    = :periodic

u0, x, y, p, jac_prototype = build_problem(
    nx = 16,
    ny = 16,
    eps = 0.05,
    left_bc = left_bc,
    right_bc = right_bc,
    bottom_bc = bottom_bc,
    top_bc = top_bc
)
u0, x, y, p, jac_prototype = build_problem(
    nx = 128,
    ny = 128,
    eps = 0.05,
    left_bc = left_bc,
    right_bc = right_bc,
    bottom_bc = bottom_bc,
    top_bc = top_bc
)

u_final, solve_stats, nsteps_done =
    solve_backward_euler(
        u0,
        p,
        tspan,
        jac_prototype;
        dt = p.dx,
        tol = tol,
        jacobian_builder = assemble_global_jacobian,
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

println("Solved 2D relaxation Euler system")
println("Final time = ", tspan[2])

println("Mean density = ", sum(rho_final) / ncells)
println("Mean ux = ", sum(ux_final) / ncells)
println("Mean uy = ", sum(uy_final) / ncells)

print_run_stats("Solve", solve_stats, nsteps_done, p)

rho_plot = heatmap(
    x,
    y,
    permutedims(rho_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final density rho",
    aspect_ratio = :equal
)

savefig(rho_plot, "plots/rho_final_2d.png")

ux_plot = heatmap(
    x,
    y,
    permutedims(ux_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final velocity ux",
    aspect_ratio = :equal
)

savefig(ux_plot, "plots/ux_final_2d.png")

uy_plot = heatmap(
    x,
    y,
    permutedims(uy_grid);
    xlabel = "x",
    ylabel = "y",
    title = "Final velocity uy",
    aspect_ratio = :equal
)

savefig(uy_plot, "plots/uy_final_2d.png")

println("Saved rho_final_2d.png")
println("Saved ux_final_2d.png")
println("Saved uy_final_2d.png")