using EulerAP
using Plots

tspan = (0.0, 0.01)
tol = 1e-8
gamma = 1.4

# Boundary condition choices: :periodic, :dirichlet, :neumann
left_bc   = :dirichlet
right_bc  = :dirichlet

# Initial condition (Riemann / shock-tube style)
# Domain: x in [0, 1.0]
# Initial density: rho_L for x <= x_m, rho_R for x > x_m
# Velocity: zero everywhere
const rho_L = 1.0
const rho_R = 0.5
const x_m = 0.5

function initial_condition_riemann(x)
    rho0 = x <= x_m ? rho_L : rho_R
    return rho0, 0.0
end

function homogeneous_dirichlet(x, t)
    # return (0.0, 0.0) # returning zero gradient for rho and mx
    return initial_condition_riemann(x)
end

bc_funcs = Dict(
    :left => homogeneous_dirichlet,
    :right => homogeneous_dirichlet
)

# warm-up run for julia
u0, x, _, p, jac_cache = build_problem(
    size = (8,),
    eps = 1e-3,
    left_bc = left_bc,
    right_bc = right_bc,
    bc_funcs = bc_funcs,
    ic_func = initial_condition_riemann,
    flux = :rusanov,
    gamma = gamma,
    xmin = 0.0,
    xmax = 1.0
)

# Actual run
u0, x, _, p, jac_cache = build_problem(
    size = (512,),
    eps = 1e-3,
    left_bc = left_bc,
    right_bc = right_bc,
    bc_funcs = bc_funcs,
    ic_func = initial_condition_riemann,
    flux = :rusanov,
    gamma = gamma,
    xmin = 0.0,
    xmax = 1.0
)

u_init = copy(u0)

u_final, solve_stats, nsteps_done =
    solve_backward_euler(
        u0,
        p,
        tspan,
        jac_cache;
        dt = minimum(p.dx),
        tol = tol,
        flux = :rusanov,
        gamma = gamma
    )

_ncells = p.nx * p.ny

println("Solved 1D relaxation Euler system (MKL)")
println("Final time = ", tspan[2])

rho_final = @view u_final[1:_ncells]
mx_final  = @view u_final[_ncells + 1:2 * _ncells]
ux_final  = mx_final ./ rho_final

println("Mean density = ", sum(rho_final) / _ncells)
println("Mean ux = ", sum(ux_final) / _ncells)

print_run_stats("Solve", solve_stats, nsteps_done, p; gamma = gamma)
n_threads = get(ENV, "MKL_NUM_THREADS", string(Threads.nthreads()))

sol = sol1D(x, u_init, u_final, _ncells)
figures = plot(sol, size=(1400, 550), plot_title="1D sol $(p.size) & $(p.eps)")

mkpath("plots") 
savefig(figures, "plots/sol1D_$(p.nx).png")
println("Solution is saved in plots/sol1D_$(p.nx).png")