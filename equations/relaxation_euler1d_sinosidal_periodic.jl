using EulerAP
# using Plots

# This initial condition does not seem to work currently. We may need to update flux for it to work probably 
# since all other conditions are working.

tspan = (0.0, 0.3)
tol   = 1e-8
gamma = 3.0
eps   = 1e-4

eta = 10

function initial_condition_sinosidal(x, t)
    rho = 1.0 + 0.2 * sin(8.0 * π * x)
    u   = -0.2 * π * sin(8.0 * π * x)
    return rho, u
end

left_bc   = :periodic
right_bc  = :periodic


u0, coords, p, jac_cache = build_problem(
    size        = (100,),
    eps         = eps,
    domain_min  = (-1.0,),
    domain_max  = (1.0,),
    left_bc     = left_bc,
    right_bc    = right_bc,
    # bc_funcs    = bc_funcs,
    ic_func     = initial_condition_sinosidal,
    tspan       = tspan,
    flux        = :energy_stable,
    gamma       = gamma,
    eta         = eta
)

u_init = copy(u0)

u_final, solve_stats, nsteps_done =
    solve_backward_euler(
        u0,
        p,
        tspan,
        jac_cache;
        dt    = minimum(p.dx),
        tol   = tol,
    )

_ncells = prod(p.size)

println("Solved 1D relaxation Euler system (MKL)")
println("Final time = ", tspan[2])

rho_final = @view u_final[1:_ncells]
mx_final  = @view u_final[_ncells + 1:2 * _ncells]
ux_final  = mx_final ./ rho_final

println("Mean density = ", sum(rho_final) / _ncells)
println("Mean ux      = ", sum(ux_final) / _ncells)

print_run_stats("Solve", 
                solve_stats, 
                nsteps_done, 
                p; 
                gamma = gamma)

n_threads = get(ENV, "MKL_NUM_THREADS", string(Threads.nthreads()))

sol = sol1D(coords[1], 
            u_init, 
            u_final, 
            _ncells)

save_solution_h5(sol, p ; t_final = tspan[2])

# figures = plot(sol, 
#                size=(1400, 550), 
#                plot_title="1D sol $(p.size) & $(p.eps)")

# mkpath("plots") 
# savefig(figures, "plots/sol1D_$(p.size[1])_$(p.eps).png")
# println("Solution is saved in plots/sol1D_$(p.size[1])_$(p.eps).png")
