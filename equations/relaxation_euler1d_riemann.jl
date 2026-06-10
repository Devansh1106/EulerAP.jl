using EulerAP
# using Plots

tspan = (0.0, 0.1)
tol   = 1e-8
gamma = 1.4
eps   = 1e-2

const rho_L = 1.0
const rho_R = 0.5
const x_m   = 0.5

function initial_condition_riemann(x, t)
    rho0 = x <= x_m ? rho_L : rho_R
    return rho0, 0.0
end

left_bc   = :extrapolate
right_bc  = :extrapolate


u0, coords, p, jac_cache = build_problem(
    size        = (100,),
    eps         = eps,
    domain_min  = (-0.5,),
    domain_max  = (0.5,),
    left_bc     = left_bc,
    right_bc    = right_bc,
    # bc_funcs    = bc_funcs,
    ic_func     = initial_condition_riemann,
    tspan       = tspan,
    flux        = :rusanov,
    gamma       = gamma,
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
        flux  = :rusanov,
        gamma = gamma
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

save_solution_h5(sol, p)

# figures = plot(sol, 
#                size=(1400, 550), 
#                plot_title="1D sol $(p.size) & $(p.eps)")

# mkpath("plots") 
# savefig(figures, "plots/sol1D_$(p.size[1])_$(p.eps).png")
# println("Solution is saved in plots/sol1D_$(p.size[1])_$(p.eps).png")
