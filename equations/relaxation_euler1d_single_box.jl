################################################################
# tspan tested = (0.0, 0.05) (for large time: density goes -ve)
# Parameter values tested: RHO_FLOOR = 1e-10
#                          DELTA     = 0.1
#                          a         = -2.0
#                          b         = 2.0
#                          γ         = 3.0
# Boundary condition used: left_bc   = :extrapolate
#                          right_bc  = :extrapolate
# dt for test = min(dx)
# Domain for test = [-6.0, 6.0]
#################################################################

using EulerAP
# using Plots

tspan = (0.0, 0.05)
tol   = 1e-8
γ = 3.0
eps   = 1.0
const RHO_FLOOR = 1e-8
const DELTA = 0.1
const a = -2.0
const b = 2.0

left_bc   = :extrapolate
right_bc  = :extrapolate

function heaviside_smooth(x::Real)
    return 0.5*(1+tanh(x/DELTA))
end

function initial_box(x::Real, t::Real)
    X = x-a
    Y = x-b
    ha = heaviside_smooth(X)
    hb = heaviside_smooth(Y)
    ρ = max(RHO_FLOOR, ha - hb)

    u = -tanh(X/DELTA) * tanh(X/DELTA)
    u += tanh(Y/DELTA) * tanh(Y/DELTA)
    u = u / (2 * DELTA)
    u = u * (-γ) * ρ^(γ - 2.0)
    mx = ρ * u
    return ρ, mx
end

u0, coords, p, jac_cache = build_problem(
    size        = (100,),
    eps         = eps,
    domain_min  = (-6.0,),
    domain_max  = (6.0,),
    left_bc     = left_bc,
    right_bc    = right_bc,
    # bc_funcs    = bc_funcs,
    ic_func     = initial_box,
    tspan       = tspan,
    flux        = :rusanov,
    gamma       = γ,
)

u_init = copy(u0)

_ncells = prod(p.size)

u_final, solve_stats, nsteps_done =
    solve_backward_euler(
        u0,
        p,
        tspan,
        jac_cache;
        dt    = minimum(p.dx),
        tol   = tol,
    )

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
                gamma = γ)

n_threads = get(ENV, "MKL_NUM_THREADS", string(Threads.nthreads()))

sol = sol1D(coords[1], 
            u_init, 
            u_final, 
            _ncells)

save_solution_h5(sol, p; t_final = tspan[2])
# save_solution_h5(sol, p; t_final = nothing)


# figures = plot(sol, 
#                size=(1400, 550), 
#                plot_title="1D sol $(p.size) & $(p.eps)")

# mkpath("plots") 
# savefig(figures, "plots/sol1D_$(p.size[1])_$(p.eps).png")
# println("Solution is saved in plots/sol1D_$(p.size[1])_$(p.eps).png")