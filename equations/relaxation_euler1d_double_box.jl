################################################################
# tspan tested = (0.0, 0.05) (for large time: density goes -ve)
# Parameter values tested: RHO_FLOOR = 1e-10
#                          DELTA     = 0.005
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

tspan = (0.0, 3.0)
tol   = 1e-8
γ = 3.0
eps   = 1e-4
const RHO_FLOOR = 1e-8
const DELTA = 0.005
const a = -2.0
const b = -1.0

const c = 1.0
const d = 2.0

left_bc   = :extrapolate
right_bc  = :extrapolate

function heaviside_smooth(x::Real)
    return 0.5*(1+tanh(x/DELTA))
end

function initial_box(x::Real, t::Real)
    X = x-a
    Y = x-b

    _X = x-c
    _Y = x-d

    ha = heaviside_smooth(X)
    hb = heaviside_smooth(Y)

    hc = heaviside_smooth(_X)
    hd = heaviside_smooth(_Y)

    if a <= x <= b
        ρ = max(RHO_FLOOR, ha - hb)
    elseif c <= x <= d
        ρ = max(RHO_FLOOR, hc - hd)
    else
        ρ = RHO_FLOOR
    end

    # ρ = max(RHO_FLOOR, ha - hb)

    uab = -tanh(X/DELTA) * tanh(X/DELTA)
    uab += tanh(Y/DELTA) * tanh(Y/DELTA)
    uab = uab / (2 * DELTA)
    uab = uab * (-γ) * ρ^(γ - 2.0)

    ucd = -tanh(_X/DELTA) * tanh(_X/DELTA)
    ucd += tanh(_Y/DELTA) * tanh(_Y/DELTA)
    ucd = ucd / (2 * DELTA)
    ucd = ucd * (-γ) * ρ^(γ - 2.0)

    if a <= x <= b
        u = uab
    elseif c <= x <= d
        u = ucd
    else
        u = 0.0
    end
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