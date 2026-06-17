using EulerAP
# using Plots

# The result from this IC is not what we are expecting. With ϵ=1, we expect the solution to be far away from exact sol (BarenBlatt is 
# an exact sol) and as ϵ ≪ 1, we expect the solution to approach the exact sol. But currently the behavior is exactly opposite, it 
# matches with ϵ=1 and moves away as we reduce ϵ ≪ 1.

# Why we expect whatever is mentioned above: Since as ϵ→0, the relaxed Euler system converges to Porous Medium Equation (PME) whose
# exact solution is the BarenBlatt hence for ϵ ≪ 1, we expect it go closer to the BarenBlatt exact solution and be far away for ϵ=1.

tspan = (1.0, 1.2)
tol   = 1e-8
gamma = 3.0
eps   = 1e-4
const RHO_FLOOR = 1e-10

left_bc   = :extrapolate
right_bc  = :extrapolate

function barenblatt(x::Real, t::Real, Γ::Real, γ::Real)
    # Avoid division by zero at the exact start t = 0
    t_eff = t <= 0.0 ? error("division by zero since t is 0") : Float64(t)
    
    β = 1.0 / (γ + 1.0)
    ξ = x / (t_eff^β)
    
    factor = (γ - 1.0) / (2.0 * γ * (γ + 1.0))
    bracket_value = Γ - factor * (ξ^2)
    
    # Apply the positive-part operator max(value, 0)
    positive_part = max(bracket_value, 0.0)
    ρ = t_eff^(-β) * (positive_part^(1.0 / (γ - 1.0)))
    
    return ρ
end

function initial_u(x::Real, t::Real, Γ::Real, γ::Real)
    ρ = barenblatt(x, t, Γ, γ)
    β = 1.0 / (γ + 1.0)

    if ρ > 0.0
        u = β * x / t
        mx = ρ * u
    else
        # In the vacuum region, keep momentum zero.
        mx = 0.0
    end
    return mx
end

function initial_condition_barenblatt(x, t)
    gamma = 3.0
    ρ = barenblatt(x, t, 1.0, gamma)
    ρ = max(ρ, RHO_FLOOR)  # density floor to prevent vacuum
    mx = initial_u(x, t, 1.0, gamma)
    return ρ, mx
end

# function initial_condition_barenblatt(x,t)
#     ρ = barenblatt(x, t, 1.0, gamma)
#     ρ = max(ρ, RHO_FLOOR)  # density floor to prevent vacuum
#     mx = initial_mx(x, t, 1.0, gamma)
#     return ρ, mx
# end

u0, coords, p, jac_cache = build_problem(
    size        = (1000,),
    eps         = eps,
    domain_min  = (-6.0,),
    domain_max  = (6.0,),
    left_bc     = left_bc,
    right_bc    = right_bc,
    # bc_funcs    = bc_funcs,
    ic_func     = initial_condition_barenblatt,
    tspan       = tspan,
    flux        = :rusanov,
    gamma       = gamma,
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
                gamma = gamma)

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
