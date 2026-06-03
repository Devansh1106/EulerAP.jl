# const linsolve = KLUFactorization()
const linsolve = MKLPardisoFactorize()

"""
    backward_euler_residual!(res, u, p::ImplicitStepData)

Residual callback used by the nonlinear solver for one backward-Euler step.
"""
function backward_euler_residual!(res, u, p::ImplicitStepData)
    # 1. Compute the spatial fluxes directly into the solver's thread-safe `res` vector.
    # This completely overwrites `res` with the spatial RHS.
    implicit_part!(res, u, p.model, p.t; flux = p.flux)
    
    # 2. Convert the spatial RHS into the Backward Euler residual IN-PLACE.
    # Julia's `@.` loop safely reads from `res` and writes back to `res` 
    # element-by-element without allocating a single byte.
    @. res = u - p.u_prev - p.dt * res
    
    return nothing
end

"""
    solve_backward_euler(u0, p, tspan, jac_cache; kwargs...)

Advance the solution with fixed-step backward Euler and return the final state,
run statistics, and number of steps completed.
"""
function solve_backward_euler(
    u0,
    p::RelaxationParams,
    tspan,
    jac_cache::SparseJacobianCache; # Pass the struct instead of the raw matrix
    dt = 5.0e-2,
    tol = 1e-8,
    flux = :rusanov,
    jacobian_builder! = assemble_global_jacobian!
)
    resolved_flux = resolve_flux(flux)

    # The closure captures our jac_cache struct. 
    # J_internal is the matrix NonlinearSolve tracks, but we update our cached matrix and copy it to J_internal
    jac_fun = (J_internal, u, step_data) -> begin
        # assembled Jacobian matches the backward-Euler residual (I - dt*dF/du).
        jacobian_builder!(jac_cache, u, step_data.model, step_data.dt, step_data.t; flux = step_data.flux)
        copyto!(J_internal, jac_cache.J)
        return J_internal
    end

    nls_function = NonlinearFunction(
        backward_euler_residual!;
        jac = jac_fun,
        jac_prototype = jac_cache.J # Pass the raw matrix from the cache here
    )

    nls_algorithm = NewtonRaphson(
        concrete_jac = true,
        linsolve = linsolve,
        linesearch = BackTracking() # required since NR was stuck w/o this for Riemann problem/stiff cases
    )

    u = copy(u0)

    step_data = ImplicitStepData(
        p, dt, tspan[1], copy(u0), resolved_flux
    )

    nsteps_target = ceil(Int, (tspan[2] - tspan[1]) / dt)
    # Add one extra slot to guard against floating-point rounding producing
    # an extra (very small) final step when summing dt steps.
    stats = RunStats(nsteps_target + 1)

    nonlinear_problem = NonlinearProblem(nls_function, u, step_data)

    cache = init(
        nonlinear_problem,
        nls_algorithm;
        abstol = tol,
        reltol = tol
    )

    t = tspan[1]
    nsteps_done = 0

    total_timed = @timed while t < tspan[2]
        dt_step = min(dt, tspan[2] - t)
        step_data.dt = dt_step
        step_data.t = t + dt_step
        step_data.u_prev .= u
        nsteps_done += 1

        step_timed = @timed begin
            reinit!(cache; u0 = u, p = step_data) # in-place (no allocation)
            nonlinear_solution = SciMLBase.solve!(cache)
            if !SciMLBase.successful_retcode(nonlinear_solution.retcode)
                error(
                    "Backward Euler nonlinear solve failed at step $nsteps_done, " *
                    "t=$(step_data.t), dt=$(step_data.dt), retcode=$(nonlinear_solution.retcode)"
                )
            end
            u .= nonlinear_solution.u
        end
        
        stats.step_times[nsteps_done] = step_timed.time
        stats.step_bytes[nsteps_done] = step_timed.bytes
        stats.step_gctimes[nsteps_done] = step_timed.gctime

        t += dt_step
    end

    stats.total_gctime = total_timed.gctime
    stats.total_time   = total_timed.time
    stats.total_bytes  = total_timed.bytes
    return u, stats, nsteps_done
end
