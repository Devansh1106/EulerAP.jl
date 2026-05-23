const linsolve = KLUFactorization()

function backward_euler_residual!(res, u, p::ImplicitStepData)
    if !(eltype(p.rhs_cache) === eltype(u) &&
         length(p.rhs_cache) == length(u))

        p.rhs_cache = similar(u)
    end
    implicit_part!(p.rhs_cache, u, p.model, p.t; flux = p.flux)
    @. res = u - p.u_prev - p.dt * p.rhs_cache
    return nothing
end

function solve_backward_euler(
    u0,
    p::RelaxationParams,
    tspan,
    jac_prototype;
    dt = 5.0e-2,    # default values when user does not provide any
    tol = 1e-8,      # default values when user does not provide any
    flux = :rusanov,
    jacobian_builder = assemble_global_jacobian
)
    resolved_flux = resolve_flux(flux)

    # jacobian_builder is expected to have signature jacobian_builder(J_prototype, u, model; flux=..., t=...)
    # It should fill J_prototype.nzval in-place and return it. jac_fun will copy into the solver's J.
    jac_fun = (J, u, step_data) -> begin
        jacobian_builder(J, u, step_data.model; flux = step_data.flux, t = step_data.t)
        return J
    end

    nls_function = NonlinearFunction(
                   backward_euler_residual!;
                   jac = jac_fun,
                   jac_prototype = jac_prototype)

    nls_algorithm = NewtonRaphson(
                    concrete_jac = true,
                    linsolve = linsolve)

    u = copy(u0)

    step_data = ImplicitStepData(
                p,
                dt,
                tspan[1],
                copy(u0),
                similar(u0),
                resolved_flux)

    nsteps_target = ceil(Int, (tspan[2] - tspan[1]) / dt)

    stats = RunStats(nsteps_target)

    nonlinear_problem = NonlinearProblem(
                        nls_function,
                        u,
                        step_data)

    cache = init(
            nonlinear_problem,
            nls_algorithm;
            abstol = tol,
            reltol = tol)

    t = tspan[1]
    nsteps_done = 0

    total_timed = @timed while t < tspan[2]
        dt_step = min(dt, tspan[2] - t)
        step_data.dt = dt_step
        step_data.t = t + dt_step
        step_data.u_prev .= u
        nsteps_done += 1

        step_timed = @timed begin
            cache.u .= u
            nonlinear_solution = solve!(cache)
            u .= nonlinear_solution.u
        end
        stats.step_times[nsteps_done] = step_timed.time
        stats.step_bytes[nsteps_done] = step_timed.bytes
        stats.step_gctimes[nsteps_done] = step_timed.gctime

        t += dt_step
    end

    stats.total_gctime = total_timed.gctime
    return u, stats, nsteps_done
end