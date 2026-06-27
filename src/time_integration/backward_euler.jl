# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

const linsolve = MKLPardisoFactorize()

function backward_euler_residual!(F, u_new, u_old,
                                  semi::AbstractSemidiscretization,
                                  dt,
                                  t)

    rhs_vec = similar(u_new)
    rhs!(rhs_vec,
        u_new,
        semi.solver,
        semi,
        t)

    @. F = u_new - u_old - dt * rhs_vec

    return nothing
end

function backward_euler_jacobian!(J, u_new, 
                                  semi::AbstractSemidiscretization,
                                  dt,
                                  t)

    assemble_jacobian!(J, u_new, 
                       semi,
                       semi.cache,
                       t)

    J.nzval .*= -dt
    n = size(J, 1)

    @inbounds for i in 1:n
        J[i, i] += 1
    end

    return nothing
end

function backward_euler_step!(u, 
                              semi::AbstractSemidiscretization,
                              dt,
                              t;
                              abstol = 1e-8,
                              reltol = 1e-8)

    u_old = copy(u)

    function residual!(F, x, p = nothing)
        backward_euler_residual!(F, x, u_old,
                                 semi::AbstractSemidiscretization,
                                 dt,
                                 t + dt)
    end

    function jacobian!(J, x, p = nothing)
        backward_euler_jacobian!(J, x,
                                 semi::AbstractSemidiscretization,
                                 dt,
                                 t + dt)
    end

    jac = copy(semi.cache.jac_prototype)

    nlf = NonlinearFunction(residual!;
                            jac = jacobian!,
                            jac_prototype = jac)

    prob = NonlinearProblem(nlf, u_old,
                            nothing)

    sol = NonlinearSolve.solve(prob, NewtonRaphson();
                linsolve_kwargs = (linsolve = linsolve,),
                abstol = abstol,
                reltol = reltol)

    copyto!(u, sol.u)

    return nothing
end

function solve_implicit_euler(semi::AbstractSemidiscretization,
                              tspan;
                              dt,
                              abstol = 1e-8,
                              reltol = 1e-8)

    # Build Jacobian cache if not yet initialized (e.g. when called directly via
    # solve(semi, tspan, ImplicitEulerCustom()) instead of semidiscretize(...; jac_prototype=true))
    if semi.cache.config === nothing
        build_jacobian_cache!(semi)
    end

    t = first(tspan)
    u = initial_condition(t, semi)

    while t < last(tspan) - eps(t)
        # Clip dt to avoid overshooting the final time
        actual_dt = min(dt, last(tspan) - t)

        backward_euler_step!(u,
                             semi::AbstractSemidiscretization,
                             actual_dt,
                             t;
                             abstol = abstol,
                             reltol = reltol)

        t += actual_dt
    end

    return EulerAPSolution(u, t)
end

end # @muladd