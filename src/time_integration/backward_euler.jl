# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

function backward_euler_residual!(
    F,
    u_new,
    u_old,
    semi,
    dt,
    t
)

    rhs_vec = similar(u_new)

    rhs!(
        rhs_vec,
        u_new,
        semi.solver,
        semi,
        t
    )

    @. F = u_new - u_old - dt * rhs_vec

    return nothing
end

function backward_euler_jacobian!(
    J,
    u_new,
    semi,
    dt,
    t
)

    assemble_jacobian!(
        J,
        u_new,
        semi,
        t
    )

    J.nzval .*= -dt

    n = size(J, 1)

    @inbounds for i in 1:n
        J[i, i] += 1
    end

    return nothing
end

function backward_euler_step!(
    u,
    semi,
    dt,
    t;
    abstol = 1e-8,
    reltol = 1e-8
)

    u_old = copy(u)

    function residual!(F, x)
        backward_euler_residual!(
            F,
            x,
            u_old,
            semi,
            dt,
            t + dt
        )
    end

    function jacobian!(J, x)
        backward_euler_jacobian!(
            J,
            x,
            semi,
            dt,
            t + dt
        )
    end

    jac =
        copy(
            semi.cache.jac_prototype
        )

    nlf =
        NonlinearFunction(
            residual!;
            jac = jacobian!,
            jac_prototype = jac
        )

    prob =
        NonlinearProblem(
            nlf,
            u_old
        )

    sol =
        solve(
            prob,
            NewtonRaphson();
            abstol = abstol,
            reltol = reltol
        )

    copyto!(u, sol.u)

    return nothing
end

function solve_implicit_euler(
    semi,
    tspan;
    dt,
    abstol = 1e-8,
    reltol = 1e-8
)

    u =
        initial_condition(
            first(tspan),
            semi
        )

    t = first(tspan)

    while t < last(tspan) - eps(t)

        backward_euler_step!(
            u,
            semi,
            dt,
            t;
            abstol = abstol,
            reltol = reltol
        )

        t += dt
    end

    return EulerAPSolution(u, t)
end

end # @muladd