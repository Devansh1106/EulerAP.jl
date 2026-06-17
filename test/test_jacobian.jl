using Test
using Random
using EulerAP
using SparseArrays
using LinearAlgebra

@testset "Jacobian FD vs Assembled" begin
    Random.seed!(1234)

    size = (4, 4)
    eps = 0.05

    function initial_condition(x, y, t)
        r2 = x^2 + y^2
        rho0 = 1 - 0.1 * exp(2 * (1 - r2))
        ux0 = y * exp(1 - r2)
        uy0 = -x * exp(1 - r2)
        return rho0, rho0 * ux0, rho0 * uy0
    end

    u0, coords, p, cache = build_problem(size=size, eps=eps, left_bc = :periodic, right_bc=:periodic, bottom_bc=:periodic, top_bc=:periodic, ic_func = initial_condition, tspan = (0.0, 1.0))

    dof = length(u0)

    # pick a test state: slightly perturb u0
    u = copy(u0)
    for i in 1:dof
        u[i] += 1e-3 * (rand() - 0.5)
    end

    dt = 0.1 * p.dx[1]
    t = 0.0

    J_assembled = assemble_global_jacobian!(cache, u, p, dt, t)

    resolved_flux_test = resolve_flux(:rusanov; gamma = 1.4)

    function eval_F(u)
        du = similar(u)
        EulerAP.implicit_part!(du, u, p, t; resolved_flux = resolved_flux_test)
        return du
    end

    eps_fd = 1e-6
    F0 = eval_F(u)

    # Compute FD Jacobian (dense) - small problem only
    JF = zeros(Float64, dof, dof)
    for j in 1:dof
        u_pert = copy(u)
        u_pert[j] += eps_fd
        Fp = eval_F(u_pert)
        JF[:, j] = (Fp .- F0) ./ eps_fd
    end

    Jfd = I - dt .* JF
    Jdense = Matrix(J_assembled)

    max_abs = maximum(abs.(Jdense .- Jfd))
    rel_norm = norm(Jdense - Jfd) / max(1e-12, norm(Jfd))

    @test max_abs < 1e-6 || rel_norm < 1e-5
end
