using Test
using EulerAP

@testset "Solver integration" begin
    # Uniform initial condition -> should be steady for periodic BCs
    ic = (x, y, t) -> (1.0, 0.0, 0.0)
    u0, coords, p, cache = build_problem(ic_func = ic, size = (8, 8), left_bc = :periodic, right_bc = :periodic, bottom_bc = :periodic, top_bc = :periodic, tspan = (0.0, 1.0))

    dt = minimum(p.dx) * 0.25
    tspan = (0.0, dt)

    u_final, stats, nsteps = solve_backward_euler(u0, p, tspan, cache; dt = dt, tol = 1e-8)

    @test length(u_final) == length(u0)
    @test nsteps >= 1
    @test isfinite.(u_final) |> all
    # For uniform initial condition, solution should remain (approximately) unchanged
    @test maximum(abs.(u_final .- u0)) < 1e-10
end
