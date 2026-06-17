using Test
using .TestUtils: small_problem, set_seed!
using EulerAP

@testset "Types and helpers" begin
    set_seed!(123)

    u0, x, y, p, cache = small_problem(nx=8, ny=8)

    @test p.size == (8, 8)
    @test isapprox(EulerAP.cell_index(1, 1, p), 1)

    rs = EulerAP.RunStats(3)
    @test length(rs.step_times) == 3
end
