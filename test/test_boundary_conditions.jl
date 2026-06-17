using Test
using EulerAP

@testset "Boundary conditions" begin
    ic = (x, y, t) -> (1.0, 0.0, 0.0)

    @testset "Periodic BC" begin
        # Periodic wrapping: left ghost wraps to cell at i = p.size[1]
        uP, coords, pP, cacheP = build_problem(ic_func = ic, size = (8, 8), left_bc = :periodic, right_bc = :periodic, tspan = (0.0, 1.0))
        idxwrap = EulerAP.cell_index(pP.size[1], 2, pP)
        nc = prod(pP.size)
        # set a distinguishable value at wrapped cell
        uP[idxwrap] = 7.5
        uP[nc + idxwrap] = 0.3
        uP[2*nc + idxwrap] = -0.4
        got = get_boundary_state(0, 2, pP, uP, 0.0)
        @test Tuple(got) == (uP[idxwrap], uP[nc + idxwrap], uP[2*nc + idxwrap])

        corner = get_boundary_state(CartesianIndex(0, 0), pP, uP, 0.0)
        corner_idx = EulerAP.cell_index(pP.size[1], pP.size[2], pP)
        @test Tuple(corner) == (uP[corner_idx], uP[nc + corner_idx], uP[2*nc + corner_idx])
    end

    @testset "Dirichlet BC" begin
        bc_funcs = Dict(:left => ((x, y, t) -> (2.0, 0.1, 0.2)))
        u0, coords, p, cache = build_problem(ic_func = ic, size = (8, 8), left_bc = :dirichlet, bc_funcs = bc_funcs, tspan = (0.0, 1.0))
        state = get_boundary_state(0, 2, p, u0, 0.0)  # left ghost cell
        @test state == (2.0, 0.1, 0.2)
    end

    @testset "Neumann BC" begin
        # Neumann BC: first-order extrapolation ghost = interior + grad * delta
        grad = (1.0, 0.5, -0.2)
        nbcs = Dict(:left => ((x, y, t) -> grad))
        uN, coords, pN, cacheN = build_problem(ic_func = ic, size = (8, 8), left_bc = :neumann, bc_funcs = nbcs, tspan = (0.0, 1.0))
        idx_interior = EulerAP.cell_index(1, 2, pN)
        ncN = prod(pN.size)
        interior = (uN[idx_interior], uN[ncN + idx_interior], uN[2*ncN + idx_interior])
        delta = -pN.dx[1]
        expected = (interior[1] + grad[1]*delta,
                    interior[2] + grad[2]*delta,
                    interior[3] + grad[3]*delta)
        gotN = get_boundary_state(0, 2, pN, uN, 0.0)
        @test isapprox(gotN[1], expected[1]; atol=1e-12)
        @test isapprox(gotN[2], expected[2]; atol=1e-12)
        @test isapprox(gotN[3], expected[3]; atol=1e-12)
    end

    @testset "Extrapolate BC" begin
        # Extrapolate: all components copied from interior (zero normal gradient)
        # For left wall (axis=1): normal momentum = mx, tangential = my
        ic_slip = (x, y, t) -> (1.0, 0.5, -0.3)  # rho=1, mx=0.5, my=-0.3
        uS, coords, pS, cacheS = build_problem(ic_func = ic_slip, size = (8, 8), left_bc = :extrapolate, tspan = (0.0, 1.0))
        idx_interior = EulerAP.cell_index(1, 2, pS)
        ncS = prod(pS.size)
        interior = (uS[idx_interior], uS[ncS + idx_interior], uS[2*ncS + idx_interior])
        
        # Left ghost cell (i=0, j=2)
        gotS = get_boundary_state(0, 2, pS, uS, 0.0)
        
        # Expected: rho = interior, mx = -interior (ghost = -interior for zero face flux), my = interior
        expected = (interior[1], -interior[2], interior[3])
        @test gotS == expected
    end
end
