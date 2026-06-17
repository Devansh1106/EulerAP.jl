using Test
using EulerAP
using StaticArrays

@testset "Fluxes" begin
    eps = 0.05
    rho = 1.0
    mx = 0.0
    my = 0.0

    fx = EulerAP.rusanov_flux(SVector(rho, mx, my), SVector(rho, mx, my), 1, eps)
    @test isapprox(fx[1], 0.0; atol=1e-12)
    @test isapprox(fx[2], 1 / eps; rtol=1e-12)
    @test isapprox(fx[3], 0.0; atol=1e-12)

    fy = EulerAP.rusanov_flux(SVector(rho, mx, my), SVector(rho, mx, my), 2, eps)
    @test isapprox(fy[1], 0.0; atol=1e-12)
    @test isapprox(fy[2], 0.0; atol=1e-12)
    @test isapprox(fy[3], 1 / eps; rtol=1e-12)

    fx_gamma = EulerAP.rusanov_flux(SVector(rho, mx, my), SVector(rho, mx, my), 1, eps; gamma = 1.6)
    @test isapprox(fx_gamma[2], 1 / eps; rtol=1e-12)

    # resolve_flux accepts symbol and string
    fp_sym = resolve_flux(:rusanov; gamma = 1.4)
    fp_str = resolve_flux("rusanov"; gamma = 1.4)
    @test typeof(fp_sym) == typeof(fp_str)
    @test fp_sym.flux == fp_str.flux

    fp_gamma = resolve_flux(:rusanov; gamma = 1.6)
    fx_gamma_flux = fp_gamma(SVector(rho, mx, my), SVector(rho, mx, my), 1, eps)
    @test isapprox(fx_gamma_flux[2], 1 / eps; rtol=1e-12)
end
