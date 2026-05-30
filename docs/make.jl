using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using EulerAP

makedocs(
    sitename = "EulerAP Manual",
    modules = [EulerAP],
    clean = true,
    pages = [
        "Home" => "index.md",
        "Library" => [
            "Types" => "library/types.md",
            "Fluxes" => "library/fluxes.md",
            "Boundary Conditions" => "library/boundary_conditions.md",
            "Operators" => "library/operators.md",
            "Jacobian" => "library/jacobian.md",
            "Problem Setup" => "library/build_problem.md",
            "Solver" => "library/solver.md",
            "Statistics" => "library/stats.md",
        ],
    ]
)

deploydocs(
    repo = "://github.com",
    devbranch = "main"
)
