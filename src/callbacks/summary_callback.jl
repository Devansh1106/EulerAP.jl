# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SummaryCallback()

Print a summary of the simulation before time integration begins.
"""
struct SummaryCallback <: AbstractCallback
end


function initialize!(::SummaryCallback,
                     context::CallbackContext)

    simulation = context.simulation

    semi = simulation.semi
    mesh = semi.mesh
    equations = semi.equations
    solver = semi.solver

    println()
    println("============================================================")
    println("                     EulerAP Simulation")
    println("============================================================")

    println()
    println("Spatial Discretization")
    println("----------------------")

    print_summary_line("Mesh", mesh)
    print_summary_line("Equations", equations)
    print_summary_line("Solver", solver)
    print_summary_line("Flux", solver.flux)

    println()

    print_summary_line("Dimensions", ndims(mesh))
    print_summary_line("Grid", mesh.cells_per_dimension)
    print_summary_line("Domain", "$(mesh.coordinates_min) → $(mesh.coordinates_max)")
    print_summary_line("Cell size", mesh.dx)

    println()

    print_summary_line("Cells", ndofs(mesh))
    print_summary_line("Unknowns",
                       ndofs(mesh) * nvariables(equations))

    println()
    println("Boundary Conditions")
    println("-------------------")

    if ndims(mesh) == 1

        bc = semi.boundary_conditions

        print_summary_line("Left", bc.left)
        print_summary_line("Right", bc.right)

    elseif ndims(mesh) == 2

        bc = semi.boundary_conditions

        print_summary_line("Left", bc.left)
        print_summary_line("Right", bc.right)
        print_summary_line("Bottom", bc.bottom)
        print_summary_line("Top", bc.top)

    end

    println()
    println("Time Integration")
    println("----------------")

    print_summary_line("Integrator",
                       simulation.integrator)

    print_summary_line("Time span",
                       simulation.tspan)

    print_summary_line("Time step",
                       simulation.dt)

    print_summary_line("Absolute tolerance",
                       simulation.abstol)

    print_summary_line("Relative tolerance",
                       simulation.reltol)

    println("============================================================")
    println()

    return nothing
end

function finalize!(::SummaryCallback,
                   context::CallbackContext)

    stats = context.stats

    println()
    println("============================================================")
    println("                     EulerAP Simulation")
    println("============================================================")
    println()
    print_summary_line("Iterations completed", stats.iteration)
    print_summary_line("Final time", stats.time)
    print_summary_line("Total runtime (s)", stats.total_runtime)
    println("============================================================")
    println()

    return nothing
end

end # @muladd
