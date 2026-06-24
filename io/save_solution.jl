# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    save_solution(sol, semi, filename)

Save the final solution state to an HDF5 file.
"""
function save_solution(sol,
                       semi::SemidiscretizationHyperbolic,
                       filename::AbstractString)

    mesh      = semi.mesh
    equations = semi.equations

    nvars = nvariables(equations)

    u = solution_vector(sol)
    time = solution_time(sol)

    h5open(filename, "w") do file

        # --------------------------------------------------
        # Mesh
        # --------------------------------------------------

        mesh_group = create_group(file, "mesh")

        mesh_group["cells_per_dimension"] =
            collect(mesh.cells_per_dimension)

        mesh_group["coordinates_min"] =
            collect(mesh.coordinates_min)

        mesh_group["coordinates_max"] =
            collect(mesh.coordinates_max)

        mesh_group["dx"] =
            collect(mesh.dx)

        mesh_group["periodicity"] =
            collect(mesh.periodicity)

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata_group =
            create_group(file, "metadata")

        metadata_group["time"] =
            sol.t[end]

        metadata_group["ndims"] =
            ndims(mesh)

        metadata_group["nvariables"] =
            nvars

        # --------------------------------------------------
        # Solution
        # --------------------------------------------------

        solution_group = create_group(file, "solution")

        solution_group["u"] = reshape(u, nvars, ndofs(mesh))

    end

    return nothing
end

@inline solution_vector(sol::EulerAPSolution) = sol.u

@inline solution_time(sol::EulerAPSolution) = sol.t

@inline solution_vector(sol::SciMLBase.AbstractODESolution) =
    sol.u[end]

@inline solution_time(sol::SciMLBase.AbstractODESolution) =
    sol.t[end]

end # @muladd