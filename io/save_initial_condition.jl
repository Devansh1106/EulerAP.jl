# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    save_initial_condition(semi, filename; t = 0.0)

Compute and save the initial state of a semidiscretization to an HDF5 file.

The initial condition function stored in `semi` is evaluated at time `t`
(default 0.0) and saved in the same HDF5 layout as `save_solution`, so the
resulting files are interchangeable for plotting etc.
"""
function save_initial_condition(semi::SemidiscretizationHyperbolic,
                                filename::AbstractString;
                                t)

    mesh      = semi.mesh
    equations = semi.equations

    nvars = nvariables(equations)
    nd    = ndims(mesh)

    u = initial_condition(t, semi)

    h5open(filename, "w") do file

        # --------------------------------------------------
        # Convenience top-level scalars
        # --------------------------------------------------

        file["eps"] = equations.epsilon
        mesh_str = join(mesh.cells_per_dimension, "x")
        file["ncells"] = mesh_str

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
            t

        metadata_group["ndims"] =
            nd

        metadata_group["nvariables"] =
            nvars

        # --------------------------------------------------
        # Equation parameters
        # --------------------------------------------------

        equations_group = create_group(file, "equations")
        equations_group["gamma"] = equations.gamma
        equations_group["epsilon"] = equations.epsilon

        # --------------------------------------------------
        # Solution
        # --------------------------------------------------

        solution_group = create_group(file, "solution")

        solution_group["u"] = reshape(u, nvars, ndofs(mesh))

    end

    println("Saved initial condition to ", filename)
    return nothing
end

end # @muladd