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
    nd    = ndims(mesh)

    u = solution_vector(sol)
    time = solution_time(sol)
    mkpath(dirname(filename))

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
            time

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

    println("Saved solution to ", filename)
    return nothing
end

@inline solution_vector(sol::EulerAPSolution) = sol.u
@inline solution_time(sol::EulerAPSolution) = sol.t

# For SciMLBase.AbstractODESolution
@inline solution_vector(sol::AbstractODESolution) = sol.u[end]
@inline solution_time(sol::AbstractODESolution) = sol.t[end]

function save_solution(sol,
                       semi::SemidiscretizationHyperbolicElliptic,
                       filename::AbstractString)

    mesh = semi.mesh
    equations_hyperbolic = semi.equations
    equations_elliptic = semi.equations_elliptic

    nvars_hyper    = nvariables(equations_hyperbolic)
    nvars_elliptic = nvariables(equations_elliptic)
    nvars_total    = nvars_hyper + nvars_elliptic
    nd    = ndims(mesh)

    u = solution_vector(sol)
    time = solution_time(sol)
    mkpath(dirname(filename))

    # The solution is stored in block layout:
    #   [ρ₁, m₁, ρ₂, m₂, ..., ρₙ, mₙ, φ₁, φ₂, ..., φₙ]
    # Reorder to interleaved for HDF5 storage:
    #   [ρ₁, m₁, φ₁, ρ₂, m₂, φ₂, ...]
    nc = ndofs(mesh)
    u_block = reorder_block_to_interleaved(u, nvars_hyper, nvars_elliptic, nc)

    h5open(filename, "w") do file

        mesh_str = join(mesh.cells_per_dimension, "x")
        file["ncells"] = mesh_str

        mesh_group = create_group(file, "mesh")
        mesh_group["cells_per_dimension"] = collect(mesh.cells_per_dimension)
        mesh_group["coordinates_min"]    = collect(mesh.coordinates_min)
        mesh_group["coordinates_max"]    = collect(mesh.coordinates_max)
        mesh_group["dx"]                 = collect(mesh.dx)
        mesh_group["periodicity"]        = collect(mesh.periodicity)

        metadata_group = create_group(file, "metadata")
        metadata_group["time"]       = time
        metadata_group["ndims"]      = nd
        metadata_group["nvariables"] = nvars_total

        equations_group = create_group(file, "equations")
        equations_group["gamma"]  = equations_hyperbolic.gamma
        equations_group["lambda"] = equations_elliptic.lambda

        solution_group = create_group(file, "solution")
        solution_group["u"] = u_block
    end

    println("Saved solution to ", filename)
    return nothing
end

"""
    reorder_block_to_interleaved(u, nvars_hyper, nvars_elliptic, nc)

Convert block layout [hyper_vars... | elliptic_vars...] to
interleaved-by-cell layout, writing into a pre-allocated
(nvars_hyper + nvars_elliptic, nc) array.
"""
function reorder_block_to_interleaved(u, nvars_hyper, nvars_elliptic, nc)
    nvars_total = nvars_hyper + nvars_elliptic
    result = similar(u, nvars_total, nc)
    n_hyper = nvars_hyper * nc
    @inbounds for cell in 1:nc
        for v in 1:nvars_hyper
            result[v, cell] = u[(cell - 1) * nvars_hyper + v]
        end
        for v in 1:nvars_elliptic
            result[nvars_hyper + v, cell] = u[n_hyper + (cell - 1) * nvars_elliptic + v]
        end
    end
    return result
end

end # @muladd
