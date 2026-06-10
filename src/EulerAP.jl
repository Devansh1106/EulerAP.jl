module EulerAP

using ForwardDiff
using NonlinearSolve
using SparseArrays
using LinearSolve
using Pardiso
using HDF5
using RecipesBase

# --- 1D Solution Structure (Zero-Allocation Layout) ---
struct sol1D{VX, VU}
    x::VX
    u_init::VU
    u_final::VU
    _ncells::Int

    function sol1D(x::AbstractVector{Float64}, 
                   u_init::AbstractVector, 
                   u_final::AbstractVector, 
                   _ncells::Int)

        return new{typeof(x), typeof(u_init)}(x, 
                                              u_init, 
                                              u_final, 
                                              _ncells)
    end
end

# --- 2D Solution Structure (Zero-Allocation Layout) ---
struct sol2D{VX, VY, VU}
    x::VX
    y::VY
    u_init::VU
    u_final::VU
    _ncells::Int

    function sol2D(x::AbstractVector{Float64},
                   y::AbstractVector{Float64}, 
                   u_init::AbstractVector, 
                   u_final::AbstractVector, 
                   _ncells::Int)
                   
        return new{typeof(x), typeof(y), typeof(u_init)}(x, 
                                                         y, 
                                                         u_init, 
                                                         u_final, 
                                                         _ncells)
    end
end

export RelaxationParams
export FluxPair
export rusanov_flux
export build_problem
export solve_backward_euler
export print_run_stats
export save_solution_h5
export resolve_flux
export gather_local_state
export local_residual
export assemble_global_jacobian!
export BCConfig, PeriodicBC, DirichletBC, NeumannBC
export apply_bc, get_bc_config, get_boundary_state
export sol1D, sol2D

include("types.jl")
include("fluxes.jl")
include("boundary_conditions.jl")
include("operators.jl")
include("jacobian.jl")
include("build_problem.jl")
include("solver.jl")
include("stats.jl")
include("recipes.jl")

end # end module EulerAP