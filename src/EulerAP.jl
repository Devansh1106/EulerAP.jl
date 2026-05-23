module EulerAP

using ForwardDiff
using NonlinearSolve
using SparseArrays
using StaticArrays
using LinearSolve

export RelaxationParams
export FluxPair
export build_problem
export solve_backward_euler
export print_run_stats
export initial_condition
export resolve_flux
export gather_local_state
export local_residual
export local_jacobian
export assemble_global_jacobian
export BCConfig, PeriodicBC, DirichletBC, NeumannBC
export apply_bc, get_bc_config, get_boundary_state

include("types.jl")
include("fluxes.jl")
include("boundary_conditions.jl")
include("operators.jl")
include("jacobian.jl")
include("initial_condition.jl")
include("solver.jl")
include("stats.jl")

end # end module EulerAP