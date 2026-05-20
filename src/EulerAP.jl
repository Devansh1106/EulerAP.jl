module EulerAP

using ADTypes
using ForwardDiff
using NonlinearSolve
using SparseArrays
using StaticArrays
using SparseMatrixColorings
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

include("types.jl")
include("fluxes.jl")
include("operators.jl")
include("jacobian.jl")
include("initial_condition.jl")
include("solver.jl")
include("stats.jl")

end # end module EulerAP