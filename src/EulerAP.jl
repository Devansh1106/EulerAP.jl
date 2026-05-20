module EulerAP

using ADTypes
using NonlinearSolve
using SparseArrays
using SparseMatrixColorings
using LinearSolve

export RelaxationParams
export build_problem
export solve_backward_euler
export print_run_stats
export initial_condition

include("types.jl")
include("fluxes.jl")
include("operators.jl")
include("jacobian.jl")
include("initial_condition.jl")
include("solver.jl")
include("stats.jl")

end # end module EulerAP