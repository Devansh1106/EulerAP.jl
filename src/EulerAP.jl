module EulerAP

using StaticArrays

using LinearAlgebra: Tridiagonal
using SparseArrays: spzeros, sparse
using ForwardDiff

using SciMLBase: ODEFunction, ODEProblem, FullSpecialize, AbstractODESolution, NonlinearProblem, reinit!, solve!
using NonlinearSolve: NonlinearFunction, NonlinearProblem, NewtonRaphson, NonlinearSolve, init

using Pardiso
using LinearSolve: MKLPardisoFactorize

using MuladdMacro
using HDF5: h5open, create_group
using Printf: @printf
using TimerOutputs: TimerOutput, @timeit

# --------------------------------------------------
# Core infrastructure
# --------------------------------------------------

include("basic_types.jl")

include("meshes/cartesian_mesh.jl")

include("equations/equations.jl")

# ----------------------------------------------------------------------
# Callbacks must be included before solvers and semidiscretizations since
# FVCache references the CallbackStats type.
# ----------------------------------------------------------------------

include("callbacks/callbacks.jl")

include("callbacks/summary_callback.jl")
include("callbacks/alive_callback.jl")
include("callbacks/analysis_callback.jl")
include("callbacks/save_solution_callback.jl")
include("callbacks/performance_callback.jl")

include("semidiscretization/semidiscretization.jl")

include("semidiscretization/semidiscretization_hyperbolic.jl")

include("semidiscretization/semidiscretization_hyperbolic_elliptic.jl")

# Numerical fluxes must be included before solvers
include("equations/numerical_fluxes.jl")

include("solvers/solvers.jl")

include("solvers/fv_1d.jl")
include("solvers/fv_2d.jl")
include("solvers/newton_solver.jl") 

# --------------------------------------------------
# Time integration
# --------------------------------------------------

include("time_integration/time_integration.jl")

include("time_integration/implicit_euler.jl")
include("time_integration/imex.jl")

# --------------------------------------------------
# IO
# --------------------------------------------------

include("../io/save_solution.jl")
include("../io/save_initial_condition.jl")

# --------------------------------------------------
# Plotting
# --------------------------------------------------

include("visualization/recipes.jl")

# --------------------------------------------------
# Postprocessing
# --------------------------------------------------

include("postprocessing/postprocessing.jl")

include("postprocessing/norms.jl")
include("postprocessing/errors.jl")
include("postprocessing/convergence.jl")

# --------------------------------------------------
# Exports
# --------------------------------------------------

# Meshes
export CartesianMesh

# Equations
export RelaxationEulerEquations1D
export RelaxationEulerEquations2D
export EulerPressureLess1D
export EulerPressureLess2D
export PoissonBoltzmann

# Numerical Fluxes
export FluxRusanov
export FluxEnergyStable

# Solvers
export FVSolver
export EllipticSolver

# Boundary Conditions
export BoundaryConditions1D
export BoundaryConditions2D
export PeriodicBC
export DirichletBC
export NeumannBC
export ExtrapolateBC

# Semidiscretizations
export SemidiscretizationHyperbolicElliptic
export SemidiscretizationHyperbolic
export semidiscretize
export solve

# Time Integrators
export AbstractTimeIntegrator
export ImplicitEulerCustom
export IMEXIntegrator
export FirstOrderThreeStagesIMEX
export ExplicitCorrectionStage
export ImplicitCorrectionStage
export ImplicitPredictionStage

# Solutions
export EulerAPSolution

# Equations IO
export save_solution
export save_initial_condition

# Initial conditions
# Relaxation Euler
export initial_condition_riemann
export initial_condition_single_box
export initial_condition_double_box
export initial_condition_sinosidal
export initial_condition_sinosidal_riemann
export initial_condition_barenblatt
# EPB system
export initial_condition_riemann_epb
export initial_condition_five_branch
export make_initial_condition_soliton

# Source terms
export source_terms
export source_terms_hyperbolic

# Newton solver types
export NewtonParameters
export NewtonCache

# Postprocessing
export compute_errors
export convergence_table
export convergence_test

# Callbacks
export CallbackSet

export SummaryCallback
export AliveCallback
export AnalysisCallback
export SaveSolutionCallback
export PerformanceCallback

export initialize_callbacks!
export perform_callbacks!
export finalize_callbacks!

end # module EulerAP