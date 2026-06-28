module EulerAP

using StaticArrays

using SparseArrays
using ForwardDiff

using SciMLBase
using OrdinaryDiffEq
using NonlinearSolve

using LinearSolve
using Pardiso

using MuladdMacro
using HDF5
using RecipesBase
using Printf

# --------------------------------------------------
# Core infrastructure
# --------------------------------------------------

include("basic_types.jl")

include("meshes/cartesian_mesh.jl")

include("equations/equations.jl")

include("semidiscretization/semidiscretization_hyperbolic.jl")

# ----------------------------------------------------------------------
# Callbacks must be included before semidiscretization.jl since
# FVCache references the CallbackStats type.
# ----------------------------------------------------------------------

include("callbacks/callbacks.jl")

include("callbacks/summary_callback.jl")
include("callbacks/alive_callback.jl")
include("callbacks/analysis_callback.jl")
include("callbacks/save_solution_callback.jl")
include("callbacks/performance_callback.jl")

include("semidiscretization/semidiscretization.jl")

# Numerical fluxes must be included before solvers
include("equations/numerical_fluxes.jl")

include("solvers/solvers.jl")

include("solvers/fv_1d.jl")
include("solvers/fv_2d.jl")

# --------------------------------------------------
# Time integration
# --------------------------------------------------

include("time_integration/time_integration.jl")

include("time_integration/implicit_euler.jl")

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

# Numerical Fluxes
export FluxRusanov

# Solvers
export FVSolver

# Boundary Conditions
export PeriodicBC
export DirichletBC
export NeumannBC
export ExtrapolateBC

# Semidiscretizations
export SemidiscretizationHyperbolic
export semidiscretize
export solve

# Time Integrators
export AbstractTimeIntegrator
export ImplicitEulerCustom

# Solutions
export EulerAPSolution

# Equations IO
export save_solution
export save_initial_condition

# Initial conditions & source terms
export initial_condition_riemann
export initial_condition_single_box
export initial_condition_double_box
export initial_condition_sinosidal
export initial_condition_sinosidal_riemann
export initial_condition_barenblatt
export source_terms

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