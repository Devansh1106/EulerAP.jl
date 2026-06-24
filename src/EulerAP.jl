module EulerAP

using StaticArrays

using SparseArrays
using ForwardDiff

using SciMLBase
using OrdinaryDiffEq
using NonlinearSolve

using LinearSolve
using Pardiso

using HDF5
using RecipesBase

# --------------------------------------------------
# Core infrastructure
# --------------------------------------------------

include("basic_types.jl")

include("meshes/cartesian_mesh.jl")

include("equations/equations.jl")

include("solvers/solvers.jl")
include("solvers/fv_1d.jl")
include("solvers/fv_2d.jl")

include("semidiscretization/semidiscretization.jl")
include("semidiscretization/semidiscretization_hyperbolic.jl")

# --------------------------------------------------
# Time integration
# --------------------------------------------------

include("time_integration/time_integration.jl")
include("time_integration/backward_euler.jl")

# --------------------------------------------------
# IO
# --------------------------------------------------

include("io/save_solution.jl")

# --------------------------------------------------
# Plotting
# --------------------------------------------------

include("visualization/recipes.jl")

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

# Time Integrators
export AbstractTimeIntegrator
export ImplicitEulerCustom

# Solutions
export EulerAPSolution

# IO
export save_solution

end # module EulerAP