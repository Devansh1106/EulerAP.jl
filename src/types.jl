"""
    FluxPair{FX,FY}

Container for a pair of numerical flux functions. `flux_x` is used for
vertical interfaces and `flux_y` is used for horizontal interfaces.
"""
struct FluxPair{FX,FY}
    flux_x::FX
    flux_y::FY
end


"""
    RelaxationParams

Model and grid parameters for the 2D relaxation-Euler problem.
"""
struct RelaxationParams{BC}
    eps::Float64
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    bc_config::BC  # BCConfig (defined in boundary_conditions.jl)
end

@inline function cell_index(i, j, p::RelaxationParams)
    return (j - 1) * p.nx + i
end

"""
    ImplicitStepData

Mutable container passed through the nonlinear solver for one backward-Euler
step.
"""
mutable struct ImplicitStepData{M, F}
    model::M
    dt::Float64
    t::Float64
    u_prev::Vector{Float64}
    flux::F
end

"""
    RunStats

Timing and allocation statistics collected while advancing the solution.
"""
mutable struct RunStats
    total_time::Float64
    total_bytes::Int
    total_gctime::Float64
    step_times::Vector{Float64}
    step_bytes::Vector{Int}
    step_gctimes::Vector{Float64}
end

"""
    RunStats(nsteps::Int)

Create an empty statistics container sized for `nsteps` time steps.
"""
function RunStats(nsteps::Int)
    return RunStats(
        0.0,
        0,
        0.0,
        zeros(Float64, nsteps),
        zeros(Int, nsteps),
        zeros(Float64, nsteps)
    )
end