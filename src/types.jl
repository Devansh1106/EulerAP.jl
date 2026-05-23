struct FluxPair{FX,FY}
    flux_x::FX
    flux_y::FY
end

# FluxPair(flux_x, flux_y) = FluxPair{typeof(flux_x), typeof(flux_y)}(flux_x, flux_y)

struct RelaxationParams
    eps::Float64
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    bc_config::Any  # BCConfig (defined in boundary_conditions.jl)
end

@inline function cell_index(i, j, p::RelaxationParams)
    return (j - 1) * p.nx + i
end

mutable struct ImplicitStepData
    model::RelaxationParams
    dt::Float64
    t::Float64
    u_prev::Vector{Float64}
    rhs_cache::AbstractVector
    flux::FluxPair
end

mutable struct RunStats
    total_time::Float64
    total_bytes::Int
    total_gctime::Float64
    step_times::Vector{Float64}
    step_bytes::Vector{Int}
    step_gctimes::Vector{Float64}
end

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