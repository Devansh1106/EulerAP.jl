"""
    FluxPair{F}

Container for a generic numerical flux function.
"""
struct FluxPair{F}
    flux::F
end

"""
    RelaxationParams{NDIMS,BC}

Model and grid parameters for the relaxation-Euler problem in `NDIMS`
spatial dimensions.
"""
struct RelaxationParams{NDIMS, BC}
    eps::Float64
    size::NTuple{NDIMS, Int}
    dx::NTuple{NDIMS, Float64}
    domain_min::NTuple{NDIMS, Float64}
    domain_max::NTuple{NDIMS, Float64}
    bc_config::BC  # BCConfig (defined in boundary_conditions.jl)
end

@inline ndims(p::RelaxationParams{NDIMS}) where {NDIMS} = NDIMS
@inline nvars(p::RelaxationParams{NDIMS}) where {NDIMS} = NDIMS + 1
@inline ncells(p::RelaxationParams)                     = prod(p.size)

@inline function cell_index(I::CartesianIndex{NDIMS}, 
                            p::RelaxationParams{NDIMS}) where {NDIMS}

    return LinearIndices(p.size)[I]
end

@inline function cell_index(p::RelaxationParams{NDIMS}, 
                            indices::Vararg{Int, NDIMS}) where {NDIMS}

    return cell_index(CartesianIndex(indices), p)
end

@inline function cell_coords(I::CartesianIndex{NDIMS}, 
                             p::RelaxationParams{NDIMS}) where {NDIMS}

    return ntuple(d -> 
                  p.domain_min[d] + (Tuple(I)[d] - 0.5) * p.dx[d], 
                  NDIMS)
end

function Base.getproperty(p::RelaxationParams, s::Symbol)
    if s === :nx
        return getfield(p, :size)[1]

    elseif s === :ny
        return length(getfield(p, :size)) >= 2 ? getfield(p, :size)[2] : 1

    elseif s === :xmin
        return getfield(p, :domain_min)[1]

    elseif s === :xmax
        return getfield(p, :domain_max)[1]

    elseif s === :ymin
        return length(getfield(p, :domain_min)) >= 2 ? getfield(p, :domain_min)[2] : getfield(p, :domain_min)[1]

    elseif s === :ymax
        return length(getfield(p, :domain_max)) >= 2 ? getfield(p, :domain_max)[2] : getfield(p, :domain_max)[1]

    else
        return getfield(p, s)
    end
end

# For 2D
@inline cell_index(i::Int, 
                   j::Int, 
                   p::RelaxationParams{2}) = cell_index(CartesianIndex(i, j), p)

# For 1D
@inline cell_index(i::Int, 
                   p::RelaxationParams{1}) = cell_index(CartesianIndex(i), p)

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