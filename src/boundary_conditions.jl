using StaticArrays

"""
    AbstractBC

Abstract boundary-condition type used by the ghost-cell lookup machinery.
"""
abstract type AbstractBC end

"""
    PeriodicBC

Periodic boundary condition.
"""
struct PeriodicBC <: AbstractBC end

"""
    DirichletBC(f)

Dirichlet boundary condition defined by a user function `f(x, y, t)` that
returns `(rho, mx, my)`.
"""
struct DirichletBC{F} <: AbstractBC
    bc_func::F
end

"""
    NeumannBC(f)

Neumann boundary condition defined by a user function `f(x, y, t)` returning
the normal derivatives `(d_rho/dn, d_mx/dn, d_my/dn)`.
"""
struct NeumannBC{F} <: AbstractBC
    bc_func::F
end

"""
    ExtrapolateBC

Extrapolation boundary condition.
For state vector (ρ, mx, my, ...):
- All components are extrapolated from the interior (zero normal gradient).
"""
struct ExtrapolateBC <: AbstractBC end

"""
    BCConfig

Configuration specifying boundary conditions on each side of the domain.
"""
struct BCConfig{L <:AbstractBC, R <:AbstractBC, B <:AbstractBC, T <:AbstractBC}
    left::L
    right::R
    bottom::B
    top::T
end

"""
    BCConfig(; left, right, bottom, top, bc_funcs=nothing)

Outer constructor for the `BCConfig` struct that parses and stores boundary condition 
specifications for a 2D computational domain.

This function processes boundary identifiers (symbols, strings, or tuple-wrapped functions) 
and instantiates concrete, type-stable boundary condition types (`PeriodicBC`, `DirichletBC`, 
`NeumannBC`) for each spatial grid boundary.

# Boundary Representation Layout
- `left`: Coordinate boundary at \$x = x_{\\text{min}}\$ (Axis 1, Sign -1)
- `right`: Coordinate boundary at \$x = x_{\\text{max}}\$ (Axis 1, Sign +1)
- `bottom`: Coordinate boundary at \$y = y_{\\text{min}}\$ (Axis 2, Sign -1)
- `top`: Coordinate boundary at \$y = y_{\\text{max}}\$ (Axis 2, Sign +1)


# Keyword Arguments
- `left`: Boundary type for the left wall.
- `right`: Boundary type for the right wall.
- `bottom`: Boundary type for the bottom wall.
- `top`: Boundary type for the top wall.
- `bc_funcs::Union{Nothing, NamedTuple}`: Named tuple mapping boundary keys to space-time 
  coordinate functions `f(x, y, t)` that return state vectors or normal derivatives.

# Returns
- `BCConfig`: A structural container holding the concrete boundary sub-types for dispatch 
  inside spatial flux routines.
"""
function BCConfig(;
    left,
    right,
    bottom,
    top,
    bc_funcs = nothing
)
    left_bc   = _parse_bc_spec(left, bc_funcs, :left)
    right_bc  = _parse_bc_spec(right, bc_funcs, :right)
    bottom_bc = _parse_bc_spec(bottom, bc_funcs, :bottom)
    top_bc    = _parse_bc_spec(top, bc_funcs, :top)
    
    return BCConfig(left_bc, right_bc, bottom_bc, top_bc)
end

function _parse_bc_spec(spec::Union{Symbol, Nothing}, 
                        bc_funcs::Union{Nothing, AbstractDict}, 
                        side::Symbol)
    
    # In the case of 1D when bottom and top are `Nothing`, it default to PeriodicBC 
    # No harm in this since 1D will not access them anyway.
    # TODO: is there a better for this which is dimension agnostic too?
    if spec === nothing
        return PeriodicBC()
    end

    if spec == :periodic
        return PeriodicBC()

    elseif spec == :dirichlet
        if bc_funcs === nothing || !haskey(bc_funcs, side)
            error("`bc_funcs` must provide key `:$side` for Dirichlet BC")
        end
        return DirichletBC(bc_funcs[side])

    elseif spec == :neumann
        if bc_funcs === nothing || !haskey(bc_funcs, side)
            error("`bc_funcs` must provide key `:$side` for Neumann BC")
        end
        return NeumannBC(bc_funcs[side])

    elseif spec == :extrapolate
        return ExtrapolateBC()

    else
        error("Unknown BC type: $spec. Must be :periodic, :dirichlet, :neumann, or :extrapolate")
    end
end

"""
    apply_bc(bc::AbstractBC, state_val, x, y, t, p::RelaxationParams)

Apply a boundary condition to get the value at a boundary ghost cell.
"""

function apply_bc(bc::PeriodicBC, 
                  state_val::AbstractVector, 
                  coords::Tuple, 
                  t::Real, 
                  p::RelaxationParams, 
                  axis::Int)

    return state_val
end

function apply_bc(bc::DirichletBC, 
                  state_val::AbstractVector, 
                  coords::Tuple, 
                  t::Real, 
                  p::RelaxationParams, 
                  axis::Int)

    # Explicitly casting ensures Dual numbers for Auto-Diff stay intact
    return SVector{length(state_val), eltype(state_val)}(bc.bc_func(coords..., t))
end

# Needs normal derivative from bc_func()
function apply_bc(bc::NeumannBC, 
                  state_val::AbstractVector, 
                  coords::Tuple, 
                  t::Real, 
                  p::RelaxationParams, 
                  axis::Int)
                  
    grad  = bc.bc_func(coords..., t)
    delta = p.dx[axis]
    let M = length(state_val), T = eltype(state_val)
        return SVector{M, T}(ntuple(i -> 
                             state_val[i] + grad[i] * delta, 
                             M))
    end
end

function apply_bc(bc::ExtrapolateBC, 
                  state_val::AbstractVector, 
                  coords::Tuple, 
                  t::Real, 
                  p::RelaxationParams, 
                  axis::Int)
    M = length(state_val)
    T = eltype(state_val)

    # Pure extrapolation: ghost cell perfectly mirrors the interior cell
    # This sets ∂ρ/∂n = 0, ∂mx/∂n = 0, ∂my/∂n = 0
    return SVector{M, T}(ntuple(i -> state_val[i], M))
end

"""
    get_bc_config(p::RelaxationParams) -> BCConfig

Retrieve BC configuration from RelaxationParams.
"""
get_bc_config(p::RelaxationParams) = p.bc_config

@inline function _side_bc(bcfg::BCConfig, axis::Int, sign::Int)
    if axis == 1
        return sign < 0 ? bcfg.left : bcfg.right

    elseif axis == 2
        return sign < 0 ? bcfg.bottom : bcfg.top

    else
        error("Boundary conditions are only defined for the first two axes in this model")
    end
end

@inline function _wrap_index(idx::Int, n::Int)
    return mod(idx - 1, n) + 1
end

@inline function _state_at(u::AbstractVector, 
                           I::CartesianIndex{NDIMS}, 
                           p::RelaxationParams{NDIMS}) where {NDIMS}

    idx = cell_index(I, p)
    nc  = ncells(p)
    return SVector{NDIMS + 1}(ntuple(v -> 
                              u[(v - 1) * nc + idx], 
                              NDIMS + 1))
end

"""
    boundary_sides(I::CartesianIndex, p::RelaxationParams)

Return all domain sides touched by `I`. This keeps corner cells from
collapsing to only the first out-of-bounds axis.
"""
function boundary_sides(I::CartesianIndex{NDIMS}, 
                        p::RelaxationParams{NDIMS}) where {NDIMS}

    coords = Tuple(I)
    sides  = Symbol[]

    for axis in 1:NDIMS
        if coords[axis] < 1
            push!(sides, axis == 1 ? :left : :bottom)
        elseif coords[axis] > p.size[axis]
            push!(sides, axis == 1 ? :right : :top)
        end
    end

    return Tuple(sides)
end

"""
    which_side(I::CartesianIndex, p::RelaxationParams) -> Union{Symbol, Nothing}

Return the first boundary side touched by `I`, or `nothing` if `I` is
interior. Use `boundary_sides` when corner awareness is required.
"""
function which_side(I::CartesianIndex{NDIMS}, 
                    p::RelaxationParams{NDIMS}) where {NDIMS}

    sides = boundary_sides(I, p)
    return isempty(sides) ? nothing : sides[1]
end

"""
    get_boundary_state(neighbor_i::Int, neighbor_j::Int, p::RelaxationParams, u, t::Float64)

Get the state at a neighbor cell, applying BCs if it's a ghost cell outside the domain.
"""
@inline function get_boundary_state(I::CartesianIndex{NDIMS}, 
                                    p::RelaxationParams{NDIMS}, 
                                    u::AbstractVector, 
                                    t::Real) where {NDIMS}
                                    
    coords = Tuple(I)

    # Finds the number of out-of-bounds (ghost) axes
    out_flags = ntuple(d -> 
                       coords[d] < 1 || coords[d] > p.size[d], 
                       Val(NDIMS))
    num_out   = sum(out_flags)

    # Fast path: Interior cell
    if num_out == 0
        return _state_at(u, I, p)
    end

    # Safety Guard: Explicitly reject corner ghost cells for 5-point stencils 
    if num_out > 1
        error(
            "Corner ghost cell accessed at $coords. " *
            "The current 5-point stencil configuration should not access multi-axis corners."
        )
    end

    # Find the single out-of-bounds axis type-stably (double out-of-bounds are corner ghost cells, hence rejected above)
    axis = sum(ntuple(d -> 
                      out_flags[d] ? d : 0, 
                      Val(NDIMS)))
    sign = coords[axis] < 1 ? -1 : 1

    # Fetch the BC via union splitting
    bc = _side_bc(get_bc_config(p), axis, sign)

    if bc isa PeriodicBC
        wrapped = ntuple(d -> 
                         d == axis ? _wrap_index(coords[d], p.size[d]) : clamp(coords[d], 1, p.size[d]), 
                         Val(NDIMS))
        return _state_at(u, 
                         CartesianIndex(wrapped), 
                         p)
    end

    # Type-stable geometry calculations for physical boundaries
    interior       = ntuple(d -> 
                            clamp(coords[d], 1, p.size[d]), 
                            Val(NDIMS))

    interior_state = _state_at(u, 
                               CartesianIndex(interior), 
                               p)
    
    # finds centre of the boundary face i.e. centre of left face; top face etc. given that cell is touching the boundary
    point = ntuple(d -> 
                   d == axis ? (sign < 0 ? p.domain_min[d] : p.domain_max[d]) : p.domain_min[d] + (interior[d] - 0.5) * p.dx[d],
                   Val(NDIMS))

    # Dirichlet and Neumann BCs are handled here using multiple dispatch.
    return SVector{NDIMS + 1, eltype(u)}(apply_bc(bc, 
                                                  interior_state, 
                                                  point, 
                                                  t, 
                                                  p, 
                                                  axis))
end

function get_boundary_state(neighbor_i::Int, 
                            neighbor_j::Int, 
                            p::RelaxationParams{2}, 
                            u::AbstractVector, 
                            t::Real)

    return get_boundary_state(CartesianIndex(neighbor_i, neighbor_j), 
                              p, 
                              u, 
                              t)
end

@inline function neighbor_index(I::CartesianIndex{NDIMS}, 
                                p::RelaxationParams{NDIMS}, 
                                axis::Int, 
                                sign::Int) where {NDIMS}

    coords  = Tuple(I)
    shifted = ntuple(d -> 
                     d == axis ? coords[d] + sign : 
                     coords[d], NDIMS)

    # For interior cells
    if all(1 <= shifted[d] <= p.size[d] for d in 1:NDIMS)
        return cell_index(CartesianIndex(shifted), p)
    end

    # For boundary cells; this change of index make sure build_jacobian_cache gets right neighbour index based on the boundary conditions. 
    bc = _side_bc(get_bc_config(p), axis, sign)

    if isa(bc, PeriodicBC)
        wrapped = ntuple(d -> 
                         d == axis ? _wrap_index(shifted[d], p.size[d]) : clamp(shifted[d], 1, p.size[d]), 
                         NDIMS)
        return cell_index(CartesianIndex(wrapped), p)

    elseif isa(bc, NeumannBC) || isa(bc, ExtrapolateBC)
        # Neumann and ExtrapolateBC ghost states depend on interior state (derivatives ≠ 0)
        # Return interior cell index so build_jacobian_cache maps derivatives correctly
        clamped = ntuple(d -> 
                        clamp(shifted[d], 1, p.size[d]), 
                        NDIMS)
        return cell_index(CartesianIndex(clamped), p)

    else
        # Dirichlet ghost states are constant (du_ghost/du_interior = 0).
        # Return 0 so build_jacobian_cache drops the derivative mapping.
        return 0
    end
end

function neighbor_index(i::Int, 
                        j::Int, 
                        p::RelaxationParams{2}, 
                        side::Symbol)

    # Format is: (axis, sign)
    axis, sign = side == :left   ? (1, -1) : 
                 side == :right  ? (1, 1)  : 
                 side == :bottom ? (2, -1) :
                 side == :top    ? (2, 1)  : 
                 error("Unknown side")

    return neighbor_index(CartesianIndex(i, j), 
                          p, 
                          axis, 
                          sign)
end
