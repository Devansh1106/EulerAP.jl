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
    bc_func::F  # (x, y, t) -> (rho, mx, my)
end

"""
    NeumannBC(f)

Neumann boundary condition defined by a user function `f(x, y, t)` returning
the normal derivatives `(d_rho/dn, d_mx/dn, d_my/dn)`.
"""
struct NeumannBC{F} <: AbstractBC
    bc_func::F  # (x, y, t) -> (rho, mx, my) - gradient values
end

"""
    BCConfig

Configuration specifying boundary conditions on each side of the domain.
"""
struct BCConfig
    left::AbstractBC
    right::AbstractBC
    bottom::AbstractBC
    top::AbstractBC
end

"""
    BCConfig(; left=:periodic, right=:periodic, bottom=:periodic, top=:periodic)

Convenience constructor that accepts symbols and creates default periodic BCs.
"""
function BCConfig(; # default values
    left = :periodic,
    right = :periodic,
    bottom = :periodic,
    top = :periodic,
    bc_funcs = nothing
)
    default_func = (x, y, t) -> (0.0, 0.0, 0.0)
    
    left_bc = _parse_bc_spec(left, bc_funcs, :left, default_func)
    right_bc = _parse_bc_spec(right, bc_funcs, :right, default_func)
    bottom_bc = _parse_bc_spec(bottom, bc_funcs, :bottom, default_func)
    top_bc = _parse_bc_spec(top, bc_funcs, :top, default_func)
    
    return BCConfig(left_bc, right_bc, bottom_bc, top_bc)
end

function _parse_bc_spec(spec, bc_funcs, side, default_func)
    if spec == :periodic
        return PeriodicBC()
    elseif spec == :dirichlet
        func = (bc_funcs !== nothing && haskey(bc_funcs, side)) ? bc_funcs[side] : default_func
        return DirichletBC(func)
    elseif spec == :neumann
        func = (bc_funcs !== nothing && haskey(bc_funcs, side)) ? bc_funcs[side] : default_func
        return NeumannBC(func)
    else
        error("Unknown BC type: $spec. Must be :periodic, :dirichlet, or :neumann")
    end
end

"""
    apply_bc(bc::AbstractBC, state_val, x, y, t, p::RelaxationParams)

Apply a boundary condition to get the value at a boundary ghost cell.
"""
function apply_bc(bc::PeriodicBC, state_val, x, y, t, p)
    # Periodic wrapping is handled by the ghost-cell lookup.
    # This fallback keeps the interior state unchanged.
    return state_val
end

function apply_bc(bc::DirichletBC, state_val, x, y, t, p)
    # Dirichlet: enforce the BC function value at the boundary
    return bc.bc_func(x, y, t)
end

function apply_bc(bc::NeumannBC, state_val, x, y, t, p)
    # Neumann: user provides normal derivative values via `bc.bc_func(x,y,t)`
    # We perform a first-order extrapolation for the ghost cell:
    #   ghost = interior + (d/dn)*delta
    # where `delta` is the signed distance from the interior cell center to
    # the ghost cell center (±dx or ±dy depending on the side).
    grad = bc.bc_func(x, y, t)
    # grad is expected to be a 3-tuple: (d_rho/dn, d_mx/dn, d_my/dn)
    d_rho, d_mx, d_my = grad

    # Determine signed delta based on which boundary location was passed
    # (get_boundary_state passes x=p.xmin/p.xmax for left/right interior)
    delta = 0.0
    if isapprox(x, p.xmin; atol = 0)
        delta = -p.dx
    elseif isapprox(x, p.xmax; atol = 0)
        delta = p.dx
    elseif isapprox(y, p.ymin; atol = 0)
        delta = -p.dy
    elseif isapprox(y, p.ymax; atol = 0)
        delta = p.dy
    else
        # Fallback: if we cannot determine side, assume zero change
        delta = 0.0
    end

    return (state_val[1] + d_rho * delta,
            state_val[2] + d_mx * delta,
            state_val[3] + d_my * delta)
end

"""
    get_bc_config(p::RelaxationParams) -> BCConfig

Retrieve BC configuration from RelaxationParams.
"""
get_bc_config(p) = p.bc_config

@inline function _wrap_index(idx::Int, n::Int)
    return mod(idx - 1, n) + 1
end

@inline function _state_at(u, i::Int, j::Int, p::RelaxationParams)
    idx = cell_index(i, j, p)
    ncells = p.nx * p.ny
    return (u[idx], u[ncells + idx], u[2 * ncells + idx])
end

"""
    which_side(i::Int, j::Int, p::RelaxationParams) -> Union{Symbol, Nothing}

Determine which boundary side a cell index (i,j) is on, if any.
Returns :left, :right, :bottom, :top, or nothing if interior. Corners are also taken care here.
"""
function which_side(i::Int, j::Int, p::RelaxationParams)
    on_left = (i == 1)
    on_right = (i == p.nx)
    on_bottom = (j == 1)
    on_top = (j == p.ny)
    
    # If on a corner, prioritize directions
    # corners are on either left or right boundary
    # here corners on left are considered on left boundary
    # corners on right are considered on right boundary
    if on_left return :left
    elseif on_right return :right
    elseif on_bottom return :bottom
    elseif on_top return :top
    else return nothing
    end
end

"""
    get_boundary_state(neighbor_i::Int, neighbor_j::Int, p::RelaxationParams, u, t::Float64)

Get the state at a neighbor cell, applying BCs if it's a ghost cell outside the domain.
"""
function get_boundary_state(neighbor_i::Int, neighbor_j::Int, p::RelaxationParams, u, t::Float64)
    
    bcfg = get_bc_config(p)
    
    # Interior cell - return normally
    if 1 <= neighbor_i <= p.nx && 1 <= neighbor_j <= p.ny
        return _state_at(u, neighbor_i, neighbor_j, p)
    end
    
    # Determine which boundary this is and apply the appropriate BC
    if neighbor_i < 1  # Left ghost cell
        if isa(bcfg.left, PeriodicBC)
            wrapped_j = (1 <= neighbor_j <= p.ny) ? neighbor_j : _wrap_index(neighbor_j, p.ny)
            return _state_at(u, p.nx, wrapped_j, p)
        end

        # Interior cell just inside boundary
        interior_i = 1 
        interior_j = neighbor_j
        if interior_j < 1 || interior_j > p.ny # if it's a ghost cell in j
            interior_j = mod(interior_j - 1, p.ny) + 1  # Wrap if needed
        end
        x_interior = p.xmin
        interior_state = _state_at(u, interior_i, interior_j, p)
        y_val = p.ymin + (interior_j - 0.5) * p.dy
        bc_state = apply_bc(bcfg.left, interior_state, x_interior, y_val, t, p)
        return bc_state
        
    elseif neighbor_i > p.nx  # Right ghost cell
        if isa(bcfg.right, PeriodicBC)
            wrapped_j = (1 <= neighbor_j <= p.ny) ? neighbor_j : _wrap_index(neighbor_j, p.ny)
            return _state_at(u, 1, wrapped_j, p)
        end

        interior_i, interior_j = p.nx, neighbor_j
        if interior_j < 1 || interior_j > p.ny
            interior_j = mod(interior_j - 1, p.ny) + 1
        end
        x_interior = p.xmax
        interior_state = _state_at(u, interior_i, interior_j, p)
        y_val = p.ymin + (interior_j - 0.5) * p.dy
        bc_state = apply_bc(bcfg.right, interior_state, x_interior, y_val, t, p)
        return bc_state
        
    elseif neighbor_j < 1  # Bottom ghost cell
        if isa(bcfg.bottom, PeriodicBC)
            wrapped_i = 1 <= neighbor_i <= p.nx ? neighbor_i : _wrap_index(neighbor_i, p.nx)
            return _state_at(u, wrapped_i, p.ny, p)
        end

        interior_i, interior_j = neighbor_i, 1
        if interior_i < 1 || interior_i > p.nx
            interior_i = mod(interior_i - 1, p.nx) + 1
        end
        y_interior = p.ymin
        interior_state = _state_at(u, interior_i, interior_j, p)
        x_val = p.xmin + (interior_i - 0.5) * p.dx
        bc_state = apply_bc(bcfg.bottom, interior_state, x_val, y_interior, t, p)
        return bc_state
        
    elseif neighbor_j > p.ny  # Top ghost cell
        if isa(bcfg.top, PeriodicBC)
            wrapped_i = 1 <= neighbor_i <= p.nx ? neighbor_i : _wrap_index(neighbor_i, p.nx)
            return _state_at(u, wrapped_i, 1, p)
        end

        interior_i, interior_j = neighbor_i, p.ny
        if interior_i < 1 || interior_i > p.nx
            interior_i = mod(interior_i - 1, p.nx) + 1
        end
        y_interior = p.ymax
        interior_state = _state_at(u, interior_i, interior_j, p)
        x_val = p.xmin + (interior_i - 0.5) * p.dx
        bc_state = apply_bc(bcfg.top, interior_state, x_val, y_interior, t, p)
        return bc_state
    end
    
    error("Unexpected neighbor location: ($neighbor_i, $neighbor_j)")
end
