"""
    build_problem(; kwargs...)

Construct the initial state, grid coordinates, model parameters, and Jacobian
cache.
"""
function build_problem(;
    ic_func,
    size = nothing,
    nx = 32,        # optional parameters when nothing is provided by the user
    ny = 32,
    eps = 0.05,
    xmin = -1.0,
    xmax = 1.0,
    ymin = -1.0,
    ymax = 1.0,
    left_bc = :periodic,
    right_bc = :periodic,
    bottom_bc = :periodic,
    top_bc = :periodic,
    bc_funcs = nothing,
    flux = :rusanov,
    gamma = 1.4
)

    _normalize_tuple(value, ndims) = value isa Number ? 
        ntuple(_ -> Float64(value), ndims) :
        begin
            tuple_value = Tuple(value)
            length(tuple_value) == ndims || error("Expected $ndims entries, got $(length(tuple_value))")
            ntuple(d -> Float64(tuple_value[d]), ndims)
        end

    grid_size = size === nothing ? (nx, ny) : Tuple(size)
    ndims = length(grid_size)

    # Only for 1D and 2D for now
    domain_min = ndims == 2 &&
                 xmin isa Number && 
                 xmax isa Number ? 
                (Float64(xmin), Float64(ymin)) :
                 _normalize_tuple(xmin, ndims)

    domain_max = ndims == 2 && 
                 xmin isa Number && 
                 xmax isa Number ?
                (Float64(xmax), Float64(ymax)) : 
                 _normalize_tuple(xmax, ndims)

    dx = ntuple(d -> 
               (domain_max[d] - domain_min[d]) / grid_size[d], 
                ndims)

    # Generate arrays representing the cell centers for each ndims
    # Tuple of two arrays containing cell centres in each dimension.
    coords = ntuple(d -> range(domain_min[d] + dx[d] / 2, 
                               domain_max[d] - dx[d] / 2; 
                               length = grid_size[d]), ndims)

    # Create boundary condition configuration
    bc_config    = BCConfig(
        left     = left_bc,
        right    = right_bc,
        bottom   = bottom_bc,
        top      = top_bc,
        bc_funcs = bc_funcs
    )

    p = RelaxationParams(
        eps,
        grid_size,
        dx,
        domain_min,
        domain_max,
        bc_config
    )
    ncells = prod(grid_size)
    nvars = ndims + 1
    u0 = zeros(nvars * ncells)

    for I in CartesianIndices(grid_size)
        idx   = cell_index(I, p)
        point = ntuple(d -> coords[d][I[d]], ndims)
        state = Tuple(ic_func(point...))

        for v in 1:nvars
            u0[(v - 1) * ncells + idx] = state[v]
        end
    end

    if ndims == 1
        return u0, coords[1], nothing, p, build_jacobian_cache(p; 
                                                               flux = flux,
                                                               gamma = gamma)
    else
        return u0, coords[1], coords[2], p, build_jacobian_cache(p; 
                                                                 flux = flux,
                                                                 gamma = gamma)
    end
end