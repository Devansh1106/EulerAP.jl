"""
    build_problem(; kwargs...)

Construct the initial state, grid coordinates, model parameters, and Jacobian
cache.

# Keyword Arguments
- `ic_func`: Initial condition function `f(x..., t) -> (rho, mx, my, ...)`.  
             Called with spatial coordinates and initial time: `ic_func(x..., tspan[1])`.
- `size`: Tuple of grid sizes per dimension (e.g., (32,) for 1D, (32, 32) for 2D)
- `eps`: Relaxation parameter
- `domain_min`: Tuple of domain minimums per dimension
- `domain_max`: Tuple of domain maximums per dimension
- `tspan`: Time span `(t0, t1)`. `ic_func` is called with time as the last argument:  
           `ic_func(x..., t0)`.
- `left_bc, right_bc, bottom_bc, top_bc`: Boundary condition symbols
- `bc_funcs`: Dict of boundary functions for non-periodic BCs
- `flux`: Flux scheme (e.g., `:rusanov`, `:energy_stable`).
- `gamma`: Adiabatic exponent.
- `eta`: Numerical diffusion coefficient η (only used with `:energy_stable` flux).
  The time-step size `dt` is provided to `solve_backward_euler` and the flux
  computes `η·Δt` at each step automatically.
"""
function build_problem(;
    ic_func,
    size,
    eps,
    domain_min,
    domain_max,
    tspan,
    left_bc,
    right_bc,
    bottom_bc = nothing,
    top_bc    = nothing,
    bc_funcs   = nothing,
    flux,
    gamma,
    eta        = nothing
)

    grid_size = Tuple(size)
    ndims     = length(grid_size)

    @assert length(domain_min) == ndims "domain_min must have $ndims entries"
    @assert length(domain_max) == ndims "domain_max must have $ndims entries"

    domain_min = Tuple(Float64.(domain_min))
    domain_max = Tuple(Float64.(domain_max))

    dx = ntuple(d -> (domain_max[d] - domain_min[d]) / grid_size[d], ndims)

    # Cell centers for each dimension
    coords = ntuple(d -> range(domain_min[d] + dx[d] / 2, 
                               domain_max[d] - dx[d] / 2; 
                               length = grid_size[d]), ndims)

    # Create boundary condition configuration
    bc_config = BCConfig(
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
        state = Tuple(ic_func(point..., tspan[1]))

        for v in 1:nvars
            u0[(v - 1) * ncells + idx] = state[v]
        end
    end

    resolved_flux = resolve_flux(flux; gamma = gamma, eta = eta)
    return u0, coords, p, build_jacobian_cache(p; resolved_flux = resolved_flux)
end