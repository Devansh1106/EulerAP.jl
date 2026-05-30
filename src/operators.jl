"""
    implicit_part!(du, u, p::RelaxationParams, t; flux=:rusanov)

Assemble the finite-volume spatial operator and relaxation source terms into
`du` for the stacked state vector `u`.
"""
function implicit_part!(du, u, p::RelaxationParams, t; flux = :rusanov)

    flux = resolve_flux(flux)
    bcfg = get_bc_config(p)

    ncells = p.nx * p.ny

    rho = @view u[1:ncells]
    mx  = @view u[ncells + 1:2 * ncells]
    my  = @view u[2 * ncells + 1:3 * ncells]

    drho = @view du[1:ncells]
    dmx  = @view du[ncells + 1:2 * ncells]
    dmy  = @view du[2 * ncells + 1:3 * ncells]

    fill!(drho, 0.0)
    fill!(dmx, 0.0)
    fill!(dmy, 0.0)

    if !isa(bcfg.left, PeriodicBC)
        for j in 1:p.ny
            l = cell_index(1, j, p)
            rho_r = rho[l]; mx_r = mx[l]; my_r = my[l]
            rho_l, mx_l, my_l = get_boundary_state(0, j, p, u, t)

            f1, f2, f3 = flux.flux_x(
                rho_l, mx_l, my_l,
                rho_r, mx_r, my_r,
                p.eps
            )
            drho[l] += f1 / p.dx
            dmx[l]  += f2 / p.dx
            dmy[l]  += f3 / p.dx
        end
    end

    if !isa(bcfg.bottom, PeriodicBC)
        for i in 1:p.nx
            b = cell_index(i, 1, p)

            rho_t = rho[b]; mx_t = mx[b]; my_t = my[b]
            rho_b, mx_b, my_b = get_boundary_state(i, 0, p, u, t)

            f1, f2, f3 = flux.flux_y(
                rho_b, mx_b, my_b,
                rho_t, mx_t, my_t,
                p.eps
            )
            drho[b] += f1 / p.dy
            dmx[b]  += f2 / p.dy
            dmy[b]  += f3 / p.dy
        end
    end

    for j in 1:p.ny
        for i in 1:p.nx
            # neighbor to the right; allow indices outside domain so BC handler
            # (`get_boundary_state`) can supply ghost values for non-periodic BCs
            i_right = i + 1
            l = cell_index(i, j, p)

            # left (interior) state
            rho_l = rho[l]; mx_l = mx[l]; my_l = my[l]

            # right state: either interior/periodic (has index) or obtained from BCs
            if 1 <= i_right <= p.nx
                r = cell_index(i_right, j, p)
                rho_r = rho[r]; mx_r = mx[r]; my_r = my[r]
                right_interior = true
            elseif isa(get_bc_config(p).right, PeriodicBC)
                r = cell_index(1, j, p)
                rho_r = rho[r]; mx_r = mx[r]; my_r = my[r]
                right_interior = true
            else
                rho_r, mx_r, my_r = get_boundary_state(i_right, j, p, u, t)
                right_interior = false
            end

            f1, f2, f3 = flux.flux_x(
                rho_l, mx_l, my_l,
                rho_r, mx_r, my_r,
                p.eps
            )

            drho[l] -= f1 / p.dx
            dmx[l]  -= f2 / p.dx
            dmy[l]  -= f3 / p.dx

            if right_interior
                drho[r] += f1 / p.dx
                dmx[r]  += f2 / p.dx
                dmy[r]  += f3 / p.dx
            end
        end
    end

    for j in 1:p.ny
        # neighbor above; allow out-of-domain index so BCs can be applied
        j_top = j + 1

        for i in 1:p.nx
            b = cell_index(i, j, p)
            rho_b = rho[b]; mx_b = mx[b]; my_b = my[b]

            if 1 <= j_top <= p.ny
                top_cell = cell_index(i, j_top, p)
                rho_t = rho[top_cell]; mx_t = mx[top_cell]; my_t = my[top_cell]
                top_interior = true
            elseif isa(get_bc_config(p).top, PeriodicBC)
                top_cell = cell_index(i, 1, p)
                rho_t = rho[top_cell]; mx_t = mx[top_cell]; my_t = my[top_cell]
                top_interior = true
            else
                rho_t, mx_t, my_t = get_boundary_state(i, j_top, p, u, t)
                top_interior = false
            end

            f1, f2, f3 = flux.flux_y(
                rho_b, mx_b, my_b,
                rho_t, mx_t, my_t,
                p.eps
            )

            drho[b] -= f1 / p.dy
            dmx[b]  -= f2 / p.dy
            dmy[b]  -= f3 / p.dy

            if top_interior
                drho[top_cell] += f1 / p.dy
                dmx[top_cell]  += f2 / p.dy
                dmy[top_cell]  += f3 / p.dy
            end
        end
    end
    # TODO: Make it more general so that if there is a source for \rho that can also be handled without any further changes to the core code.
    # Relaxation source terms: rho has no source.
    @. dmx -= mx / p.eps
    @. dmy -= my / p.eps
    return nothing
end

"""
    gather_local_state(u, i, j, p, t=0.0)

Collect the 3×5 stencil around cell `(i, j)` in the layout expected by
`local_residual`.
"""
function gather_local_state(u, i::Int, j::Int, p::RelaxationParams, t::Float64 = 0.0)
    ncells = p.nx * p.ny
    center = cell_index(i, j, p)
    # Use neighbor indices that may lie outside the domain boundary so that
    # `get_boundary_state` can apply the configured BC (periodic, Dirichlet,
    # Neumann)
    left_i   = i - 1
    right_i  = i + 1
    bottom_j = j - 1
    top_j    = j + 1

    # Get states at center and neighbors, applying BCs if needed
    rho_c, mx_c, my_c = get_boundary_state(i, j, p, u, t)
    rho_l, mx_l, my_l = get_boundary_state(left_i, j, p, u, t)
    rho_r, mx_r, my_r = get_boundary_state(right_i, j, p, u, t)
    rho_b, mx_b, my_b = get_boundary_state(i, bottom_j, p, u, t)
    rho_t, mx_t, my_t = get_boundary_state(i, top_j, p, u, t)

    return ( # this tuple is actually local_u
        rho_c, mx_c, my_c,
        rho_l, mx_l, my_l,
        rho_r, mx_r, my_r,
        rho_b, mx_b, my_b,
        rho_t, mx_t, my_t,
    )
end

"""
    local_residual(local_u, p::RelaxationParams; flux=:rusanov)

Return the three-component residual for a single cell from its local stencil.
"""
function local_residual(local_u, p::RelaxationParams; flux = :rusanov)
    flux = resolve_flux(flux)

    # local_u layout: [center(1:3), left(4:6), right(7:9), bottom(10:12), top(13:15)]

    # center
    rho_c = local_u[1]
    mx_c  = local_u[2]
    my_c  = local_u[3]

    # left
    rho_l = local_u[4]
    mx_l  = local_u[5]
    my_l  = local_u[6]

    # right
    rho_r = local_u[7]
    mx_r  = local_u[8]
    my_r  = local_u[9]

    # bottom
    rho_b = local_u[10]
    mx_b  = local_u[11]
    my_b  = local_u[12]

    # top
    rho_t = local_u[13]
    mx_t  = local_u[14]
    my_t  = local_u[15]

    # fluxes in x-direction: between left-center and center-right
    f_left = flux.flux_x(rho_l, mx_l, my_l, rho_c, mx_c, my_c, p.eps)
    f_right = flux.flux_x(rho_c, mx_c, my_c, rho_r, mx_r, my_r, p.eps)

    # fluxes in y-direction: between bottom-center and center-top
    f_bottom = flux.flux_y(rho_b, mx_b, my_b, rho_c, mx_c, my_c, p.eps)
    f_top = flux.flux_y(rho_c, mx_c, my_c, rho_t, mx_t, my_t, p.eps)

    # residual contributions: center receives +f_left - f_right (x) and +f_bottom - f_top (y)
    drho = -(f_right[1] - f_left[1]) / p.dx - (f_top[1] - f_bottom[1]) / p.dy
    dmx  = -(f_right[2] - f_left[2]) / p.dx - (f_top[2] - f_bottom[2]) / p.dy
    dmy  = -(f_right[3] - f_left[3]) / p.dx - (f_top[3] - f_bottom[3]) / p.dy

    # TODO: Make it more general so that if there is a source for \rho that can also be handled without any further changes to the core code.
    # Relaxation source terms for the momentum equations. rho has no source.
    dmx -= mx_c / p.eps
    dmy -= my_c / p.eps
    return (drho, dmx, dmy)
end
