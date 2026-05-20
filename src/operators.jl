function implicit_part!(du, u, p::RelaxationParams, t; flux = :rusanov)

    flux = resolve_flux(flux)

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

    for j in 1:p.ny
        for i in 1:p.nx

            i_right = i == p.nx ? 1 : i + 1 # TODO: is it because of periodic boundary?

            l = cell_index(i, j, p)
            r = cell_index(i_right, j, p)

            f1, f2, f3 = flux.flux_x(
                rho[l], mx[l], my[l],
                rho[r], mx[r], my[r],
                p.eps
            )

            drho[l] -= f1 / p.dx
            dmx[l]  -= f2 / p.dx
            dmy[l]  -= f3 / p.dx

            drho[r] += f1 / p.dx
            dmx[r]  += f2 / p.dx
            dmy[r]  += f3 / p.dx
        end
    end

    for j in 1:p.ny

        j_top = j == p.ny ? 1 : j + 1 # TODO: is it because of periodic boundary?

        for i in 1:p.nx

            b = cell_index(i, j, p)
            t = cell_index(i, j_top, p)

            f1, f2, f3 = flux.flux_y(
                rho[b], mx[b], my[b],
                rho[t], mx[t], my[t],
                p.eps
            )

            drho[b] -= f1 / p.dy
            dmx[b]  -= f2 / p.dy
            dmy[b]  -= f3 / p.dy

            drho[t] += f1 / p.dy
            dmx[t]  += f2 / p.dy
            dmy[t]  += f3 / p.dy
        end
    end

    return nothing
end

function gather_local_state(u, i::Int, j::Int, p::RelaxationParams)

    ncells = p.nx * p.ny
    left   = cell_index(i == 1    ? p.nx : i - 1, j, p)
    right  = cell_index(i == p.nx ? 1    : i + 1, j, p)
    bottom = cell_index(i, j == 1    ? p.ny : j - 1, p)
    top    = cell_index(i, j == p.ny ? 1    : j + 1, p)
    center = cell_index(i, j, p)

    idxs = (
        center,
        left,
        right,
        bottom,
        top
    )
    local_u = zeros(eltype(u), 15)
    k = 1
    for idx in idxs
        local_u[k]      = u[idx]
        local_u[k + 1]  = u[ncells + idx]
        local_u[k + 2]  = u[2 * ncells + idx]
        k += 3
    end
    return local_u
end

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

    return (drho, dmx, dmy)
end