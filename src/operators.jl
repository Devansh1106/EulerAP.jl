function implicit_part!(du, u, p::RelaxationParams, t)

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

            f1, f2, f3 = rusanov_flux_x(
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

            f1, f2, f3 = rusanov_flux_y(
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