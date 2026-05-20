function build_jacobian_prototype(p::RelaxationParams)

    ncells = p.nx * p.ny

    jac_prototype = spzeros(
        Float64,
        3 * ncells,
        3 * ncells
    )

    for j in 1:p.ny
        for i in 1:p.nx

            cell = cell_index(i, j, p)
            
            # boundary conditions
            left = cell_index(
                i == 1 ? p.nx : i - 1,
                j,
                p
            )

            right = cell_index(
                i == p.nx ? 1 : i + 1,
                j,
                p
            )

            bottom = cell_index(
                i,
                j == 1 ? p.ny : j - 1,
                p
            )

            top = cell_index(
                i,
                j == p.ny ? 1 : j + 1,
                p
            )

            for row_var in 0:2

                row = row_var * ncells + cell

                for neighbor in (cell, left, right, bottom, top)
                    for col_var in 0:2

                        col = col_var * ncells + neighbor

                        jac_prototype[row, col] = 1.0
                    end
                end
            end
        end
    end

    return jac_prototype
end