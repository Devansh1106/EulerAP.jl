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

# Cache mapping from a prototype SparseMatrixCSC -> positions array
const J_POS_CACHE = IdDict{SparseMatrixCSC{Float64,Int}, Array{Int,3}}()

function _build_positions_from_prototype(jp::SparseMatrixCSC{Float64,Int}, p::RelaxationParams)
    ncells = p.nx * p.ny
    # positions: row_var (1..3) x local_col (1..15) x cell (1..ncells)
    positions = Array{Int,3}(undef, 3, 15, ncells)

    colptr = jp.colptr
    rowval = jp.rowval

    # helper to find nz index for (row, col)
    function find_pos(col::Int, row::Int)
        start = colptr[col]
        stop = colptr[col+1] - 1
        for k in start:stop
            if rowval[k] == row
                return k
            end
        end
        error("Pattern mismatch: (row=$row, col=$col) not found in prototype")
    end

    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            left   = cell_index(i == 1    ? p.nx : i - 1, j, p)
            right  = cell_index(i == p.nx ? 1    : i + 1, j, p)
            bottom = cell_index(i, j == 1    ? p.ny : j - 1, p)
            top    = cell_index(i, j == p.ny ? 1    : j + 1, p)

            neighbors = (center, left, right, bottom, top)

            for row_var in 0:2
                row = row_var * ncells + center
                for local_col_idx in 1:15
                    neighbor_idx = Int(div(local_col_idx - 1, 3)) + 1 # 1..5
                    col_var = Int((local_col_idx - 1) % 3)
                    neighbor = neighbors[neighbor_idx]
                    col = col_var * ncells + neighbor
                    positions[row_var+1, local_col_idx, center] = find_pos(col, row)
                end
            end
        end
    end

    return positions
end

function local_jacobian(local_u, p::RelaxationParams; flux = :rusanov)

    flux = resolve_flux(flux)

    # convert local state (length 15) to a Static Vector for fast AD
    lu = SVector{15, eltype(local_u)}(local_u...)

    # local_residual expects a 15-element vector; wrap to return an SVector{3}
    f = u -> begin
        drho, dmx, dmy = local_residual(u, p; flux = flux)
        return SVector(drho, dmx, dmy)
    end

    J = ForwardDiff.jacobian(f, lu)

    # ensure result is an SMatrix{3,15}
    return SMatrix{3,15,eltype(J)}(J)
end

function assemble_global_jacobian(Jp::SparseMatrixCSC{Float64,Int}, u, p::RelaxationParams; flux = :rusanov, t = 0.0)

    flux = resolve_flux(flux)

    ncells = p.nx * p.ny

    # get or build positions mapping for this prototype
    if haskey(J_POS_CACHE, Jp)
        positions = J_POS_CACHE[Jp]
    else
        positions = _build_positions_from_prototype(Jp, p)
        J_POS_CACHE[Jp] = positions
    end

    nz = Jp.nzval

    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            local_u = gather_local_state(u, i, j, p, t)
            Jloc = local_jacobian(local_u, p; flux = flux)

            for row_var in 0:2
                for local_col_idx in 1:15
                    pos = positions[row_var+1, local_col_idx, center]
                    nz[pos] = Jloc[row_var + 1, local_col_idx]
                end
            end
        end
    end

    return Jp
end