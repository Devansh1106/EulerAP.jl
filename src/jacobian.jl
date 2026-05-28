@inline function neighbor_index(i, j, p, side)

    bcfg = get_bc_config(p)

    if side == :left

        if i == 1
            if isa(bcfg.left, PeriodicBC)
                return cell_index(p.nx, j, p)
            else
                return 0
            end
        end

        return cell_index(i - 1, j, p)

    elseif side == :right

        if i == p.nx
            if isa(bcfg.right, PeriodicBC)
                return cell_index(1, j, p)
            else
                return 0
            end
        end

        return cell_index(i + 1, j, p)

    elseif side == :bottom

        if j == 1
            if isa(bcfg.bottom, PeriodicBC)
                return cell_index(i, p.ny, p)
            else
                return 0
            end
        end

        return cell_index(i, j - 1, p)

    elseif side == :top

        if j == p.ny
            if isa(bcfg.top, PeriodicBC)
                return cell_index(i, 1, p)
            else
                return 0
            end
        end

        return cell_index(i, j + 1, p)

    end

    error("Unknown side")
end

build_jacobian_prototype(p::RelaxationParams) = build_jacobian_cache(p)

function build_jacobian_cache(p::RelaxationParams)

    ncells = p.nx * p.ny
    n = 3 * ncells

    rows = Int[]
    cols = Int[]

    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            left   = neighbor_index(i, j, p, :left)
            right  = neighbor_index(i, j, p, :right)
            bottom = neighbor_index(i, j, p, :bottom)
            top    = neighbor_index(i, j, p, :top)

            neighbors = (center, left, right, bottom, top)

            for row_var in 0:2
                row = row_var * ncells + center
                for neighbor in neighbors
                    if neighbor == 0
                        continue
                    end
                    for col_var in 0:2
                        col = col_var * ncells + neighbor
                        push!(rows, row)
                        push!(cols, col)
                    end
                end
            end
        end
    end

    J = sparse(rows, cols, ones(Float64, length(rows)), n, n)
    J.nzval .= 0.0

    posmap = Dict{Tuple{Int,Int},Int}()
    for col in 1:n
        for k in J.colptr[col]:(J.colptr[col+1] - 1)
            posmap[(col, J.rowval[k])] = k
        end
    end

    positions = Array{Int,3}(undef, 3, 15, ncells)
    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)

            left   = neighbor_index(i, j, p, :left)
            right  = neighbor_index(i, j, p, :right)
            bottom = neighbor_index(i, j, p, :bottom)
            top    = neighbor_index(i, j, p, :top)

            neighbors = (center, left, right, bottom, top)

            for row_var in 0:2
                row = row_var * ncells + center
                for local_col_idx in 1:15
                    neighbor_idx = Int(div(local_col_idx - 1, 3)) + 1 # 1..5
                    col_var = Int((local_col_idx - 1) % 3)
                    neighbor = neighbors[neighbor_idx]
                    if neighbor == 0
                        positions[row_var+1, local_col_idx, center] = 0
                    else
                        col = col_var * ncells + neighbor
                        positions[row_var+1, local_col_idx, center] = get(posmap, (col, row), 0)
                    end
                end
            end
        end
    end

    J_POS_CACHE[J] = positions

    return J
end

# Cache mapping from a prototype SparseMatrixCSC -> positions array
const J_POS_CACHE = IdDict{SparseMatrixCSC{Float64,Int}, Array{Int,3}}()

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

    positions = get(J_POS_CACHE, Jp, nothing)
    positions === nothing && error("Jacobian prototype was not built with build_jacobian_cache")

    nz = Jp.nzval

    for j in 1:p.ny
        for i in 1:p.nx
            center = cell_index(i, j, p)
            local_u = gather_local_state(u, i, j, p, t)
            Jloc = local_jacobian(local_u, p; flux = flux)

            for row_var in 0:2
                for local_col_idx in 1:15
                    pos = positions[row_var+1, local_col_idx, center]
                    if pos != 0
                        nz[pos] = Jloc[row_var + 1, local_col_idx]
                    end
                end
            end
        end
    end

    return Jp
end