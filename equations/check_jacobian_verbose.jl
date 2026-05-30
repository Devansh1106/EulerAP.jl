using EulerAP
using SparseArrays
using LinearAlgebra

# Small problem for verbose Jacobian verification
nx = 4
ny = 4
p = RelaxationParams(0.05, nx, ny, 1.0/nx, 1.0/ny, 0.0, 1.0, 0.0, 1.0, BCConfig())

function initial_condition(x, y)
    r2 = (2x - 1)^2 + (2y - 1)^2
    rho0 = 1 - 0.25 * exp(2 * (1 - r2))
    ux0 = y * exp(1 - r2)
    uy0 = -x * exp(1 - r2)
    return rho0, rho0 * ux0, rho0 * uy0
end

u0, x, y, p2, cache = build_problem(nx=nx, ny=ny, eps=0.05, left_bc = :periodic, right_bc=:periodic, bottom_bc=:periodic, top_bc=:periodic, ic_func=initial_condition)

# pick a test state
u = copy(u0)
for i in 1:length(u)
    u[i] += 1e-3 * (rand() - 0.5)
end

dt = 0.1 * p.dx
t = 0.0

J_assembled = assemble_global_jacobian!(cache, u, p, dt, t; flux=:rusanov)
Jdense = Matrix(J_assembled)

# FD Jacobian
function eval_F(u)
    du = similar(u)
    EulerAP.implicit_part!(du, u, p, t; flux=:rusanov)
    return du
end

eps = 1e-6
F0 = eval_F(u)
JF = zeros(Float64, length(u), length(u))
for j in 1:length(u)
    u_pert = copy(u)
    u_pert[j] += eps
    Fp = eval_F(u_pert)
    JF[:, j] = (Fp .- F0) ./ eps
end
Jfd = I - dt .* JF

D = Jdense .- Jfd

# find top mismatches
inds = sortperm(abs.(vec(D)); rev=true)[1:20]

println("Top mismatches (global row, col, assembled, fd, diff):")
for idx in inds
    r = fld(idx-1, size(D,1)) + 1
    c = (idx-1) % size(D,1) + 1
    println(r, ",", c, ", ", Jdense[r,c], ", ", Jfd[r,c], ", ", D[r,c])
end

# Also print mapping info for the first few columns to inspect positions
println("\nPositions mapping for first 10 cells (row_var, local_col_idx, global pos):")
for cell in 1:min(10, p.nx*p.ny)
    println("Cell ", cell)
    for rv in 1:3
        for lc in 1:15
            pos = cache.positions[rv, lc, cell]
            if pos != 0
                # convert pos (index into nzval) to (row,col)
                # find column by searching colptr
                col = searchsortedlast(cache.J.colptr, pos)
                row = cache.J.rowval[pos]
                println("  rv=", rv, " lc=", lc, " -> global (", row, ",", col, ") = nzval_idx ", pos)
            end
        end
    end
end

println("\nDone diagnostic")
