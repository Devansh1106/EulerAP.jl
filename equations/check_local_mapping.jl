using EulerAP
using ForwardDiff
using LinearAlgebra

# Focused diagnostic for a single cell's local jacobian vs FD and mapping
nx = 4; ny = 4

function initial_condition(x, y)
    r2 = (2x - 1)^2 + (2y - 1)^2
    rho0 = 1 - 0.25 * exp(2 * (1 - r2))
    ux0 = y * exp(1 - r2)
    uy0 = -x * exp(1 - r2)
    return rho0, rho0 * ux0, rho0 * uy0
end

u0, x, y, p, cache = build_problem(nx=nx, ny=ny, eps=0.05, left_bc=:periodic, right_bc=:periodic, bottom_bc=:periodic, top_bc=:periodic, ic_func=initial_condition)

# choose a cell to inspect (center-ish)
cell_i = 2
cell_j = 2
cell = EulerAP.cell_index(cell_i, cell_j, p)

# gather local state and compute Jloc inline
local_u = gather_local_state(u0, cell_i, cell_j, p, 0.0)
Jloc = ForwardDiff.jacobian(
    x -> collect(local_residual(x, p; flux = :rusanov)),
    collect(local_u)
)

# assemble global Jacobian so cache.J.nzval contains assembled values
dt = 0.1 * p.dx
t = 0.0
assemble_global_jacobian!(cache, u0, p, dt, t; flux = :rusanov)

println("Local Jacobian for cell $cell (3x15):")
for r in 1:3
    println([Jloc[r,c] for c in 1:15])
end

# compute FD local Jacobian by perturbing assembled local state used in residual
function local_residual_vec(local_u)
    drho, dmx, dmy = local_residual(local_u, p; flux=:rusanov)
    return vcat(drho, dmx, dmy)
end

eps = 1e-6
lu = collect(local_u)
Lf0 = local_residual_vec(lu)
Jf_local = zeros(Float64, 3, 15)
for col in 1:15
    lu_pert = copy(lu)
    lu_pert[col] += eps
    Lfp = local_residual_vec(lu_pert)
    Jf_local[:, col] = (Lfp .- Lf0) ./ eps
end

println("\nFD Local Jacobian (3x15):")
for r in 1:3
    println([Jf_local[r,c] for c in 1:15])
end

# Now, for each local column that FD says non-zero, inspect mapping
println("\nMapping & global values for nonzero FD local columns:")
for local_col in 1:15
    colnorm = maximum(abs.(Jf_local[:, local_col]))
    if colnorm > 1e-12
        println("\nLocal column: $local_col, max abs = $colnorm")
        for row_var in 0:2
            pos = cache.positions[row_var+1, local_col, cell]
            println(" row_var=", row_var, " pos=", pos)
            if pos != 0
                # decode pos -> (row, col)
                col = searchsortedlast(cache.J.colptr, pos)
                row = cache.J.rowval[pos]
                println("   maps to global (", row, ",", col, "), assembled value = ", cache.J.nzval[pos])
            end
            println("   Jloc entry = ", Jloc[row_var+1, local_col])
            println("   FD local entry = ", Jf_local[row_var+1, local_col])
        end
    end
end

println("\nDone local mapping check for cell $cell")
