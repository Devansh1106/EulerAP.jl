using EulerAP
using SparseArrays
using LinearAlgebra

# Small problem for Jacobian verification
nx = 4
ny = 4
p = RelaxationParams(0.05, nx, ny, 1.0/nx, 1.0/ny, 0.0, 1.0, 0.0, 1.0, BCConfig())

# Local initial condition used for this test (radial-like)
function initial_condition(x, y)
    # r2 = (2x - 1)^2 + (2y - 1)^2
    r2 = x^2 + y^2

    rho0 = 1 - 0.1 * exp(2 * (1 - r2))
    ux0 = y * exp(1 - r2)
    uy0 = -x * exp(1 - r2)
    return rho0, rho0 * ux0, rho0 * uy0
end

u0, x, y, p2, cache = build_problem(nx=nx, ny=ny, eps=0.05, left_bc = :periodic, right_bc=:periodic, bottom_bc=:periodic, top_bc=:periodic, ic_func = initial_condition)

# sanity: p and p2 should match
@assert p.nx == p2.nx

dof = length(u0)

# pick a test state: slightly perturb u0
u = copy(u0)
for i in 1:dof
    u[i] += 1e-3 * (rand() - 0.5)
end

# choose dt and t
dt = 0.1 * p.dx
t = 0.0

# Assemble cached Jacobian (I - dt*dF/du)
J_assembled = assemble_global_jacobian!(cache, u, p, dt, t; flux=:rusanov)

# Build FD Jacobian of residual R(u) = u - u_prev - dt*F(u)
# Here u_prev = u (linearization about current state), so R(u) derivative = I - dt * dF/du

function eval_F(u)
    du = similar(u)
    EulerAP.implicit_part!(du, u, p, t; flux=:rusanov)
    return du
end

# We'll compute columns of dF/du via forward difference
eps = 1e-6

# Preallocate dense FD Jacobian (small test)
JF = zeros(Float64, dof, dof)
F0 = eval_F(u)

for j in 1:dof
    u_pert = copy(u)
    u_pert[j] += eps
    Fp = eval_F(u_pert)
    JF[:, j] = (Fp .- F0) ./ eps
end

Jfd = I - dt .* JF

# Convert assembled to dense
Jdense = Matrix(J_assembled)

max_abs = maximum(abs.(Jdense .- Jfd))
norm_diff = norm(Jdense - Jfd) / max(1e-12, norm(Jfd))

println("Jacobian FD check on $(nx)x$(ny) grid")
println("DOFs = $dof")
println("dt = $dt")
println("max abs difference = ", max_abs)
println("relative norm difference = ", norm_diff)

if max_abs < 1e-6
    println("PASS: assembled Jacobian matches FD (within tol)")
else
    println("FAIL: assembled Jacobian differs from FD; investigate positions/assembly")
end
