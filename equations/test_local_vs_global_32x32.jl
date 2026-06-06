using EulerAP
using LinearAlgebra
# using BenchmarkTools

function initial_condition(x, y)
    r2 = (2x - 1)^2 + (2y - 1)^2
    rho0 = 1 - 0.25 * exp(2 * (1 - r2))
    ux0 = y * exp(1 - r2)
    uy0 = -x * exp(1 - r2)
    return rho0, rho0 * ux0, rho0 * uy0
end

println("=" ^ 70)
println("Correctness Test: Local-AD vs Global-AD on 32×32 grid")
println("=" ^ 70)

tspan = (0.0, 0.01)  # short run for testing
gamma = 1.4

# Build problem with 32x32 grid
# Boundary condition knobs for the test (change as needed)
left_bc_test   = :periodic
right_bc_test  = :periodic
bottom_bc_test = :periodic
top_bc_test    = :periodic

u0_test, x_test, y_test, p_test, jac_proto_test = build_problem(
    nx = 32*2,
    ny = 32*2,
    eps = 0.05,
    left_bc = left_bc_test,
    right_bc = right_bc_test,
    bottom_bc = bottom_bc_test,
    top_bc = top_bc_test,
    ic_func = initial_condition,
    gamma = gamma
)

println("\nProblem size: nx=$(p_test.nx), ny=$(p_test.ny)")
println("Total DOFs: $(length(u0_test))")
println("Time span: $tspan")
println()

# --- Run 1: Global AD (default behavior) ---
println("Running solver with GLOBAL AD (AutoForwardDiff)...")
@time u_global, stats_global, steps_global = solve_backward_euler(
    copy(u0_test),
    p_test,
    tspan,
    jac_proto_test;
    dt = minimum(p_test.dx),
    tol = 1e-8,
    flux = :rusanov,
    gamma = gamma
)

print_run_stats("Global-AD", stats_global, steps_global)

# --- Run 2: Local-AD (local Jacobian assembly) ---
println("\nRunning solver with LOCAL-AD (assemble_global_jacobian)...")
@time u_local, stats_local, steps_local = solve_backward_euler(
    copy(u0_test),
    p_test,
    tspan,
    jac_proto_test;
    dt = minimum(p_test.dx),
    tol = 1e-8,
    jacobian_builder! = assemble_global_jacobian!,
    flux = :rusanov,
    gamma = gamma
)

print_run_stats("Local-AD", stats_local, steps_local)

# --- Comparison ---
println("\n" ^ 1)
println("=" ^ 70)
println("COMPARISON")
println("=" ^ 70)

# Compare number of steps
println("Steps taken:")
println("  Global-AD: $steps_global")
println("  Local-AD:  $steps_local")
if steps_global == steps_local
    println("  ✓ Steps match")
else
    println("  ✗ Steps differ!")
end

# Compare solutions
diff_u = norm(u_global - u_local)
rel_diff = diff_u / norm(u_global)

println("\nSolution difference:")
println("  ||u_global - u_local|| = $diff_u")
println("  Relative error = $rel_diff")

if rel_diff < 1e-10
    println("  ✓ Solutions match (within numerical precision)")
elseif rel_diff < 1e-6
    println("  ✓ Solutions match (within 1e-6 relative tolerance)")
elseif rel_diff < 1e-3
    println("  ~ Solutions close (within 1e-3)")
else
    println("  ✗ Solutions differ significantly")
end

# Compare timing
global_time = sum(stats_global.step_times[1:steps_global])
local_time = sum(stats_local.step_times[1:steps_local])

println("\nTiming (sum of step times):")
println("  Global-AD: $global_time s")
println("  Local-AD:  $local_time s")

speedup = global_time / local_time
println("  Speedup: $(round(speedup, digits=2))x")

if speedup > 1.0
    println("  ✓ Local-AD is faster")
else
    println("  ~ Global-AD is faster (initialization overhead?)")
end

# Compare allocations
global_alloc = sum(stats_global.step_bytes[1:steps_global])
local_alloc = sum(stats_local.step_bytes[1:steps_local])

println("\nAllocations (sum of step bytes):")
println("  Global-AD: $(round(global_alloc / 2^20, digits=2)) MiB")
println("  Local-AD:  $(round(local_alloc / 2^20, digits=2)) MiB")

alloc_ratio = local_alloc / global_alloc
println("  Ratio: $(round(alloc_ratio, digits=2))x")

if alloc_ratio < 1.0
    println("  ✓ Local-AD allocates less")
else
    println("  ~ Global-AD allocates less")
end

println("\n" ^ 1)
println("=" ^ 70)
println("Test completed successfully")
println("=" ^ 70)
