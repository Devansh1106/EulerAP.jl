# const linsolve = KLUFactorization()
const linsolve = MKLPardisoFactorize()

"""
    backward_euler_residual!(res, u, p::ImplicitStepData)

Evaluate the global nonlinear system residual vector `res` for a single implicit 
Backward Euler time step.

This function serves as the primary residual callback executed at every iteration 
of your nonlinear solver (e.g., Newton-Raphson). It computes the difference 
between the current state candidate `u` and an implicitly advanced state calculated 
via the semi-discrete spatial operator.

# Mathematical Formulation
The Backward Euler scheme solves the implicit algebraic system \$\\mathbf{G}(\\mathbf{u}) = \\mathbf{0}\$. 
For a given state candidate \$\\mathbf{u}^k\$ at time step \$n+1\$, the global residual 
vector \$\\mathbf{R}\\mathbf{e}\\mathbf{s}\$ is defined as:

    res = u^k - u^{n} - Δt · F(u^k, t^{n+1})

where:
- `u` (\$\\mathbf{u}^k\$) is the current candidate solution vector for the next time step.
- `p.u_prev` (\$\\mathbf{u}^{n}\$) is the frozen, fully converged solution from the previous time step.
- `p.dt` (\$\\Delta t\$) is the current integration time step size.
- `F(u^k, t^{n+1})` is the spatial operator evaluated via `implicit_part!`.

# Memory & Allocation Invariants
To maintain optimal efficiency and compatibility with non-linear frameworks, this 
function executes completely in-place with **zero heap allocations**:
1. `implicit_part!` is called with `res` as the destination array, temporarily storing 
   the spatial operator evaluations \$F(\\mathbf{u}^k)\$ directly into the pre-allocated vector.
2. The Julia broadcast macro `@.` fuses the vector updates into a single contiguous loop:

       res[i] = u[i] - u_prev[i] - dt * res[i]

   This safely overwrites the spatial values with the finalized physical residual vector 
   without allocating temporary cache arrays.

# Arguments
- `res::AbstractVector`: Destination vector to store the computed global nonlinear residual.
- `u::AbstractVector`: The component-blocked candidate state vector \$\\mathbf{u}^k\$.
- `p::ImplicitStepData`: Structural configuration wrapper containing step parameters 
  (`dt`, `t`), the underlying model parameters (`p.model`), and historical step vectors (`u_prev`).
"""
function backward_euler_residual!(res, u, p::ImplicitStepData)
    # Positivity preservation: clamp density to prevent negative values
    # during Newton iterations. The Jacobian is assembled separately via
    # local_residual, so this does NOT affect the Jacobian computation.
    # ncells = prod(p.model.size)
    # @views @. u[1:ncells] = max(u[1:ncells], 1e-12)

    implicit_part!(res, u, p.model, p.t; resolved_flux = p.flux)
    @. res = u - p.u_prev - p.dt * res
    
    return nothing
end

"""
    solve_backward_euler(u0, p, tspan, jac_cache; kwargs...)

Integrate the multi-dimensional hyperbolic relaxation system forward in time using 
an implicit first-order Backward Euler method paired with a robust Newton-Raphson nonlinear solver.

This function implements a fully-staged, allocation-optimized implicit time-stepping loop. 
It interfaces with `NonlinearSolve.jl` using stateful structure caches to eliminate memory 
overhead across temporal iterations, making it highly effective for stiff relaxation timelines.

# Mathematical Formulation
For each discrete time step from \$t^n\$ to \$t^{n+1} = t^n + \\Delta t\$, this function 
drives the global non-linear system residual \$\\mathbf{G}(\\mathbf{u}^{n+1})\$ to zero:

    \\mathbf{G}(\\mathbf{u}^{n+1}) = \\mathbf{u}^{n+1} - \\mathbf{u}^n - \\Delta t \\cdot F(\\mathbf{u}^{n+1}, t^{n+1}) = \\mathbf{0}

A non-linear Newton-Raphson routine updates candidates via:
    \\mathbf{J} \\cdot \\Delta \\mathbf{u}^k = -\\mathbf{G}(\\mathbf{u}^k)
where the true structural Jacobian matching this residual is:
    \\mathbf{J} = \\mathbf{I} - \\Delta t \\cdot \\frac{\\partial F}{\\partial \\mathbf{u}}

# Arguments
- `u0::AbstractVector`: Initial component-blocked state vector distribution.
- `p::RelaxationParams{NDIMS}`: Structural solver configuration parameters.
- `tspan::Tuple{Real, Real}`: Integration interval boundaries `(t_start, t_end)`.
- `jac_cache::SparseJacobianCache`: Pre-allocated sparse matrix framework used for cell-by-cell 
  assembly without triggering global automatic differentiation.

# Keyword Arguments
- `dt::Float64`: Nominal time step size (defaults to `5.0e-2`).
- `tol::Float64`: Absolute and relative convergence tolerance thresholds for the Newton loop (defaults to `1e-8`).
- `flux::Symbol`: Shortcut identifier for the underlying numerical interface flux (defaults to `:rusanov`).
- `gamma::Float64`: Adiabatic exponent parameter forwarded to the pressure law functions (defaults to `1.4`).
- `jacobian_builder!`: Callback function handling local stencil differentiation and sparse matrix mapping.

# Returns
- `u::AbstractVector`: Finalized component-blocked solution vector at `t = tspan[2]`.
- `stats::RunStats`: Detailed diagnostic telemetry tracking execution time, allocation bytes, 
  and garbage collection cycles per step.
- `nsteps_done::Int`: Total number of implicit temporal integration steps performed.
"""
function solve_backward_euler(
    u0,
    p::RelaxationParams,
    tspan,
    jac_cache::SparseJacobianCache;
    dt = 5.0e-2,
    tol = 1e-8,
    flux = :rusanov,
    gamma = 1.4,
    jacobian_builder! = assemble_global_jacobian!
)
    resolved_flux = resolve_flux(flux; gamma = gamma)

    # The closure captures our jac_cache struct. 
    # J_internal is the matrix NonlinearSolve tracks, but we update our cached matrix and copy it to J_internal
    jac_fun = (J_internal, u, step_data) -> 
    begin
        # assembled Jacobian matches the backward-Euler residual (I - dt*dF/du).
        jacobian_builder!(jac_cache, 
                          u, 
                          step_data.model, 
                          step_data.dt, 
                          step_data.t; 
                          flux = step_data.flux)

        copyto!(J_internal, jac_cache.J)
        return J_internal
    end

    nls_function = NonlinearFunction(backward_euler_residual!;
                                     jac = jac_fun,
                                     jac_prototype = jac_cache.J) # Pass the raw matrix from the cache here

    nls_algorithm = NewtonRaphson(concrete_jac = true,
                                  linsolve = linsolve,
                                  linesearch = BackTracking()) # required since NR was stuck w/o this for Riemann problem/stiff cases

    u = copy(u0)

    step_data = ImplicitStepData(
        p, dt, tspan[1], copy(u0), resolved_flux
    )

    nsteps_target = ceil(Int, (tspan[2] - tspan[1]) / dt)
    # Add one extra slot to guard against floating-point rounding producing
    # an extra (very small) final step when summing dt steps.
    stats = RunStats(nsteps_target + 1)

    nonlinear_problem = NonlinearProblem(nls_function, u, step_data)

    cache = init(
        nonlinear_problem,
        nls_algorithm;
        abstol = tol,
        reltol = tol
    )

    t = tspan[1]
    nsteps_done = 0

    total_timed = @timed while t < tspan[2]
        dt_step           = min(dt, tspan[2] - t)
        step_data.dt      = dt_step
        step_data.t       = t + dt_step
        step_data.u_prev .= u
        nsteps_done      += 1

        step_timed = @timed begin
            reinit!(cache; 
                    u0 = u, 
                    p = step_data) # in-place (no allocation)

            nonlinear_solution = SciMLBase.solve!(cache)
            if !SciMLBase.successful_retcode(nonlinear_solution.retcode)
                error(
                    "Backward Euler nonlinear solve failed at step $nsteps_done, " *
                    "t=$(step_data.t), dt=$(step_data.dt), retcode=$(nonlinear_solution.retcode)"
                )
            end
            u .= nonlinear_solution.u
        end
        
        stats.step_times[nsteps_done]   = step_timed.time
        stats.step_bytes[nsteps_done]   = step_timed.bytes
        stats.step_gctimes[nsteps_done] = step_timed.gctime

        t += dt_step
    end

    stats.total_gctime = total_timed.gctime
    stats.total_time   = total_timed.time
    stats.total_bytes  = total_timed.bytes
    return u, stats, nsteps_done
end
