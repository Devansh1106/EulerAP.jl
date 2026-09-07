# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function stencil_indices(I::CartesianIndex{1},
                                 semi::AbstractSemidiscretization)

    center = I

    # (I, semi, axis, sign); `x`: axis = 1
    left   = neighbor_index(I, semi, 1, -1)
    right  = neighbor_index(I, semi, 1, 1)

    return (center, left, right)
end

function neighbor_index(I::CartesianIndex{1},
                        semi::AbstractSemidiscretization,
                        axis::Int,
                        sign::Int)
    
    @assert axis == 1
    shifted = I[1] + sign

    # Interior cell
    if 1 <= shifted && shifted <= size(semi.mesh, 1)
        return CartesianIndex(shifted)
    end

    nx = size(semi.mesh, 1)
    bc = _side_bc(semi.boundary_conditions,
                  axis,
                  sign)

    if isa(bc, PeriodicBC)
        wrapped = _wrap_index(shifted, nx)
        return CartesianIndex(wrapped)

    elseif bc isa NeumannBC || bc isa ExtrapolateBC
        # Ghost state depends on interior state
        return CartesianIndex(clamp(shifted, 1, nx))

    elseif bc isa DirichletBC || bc isa MixedBC
        # Ghost state independent of interior state (or, for `MixedBC`,
        # a per-variable mix that must be resolved by `apply_bc` rather
        # than by pre-clamping the neighbor index here — see the note in
        # the `MixedBC` docstring).
        return CartesianIndex(shifted)

    else
        error("Unknown boundary condition type $(typeof(bc))")
    end 
end

@inline function _side_bc(bcfg,
                          axis::Int,
                          sign::Int)

    @assert axis == 1
    return sign < 0 ? bcfg.left : bcfg.right
end

@inline function boundary_side(I::CartesianIndex{1},
                               semi::AbstractSemidiscretization)

    i = I[1]
    if i < 1
        return :left
    elseif i > size(semi.mesh, 1)
        return :right
    else
        return nothing
    end
end

function rhs!(du, u,
              solver::FVSolver{1, TFlux},
              semi::SemidiscretizationHyperbolic,
              t;
              dt=0.0) where {TFlux}

    cache = semi.cache
    fill!(du, zero(eltype(du)))
    nvars = nvariables(semi.equations)
    for I in eachcell(semi.mesh)
        gather_local_state!(cache.x_cache,
                            u,
                            I,
                            semi,
                            t)

        local_residual!(cache.residual_buffer,
                        cache.x_cache,
                        semi.solver,
                        semi;
                        dt=dt)
        cell = cell_index(I, semi)

        @inbounds for v in 1:nvars
            du[global_dof(cell, v, nvars)] = cache.residual_buffer[v]
        end
    end

    return nothing
end

@inline function gather_local_state!(x, u, I::CartesianIndex{1},
                                     semi::AbstractSemidiscretization,
                                     t)

    nvars = nvariables(semi.equations)
    center, left, right = stencil_indices(I, semi)

    offset = 0

    for cell in (center, left, right)
        state = cell_state(u, cell, semi, t)

        @inbounds for v in 1:nvars
            x[offset + v] = state[v]
        end
        offset += nvars
    end

    return nothing
end

@inline function cell_state(u, I::CartesianIndex{1},
                            semi::AbstractSemidiscretization,
                            t)

    nx = size(semi.mesh, 1)

    # --------------------------------------------------
    # Interior cell
    # --------------------------------------------------
    if 1 <= I[1] <= nx
        return extract_cell_state(u, I, semi)
    end

    # --------------------------------------------------
    # Ghost cell
    # --------------------------------------------------

    side = boundary_side(I, semi)

    bc = side === :left ?
         semi.boundary_conditions.left :
         semi.boundary_conditions.right

    return apply_bc(bc, u, I, semi, t)
end

@inline function extract_cell_state(u, I::CartesianIndex{1},
                                    semi::AbstractSemidiscretization)

    nvars = nvariables(semi.equations)
    cell  = cell_index(I, semi)

    return SVector{nvars}(ntuple(v -> 
                                 u[global_dof(cell, v, nvars)],
                                 nvars))
end

@inline function local_residual!(y, x, solver::FVSolver{1, TFlux}, semi::AbstractSemidiscretization; dt=0.0) where {TFlux}

    equations = semi.equations
    source_terms = semi.source_terms
    flux = semi.solver.flux

    # --------------------------------------------------
    # Extract stencil states
    # --------------------------------------------------

    u_center = local_state(x, 1, equations)
    u_left   = local_state(x, 2, equations)
    u_right  = local_state(x, 3, equations)

    dx = semi.mesh.dx[1]
    
    # --------------------------------------------------
    # Numerical fluxes
    # --------------------------------------------------

    flux_left = flux(u_left, u_center, 1, equations, dt, dx)

    flux_right = flux(u_center, u_right, 1, equations, dt, dx)


    # --------------------------------------------------
    # FV divergence
    # --------------------------------------------------

    @inbounds for v in eachindex(y)
        y[v] = -(flux_right[v] - flux_left[v]) / dx
    end

    # --------------------------------------------------
    # Source terms
    # --------------------------------------------------

    if source_terms !== nothing
        src = source_terms(u_center,
                           equations)

        @inbounds for v in eachindex(y)
            y[v] += src[v]
        end
    end

    return nothing
end

"""
    apply_bc(bc::ExtrapolateBC{1}, u, I, semi, t)

Zero-gradient (even reflection) ghost state: copy the cell average of ghost
`I`'s mirror partner across the boundary face.

    left  boundary:  U[0]    = U[1]     U[-1]   = U[2]
    right boundary:  U[nx+1] = U[nx]    U[nx+2] = U[nx-1]

Using the mirror partner rather than clamping to the nearest interior cell is
what makes the *outer* layer meaningful. Clamping gave `U[-1] = U[1]`, re-using
the innermost cell, which breaks the reflection symmetry the ghost slope should
satisfy — for an even reflection the ghost's slope must be minus the slope of
the cell it mirrors:

    slope[0] == -slope[1]

Clamping violates that for limiters that do not clip (measured with
`nolimiter`: `slope[1] = +3.617150e-07` against `slope[0] = 0`, so the ghost
reconstructs flat while its partner does not); the mirror partner restores it
exactly (`slope[0] = -3.617150e-07`). Under `minmod` both forms give zero,
because `U[0] == U[1]` makes one of the one-sided differences vanish and minmod
returns zero — so this correction is invisible to `minmod` and matters for
`nolimiter`/`CWENO`.

The two forms agree at depth 1, so the first-order scheme is unaffected.

This is now exactly homogeneous Neumann (`q = 0`) — `gradient_ghost_state` with
a zero slope reduces to the same copy — which is the intended meaning of a
zero-gradient outflow condition.

Note this is *not* LeVeque's first-order extrapolation (7.4)
`U[nx+1] = 2U[nx] - U[nx-1]`, which continues the interior trend instead of
flattening it; that is a different condition, deliberately not used here.
"""
@inline function apply_bc(bc::ExtrapolateBC{1},
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    return extract_cell_state(u,
                              CartesianIndex(mirror_partner_index(I, semi)),
                              semi)
end

# ----------------------------------------------------------------------------
# Dirichlet boundary condition
#
# Built from two independent pieces — `dirichlet_boundary_value` obtains the
# prescribed value `g`, `mirror_ghost_state` turns `g` into a ghost state —
# composed by the `apply_bc` method at the bottom.
#
# ONE implementation serves both the first- and second-order schemes; there is
# deliberately no per-order variant. Nothing in the construction depends on the
# scheme: the only difference is *how many ghost layers each scheme asks for*,
# and that is already carried by the ghost index `I` itself.
#
#     first-order   asks for depth 1 only     ->  I = 0        and I = nx + 1
#     second-order  asks for depths 1 and 2   ->  I = 0, -1    and I = nx+1, nx+2
#
# `mirror_ghost_state` maps any depth to its interior partner, so the same call
# answers both. (`apply_bc` also has no access to the scheme: the integrator is
# an argument to `solve`, not a field of `semi`, so a per-order split would
# require threading an order argument through `cell_state` and every one of its
# callers to duplicate a formula that does not vary by order.)
# ----------------------------------------------------------------------------

"""
    dirichlet_boundary_value(bc, I, semi, t)

The prescribed Dirichlet value `g` itself — nothing more: evaluate the user's
boundary function and return it as a state vector.

Kept separate from the ghost construction so that *obtaining* `g` and *building
a ghost state from* `g` can be read, reused, and changed independently.

`g` is sampled at `coordinates(I, mesh)`, the ghost cell centre. For a constant
boundary function — every current use — that equals the value at the boundary
face and the reflection below is exact. For a *spatially varying* `g` the two
ghost layers would reflect through two different samples of it; evaluating at
the face instead (`coordinates_min` / `coordinates_max`) is the one-line change
that would make that case consistent.
"""
@inline function dirichlet_boundary_value(bc::DirichletBC{1},
                                          I::CartesianIndex{1},
                                          semi,
                                          t)

    x = coordinates(I, semi.mesh)
    nvars = nvariables(semi.equations)
    sol = bc.boundary_value(x, t, semi.equations)

    return SVector{nvars}(ntuple(v -> sol[v], Val(nvars)))
end

"""
    mirror_partner_index(I, semi)

Interior cell that ghost `I` reflects onto across the nearest boundary face:

    left  face sits at index 1/2,      so i -> 1 - i        ( 0 -> 1,  -1 -> 2)
    right face sits at index nx + 1/2, so i -> 2nx + 1 - i  (nx+1 -> nx, nx+2 -> nx-1)

Shared by the Dirichlet and Neumann ghost constructions, which differ only in
what they do once the partner is known. Works at any ghost depth, so the
first-order scheme's single layer and the second-order scheme's two layers use
the same mapping.

`clamp` only bites on a mesh too small to hold the stencil (`nx < 2` for the
outer layer); for any usable mesh the mirrored index is already interior.
"""
@inline function mirror_partner_index(I::CartesianIndex{1}, semi)

    nx   = size(semi.mesh, 1)
    side = boundary_side(I, semi)

    mirrored = side === :left ? (1 - I[1]) : (2 * nx + 1 - I[1])

    return clamp(mirrored, 1, nx)
end

"""
    mirror_ghost_state(g, u, I, semi)

Ghost state obtained by reflecting ghost `I`'s interior partner through the
prescribed value `g`:

    Q_ghost = 2g - Q_interior

so that the average of a ghost and its partner is exactly `g` at the boundary
face. The partner is the mirror of `I` about that face:

    left  face sits at index 1/2,      so i -> 1 - i        ( 0 -> 1,  -1 -> 2)
    right face sits at index nx + 1/2, so i -> 2nx + 1 - i  (nx+1 -> nx, nx+2 -> nx-1)

Because the partner is computed from `I`, this covers *any* ghost depth: the
first-order scheme's single layer and the second-order scheme's two layers are
the same expression evaluated at different `I`.

With `g = 0` on momentum this reduces to the exact solid wall
`m_ghost = -m_interior`, which makes the boundary face carry zero mass flux —
contrast a piecewise-constant `Q_ghost = g`, which leaks mass through a wall
and limits the ghost slope to zero, dropping the boundary to first order.

`g` is taken as an argument rather than a `bc`, so this knows nothing about
`DirichletBC` and can serve any condition that prescribes a boundary value.

`clamp` only bites on a mesh too small to hold the stencil (`nx < 2` for the
outer layer); for any usable mesh the mirrored index is already interior.
"""
@inline function mirror_ghost_state(g,
                                    u,
                                    I::CartesianIndex{1},
                                    semi)

    nvars      = nvariables(semi.equations)
    interior_i = mirror_partner_index(I, semi)

    interior = extract_cell_state(u, CartesianIndex(interior_i), semi)

    return SVector{nvars}(ntuple(v -> 2 * g[v] - interior[v], Val(nvars)))
end

"""
    apply_bc(bc::DirichletBC{1}, u, I, semi, t)

Dirichlet ghost state: get `g`, then reflect through it.

The composition must live here, on the `apply_bc` dispatch, rather than in a
caller: `MixedBC` builds its state per variable via
`apply_bc(bc.bcs[v], u, I, semi, t)[v]`, so a Dirichlet *component* is only
mirrored if the mirror happens inside this method. That is the common case in
practice — a solid wall is `MixedBC(ExtrapolateBC, DirichletBC(0))`, density
extrapolated and momentum pinned.
"""
@inline function apply_bc(bc::DirichletBC{1},
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    g = dirichlet_boundary_value(bc, I, semi, t)

    return mirror_ghost_state(g, u, I, semi) # for U[0] = 2g - U[1]
end

# ----------------------------------------------------------------------------
# Neumann boundary condition
#
# Same three-part shape as Dirichlet above — `neumann_boundary_gradient` obtains
# the prescribed slope `q`, `gradient_ghost_state` turns `q` into a ghost state,
# and the `apply_bc` method composes them — and likewise ONE implementation
# serves both schemes, the ghost depth being carried by `I`.
#
# SIGN CONVENTION: `q` is the **outward normal** derivative du/dn, the standard
# form in which a Neumann condition is stated. `n` points out of the domain on
# both sides (-x on the left, +x on the right), so one `q` steps *away from the
# domain* at either end and both sides carry the same sign:
#
#     left  boundary:  U[0]    = U[1]   + q*h      U[-1]   = U[2]    + 3*q*h
#     right boundary:  U[nx+1] = U[nx]  + q*h      U[nx+2] = U[nx-1] + 3*q*h
#
# In Cartesian terms that is du/dx = -q on the left and du/dx = +q on the right.
# ----------------------------------------------------------------------------

"""
    neumann_boundary_gradient(bc, I, semi, t)

The prescribed boundary slope `q` itself — nothing more: evaluate the user's
gradient function and return it as a state vector. `q` is the outward normal
derivative du/dn (see the sign convention note above).
"""
@inline function neumann_boundary_gradient(bc::NeumannBC{1},
                                           I::CartesianIndex{1},
                                           semi,
                                           t)

    nvars = nvariables(semi.equations)
    grad = bc.boundary_gradient(coordinates(I, semi.mesh),
                                t,
                                semi.equations)

    return SVector{nvars}(ntuple(v -> grad[v], Val(nvars)))
end

"""
    gradient_ghost_state(q, u, I, semi)

Ghost state carrying the prescribed outward normal slope `q` across the
boundary face: take ghost `I`'s mirror partner and step out to `I` along a line
of slope `q`,

    Q_ghost = Q_partner + |x_ghost - x_partner| * q

The offset magnitude `|I - partner| * h` is used because `q` is already
measured along the outward normal, which points in opposite Cartesian
directions at the two ends:

    left  boundary:  U[0]    = U[1]   + q*h      U[-1]   = U[2]    + 3*q*h
    right boundary:  U[nx+1] = U[nx]  + q*h      U[nx+2] = U[nx-1] + 3*q*h

Writing it against the mirror partner rather than the nearest interior cell is
what makes the outer layer reach real interior data (`U[2]`, `U[nx-1]`) instead
of re-using the innermost cell. Each pair straddles the face symmetrically, so
the outward difference across it recovers `q` for both layers —
`(U[0]-U[1])/h = q` and `(U[-1]-U[2])/(3h) = q`. Extrapolating both layers from
`U[1]` instead (`U[-1] = U[1] + 2q*h`) would pin the ghost slope to `q`
regardless of the interior solution, costing the second-order scheme its
accuracy at the boundary.

`q` is taken as an argument rather than a `bc`, so this knows nothing about
`NeumannBC`.
"""
@inline function gradient_ghost_state(q,
                                      u,
                                      I::CartesianIndex{1},
                                      semi)

    nvars      = nvariables(semi.equations)
    h          = semi.mesh.dx[1]
    interior_i = mirror_partner_index(I, semi)

    # Centre-to-centre distance from the partner cell out to the ghost cell.
    # Magnitude, not signed: `q` is the outward normal derivative, so stepping
    # away from the domain adds `+q` per unit length at either boundary.
    offset = abs(I[1] - interior_i) * h

    interior = extract_cell_state(u, CartesianIndex(interior_i), semi)

    return SVector{nvars}(ntuple(v -> interior[v] + offset * q[v], Val(nvars)))
end

"""
    apply_bc(bc::NeumannBC{1}, u, I, semi, t)

Neumann ghost state: get `q`, then step across the face with it.

As with Dirichlet, the composition stays on the `apply_bc` dispatch so that a
Neumann component inside a `MixedBC` is handled too.
"""
@inline function apply_bc(bc::NeumannBC{1},
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t)

    q = neumann_boundary_gradient(bc, I, semi, t)

    return gradient_ghost_state(q, u, I, semi)
end

@inline function apply_bc(bc::MixedBC{1, N},
                          u,
                          I::CartesianIndex{1},
                          semi,
                          t) where {N}

    # Evaluate each variable's own sub-BC independently and keep only the
    # component that variable is responsible for. Each sub-BC's `apply_bc`
    # method already knows how to compute its full ghost state (e.g.
    # `ExtrapolateBC` clamps to the nearest interior cell, `DirichletBC`
    # evaluates its boundary function), so this reuses the existing
    # per-condition numerics rather than reimplementing them.
    return SVector{N}(ntuple(v -> apply_bc(bc.bcs[v], u, I, semi, t)[v], N))
end

end # @muladd