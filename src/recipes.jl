# ==============================================================================
# 1D OPTIMIZED RECIPE (Overlaid Stacking)
# 1 Row, 2 Columns -> Initial and Final states drawn inside the same window
# ==============================================================================
@recipe function f(sol::sol1D)
    _ncells = sol._ncells
    x = sol.x

    rho_init  = @view sol.u_init[1:_ncells]
    mx_init   = @view sol.u_init[_ncells + 1:2 * _ncells]
    rho_final = @view sol.u_final[1:_ncells]
    mx_final  = @view sol.u_final[_ncells + 1:2 * _ncells]

    ux_init = mx_init ./ rho_init
    ux_final = mx_final ./ rho_final

    # Define a 1x2 horizontal layout (Panel 1: Density, Panel 2: Velocity)
    layout --> (1, 2)
    linewidth --> 2
    xlabel --> "x"

    # --- SUBPLOT 1: DENSITY (ρ) OVERLAY ---
    @series begin
        subplot := 1
        title := "Density (ρ)"
        label := "Initial"
        seriestype := :path
        linestyle := :dash
        linecolor := :red
        x, rho_init
    end

    @series begin
        subplot := 1
        title := "Density (ρ)"
        label := "Final"
        seriestype := :path
        linestyle := :solid
        linecolor := :blue
        x, rho_final
    end

    # --- SUBPLOT 2: VELOCITY (uₓ) OVERLAY (LAZY ELEMENT EVALUATION) ---
    @series begin
        subplot := 2
        title := "Velocity (u_x)"
        label := "Initial"
        seriestype := :path
        linestyle := :dash
        linecolor := :red
        x, ux_init
    end

    @series begin
        subplot := 2
        title := "Velocity (u_x)"
        label := "Final"
        seriestype := :path
        linestyle := :solid
        linecolor := :blue
        x, ux_final
    end
end

# ==============================================================================
# 2D OPTIMIZED RECIPE (Overlaid Analysis)
# 1 Row, 3 Columns -> Final Solution as Heatmap, Initial Condition as Contour lines
# ==============================================================================
@recipe function f(sol::sol2D)
    Nx, Ny = length(sol.x), length(sol.y)
    _ncells = sol._ncells
    
    # Define a 2x3 grid: Row 1 = Initial Condition, Row 2 = Final Solution
    layout --> (2, 3)
    aspect_ratio --> :equal
    xlabel --> "x"
    ylabel --> "y"

    # 1. Unpack spatial matrix views safely
    rho_init  = reshape(@view(sol.u_init[1:_ncells]), Nx, Ny)
    mx_init   = reshape(@view(sol.u_init[_ncells + 1:2 * _ncells]), Nx, Ny)
    my_init   = reshape(@view(sol.u_init[2 * _ncells + 1:3 * _ncells]), Nx, Ny)

    rho_final = reshape(@view(sol.u_final[1:_ncells]), Nx, Ny)
    mx_final  = reshape(@view(sol.u_final[_ncells + 1:2 * _ncells]), Nx, Ny)
    my_final  = reshape(@view(sol.u_final[2 * _ncells + 1:3 * _ncells]), Nx, Ny)

    # 2. Extract relative on-the-fly velocities
    ux_init, ux_final = mx_init ./ rho_init, mx_final ./ rho_final
    uy_init, uy_final = my_init ./ rho_init, my_final ./ rho_final

    # Build an internal map of data pairs for cleaner iterative processing
    dataset = [
        (1, "Density (ρ)", rho_init, rho_final),
        (2, "Velocity X (u_x)", ux_init, ux_final),
        (3, "Velocity Y (u_y)", uy_init, uy_final)
    ]

    for (col_idx, name, init_mat, final_mat) in dataset
        
        # --- ROW 1: Initial Condition (Heatmap) ---
        @series begin
            subplot := col_idx
            title := name
            if col_idx == 1
                ylabel := "Initial\ny"
            end
            seriestype := :heatmap
            colorbar := true
            seriescolor --> :viridis 
            sol.x, sol.y, init_mat'
        end

        # --- ROW 2: Final Solution (Heatmap) ---
        @series begin
            subplot := col_idx + 3
            title := ""
            if col_idx == 1
                ylabel := "Final\ny"
            end
            seriestype := :heatmap
            colorbar := true
            seriescolor --> :viridis 
            sol.x, sol.y, final_mat'
        end
    end
end
