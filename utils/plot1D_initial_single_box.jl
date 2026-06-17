using Plots

# Definitions based on your provided function code
const DELTA = 0.1            # Smoothing parameter
const a = -2.0               # Left boundary of box
const b = 2.0                # Right boundary of box
const RHO_FLOOR = 1e-10       # Small positive floor to prevent division by zero
const γ = 3.0                 # Adiabatic exponent (matches equations/relaxation_euler1d_single_box.jl)

# 1. Smoothed Heaviside function H_Δ(x)
function heaviside_smooth(x::Real)
    return 0.5 * (1 + tanh(x / DELTA))
end

# 2. Initial density profile ρ(x)
function initial_rho(x::Real)
    X = x - a
    Y = x - b
    ha = heaviside_smooth(X)
    hb = heaviside_smooth(Y)
    ρ = max(RHO_FLOOR, ha - hb)
    return ρ
end

# 3. Initial velocity profile u_x(x)
# From the formula in initial_box: u = (tanh²(X/Δ) + tanh²(Y/Δ)) / (2Δ) · (-γ) · ρ^(γ-2)
function initial_ux(x::Real)
    X = x - a
    Y = x - b
    ρ = initial_rho(x)

    u = -tanh(X / DELTA) * tanh(X / DELTA)
    u += tanh(Y / DELTA) * tanh(Y / DELTA)
    u = u / (2 * DELTA)
    u = u * (-γ) * ρ^(γ - 2.0)
    return u
end

# Define domain
x_vals = range(-5.0, 5.0, length=500)
rho_vals = initial_rho.(x_vals)
ux_vals  = initial_ux.(x_vals)

gr() # Use the GR backend

# Density subplot
p1 = plot(x_vals, rho_vals,
          xlabel="Position (x)",
          ylabel="Density (ρ)",
          label="ρ(x)",
          lw=2.5,
          color=:blue,
          grid=:xy,
          legend=:topright,
          left_margin=5Plots.mm, bottom_margin=5Plots.mm)
vline!(p1, [a, b], color=:black, linestyle=:dash, alpha=0.5, label="Target Bounds [a,b]")

# Velocity subplot
p2 = plot(x_vals, ux_vals,
          xlabel="Position (x)",
          ylabel="Velocity (u_x)",
          label="u_x(x)",
          lw=2.5,
          color=:red,
          grid=:xy,
          legend=:topright,
          left_margin=5Plots.mm, bottom_margin=5Plots.mm)
vline!(p2, [a, b], color=:black, linestyle=:dash, alpha=0.5, label="Target Bounds [a,b]")

fig = plot(p1, p2, layout=(1, 2), size=(1400, 500),
           plot_title="Initial Smoothed Box: Δ = $(DELTA), γ = $(γ)")

savefig("utils/rho_single_box_plot.png")
println("Saved plot to utils/rho_single_box_plot.png")