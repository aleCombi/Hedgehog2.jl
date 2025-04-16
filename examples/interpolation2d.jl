using Revise
using Hedgehog  # your library
using DataInterpolations
using Plots

# Input grid (e.g. x = time, y = strike)
x = [0.25, 0.5, 1.0, 2.0]  # maturities
y = [80, 90, 100, 110, 120]  # strikes

# Synthetic values for the surface
z = [
    0.35 0.32 0.30 0.32 0.35
    0.34 0.31 0.29 0.31 0.34
    0.33 0.30 0.28 0.30 0.33
    0.32 0.29 0.27 0.29 0.32
]

# Build 2D interpolator with smooth settings
surf = Interpolator2D(x, y, z; interp_y = CubicSpline, interp_x = LinearInterpolation)

# Evaluation grid for plotting
x_plot = range(0.25, 2.0, length = 100)
y_plot = range(80, 120, length = 100)
Z = [surf[xi, yi] for xi in x_plot, yi in y_plot]

# Plot the surface
plot(
    y_plot,
    x_plot,
    Z,
    st = :surface,
    xlabel = "Strike",
    ylabel = "Maturity",
    zlabel = "Interpolated Value",
    title = "2D Interpolation Surface",
    legend = false,
    camera = (60, 30),
)
