using Revise
using Hedgehog2  # Replace with your actual module name if different
using DataInterpolations
using Dates
using Plots

# Input market data
tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
dfs = [0.995, 0.990, 0.980, 0.955, 0.890, 0.750]
ref_date = Date(2025, 1, 1)

# Build interpolated curves
curve_linear = Hedgehog2.RateCurve(ref_date, tenors, dfs;
    interp = LinearInterpolation,
    extrap = ExtrapolationType.Constant,
)

curve_cubic = Hedgehog2.RateCurve(ref_date, tenors, dfs;
    interp = CubicSpline,
    extrap = ExtrapolationType.Constant,
)

# Evaluation grid for plotting
ts = range(0.0, stop=10.0, length=300)
zr_linear = zero_rate.(Ref(curve_linear), ts)
zr_cubic = zero_rate.(Ref(curve_cubic), ts)
zr_input = @. -log(dfs) / tenors  # Implied input zero rates

# Plot interpolated zero rate curves
plot(ts, zr_linear, label="Linear Interpolation", lw=2)
plot!(ts, zr_cubic, label="Cubic Spline Interpolation", lw=2, ls=:dash)
scatter!(tenors, zr_input, label="Input Zero Rates", ms=4, color=:black)
xlabel!("Time to Maturity (Years)")
ylabel!("Zero Rate (Continuously Compounded)")
title!("Interpolated Zero Rate Curve")
