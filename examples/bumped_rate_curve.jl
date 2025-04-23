# interpolation_sandbox.jl
using DataInterpolations
using Plots

# Simple spine
tenors = [0.5, 1.0, 2.0, 5.0, 10.0]
zero_rates = [0.01, 0.015, 0.04, 0.025, 0.03]

# Choose your interpolator
interp =
    QuadraticInterpolation(zero_rates, tenors; extrapolation = ExtrapolationType.Constant)

# Evaluate interpolator
ts = range(0.0, stop = 12.0, length = 300)
zs = interp.(ts)

# Bump the 3rd zero rate
bumped_z = copy(zero_rates)
bumped_z[3] += 0.0025  # 25 bps bump
interp_bumped =
    QuadraticInterpolation(bumped_z, tenors; extrapolation = ExtrapolationType.Constant)
zs_bumped = interp_bumped.(ts)

# Plot
plot(ts, zs, label = "Original", linewidth = 2)
plot!(ts, zs_bumped, label = "Bumped", linestyle = :dash, linewidth = 2)
scatter!(tenors, zero_rates, label = "Spine", color = :blue)
scatter!(tenors, bumped_z, label = "Bumped Spine", color = :orange)
xlabel!("Tenor (years)")
ylabel!("Zero rate")
title!("Zero Rate Interpolation and Bump Effect")
