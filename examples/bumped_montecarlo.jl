using DifferentialEquations
using DiffEqNoiseProcess
using Plots

# Parameters
μ = 0.05
σ = 0.2
t0 = 0.0
T = 1.0
W0 = 100.0
tspan = (t0, T)
N = 250
dt = (T - t0) / N

# Step 1: Simulate GBM with a fixed seed
gbm = GeometricBrownianMotionProcess(μ, σ, t0, W0)
gbm_noise = NoiseProblem(gbm, tspan, rng = MersenneTwister(42))
sol = DifferentialEquations.solve(gbm_noise, dt = dt)

# Step 2: Reuse the same seed with bumped σ
μ_bumped = μ
σ_bumped = σ + 0.05
gbm_bumped = GeometricBrownianMotionProcess(μ_bumped, σ_bumped, t0, W0)
gbm_bumped_noise = NoiseProblem(gbm_bumped, tspan, seed = 1)
sol_bumped = DifferentialEquations.solve(gbm_bumped_noise, dt = dt)

# Step 3: Plot both
plot(sol.t, sol.u, label = "μ = $μ", linewidth = 2)
plot!(sol_bumped.t, sol_bumped.u, label = "μ = $μ_bumped", linestyle = :dash, linewidth = 2)
xlabel!("Time")
ylabel!("GBM Value")
title!("GBM with Drift Bump and Reused Seed")
