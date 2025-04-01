using DifferentialEquations
using DiffEqNoiseProcess
using Plots
using Random
using Hedgehog2


u0 = [100.0, 0.04]  # Initial (Stock Price, Variance)
tspan = (0.0, 1.0)  # Simulation for 1 year
# Create and solve the original problem with saved noise
rng = MersenneTwister(123)
prob = Hedgehog2.HestonProblem(0.05, 2.0, 0.04, 0.3, -0.5, u0, tspan)  # Heston parameters
sol = solve(prob, EM(), dt=1//250, save_noise=true)

# Extract the 2D noise and flip it for antithetic
W_t = sol.W.t
W_u = sol.W.W  # This is a Matrix: each row is a Brownian component

# Flip both components for antithetic version
antithetic_noise = NoiseGrid(W_t, -W_u)

# Create new problem with antithetic noise
prob_antithetic = remake(prob; noise=antithetic_noise)
sol_anti = solve(prob_antithetic, EM(), dt=1//250)

# Plot only the first component (stock price)
plot(sol.t, sol.u .|> x -> x[1], label="Original", linewidth=2)
plot!(sol_anti.t, sol_anti.u .|> x -> x[1], label="Antithetic", linestyle=:dash, linewidth=2)
xlabel!("Time")
ylabel!("Stock Price")
title!("Heston Process: Original vs Antithetic Paths")
