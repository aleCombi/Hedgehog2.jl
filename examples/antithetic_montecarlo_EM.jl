using DifferentialEquations
using DiffEqNoiseProcess
using Random
using Plots

# Parameters
μ = 0.05
σ = 0.2
t0 = 0.0
T = 1.0
S0 = 100.0
tspan = (t0, T)
N = 250
dt = (T - t0) / N

# Define GBM drift and diffusion
function drift!(du, u, p, t)
    du[1] = μ * u[1]
end

function diffusion!(du, u, p, t)
    du[1] = σ * u[1]
end

# Create noise with fixed seed
rng = MersenneTwister(123)
wiener = WienerProcess(t0, 0.0)

# Define and solve the original GBM SDE
u0 = [S0]
sde = SDEProblem(drift!, diffusion!, u0, tspan, noise=wiener)
sol = DifferentialEquations.solve(sde, dt=dt, save_noise=true)
sol = DifferentialEquations.solve(sde, dt=dt, save_noise=true)
plot(sol.W.t, sol.W.u)
# Extract the used noise path
# W_vals = sol.W[1, :]  # Extract Brownian path for the first dimension

# Flip it for antithetic path
wiener_antithetic = NoiseGrid(sol.W.t, -sol.W.W)

# # Define and solve the antithetic GBM SDE
sde_antithetic = SDEProblem(drift!, diffusion!, u0, tspan, noise=wiener_antithetic)
sol_antithetic = solve(sde_antithetic, dt=dt)

# # Plot
plot(sol, label="Original", linewidth=2)
plot!(sol_antithetic, label="Antithetic", linestyle=:dash, linewidth=2)
# xlabel!("Time")
# ylabel!("GBM Value")
# title!("GBM with Antithetic Variates via Flipped Brownian Path")
