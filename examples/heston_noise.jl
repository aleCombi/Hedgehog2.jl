using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots

# Define Heston model parameters
S0 = 100.0    # Initial stock price
V0 = 0.04     # Initial variance
κ = 2.0       # Mean reversion speed
θ = 0.04      # Long-run variance
σ = 0.3       # Volatility of variance
ρ = -0.7      # Correlation
r = 0.05      # Risk-free rate
T = 1.0       # Time to maturity

# Create the exact sampling Heston distribution
heston_dist = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)

# Construct the Heston Noise Process
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist)

# Define `NoiseProblem`
problem = NoiseProblem(heston_noise, (0.0, T))

# Solve with multiple trajectories
solution = solve(EnsembleProblem(problem), dt=T, trajectories=10)

# Directly plot the solution
plot(solution, title="Exact Heston Model Simulations",
     xlabel="Time", ylabel="Log Stock Price")
