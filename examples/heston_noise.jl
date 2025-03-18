using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots

# Define Heston model parameters
S0 = 1.0    # Initial stock price
V0 = 0.04     # Initial variance
κ = 2.0       # Mean reversion speed
θ = 0.04      # Long-run variance
σ = 0.02     # Volatility of variance
ρ = -0.7      # Correlation
r = 0.05      # Risk-free rate
T = 1.0       # Time to maturity

# Create the exact sampling Heston distribution
heston_dist = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)

# Construct the Heston Noise Process
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist)

# Define `NoiseProblem`
problem = NoiseProblem(heston_noise, (0.0, T))

trajectories = 10000
# Solve with multiple trajectories
solution_h = solve(EnsembleProblem(problem), dt=T, trajectories=trajectories)

u0 = [S0, V0]  # Initial (Stock Price, Variance)
tspan = (0.0, 1.0)  # Simulation for 1 year
process = HestonProcess(r, κ, θ, σ, ρ)  # Heston parameters
prob = SDEProblem(get_sde_function(process), u0, tspan, process)
ensemble_problem = EnsembleProblem(prob)
solution_h2 = solve(ensemble_problem, dt=T, trajectories=trajectories)

final_prices_2 = [sol.u[end][1] for sol in solution_h2]  # Extract S_T for each trajectory
final_prices = [sol.u[end] for sol in solution_h]  # Transform log-prices to prices

println(mean(final_prices_2.^3))
println(mean(final_prices.^3))
