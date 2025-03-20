using Revise, Hedgehog2, Distributions, DifferentialEquations, Random, Plots

# Define Heston model parameters
S0 = 1.0    # Initial stock price
V0 = 0.010201    # Initial variance
κ = 6.21      # Mean reversion speed
θ = 0.019      # Long-run variance
σ = 0.61   # Volatility of variance
ρ = -0.7     # Correlation
r = 0.0319      # Risk-free rate
T = 1.0       # Time to maturity

# Create the exact sampling Heston distribution
heston_dist = Hedgehog2.HestonDistribution(S0, V0, κ, θ, σ, ρ, r, T)

# Construct the Heston Noise Process
heston_noise = Hedgehog2.HestonNoise(0.0, heston_dist)

# Define `NoiseProblem`
problem = NoiseProblem(heston_noise, (0.0, T))

trajectories = 1000
# Solve with multiple trajectories
solution_exact = solve(EnsembleProblem(problem), dt=T, trajectories=trajectories)

u0 = [S0, V0]  # Initial (Stock Price, Variance)
tspan = (0.0, 1.0)  # Simulation for 1 year
process = HestonProcess(r, κ, θ, σ, ρ)  # Heston parameters
prob = SDEProblem(get_sde_function(process), u0, tspan, process)
ensemble_problem = EnsembleProblem(prob)
# solution_h2 = solve(ensemble_problem, dt=T/100.0, trajectories=trajectories)

# final_prices_2 = [sol.u[end][1] for sol in solution_h2]  # Extract S_T for each trajectory
final_prices_exact = [sol.u[end] for sol in solution_exact]  # Transform log-prices to prices

# println("Euler: ", mean(max.(final_prices_2 .- 1, 0)), " ", var(final_prices_2))
price = mean(max.(final_prices_exact.- 1, 0))
variance = var(final_prices_exact)
println("Exact: ", price, " variance: ",variance, " error ", (price - 6.8061)^2)
