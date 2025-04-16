using Revise
using Hedgehog2
using Dates
using Printf

# Set up option parameters
spot = 100.0
strike = 100.0
rate = 0.05
sigma = 0.20
reference_date = Date(2023, 1, 1)
expiry = reference_date + Year(1)

# Create the payoff (European call option)
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Create market inputs
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Create pricing problem
prob = PricingProblem(payoff, market_inputs)

# Method 1: Black-Scholes Analytic Formula
bs_method = BlackScholesAnalytic()
bs_solution = solve(prob, bs_method)

# Method 2: Monte Carlo with Exact Black-Scholes Simulation
# Using 100,000 paths for more accurate results
trajectories = 5_000
mc_exact_method =
    MonteCarlo(LognormalDynamics(), BlackScholesExact(), SimulationConfig(trajectories))
mc_exact_solution = solve(prob, mc_exact_method)

# Method 3: Monte Carlo with Euler-Maruyama discretization
# Using 100,000 paths and 100 time steps
trajectories = 10_000
steps = 100
mc_euler_method =
    MonteCarlo(LognormalDynamics(), EulerMaruyama(), SimulationConfig(trajectories, steps=steps, variance_reduction=Hedgehog2.Antithetic()))
mc_euler_solution = solve(prob, mc_euler_method)

# Print the results
println("European Call Option Pricing Comparison")
println("----------------------------------------")
println("Parameters:")
println("Spot: $spot")
println("Strike: $strike")
println("Risk-free rate: $rate")
println("Volatility: $sigma")
println("Time to maturity: 1 year\n")

println("Pricing Results:")
@printf("Black-Scholes Analytic:    %.6f\n", bs_solution.price)
@printf("Monte Carlo (Exact):       %.6f\n", mc_exact_solution.price)
@printf("Monte Carlo (Euler):       %.6f\n", mc_euler_solution.price)

# Calculate and print relative errors
rel_error_exact = abs(mc_exact_solution.price - bs_solution.price) / bs_solution.price
rel_error_euler = abs(mc_euler_solution.price - bs_solution.price) / bs_solution.price

println("\nRelative Errors:")
@printf("Monte Carlo (Exact):       %.6f%%\n", rel_error_exact * 100)
@printf("Monte Carlo (Euler):       %.6f%%\n", rel_error_euler * 100)
