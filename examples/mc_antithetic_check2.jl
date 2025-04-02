using Hedgehog2
using Dates
using Plots
using Random

# Set up option parameters
strike = 100.0
reference_date = Date(2020, 1, 1)
expiry = reference_date + Year(1)
rate = 0.05
spot = 100.0
sigma = 0.20

# Create the payoff (European call option)
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Create market inputs
market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# Create pricing problem
prob = PricingProblem(payoff, market_inputs)

# Set up Monte Carlo with antithetic variates
# Using fewer paths since we're just interested in visualizing
trajectories = 10
time_steps = 100  # More steps for smoother paths

# Use fixed seed for reproducibility
Random.seed!(1234)
seeds = rand(1:10^9, trajectories)

# Create Monte Carlo method with BlackScholesExact strategy and antithetic sampling
strategy = BlackScholesExact(trajectories, time_steps, seeds=seeds, antithetic=true)
mc_method = MonteCarlo(LognormalDynamics(), strategy)

# Solve the pricing problem
solution = solve(prob, mc_method)

# Extract paths from the solution
# With antithetic=true, the first trajectories paths are original
# and the next trajectories paths are antithetic
ensemble = solution.ensemble
all_paths = ensemble.solutions

# We know with antithetic=true, we have original paths followed by antithetic paths
# Extract the first original path and its corresponding antithetic path
original_path = all_paths[1]
antithetic_path = all_paths[trajectories + 1]  # First antithetic path

# Extract time points and stock prices
time_points = original_path.t

# Get stock prices (might be stored as vectors if multi-dimensional)
if eltype(original_path.u) <: AbstractVector
    original_prices = [u[1] for u in original_path.u]
    antithetic_prices = [u[1] for u in antithetic_path.u]
else
    original_prices = original_path.u
    antithetic_prices = antithetic_path.u
end

# Create the plot
p = plot(
    time_points, original_prices,
    label="Original Path",
    linewidth=2,
    title="Black-Scholes: Original vs Antithetic Path",
    xlabel="Time (years)",
    ylabel="Stock Price",
    legend=:topleft
)

plot!(
    time_points, antithetic_prices,
    label="Antithetic Path",
    linewidth=2,
    linestyle=:dash
)

# Add a horizontal line at initial spot price
hline!([spot], label="Initial Price", linewidth=1, linestyle=:dot, color=:black)

# Display the plot
display(p)

# Calculate terminal values for comparison
terminal_original = original_prices[end]
terminal_antithetic = antithetic_prices[end]

println("Initial stock price: $spot")
println("Terminal price (original path): $terminal_original")
println("Terminal price (antithetic path): $terminal_antithetic")
println("Product of terminals: $(terminal_original * terminal_antithetic)")
println("Square of initial: $(spot^2 * exp(2*rate))")  # Expected product with drift

# Calculate correlation between paths
correlation = cor(original_prices, antithetic_prices)
println("Correlation between paths: $correlation")