using Hedgehog2
using Dates
using Plots
using Random
using Statistics

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

# Let's increase the number of paths to get better variance estimates
trajectories = 5000
time_steps = 1

# Use fixed seed for reproducibility
Random.seed!(1234)
seeds = rand(1:10^9, trajectories)

# First with antithetic sampling
strategy_antithetic = BlackScholesExact(trajectories ÷ 2, time_steps, seeds=seeds[1:trajectories÷2], antithetic=true)
mc_method_antithetic = MonteCarlo(LognormalDynamics(), strategy_antithetic)
solution_antithetic = solve(prob, mc_method_antithetic)

# Then without antithetic sampling, using same number of total paths
strategy_standard = BlackScholesExact(trajectories, time_steps, seeds=seeds)
mc_method_standard = MonteCarlo(LognormalDynamics(), strategy_standard)
solution_standard = solve(prob, mc_method_standard)

# For visualization, we'll just use the first path from the antithetic solution
solution = solution_antithetic

# Extract paths from the solution
# With antithetic=true, the first trajectories paths are original
# and the next trajectories paths are antithetic
ensemble = solution.ensemble
all_paths = ensemble.solutions

# We know with antithetic=true, we have original paths followed by antithetic paths
# Extract an original path and its corresponding antithetic path
path_num = 2500
original_path = all_paths[path_num]
antithetic_path = all_paths[trajectories ÷ 2 + path_num]  # antithetic path

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

# ---------- Measure Variance Reduction ----------

# Obtain analytical solution as reference
analytic_method = BlackScholesAnalytic()
analytic_solution = solve(prob, analytic_method)
reference_price = analytic_solution.price

println("\n----- Variance Reduction Analysis -----")
println("Analytical price: $reference_price")
println("Standard MC price: $(solution_standard.price)")
println("Antithetic MC price: $(solution_antithetic.price)")

# Extract all payoffs to calculate variance
# For standard MC
ensemble_standard = solution_standard.ensemble
standard_paths = ensemble_standard.solutions
standard_terminal_prices = if eltype(standard_paths[1].u) <: AbstractVector
    [path.u[end][1] for path in standard_paths]
else
    [path.u[end] for path in standard_paths]
end
# Calculate undiscounted payoffs
standard_payoffs = max.(standard_terminal_prices .- strike, 0.0)

# For antithetic MC - we need to average pairs of paths
ensemble_antithetic = solution_antithetic.ensemble
antithetic_paths = ensemble_antithetic.solutions
total_paths = length(antithetic_paths)
half_paths = total_paths ÷ 2

# Extract terminal prices for both original and antithetic paths
all_terminal_prices = if eltype(antithetic_paths[1].u) <: AbstractVector
    [path.u[end][1] for path in antithetic_paths]
else
    [path.u[end] for path in antithetic_paths]
end

# Original paths are first half, antithetic are second half
original_terminal_prices = all_terminal_prices[1:half_paths]
antithetic_terminal_prices = all_terminal_prices[(half_paths+1):end]

# Calculate undiscounted payoffs
original_payoffs = max.(original_terminal_prices .- strike, 0.0)
antithetic_payoffs = max.(antithetic_terminal_prices .- strike, 0.0)

# Calculate average payoffs for each pair (original + antithetic)
antithetic_paired_payoffs = [(original_payoffs[i] + antithetic_payoffs[i])/2 for i in 1:half_paths]

# Calculate variances
standard_variance = var(standard_payoffs)
antithetic_variance = var(antithetic_paired_payoffs)
variance_reduction_ratio = standard_variance / antithetic_variance

println("\nVariance Analysis:")
println("Standard MC variance: $standard_variance")
println("Antithetic MC variance: $antithetic_variance")
println("Variance reduction ratio: $(round(variance_reduction_ratio, digits=2))x")

# Calculate the standard errors of the mean
standard_se = sqrt(standard_variance / length(standard_payoffs))
antithetic_se = sqrt(antithetic_variance / length(antithetic_paired_payoffs))
se_reduction_ratio = standard_se / antithetic_se

println("\nStandard Error Analysis:")
println("Standard MC SE: $standard_se")
println("Antithetic MC SE: $antithetic_se")
println("SE reduction ratio: $(round(se_reduction_ratio, digits=2))x")

# Calculate efficiency gain (considering we need half the paths with antithetic)
efficiency_gain = variance_reduction_ratio * 2  # Adjusting for the fact that antithetic uses half the independent paths

println("\nEfficiency Analysis:")
println("Computational efficiency gain: $(round(efficiency_gain, digits=2))x")
println("(This accounts for the fact that antithetic sampling generates 2 paths for each set of random numbers)")