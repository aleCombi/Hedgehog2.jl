using Revise
using Hedgehog2
using Random
using Plots
using Dates
using Statistics

println("=== Heston Model: Broadie-Kaya Path Simulation ===")

# --- Market Inputs ---
reference_date = Date(2020, 1, 1)

# Heston parameters
S0 = 1.0
V0 = 0.010201
κ = 4
θ = 0.019
σ = 0.15
ρ = -0.7
r = 0.0319

market_inputs = HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# --- Payoff ---
expiry = reference_date + Year(5)
strike = S0
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# --- Set up the pricing problem ---
prob = PricingProblem(payoff, market_inputs)

# --- Monte Carlo with Broadie-Kaya ---
trajectories = 1
steps = 20 # Increase steps for better visualization

# Create Broadie-Kaya strategy 
strategy = HestonBroadieKaya(trajectories, steps=steps)
bk_method = MonteCarlo(HestonDynamics(), strategy)

# Solve to get the path
solution = solve(prob, bk_method)

# --- Extract the path ---
path = solution.ensemble.solutions[1]

# Check the actual structure of the path
println("Path structure:")
println("Length of time points: ", length(path.t))
println("Length of values: ", length(path.u))

# Extract data for visualization
time_points = path.t
stock_prices = [exp(p[1]) for p in path.u]
variances = [p[2] for p in path.u]

# --- Create visualization ---
p1 = plot(
    time_points, stock_prices,
    linewidth=2,
    title="Heston Model: Stock Price Path (Broadie-Kaya)",
    xlabel="Time (years)",
    ylabel="Stock Price",
    legend=false,
    color=:blue
)

p2 = plot(
    time_points, variances,
    linewidth=2,
    title="Heston Model: Variance Path",
    xlabel="Time (years)",
    ylabel="Variance",
    legend=false,
    color=:red
)

combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
display(combined_plot)

# Calculate path statistics
if length(stock_prices) > 2
    log_returns = diff(log.(stock_prices))
    volatility = sqrt(252) * std(log_returns)  # Annualized volatility
    
    println("\n=== Path Statistics ===")
    println("Initial stock price: $S0")
    println("Initial variance: $V0")
    println("Terminal stock price: $(stock_prices[end])")
    println("Terminal variance: $(variances[end])")
    println("Mean variance: $(mean(variances))")
    println("Realized volatility: $(volatility * 100)%")
    println("Path duration: $(time_points[end]) years")
    println("Number of time points: $(length(time_points))")
end