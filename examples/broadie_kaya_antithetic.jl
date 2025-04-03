using Revise
using Hedgehog2
using Distributions
using Random
using Plots
using Dates

println("=== Heston Model: Broadie-Kaya with Antithetic Variates ===")

# --- Market Inputs ---
reference_date = Date(2020, 1, 1)

S0 = 1.0
V0 = 0.04         # Initial variance (20% vol)
κ = 1.0           # Lower mean reversion
θ = 0.04          # Long-term variance matches initial
σ = 0.2           # Lower vol-of-vol
ρ = -0.5          # Less extreme correlation
r = 0.02          # Lower interest rate

market_inputs = HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# --- Payoff ---
expiry = reference_date + Year(5)
strike = S0  # ATM call
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# --- Dynamics ---
dynamics = HestonDynamics()

# --- Carr-Madan (for reference price) ---
α = 1.0
boundary = 32.0
carr_madan_method = CarrMadan(α, boundary, dynamics)
prob = PricingProblem(payoff, market_inputs)
carr_madan_solution = solve(prob, carr_madan_method)
reference_price = carr_madan_solution.price

println("Reference price (Carr-Madan): $reference_price")

# --- Monte Carlo with Broadie-Kaya ---
# Just a few paths for visualization
trajectories = 2000
steps = 20

# Create Broadie-Kaya strategy with antithetic variates
strategy = HestonBroadieKaya(trajectories ÷ 2, steps=steps, antithetic=true)
bk_method = MonteCarlo(dynamics, strategy)
bk_solution = solve(prob, bk_method)

# --- Extract paths for visualization ---
ensemble = bk_solution.ensemble
all_paths = ensemble.solutions

# Choose a path index to visualize
path_index = 1

# Extract original path and its antithetic counterpart
original_path = all_paths[path_index]
antithetic_path = all_paths[(trajectories ÷ 2) + path_index]

# Extract time points
time_points = original_path.t

# Extract spot prices (exp of log-price)
original_prices = [exp(p[1]) for p in original_path.u]
antithetic_prices = [exp(p[1]) for p in antithetic_path.u]

# Extract variance paths
original_variance = [p[2] for p in original_path.u]
antithetic_variance = [p[2] for p in antithetic_path.u]

# --- Create plots ---
# Price paths
p1 = plot(
    time_points, original_prices,
    label="Original Path",
    linewidth=2,
    title="Heston Model: Stock Price Paths (Broadie-Kaya)",
    xlabel="Time (years)",
    ylabel="Stock Price",
    legend=:topleft
)

plot!(
    p1,
    time_points, antithetic_prices,
    label="Antithetic Path",
    linewidth=2,
    linestyle=:dash,
    color=:red
)

# Variance paths
p2 = plot(
    time_points, original_variance,
    label="Original Path",
    linewidth=2,
    title="Heston Model: Variance Paths",
    xlabel="Time (years)",
    ylabel="Variance",
    legend=:topleft,
    ylims=(0, max(maximum(original_variance), maximum(antithetic_variance)) * 1.1)
)

plot!(
    p2,
    time_points, antithetic_variance,
    label="Antithetic Path",
    linewidth=2,
    linestyle=:dash,
    color=:red
)

# Combine plots
combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))

# Display the plot
display(combined_plot)

# --- Calculate statistics ---
# Terminal values
original_terminal_price = original_prices[end]
antithetic_terminal_price = antithetic_prices[end]
original_terminal_variance = original_variance[end]
antithetic_terminal_variance = antithetic_variance[end]

# Correlations
price_correlation = cor(original_prices, antithetic_prices)
variance_correlation = cor(original_variance, antithetic_variance)

# Log returns
original_returns = diff(log.(original_prices))
antithetic_returns = diff(log.(antithetic_prices))
returns_correlation = cor(original_returns, antithetic_returns)

# --- Print summary statistics ---
println("\n=== Path Statistics ===")
println("Initial stock price: $S0")
println("Initial variance: $V0")
println()
println("Terminal stock price (original): $original_terminal_price")
println("Terminal stock price (antithetic): $antithetic_terminal_price")
println("Terminal variance (original): $original_terminal_variance")
println("Terminal variance (antithetic): $antithetic_terminal_variance")
println()
println("Stock price correlation: $price_correlation")
println("Variance correlation: $variance_correlation")
println("Log returns correlation: $returns_correlation")

# --- Price from full Monte Carlo simulation ---
println("\n=== Monte Carlo pricing (with all paths) ===")
println("Broadie-Kaya price: $(bk_solution.price)")
println("Relative error: $(abs(bk_solution.price - reference_price)/reference_price * 100)%")