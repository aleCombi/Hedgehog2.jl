using Revise
using Hedgehog2
using Distributions
using Random
using Plots
using Dates
using Statistics
using Printf

println("=== Heston Model: Antithetic Variates Variance Reduction Analysis ===")

# --- Market Inputs ---
reference_date = Date(2020, 1, 1)

# Heston parameters (chosen to satisfy Feller condition)
S0 = 1.0
V0 = 0.04         # Initial variance (20% vol)
κ = 1.0           # Mean reversion
θ = 0.04          # Long-term variance
σ = 0.2           # Vol-of-vol
ρ = -0.5          # Correlation
r = 0.02          # Risk-free rate

market_inputs = HestonInputs(reference_date, r, S0, V0, κ, θ, σ, ρ)

# --- Payoff ---
expiry = reference_date + Year(5)
strike = S0  # ATM call
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# --- Dynamics ---
dynamics = HestonDynamics()

# --- Get reference price using Carr-Madan (Fourier) method ---
α = 1.0
boundary = 32.0
carr_madan_method = CarrMadan(α, boundary, dynamics)
prob = PricingProblem(payoff, market_inputs)
carr_madan_solution = solve(prob, carr_madan_method)
reference_price = carr_madan_solution.price

println("Reference price (Carr-Madan): $reference_price")

# --- Variance Reduction Analysis ---
# We'll compare standard MC vs antithetic variates
n_trials = 30        # Number of independent trials
base_paths = 500     # Base number of paths
time_steps = 20      # Time steps for each path
Random.seed!(42)     # For reproducibility

# Arrays to store results
std_prices = Float64[]
anti_prices = Float64[]
std_times = Float64[]
anti_times = Float64[]

# Repeat trials
for trial in 1:n_trials
    # Different seed for each trial
    trial_seed = 42 + trial
    
    # Standard Monte Carlo (full number of paths)
    std_strategy = HestonBroadieKaya(base_paths, steps=time_steps, seeds=rand(MersenneTwister(trial_seed), 1:10^9, base_paths))
    std_method = MonteCarlo(dynamics, std_strategy)
    
    std_time = @elapsed begin
        std_solution = solve(prob, std_method)
        push!(std_prices, std_solution.price)
    end
    push!(std_times, std_time)
    
    # Antithetic Monte Carlo (half the number of paths × 2)
    anti_strategy = HestonBroadieKaya(base_paths ÷ 2, steps=time_steps, seeds=rand(MersenneTwister(trial_seed), 1:10^9, base_paths ÷ 2), antithetic=true)
    anti_method = MonteCarlo(dynamics, anti_strategy)
    
    anti_time = @elapsed begin
        anti_solution = solve(prob, anti_method)
        push!(anti_prices, anti_solution.price)
    end
    push!(anti_times, anti_time)
    
    # Print progress
    if trial % 5 == 0
        println("Completed $trial/$n_trials trials")
    end
end

# --- Calculate variance reduction statistics ---
std_mean = mean(std_prices)
anti_mean = mean(anti_prices)

std_var = var(std_prices)
anti_var = var(anti_prices)

std_bias = std_mean - reference_price
anti_bias = anti_mean - reference_price

std_mse = mean((std_prices .- reference_price).^2)
anti_mse = mean((anti_prices .- reference_price).^2)

var_reduction_ratio = std_var / anti_var
mse_reduction_ratio = std_mse / anti_mse
time_ratio = mean(std_times) / mean(anti_times)

# Efficiency improvement (variance reduction per unit of computation)
efficiency_gain = var_reduction_ratio * time_ratio

# --- Visualization ---
# Distribution of prices
histogram_plot = histogram(
    std_prices, 
    alpha=0.5, 
    label="Standard MC",
    bins=15,
    title="Distribution of Price Estimates",
    xlabel="Price",
    ylabel="Frequency"
)

histogram!(
    histogram_plot,
    anti_prices, 
    alpha=0.5, 
    label="Antithetic MC",
    bins=15
)

# Add reference price line
vline!(
    histogram_plot,
    [reference_price], 
    label="Reference Price",
    color=:red, 
    linewidth=2, 
    linestyle=:dash
)

# --- Print results ---
println("\n=== Variance Reduction Analysis ===")
println("Number of trials: $n_trials")
println("Paths (standard MC): $base_paths")
println("Paths (antithetic MC): $(base_paths ÷ 2) × 2")
println()
println("Reference price: $reference_price")
println()
@printf("Standard MC mean:     %.6f (bias: %.6f)\n", std_mean, std_bias)
@printf("Antithetic MC mean:   %.6f (bias: %.6f)\n", anti_mean, anti_bias)
println()
@printf("Standard MC variance: %.6e\n", std_var)
@printf("Antithetic variance:  %.6e\n", anti_var)
@printf("Variance reduction:   %.2fx\n", var_reduction_ratio)
println()
@printf("Standard MC MSE:      %.6e\n", std_mse)
@printf("Antithetic MC MSE:    %.6e\n", anti_mse)
@printf("MSE reduction:        %.2fx\n", mse_reduction_ratio)
println()
@printf("Avg. time (standard): %.3f seconds\n", mean(std_times))
@printf("Avg. time (antithetic): %.3f seconds\n", mean(anti_times))
@printf("Time ratio:           %.2fx\n", time_ratio)
println()
@printf("Efficiency gain:      %.2fx\n", efficiency_gain)

# --- Visualize a sample path pair ---
if length(std_prices) > 0 && length(anti_prices) > 0
    Random.seed!(42)  # Reset seed for consistent visualization
    
    # Simulate a single path with antithetic variates for visualization
    vis_strategy = HestonBroadieKaya(1, steps=time_steps, antithetic=true)
    vis_method = MonteCarlo(dynamics, vis_strategy)
    vis_solution = solve(prob, vis_method)
    
    # Extract paths
    original_path = vis_solution.ensemble.solutions[1]
    antithetic_path = vis_solution.ensemble.solutions[2]  # Second path is antithetic
    
    # Extract time points
    time_points = original_path.t
    
    # Extract spot prices and variances
    original_prices = [exp(p[1]) for p in original_path.u]
    antithetic_prices = [exp(p[1]) for p in antithetic_path.u]
    original_variance = [p[2] for p in original_path.u]
    antithetic_variance = [p[2] for p in antithetic_path.u]
    
    # Create path plots
    p1 = plot(
        time_points, original_prices,
        label="Original Path",
        linewidth=2,
        title="Heston Model: Stock Price Paths",
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
    
    p2 = plot(
        time_points, original_variance,
        label="Original Path",
        linewidth=2,
        title="Heston Model: Variance Paths",
        xlabel="Time (years)",
        ylabel="Variance",
        legend=:topleft
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
    path_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    # Calculate price-path correlation
    path_correlation = cor(original_prices, antithetic_prices)
    var_correlation = cor(original_variance, antithetic_variance)
    
    println("\n=== Path Correlation Analysis ===")
    @printf("Stock price path correlation: %.4f\n", path_correlation)
    @printf("Variance path correlation:    %.4f\n", var_correlation)
    
    # Display both plots
    display(histogram_plot)
    display(path_plot)
end