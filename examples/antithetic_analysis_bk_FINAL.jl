using Hedgehog2
using Dates
using Random
using Statistics
using Plots

function analyze_bk_antithetic_paths(num_paths=10, seed=42)
    # Set random seed for reproducibility
    Random.seed!(seed)
    
    # Set up common parameters
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    spot = 100.0
    strike = 100.0
    
    # Create payoff
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
    
    # Heston parameters - can be adjusted to see different behaviors
    rate_heston = 0.03
    V0 = 0.04
    κ = 2.0
    θ = 0.04
    σ = 0.3
    ρ = -0.7
    
    # Create the market inputs
    heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
    prob = PricingProblem(payoff, heston_market)
    
    # Generate seeds for paths
    path_seeds = rand(1:10^9, num_paths ÷ 2)
    
    # Create the Broadie-Kaya method with antithetic sampling
    bk_strategy = HestonBroadieKaya(num_paths ÷ 2, steps=20)
    bk_method = MonteCarlo(HestonDynamics(), bk_strategy)
    
    # Update with seeds and antithetic flag
    antithetic_method = @set bk_method.strategy.seeds = path_seeds
    antithetic_method = @set antithetic_method.strategy.kwargs = merge(antithetic_method.strategy.kwargs, (antithetic=true,))
    
    # Solve the problem
    solution = solve(prob, antithetic_method)
    
    # Return the solution for further analysis
    return solution, prob
end

function plot_bk_path_pair(solution, prob, path_index=1; apply_exp=true)
    half_paths = length(solution.ensemble.solutions) ÷ 2
    
    # Get the original and antithetic paths
    original_path = solution.ensemble.solutions[path_index]
    antithetic_path = solution.ensemble.solutions[half_paths + path_index]
    
    # Extract time points
    time_points = original_path.t
    
    # Extract stock prices (log to normal scale if requested)
    if apply_exp
        original_prices = [exp(u[1]) for u in original_path.u]
        antithetic_prices = [exp(u[1]) for u in antithetic_path.u]
        ylabel_text = "Stock Price"
    else
        original_prices = [u[1] for u in original_path.u]
        antithetic_prices = [u[1] for u in antithetic_path.u]
        ylabel_text = "Log Price"
    end
    
    # Create price plot
    p1 = plot(
        time_points, original_prices,
        label="Original Path",
        linewidth=2,
        title="Heston Broadie-Kaya: $(apply_exp ? "Stock Price" : "Log Price") Paths",
        xlabel="Time (years)",
        ylabel=ylabel_text,
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
    
    # Calculate price correlation
    price_corr = cor(original_prices, antithetic_prices)
    annotate!(p1, [(0.5, minimum(original_prices), 
                 text("Path correlation: $(round(price_corr, digits=4))", 10, :left))])
    
    return p1
end

function calculate_bk_path_statistics(solution, prob)
    half_paths = length(solution.ensemble.solutions) ÷ 2
    
    price_correlations = Float64[]
    terminal_products = Float64[]
    discount_factor = df(prob.market.rate, prob.payoff.expiry)
    
    original_terminals = Float64[]
    antithetic_terminals = Float64[]
    
    # Calculate statistics for each path pair
    for i in 1:half_paths
        original_path = solution.ensemble.solutions[i]
        antithetic_path = solution.ensemble.solutions[half_paths + i]
        
        # Extract full paths for correlation
        original_prices = [exp(u[1]) for u in original_path.u]
        antithetic_prices = [exp(u[1]) for u in antithetic_path.u]
        
        # Calculate correlation between full paths
        path_corr = cor(original_prices, antithetic_prices)
        push!(price_correlations, path_corr)
        
        # Extract terminal values
        original_terminal = exp(original_path.u[end][1])
        antithetic_terminal = exp(antithetic_path.u[end][1])
        
        # Calculate product of terminal values
        push!(terminal_products, original_terminal * antithetic_terminal)
        
        # Store terminal values for payoff calculation
        push!(original_terminals, original_terminal)
        push!(antithetic_terminals, antithetic_terminal)
    end
    
    # Calculate payoffs
    original_payoffs = prob.payoff.(original_terminals)
    antithetic_payoffs = prob.payoff.(antithetic_terminals)
    
    # Calculate standard MC estimator variance (using original paths)
    standard_estimator = discount_factor * original_payoffs
    standard_variance = var(standard_estimator)
    
    # Calculate antithetic MC estimator variance
    antithetic_pairs = [(original_payoffs[i] + antithetic_payoffs[i]) / 2 for i in 1:half_paths]
    antithetic_estimator = discount_factor * antithetic_pairs
    antithetic_variance = var(antithetic_estimator)
    
    # Calculate payoff correlation
    payoff_correlation = cor(original_payoffs, antithetic_payoffs)
    
    # Calculate variance reduction
    var_reduction_ratio = standard_variance / antithetic_variance
    
    return (
        mean_path_correlation = mean(price_correlations),
        path_correlations = price_correlations,
        mean_terminal_product = mean(terminal_products),
        payoff_correlation = payoff_correlation,
        standard_variance = standard_variance,
        antithetic_variance = antithetic_variance,
        var_reduction_ratio = var_reduction_ratio
    )
end

# Run analysis and generate plots
num_paths = 100  # Use an even number
bk_solution, bk_prob = analyze_bk_antithetic_paths(num_paths, 42)

# Plot several path pairs to observe different behaviors
p1 = plot_bk_path_pair(bk_solution, bk_prob, 1)
p2 = plot_bk_path_pair(bk_solution, bk_prob, 2)
p3 = plot_bk_path_pair(bk_solution, bk_prob, 3)
p4 = plot_bk_path_pair(bk_solution, bk_prob, 4)

# Plot them in a 2x2 grid
path_plots = plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700), title="Broadie-Kaya Antithetic Path Examples")
display(path_plots)

# Also plot some in log scale
p1_log = plot_bk_path_pair(bk_solution, bk_prob, 1, apply_exp=false)
p2_log = plot_bk_path_pair(bk_solution, bk_prob, 2, apply_exp=false)

log_plots = plot(p1_log, p2_log, layout=(1,2), size=(900, 400), title="Broadie-Kaya Paths (Log Scale)")
display(log_plots)

# Calculate statistics across all paths
stats = calculate_bk_path_statistics(bk_solution, bk_prob)

# Print summary statistics
println("=== Broadie-Kaya Antithetic Path Analysis ===")
println("Number of paths: $num_paths")
println("Average path correlation: $(round(stats.mean_path_correlation, digits=4))")
println("Payoff correlation: $(round(stats.payoff_correlation, digits=4))")
println("Standard estimator variance: $(round(stats.standard_variance, digits=8))")
println("Antithetic estimator variance: $(round(stats.antithetic_variance, digits=8))")
println("Variance reduction ratio: $(round(stats.var_reduction_ratio, digits=2))×")

# Plot distribution of path correlations
histogram!(
    stats.path_correlations, 
    bins=20, 
    title="Distribution of Broadie-Kaya Path Correlations",
    xlabel="Correlation",
    ylabel="Frequency",
    legend=false
)