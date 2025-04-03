using Hedgehog2
using Dates
using Random
using Statistics
using Plots

function analyze_antithetic_paths(prob::PricingProblem, mc_method::MonteCarlo; 
                                  num_paths=10, seed=nothing)
    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Generate seeds for paths
    path_seeds = rand(1:10^9, num_paths ÷ 2)
    
    # Create the antithetic method
    antithetic_method = @set mc_method.strategy.seeds = path_seeds
    antithetic_method = @set antithetic_method.strategy.kwargs = merge(antithetic_method.strategy.kwargs, (antithetic=true,))
    
    # Solve the problem
    solution = solve(prob, antithetic_method)
    
    # Return the ensemble solutions directly
    return solution.ensemble.solutions
end

function plot_path_pair(paths, dynamics, strategy, method_name; apply_exp=true)
    original_path = paths[1]
    antithetic_path = paths[length(paths)÷2 + 1]
    
    # Extract time points
    time_points = original_path.t
    
    # Check if paths contain vector states (like Heston) or scalar states (like some BS)
    is_vector_state = eltype(original_path.u) <: AbstractVector
    
    if is_vector_state
        # Handle paths with vector states (e.g., Heston model)
        if apply_exp
            original_prices = [exp(u[1]) for u in original_path.u]
            antithetic_prices = [exp(u[1]) for u in antithetic_path.u]
        else
            original_prices = [u[1] for u in original_path.u]
            antithetic_prices = [u[1] for u in antithetic_path.u]
        end
        
        # Extract variance components if they exist
        has_variance = length(original_path.u[1]) >= 2
        
        if has_variance
            original_vars = [u[2] for u in original_path.u]
            antithetic_vars = [u[2] for u in antithetic_path.u]
            var_corr = cor(original_vars, antithetic_vars)
        end
    else
        # Handle scalar state paths
        if apply_exp
            original_prices = exp.(original_path.u)
            antithetic_prices = exp.(antithetic_path.u)
        else
            original_prices = original_path.u
            antithetic_prices = antithetic_path.u
        end
        has_variance = false
    end
    
    # Create plot
    p = plot(
        time_points, original_prices,
        label="Original Path",
        linewidth=2,
        title="$method_name: $(apply_exp ? "Stock Price" : "Log Price") Paths",
        xlabel="Time (years)",
        ylabel=apply_exp ? "Stock Price" : "Log Price",
        legend=:topleft
    )
    
    plot!(
        p,
        time_points, antithetic_prices,
        label="Antithetic Path",
        linewidth=2,
        linestyle=:dash,
        color=:red
    )
    
    # Calculate correlation
    price_corr = cor(original_prices, antithetic_prices)
    
    # Add correlation annotation (only for price paths, ignoring variance)
    annotate!(p, [(0.5, minimum(original_prices), 
                text("Path correlation: $(round(price_corr, digits=4))", 10, :left))])
    
    return p
end

# Set up common parameters
reference_date = Date(2020, 1, 1)
expiry = reference_date + Year(1)
spot = 10.0
strike = 10.0
seed = 42

# Create payoff
payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

# Black-Scholes setup
rate_bs = 0.05
sigma_bs = 0.20
bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
bs_prob = PricingProblem(payoff, bs_market)

# Heston setup
rate_heston = 0.03
V0 = 0.04
κ = 2.0
θ = 0.04
σ = 0.3
ρ = -0.7
heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
heston_prob = PricingProblem(payoff, heston_market)

# Create Monte Carlo methods
num_paths = 10
steps_bs = 100
steps_heston = 100

# 1. Black-Scholes with Exact Simulation
bs_exact_strategy = BlackScholesExact(num_paths, steps=steps_bs)
bs_exact_method = MonteCarlo(LognormalDynamics(), bs_exact_strategy)
bs_exact_paths = analyze_antithetic_paths(bs_prob, bs_exact_method, seed=seed)

# 2. Black-Scholes with Euler-Maruyama
bs_em_strategy = EulerMaruyama(num_paths, steps=steps_bs)
bs_em_method = MonteCarlo(LognormalDynamics(), bs_em_strategy)
bs_em_paths = analyze_antithetic_paths(bs_prob, bs_em_method, seed=seed)

# 3. Heston with Euler-Maruyama
heston_em_strategy = EulerMaruyama(num_paths, steps=steps_heston)
heston_em_method = MonteCarlo(HestonDynamics(), heston_em_strategy)
heston_em_paths = analyze_antithetic_paths(heston_prob, heston_em_method, seed=seed)

# 4. Heston with Broadie-Kaya (for fewer paths due to computational complexity)
heston_bk_strategy = HestonBroadieKaya(num_paths, steps=100)
heston_bk_method = MonteCarlo(HestonDynamics(), heston_bk_strategy)
heston_bk_paths = analyze_antithetic_paths(heston_prob, heston_bk_method, seed=seed)

# Generate plots with exponential transformation (natural price scale)
p1 = plot_path_pair(bs_exact_paths, LognormalDynamics(), bs_exact_strategy, "Black-Scholes Exact", apply_exp=false)
p2 = plot_path_pair(bs_em_paths, LognormalDynamics(), bs_em_strategy, "Black-Scholes Euler-Maruyama", apply_exp=true)
p3 = plot_path_pair(heston_em_paths, HestonDynamics(), heston_em_strategy, "Heston Euler-Maruyama", apply_exp=false)
p4 = plot_path_pair(heston_bk_paths, HestonDynamics(), heston_bk_strategy, "Heston Broadie-Kaya", apply_exp=true)

# If you want to view paths in log scale, use apply_exp=false
# For example:
# p1_log = plot_path_pair(bs_exact_paths, LognormalDynamics(), bs_exact_strategy, "Black-Scholes Exact (Log Scale)", apply_exp=false)

# Plot them in a 2x2 grid
final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700))
display(final_plot)