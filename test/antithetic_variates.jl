using Hedgehog2
using Dates
using Random
using Statistics
using Test
using Printf

function analyze_path_correlation(paths, n_original)
    # Calculate correlations between original and antithetic paths
    correlations = Float64[]
    
    for i in 1:n_original
        original_path = paths[i]
        antithetic_path = paths[i + n_original]
        
        # Get price values at each time step
        if eltype(original_path.u) <: AbstractVector
            original_prices = [u[1] for u in original_path.u]
            antithetic_prices = [u[1] for u in antithetic_path.u]
        else
            original_prices = original_path.u
            antithetic_prices = antithetic_path.u
        end
        
        # Calculate correlation
        path_correlation = cor(original_prices, antithetic_prices)
        push!(correlations, path_correlation)
    end
    
    return correlations
end

function calculate_return_correlation(paths, n_original)
    # Calculate correlations between log returns of original and antithetic paths
    return_correlations = Float64[]
    
    for i in 1:n_original
        original_path = paths[i]
        antithetic_path = paths[i + n_original]
        
        # Get price values at each time step
        if eltype(original_path.u) <: AbstractVector
            original_prices = [u[1] for u in original_path.u]
            antithetic_prices = [u[1] for u in antithetic_path.u]
        else
            original_prices = original_path.u
            antithetic_prices = antithetic_path.u
        end
        
        # Calculate log returns
        original_returns = diff(log.(original_prices))
        antithetic_returns = diff(log.(antithetic_prices))
        
        # Skip if either array has issues
        if any(isnan.(original_returns)) || any(isnan.(antithetic_returns)) ||
           any(isinf.(original_returns)) || any(isinf.(antithetic_returns))
            continue
        end
        
        # Calculate correlation of returns
        return_corr = cor(original_returns, antithetic_returns)
        push!(return_correlations, return_corr)
    end
    
    return return_correlations
end

function calculate_terminal_product(paths, n_original, reference_date, expiry, r)
    # Calculate the product of terminal values for pairs of original and antithetic paths
    terminal_products = Float64[]
    
    for i in 1:n_original
        original_path = paths[i]
        antithetic_path = paths[i + n_original]
        
        # Get terminal values
        if eltype(original_path.u) <: AbstractVector
            original_terminal = original_path.u[end][1]
            antithetic_terminal = antithetic_path.u[end][1]
        else
            original_terminal = original_path.u[end]
            antithetic_terminal = antithetic_path.u[end]
        end
        
        push!(terminal_products, original_terminal * antithetic_terminal)
    end
    
    return terminal_products
end

function variance_reduction_test(method_standard, method_antithetic, prob, reference_price, n_trials=20)
    # Perform multiple trials to estimate variance reduction
    prices_std = Float64[]
    prices_anti = Float64[]
    
    for i in 1:n_trials
        # Set random seed for reproducibility while allowing multiple trials
        Random.seed!(42 + i)
        
        # Standard Monte Carlo
        sol_std = solve(prob, method_standard)
        push!(prices_std, sol_std.price)
        
        # Antithetic Monte Carlo
        sol_anti = solve(prob, method_antithetic)
        push!(prices_anti, sol_anti.price)
    end
    
    # Calculate statistics
    mean_std = mean(prices_std)
    mean_anti = mean(prices_anti)
    
    var_std = var(prices_std)
    var_anti = var(prices_anti)
    
    bias_std = mean_std - reference_price
    bias_anti = mean_anti - reference_price
    
    rmse_std = sqrt(bias_std^2 + var_std)
    rmse_anti = sqrt(bias_anti^2 + var_anti)
    
    return (
        mean_std = mean_std,
        mean_anti = mean_anti,
        var_std = var_std,
        var_anti = var_anti,
        bias_std = bias_std,
        bias_anti = bias_anti,
        rmse_std = rmse_std,
        rmse_anti = rmse_anti,
        var_reduction = var_std / var_anti,
        rmse_reduction = rmse_std / rmse_anti
    )
end

@testset "Enhanced Antithetic Path Correlation Tests" begin
    # Common parameters
    Random.seed!(42)
    trajectories = 50  # Reduced for Broadie-Kaya which is computationally intensive
    steps = 100
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    spot = 100.0
    strike = 100.0

    # Create European call option payoff
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
    
    @testset "Black-Scholes Model" begin
        # BS parameters
        rate_bs = 0.05
        sigma_bs = 0.20
        
        # Create BS market inputs
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        bs_prob = PricingProblem(payoff, bs_market)
        
        # Create Monte Carlo with antithetic sampling for BS
        bs_seeds = rand(1:10^9, trajectories)
        bs_strategy = BlackScholesExact(trajectories ÷ 2, steps, seeds=bs_seeds[1:trajectories÷2], antithetic=true)
        bs_method = MonteCarlo(LognormalDynamics(), bs_strategy)
        
        # Solve and get paths
        bs_solution = solve(bs_prob, bs_method)
        bs_paths = bs_solution.ensemble.solutions
        
        # Analyze correlations
        bs_correlations = analyze_path_correlation(bs_paths, trajectories ÷ 2)
        bs_return_correlations = calculate_return_correlation(bs_paths, trajectories ÷ 2)
        
        # Report statistics for diagnostic purposes
        println("Black-Scholes path correlations:")
        println("  Mean: ", mean(bs_correlations))
        println("  Min: ", minimum(bs_correlations))
        println("  Max: ", maximum(bs_correlations))
        
        println("\nBlack-Scholes return correlations:")
        println("  Mean: ", mean(bs_return_correlations))
        println("  Min: ", minimum(bs_return_correlations))
        println("  Max: ", maximum(bs_return_correlations))
        
        # Test that price paths show strong negative correlation
        @test mean(bs_correlations) < -0.7
        
        # Test that log returns show near-perfect negative correlation
        @test mean(bs_return_correlations) < -0.95
        
        # Test product of terminal values is consistent with theoretical value
        terminal_products = calculate_terminal_product(bs_paths, trajectories ÷ 2, reference_date, expiry, rate_bs)
        mean_product = mean(terminal_products)
        theoretical_product = spot^2 * exp(2 * rate_bs * yearfrac(reference_date, expiry))
        
        # The product should be roughly close to the theoretical value
        # Using a large tolerance due to randomness
        @test isapprox(mean_product, theoretical_product, rtol=0.2)
        
        # Run variance reduction test using a low number of trajectories for speed
        # Analytic reference
        analytic_method = BlackScholesAnalytic()
        reference_price = solve(bs_prob, analytic_method).price
        
        # Standard vs Antithetic
        std_strategy = BlackScholesExact(trajectories, steps)
        anti_strategy = BlackScholesExact(trajectories ÷ 2, steps, antithetic=true)
        
        std_method = MonteCarlo(LognormalDynamics(), std_strategy)
        anti_method = MonteCarlo(LognormalDynamics(), anti_strategy)
        
        var_results = variance_reduction_test(std_method, anti_method, bs_prob, reference_price, 10)
        
        println("\nBlack-Scholes variance reduction:")
        println("  Standard variance: ", var_results.var_std)
        println("  Antithetic variance: ", var_results.var_anti)
        println("  Variance reduction factor: ", var_results.var_reduction)
        println("  RMSE reduction factor: ", var_results.rmse_reduction)
        
        # Test for variance reduction
        @test var_results.var_reduction > 1.0
    end
    
    @testset "Heston Model with Euler-Maruyama" begin
        # Heston parameters
        rate_heston = 0.03
        V0 = 0.04
        κ = 2.0
        θ = 0.04
        σ = 0.3
        ρ = -0.7
        
        # Create Heston market inputs
        heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
        heston_prob = PricingProblem(payoff, heston_market)
        
        # Create Monte Carlo with antithetic sampling for Heston
        heston_seeds = rand(1:10^9, trajectories)
        heston_strategy = EulerMaruyama(trajectories ÷ 2, steps, seeds=heston_seeds[1:trajectories÷2], antithetic=true)
        heston_method = MonteCarlo(HestonDynamics(), heston_strategy)
        
        # Solve and get paths
        heston_solution = solve(heston_prob, heston_method)
        heston_paths = heston_solution.ensemble.solutions
        
        # Analyze correlations
        heston_correlations = analyze_path_correlation(heston_paths, trajectories ÷ 2)
        heston_return_correlations = calculate_return_correlation(heston_paths, trajectories ÷ 2)
        
        # Report statistics for diagnostic purposes
        println("\nHeston (Euler-Maruyama) path correlations:")
        println("  Mean: ", mean(heston_correlations))
        println("  Min: ", minimum(heston_correlations))
        println("  Max: ", maximum(heston_correlations))
        
        println("\nHeston (Euler-Maruyama) return correlations:")
        println("  Mean: ", mean(heston_return_correlations))
        println("  Min: ", minimum(heston_return_correlations))
        println("  Max: ", maximum(heston_return_correlations))
        
        # For Heston, the correlation might not be as strong due to stochastic volatility
        @test mean(heston_correlations) < -0.5
        
        # For return correlations, we still expect strong negative correlation
        @test mean(heston_return_correlations) < -0.7
        
        # Test percentage of negative correlations
        # In Heston, we expect most but not necessarily all paths to show negative correlation
        percent_negative = sum(heston_correlations .< 0) / length(heston_correlations) * 100
        println("  Percentage of negative correlations: $(percent_negative)%")
        @test percent_negative > 90  # At least 90% should be negative
        
        # Run variance reduction test
        # Reference price from Carr-Madan
        carr_madan_method = CarrMadan(1.0, 32.0, HestonDynamics())
        reference_price = solve(heston_prob, carr_madan_method).price
        
        # Standard vs Antithetic
        std_strategy = EulerMaruyama(trajectories, steps)
        anti_strategy = EulerMaruyama(trajectories ÷ 2, steps, antithetic=true)
        
        std_method = MonteCarlo(HestonDynamics(), std_strategy)
        anti_method = MonteCarlo(HestonDynamics(), anti_strategy)
        
        var_results = variance_reduction_test(std_method, anti_method, heston_prob, reference_price, 5)
        
        println("\nHeston (Euler-Maruyama) variance reduction:")
        println("  Standard variance: ", var_results.var_std)
        println("  Antithetic variance: ", var_results.var_anti)
        println("  Variance reduction factor: ", var_results.var_reduction)
        println("  RMSE reduction factor: ", var_results.rmse_reduction)
        
        # Test for variance reduction
        @test var_results.var_reduction > 1.0
    end
    
    @testset "Heston Model with Broadie-Kaya" begin
        # Heston parameters - using slightly different parameters for Broadie-Kaya
        rate_heston = 0.03
        V0 = 0.04
        κ = 2.0
        θ = 0.04
        σ = 0.3
        ρ = -0.7
        
        # Create Heston market inputs
        heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
        heston_prob = PricingProblem(payoff, heston_market)
        
        # Create Broadie-Kaya strategy with antithetic sampling
        # Using even fewer trajectories and steps since this is computationally expensive
        fewer_trajs = 20
        bk_seeds = rand(1:10^9, fewer_trajs)
        bk_strategy = HestonBroadieKaya(fewer_trajs ÷ 2, steps=2, seeds=bk_seeds[1:fewer_trajs÷2], antithetic=true)
        bk_method = MonteCarlo(HestonDynamics(), bk_strategy)
        
        # Solve and get paths
        bk_solution = solve(heston_prob, bk_method)
        bk_paths = bk_solution.ensemble.solutions
        
        # Analyze correlations - we expect weaker correlation due to the exact sampling nature of BK
        bk_correlations = analyze_path_correlation(bk_paths, fewer_trajs ÷ 2)
        
        # Report statistics for diagnostic purposes
        println("\nHeston (Broadie-Kaya) path correlations:")
        println("  Mean: ", mean(bk_correlations))
        println("  Min: ", minimum(bk_correlations))
        println("  Max: ", maximum(bk_correlations))
        
        # With Broadie-Kaya, the correlation might be weaker than Euler-Maruyama
        # since it's an exact simulation method
        # Still, antithetic sampling should produce negative correlation
        @test mean(bk_correlations) < 0
        
        # Test percentage of negative correlations
        # We expect a majority of paths to show negative correlation
        percent_negative = sum(bk_correlations .< 0) / length(bk_correlations) * 100
        println("  Percentage of negative correlations: $(percent_negative)%")
        @test percent_negative > 50  # At least half should be negative
        
        # Run a simple 1-trial test to check variance reduction
        # Reference price from Carr-Madan
        reference_price = solve(heston_prob, CarrMadan(1.0, 32.0, HestonDynamics())).price
        
        # Standard BK vs Antithetic BK
        std_bk_strategy = HestonBroadieKaya(fewer_trajs, steps=2)
        anti_bk_strategy = HestonBroadieKaya(fewer_trajs ÷ 2, steps=2, antithetic=true)
        
        std_bk_method = MonteCarlo(HestonDynamics(), std_bk_strategy)
        anti_bk_method = MonteCarlo(HestonDynamics(), anti_bk_strategy)
        
        # Due to computational cost, run fewer trials for BK
        var_results = variance_reduction_test(std_bk_method, anti_bk_method, heston_prob, reference_price, 3)
        
        println("\nHeston (Broadie-Kaya) variance reduction:")
        println("  Standard variance: ", var_results.var_std)
        println("  Antithetic variance: ", var_results.var_anti)
        println("  Variance reduction factor: ", var_results.var_reduction)
        println("  RMSE reduction factor: ", var_results.rmse_reduction)
        
        # We can't guarantee variance reduction with BK due to low trial count,
        # but we can check for reasonable behavior
        println("  Antithetic price: ", var_results.mean_anti)
        println("  Reference price: ", reference_price)
        println("  Relative error: ", abs(var_results.mean_anti - reference_price) / reference_price)
        
        # Check if the mean price is reasonable (within 10% of reference)
        @test isapprox(var_results.mean_anti, reference_price, rtol=0.1)
    end
    
    @testset "Correlation Comparison" begin
        # Compare correlations between BS and Heston to verify both are working
        # but may have different characteristics
        
        # Repeat setup briefly
        rate_bs = 0.05
        sigma_bs = 0.20
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        bs_prob = PricingProblem(payoff, bs_market)
        bs_seeds = rand(1:10^9, trajectories)
        bs_strategy = BlackScholesExact(trajectories ÷ 2, steps, seeds=bs_seeds[1:trajectories÷2], antithetic=true)
        bs_method = MonteCarlo(LognormalDynamics(), bs_strategy)
        bs_solution = solve(bs_prob, bs_method)
        bs_paths = bs_solution.ensemble.solutions
        
        rate_heston = 0.03
        V0 = 0.04
        κ = 2.0
        θ = 0.04
        σ = 0.3
        ρ = -0.7
        heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
        heston_prob = PricingProblem(payoff, heston_market)
        heston_seeds = rand(1:10^9, trajectories)
        heston_strategy = EulerMaruyama(trajectories ÷ 2, steps, seeds=heston_seeds[1:trajectories÷2], antithetic=true)
        heston_method = MonteCarlo(HestonDynamics(), heston_strategy)
        heston_solution = solve(heston_prob, heston_method)
        heston_paths = heston_solution.ensemble.solutions
        
        # Get path correlations
        bs_correlations = analyze_path_correlation(bs_paths, trajectories ÷ 2)
        heston_correlations = analyze_path_correlation(heston_paths, trajectories ÷ 2)
        
        # Black-Scholes should typically have stronger negative correlation
        # due to simpler model dynamics
        println("\nCorelation comparison:")
        println("  BS mean correlation: ", mean(bs_correlations))
        println("  Heston mean correlation: ", mean(heston_correlations))
        @test mean(bs_correlations) < mean(heston_correlations)
    end
end