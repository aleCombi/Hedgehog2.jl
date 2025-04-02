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

@testset "Antithetic Path Correlation Tests" begin
    # Common parameters
    Random.seed!(42)
    trajectories = 100
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
        bs_original = bs_paths[1]
        bs_antithetic = bs_paths[trajectories ÷ 2 + 1]
        
        if eltype(bs_original.u) <: AbstractVector
            bs_orig_terminal = bs_original.u[end][1]
            bs_anti_terminal = bs_antithetic.u[end][1]
        else
            bs_orig_terminal = bs_original.u[end]
            bs_anti_terminal = bs_antithetic.u[end]
        end
        
        terminal_product = bs_orig_terminal * bs_anti_terminal
        theoretical_product = spot^2 * exp(2 * rate_bs * yearfrac(reference_date, expiry))
        
        # The product should be roughly close to the theoretical value
        # Using a large tolerance due to randomness
        @test isapprox(terminal_product, theoretical_product, rtol=0.2)
    end
    
    @testset "Heston Model" begin
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
        println("\nHeston path correlations:")
        println("  Mean: ", mean(heston_correlations))
        println("  Min: ", minimum(heston_correlations))
        println("  Max: ", maximum(heston_correlations))
        
        println("\nHeston return correlations:")
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
        @test mean(bs_correlations) < mean(heston_correlations)
    end
end