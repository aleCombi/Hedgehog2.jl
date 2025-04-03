using Test
using Hedgehog2
using Dates
using Random
using Statistics

"""
    analyze_antithetic_variance_reduction(prob::PricingProblem, mc_method::MonteCarlo; 
                                         num_paths=1000, seed=nothing)

Analyzes the variance reduction achieved by antithetic variates for a given pricing problem.

Returns a named tuple with:
- payoff_correlation: Correlation between original and antithetic payoffs
- standard_variance: Variance of standard Monte Carlo estimator
- antithetic_variance: Variance of antithetic variates estimator
- var_reduction_ratio: Ratio of standard variance to antithetic variance
"""
function analyze_antithetic_variance_reduction(prob::PricingProblem, mc_method::MonteCarlo; 
                                               num_paths=1000, seed=nothing)
    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    payoff = prob.payoff
    discount_factor = df(prob.market.rate, prob.payoff.expiry)
    
    # Generate seeds for paths
    path_seeds = rand(1:10^9, num_paths ÷ 2)
    
    # Create the antithetic method - correctly updating the kwargs field
    antithetic_method = @set mc_method.strategy.seeds = path_seeds
    # Update the kwargs field to include antithetic=true
    antithetic_method = @set antithetic_method.strategy.kwargs = merge(antithetic_method.strategy.kwargs, (antithetic=true,))
    
    # Solve the problem
    solution = solve(prob, antithetic_method)
    
    # Get all paths 
    all_paths = solution.ensemble.solutions
    half_paths = num_paths ÷ 2
    
    # Extract terminal values
    original_terminals = [Hedgehog2.get_terminal_value(all_paths[i], mc_method.dynamics, mc_method.strategy) 
                           for i in 1:half_paths]
    antithetic_terminals = [Hedgehog2.get_terminal_value(all_paths[i+half_paths], mc_method.dynamics, mc_method.strategy)  
                           for i in 1:half_paths]
    
    # Calculate payoffs
    original_payoffs = payoff.(original_terminals)
    antithetic_payoffs = payoff.(antithetic_terminals)
    
    # Calculate correlation
    payoff_correlation = cor(original_payoffs, antithetic_payoffs)
    
    # Calculate standard MC estimator variance (using just original paths)
    standard_estimator = discount_factor * original_payoffs
    standard_variance = var(standard_estimator)
    
    # Calculate antithetic MC estimator variance (using averages of path pairs)
    antithetic_pairs = [(original_payoffs[i] + antithetic_payoffs[i]) / 2 for i in 1:half_paths]
    antithetic_estimator = discount_factor * antithetic_pairs
    antithetic_variance = var(antithetic_estimator)
    
    # Calculate variance reduction ratio
    var_reduction_ratio = standard_variance / antithetic_variance
    
    return (
        payoff_correlation = payoff_correlation,
        standard_variance = standard_variance,
        antithetic_variance = antithetic_variance,
        var_reduction_ratio = var_reduction_ratio
    )
end

@testset "Antithetic Variates Variance Reduction" begin
    # Set up common test parameters
    seed = 42
    num_paths = 1000  # Reduced for faster test execution
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    spot = 100.0
    strike = 100.0
    
    # Create European call option payoff
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
    
    # --- Black-Scholes with Exact simulation ---
    @testset "Black-Scholes Exact Simulation" begin
        rate_bs = 0.05
        sigma_bs = 0.20
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        bs_prob = PricingProblem(payoff, bs_market)
        bs_exact_strategy = BlackScholesExact(num_paths)
        bs_exact_method = MonteCarlo(LognormalDynamics(), bs_exact_strategy)
        
        results = analyze_antithetic_variance_reduction(bs_prob, bs_exact_method; 
                                                      num_paths=num_paths, seed=seed)
        
        # Test that payoff correlation is negative (required for variance reduction)
        @test results.payoff_correlation < 0
        
        # Test that antithetic variance is less than standard variance
        @test results.antithetic_variance < results.standard_variance
        
        # Test that variance reduction ratio is greater than 1.0
        @test results.var_reduction_ratio > 1.0
        
        # Check minimum variance reduction factor (should be at least 2.0 for BS exact)
        @test results.var_reduction_ratio > 2.0
    end
    
    # --- Black-Scholes with Euler-Maruyama ---
    @testset "Black-Scholes Euler-Maruyama" begin
        rate_bs = 0.05
        sigma_bs = 0.20
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        bs_prob = PricingProblem(payoff, bs_market)
        bs_em_strategy = EulerMaruyama(num_paths, steps=100)
        bs_em_method = MonteCarlo(LognormalDynamics(), bs_em_strategy)
        
        results = analyze_antithetic_variance_reduction(bs_prob, bs_em_method; 
                                                      num_paths=num_paths, seed=seed)
        
        # EM might not have negative correlation due to discretization
        # but should still show variance reduction
        
        # Test that antithetic variance is less than standard variance
        @test results.antithetic_variance < results.standard_variance
        
        # Test that variance reduction ratio is greater than 1.0
        @test results.var_reduction_ratio > 1.0
    end
    
    # --- Heston with Euler-Maruyama ---
    @testset "Heston Euler-Maruyama" begin
        rate_heston = 0.03
        V0 = 0.04
        κ = 2.0
        θ = 0.04
        σ = 0.3
        ρ = -0.7
        
        heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
        heston_prob = PricingProblem(payoff, heston_market)
        
        heston_em_strategy = EulerMaruyama(num_paths, steps=100)
        heston_em_method = MonteCarlo(HestonDynamics(), heston_em_strategy)
        
        results = analyze_antithetic_variance_reduction(heston_prob, heston_em_method; 
                                                       num_paths=num_paths, seed=seed)
        
        # For Heston, correlation might be weak or even positive
        # but we still expect variance reduction
        
        # Test that antithetic variance is less than standard variance
        @test results.antithetic_variance < results.standard_variance
        
        # Test that variance reduction ratio is greater than 1.0
        @test results.var_reduction_ratio > 1.0
    end
    
    # --- Heston with Broadie-Kaya ---
    @testset "Heston Broadie-Kaya" begin
        rate_heston = 0.03
        V0 = 0.04
        κ = 2.0
        θ = 0.04
        σ = 0.3
        ρ = -0.7
        
        heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
        heston_prob = PricingProblem(payoff, heston_market)
        
        # Use fewer paths for BK due to computational intensity
        bk_paths = min(num_paths, 200)  # Further reduced for test
        heston_bk_strategy = HestonBroadieKaya(bk_paths, steps=20)
        heston_bk_method = MonteCarlo(HestonDynamics(), heston_bk_strategy)
        
        results = analyze_antithetic_variance_reduction(heston_prob, heston_bk_method; 
                                                       num_paths=bk_paths, seed=seed)
        
        # Test that antithetic variance is less than standard variance
        @test results.antithetic_variance < results.standard_variance
        
        # Test that variance reduction ratio is greater than 1.0
        @test results.var_reduction_ratio > 1.0
    end
    
    # --- Test on Put option (verify it works for other payoffs) ---
    @testset "Put Option Antithetic Reduction" begin
        rate_bs = 0.05
        sigma_bs = 0.20
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        
        # Create Put option payoff
        put_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())
        put_prob = PricingProblem(put_payoff, bs_market)
        
        bs_exact_strategy = BlackScholesExact(num_paths)
        bs_exact_method = MonteCarlo(LognormalDynamics(), bs_exact_strategy)
        
        results = analyze_antithetic_variance_reduction(put_prob, bs_exact_method; 
                                                       num_paths=num_paths, seed=seed)
        
        # Test that variance reduction works for put options too
        @test results.var_reduction_ratio > 1.0
    end
    
    # --- Comparison Tests ---
    @testset "Relative Variance Reduction Efficiency" begin
        rate_bs = 0.05
        sigma_bs = 0.20
        bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
        bs_prob = PricingProblem(payoff, bs_market)
        
        # Run with exactly the same settings for fair comparison
        paths_comp = 400  # Smaller for test speed
        seed_comp = 123   # Different seed
        
        # Exact simulation
        bs_exact_strategy = BlackScholesExact(paths_comp)
        bs_exact_method = MonteCarlo(LognormalDynamics(), bs_exact_strategy)
        
        # Euler-Maruyama
        bs_em_strategy = EulerMaruyama(paths_comp, steps=100)
        bs_em_method = MonteCarlo(LognormalDynamics(), bs_em_strategy)
        
        # Compare results
        exact_results = analyze_antithetic_variance_reduction(bs_prob, bs_exact_method; 
                                                           num_paths=paths_comp, seed=seed_comp)
        
        em_results = analyze_antithetic_variance_reduction(bs_prob, bs_em_method; 
                                                        num_paths=paths_comp, seed=seed_comp)
        
        # Exact simulation should provide better variance reduction than EM
        # for Black-Scholes
        @test exact_results.var_reduction_ratio > em_results.var_reduction_ratio
    end
end