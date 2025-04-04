using Hedgehog2
using Dates
using Random
using Statistics
using Printf

function analyze_antithetic_variance_reduction(
    prob::PricingProblem,
    mc_method::MonteCarlo;
    num_paths = 1000,
    seed = nothing,
)
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
    antithetic_method = @set antithetic_method.strategy.kwargs =
        merge(antithetic_method.strategy.kwargs, (antithetic = true,))

    # Solve the problem
    solution = solve(prob, antithetic_method)

    # Get all paths 
    all_paths = solution.ensemble.solutions
    half_paths = num_paths ÷ 2

    # Extract terminal values
    original_terminals = [
        Hedgehog2.get_terminal_value(all_paths[i], mc_method.dynamics, mc_method.strategy) for i = 1:half_paths
    ]
    antithetic_terminals = [
        Hedgehog2.get_terminal_value(
            all_paths[i+half_paths],
            mc_method.dynamics,
            mc_method.strategy,
        ) for i = 1:half_paths
    ]

    # Calculate payoffs
    original_payoffs = payoff.(original_terminals)
    antithetic_payoffs = payoff.(antithetic_terminals)

    # Calculate correlation
    payoff_correlation = cor(original_payoffs, antithetic_payoffs)

    # Calculate standard MC estimator variance (using just original paths)
    standard_estimator = discount_factor * original_payoffs
    standard_variance = var(standard_estimator)

    # Calculate antithetic MC estimator variance (using averages of path pairs)
    antithetic_pairs =
        [(original_payoffs[i] + antithetic_payoffs[i]) / 2 for i = 1:half_paths]
    antithetic_estimator = discount_factor * antithetic_pairs
    antithetic_variance = var(antithetic_estimator)

    # Calculate variance reduction ratio
    var_reduction_ratio = standard_variance / antithetic_variance

    return (
        payoff_correlation = payoff_correlation,
        standard_variance = standard_variance,
        antithetic_variance = antithetic_variance,
        var_reduction_ratio = var_reduction_ratio,
    )
end

# Main script
function run_variance_reduction_tests(num_paths = 1000, seed = 42)
    println("=== Antithetic Variates Variance Reduction Analysis ===")
    println("Number of paths: $num_paths")
    println("Random seed: $seed")

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Common parameters
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    spot = 100.0
    strike = 100.0

    # Create payoff (ATM European call)
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())

    # Test configurations
    println("\n1. Black-Scholes Model with Exact Simulation")
    rate_bs = 0.05
    sigma_bs = 0.20
    bs_market = BlackScholesInputs(reference_date, rate_bs, spot, sigma_bs)
    bs_prob = PricingProblem(payoff, bs_market)
    bs_exact_strategy = BlackScholesExact(num_paths)
    bs_exact_method = MonteCarlo(LognormalDynamics(), bs_exact_strategy)

    bs_exact_results = analyze_antithetic_variance_reduction(
        bs_prob,
        bs_exact_method;
        num_paths = num_paths,
        seed = seed,
    )

    print_results(bs_exact_results)

    println("\n2. Black-Scholes Model with Euler-Maruyama")
    bs_em_strategy = EulerMaruyama(num_paths, steps = 100)
    bs_em_method = MonteCarlo(LognormalDynamics(), bs_em_strategy)

    bs_em_results = analyze_antithetic_variance_reduction(
        bs_prob,
        bs_em_method;
        num_paths = num_paths,
        seed = seed,
    )

    print_results(bs_em_results)

    println("\n3. Heston Model with Euler-Maruyama")
    # Heston parameters
    rate_heston = 0.03
    V0 = 0.04
    κ = 2.0
    θ = 0.04
    σ = 0.3
    ρ = -0.7

    heston_market = HestonInputs(reference_date, rate_heston, spot, V0, κ, θ, σ, ρ)
    heston_prob = PricingProblem(payoff, heston_market)

    heston_em_strategy = EulerMaruyama(num_paths, steps = 100)
    heston_em_method = MonteCarlo(HestonDynamics(), heston_em_strategy)

    heston_em_results = analyze_antithetic_variance_reduction(
        heston_prob,
        heston_em_method;
        num_paths = num_paths,
        seed = seed,
    )

    print_results(heston_em_results)

    println("\n4. Heston Model with Broadie-Kaya (Exact)")
    # Using fewer paths for Broadie-Kaya due to computational intensity
    bk_paths = min(num_paths, 500)
    heston_bk_strategy = HestonBroadieKaya(bk_paths, steps = 20)
    heston_bk_method = MonteCarlo(HestonDynamics(), heston_bk_strategy)

    heston_bk_results = analyze_antithetic_variance_reduction(
        heston_prob,
        heston_bk_method;
        num_paths = bk_paths,
        seed = seed,
    )

    print_results(heston_bk_results)

    println("\n=== Summary of Variance Reduction Factors ===")
    println(
        "1. Black-Scholes (Exact):      $(round(bs_exact_results.var_reduction_ratio, digits=2))×",
    )
    println(
        "2. Black-Scholes (Euler-M):    $(round(bs_em_results.var_reduction_ratio, digits=2))×",
    )
    println(
        "3. Heston (Euler-M):           $(round(heston_em_results.var_reduction_ratio, digits=2))×",
    )
    println(
        "4. Heston (Broadie-Kaya):      $(round(heston_bk_results.var_reduction_ratio, digits=2))×",
    )
end

function print_results(results)
    println("Payoff correlation:       $(round(results.payoff_correlation, digits=4))")
    println("Standard MC variance:     $(round(results.standard_variance, digits=8))")
    println("Antithetic MC variance:   $(round(results.antithetic_variance, digits=8))")
    println("Variance reduction ratio: $(round(results.var_reduction_ratio, digits=2))×")
end

# Execute test
run_variance_reduction_tests(2000, 42)
