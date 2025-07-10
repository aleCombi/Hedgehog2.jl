using Test
using Hedgehog
using Dates
using Random
using Statistics
using Printf

@testset "Heston Model Monte Carlo Test / Variance reduction test" begin
    # --------------------------
    # Setup common test parameters
    # --------------------------
    # Heston model parameters
    spot = 100.0
    strike = 100.0
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    r = 0.03             # Risk-free rate
    V0 = 0.04            # Initial variance
    κ = 2.0              # Mean reversion speed
    θ = 0.04             # Long-term variance
    σ = 0.3              # Vol-of-vol
    ρ = -0.7             # Correlation

    # Print parameter configuration for debugging
    println("Test Parameters:")
    println("  Spot: $spot")
    println("  Strike: $strike")
    println("  Rate: $r")
    println("  Initial variance (V0): $V0")
    println("  Mean reversion (κ): $κ")
    println("  Long-term variance (θ): $θ")
    println("  Vol-of-vol (σ): $σ")
    println("  Correlation (ρ): $ρ")
    println("  Time to maturity: 1 year")
    println("  Option type: European Call")

    # Define payoff and market inputs
    payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
    market_inputs = HestonInputs(reference_date, r, spot, V0, κ, θ, σ, ρ)
    prob = PricingProblem(payoff, market_inputs)

    # Define test parameters - increase trajectories for more reliable results
    trajectories = 5000
    steps = 100

    # Reference price using Carr-Madan (Fourier) method
    carr_madan_method = CarrMadan(1.0, 32.0, HestonDynamics())
    carr_madan_sol = Hedgehog.solve(prob, carr_madan_method)
    reference_price = carr_madan_sol.price

    # Print reference price
    println("Reference price (Carr-Madan): $reference_price")

    # Test scenarios for Euler-Maruyama
    scenarios = [
        (
            "Euler-Maruyama without antithetic",
            EulerMaruyama(),
            HestonDynamics(),
            SimulationConfig(trajectories, seeds = nothing, variance_reduction=Hedgehog.NoVarianceReduction())
        ),
        (
            "Euler-Maruyama with antithetic",
            EulerMaruyama(),
            HestonDynamics(),
            SimulationConfig(trajectories ÷ 2, seeds = nothing, variance_reduction=Hedgehog.Antithetic())
        ),
    ]

    results = Dict()

    # Run all scenarios
    for (scenario_name, strategy, dynamics, config) in scenarios
        # Print current scenario
        println("Running scenario: ", scenario_name)

        # Execute multiple trials to measure variance
        prices = Float64[]
        for trial = 1:5
            # Create a new RNG with a trial-specific seed
            trial_rng = MersenneTwister(42 + trial)

            # Generate random seeds for each path
            trial_seeds = rand(trial_rng, 1:10^9, trajectories)

            # Create modified strategy with trial seeds
            modified_config = @set config.seeds = trial_seeds

            # Create Monte Carlo method
            mc_method = MonteCarlo(dynamics, strategy, modified_config)

            # Solve with current seed
            sol = Hedgehog.solve(prob, mc_method)
            push!(prices, sol.price)

            # Print diagnostic info
            println("  Trial $trial: Price = $(sol.price)")
        end

        # Calculate statistics
        mean_price = mean(prices)
        price_variance = var(prices)
        error = abs(mean_price - reference_price)
        rel_error = error / reference_price

        # Store results
        results[scenario_name] = (
            mean_price = mean_price,
            reference_price = reference_price,
            error = error,
            rel_error = rel_error,
            variance = price_variance,
        )

        # Test the result with a generous tolerance (Heston MC typically needs more paths)
        @test isapprox(mean_price, reference_price, rtol = 0.05)
    end

    # Compare variance reduction
    if length(scenarios) >= 2
        std_var = results["Euler-Maruyama without antithetic"].variance
        anti_var = results["Euler-Maruyama with antithetic"].variance
        var_reduction = std_var / anti_var

        @info "Variance reduction (Euler-Maruyama): $(var_reduction)×"
        @test var_reduction > 1.0  # Antithetic should reduce variance
    end

    # Print summary
    println("\n=== Heston Model Monte Carlo Test Results ===")
    println("Reference price (Carr-Madan): $reference_price")
    println("Scenario                    | Price      | Rel Error | Variance")
    println("----------------------------|------------|-----------|----------")

    for (name, res) in results
        @printf(
            "%-28s | %.6f | %.6f | %.2e\n",
            name,
            res.mean_price,
            res.rel_error,
            res.variance
        )
    end
end
