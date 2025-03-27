@testset "Binomial tree vs Black-Scholes analytic" begin
    # Define European put option on spot
    strike = 1.1
    expiry = Date(2021, 1, 1)
    euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

    # Market inputs
    reference_date = Date(2020, 1, 1)
    rate = 0.2
    spot = 1.0
    sigma = 0.4
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # Create pricing problem
    prob = PricingProblem(euro_payoff, market_inputs)

    # Solve using Black-Scholes analytic
    analytic_sol = solve(prob, BlackScholesAnalytic())

    # Solve using binomial tree (CRR)
    crr_method = CoxRossRubinsteinMethod(100)
    crr_sol = solve(prob, crr_method)

    @test isapprox(analytic_sol.price, crr_sol.price; atol=1e-3)
end

@testset "American binomial tree regression" begin
    # American call option on spot
    strike = 1.0
    expiry = Date(2021, 1, 1)
    american_payoff = VanillaOption(strike, expiry, Hedgehog2.American(), Call(), Spot())

    reference_date = Date(2020, 1, 1)
    rate = 0.2
    spot = 1.0
    sigma = 0.4
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    prob_spot = PricingProblem(american_payoff, market_inputs)
    crr_method = CoxRossRubinsteinMethod(80)
    crr_sol_spot = solve(prob_spot, crr_method)

    @test isapprox(crr_sol_spot.price, 0.25735029973418333, atol=1e-8)

    # American put option on forward
    american_payoff_fwd = VanillaOption(strike, expiry, Hedgehog2.American(), Put(), Forward())
    prob_fwd = PricingProblem(american_payoff_fwd, market_inputs)
    crr_sol_fwd = solve(prob_fwd, crr_method)

    @test isapprox(crr_sol_fwd.price, 0.07410498956582845, atol=1e-8)
end
