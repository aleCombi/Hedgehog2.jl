@testset "Binomial Tree" begin
    @testset "American binomial tree regression" begin
        # American call option on spot
        strike = 1.0
        reference_date = Date(2020, 1, 1)
        expiry = reference_date + Day(365)
        american_payoff = VanillaOption(strike, expiry, Hedgehog.American(), Call(), Spot())

        rate = 0.2
        spot = 1.0
        sigma = 0.4
        market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

        prob_spot = PricingProblem(american_payoff, market_inputs)
        crr_method = CoxRossRubinsteinMethod(80)
        crr_sol_spot = Hedgehog.solve(prob_spot, crr_method)

        @test isapprox(crr_sol_spot.price, 0.25225758542934945, atol = 1e-8)

        # American put option on forward
        american_payoff_fwd =
            VanillaOption(strike, expiry, Hedgehog.American(), Put(), Forward())
        prob_fwd = PricingProblem(american_payoff_fwd, market_inputs)
        crr_sol_fwd = Hedgehog.solve(prob_fwd, crr_method)

        @test isapprox(crr_sol_fwd.price, 0.07409148128021317, atol = 1e-8)
    end
end