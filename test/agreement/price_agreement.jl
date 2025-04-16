@testset "Price Agreement" begin
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

        @test isapprox(analytic_sol.price, crr_sol.price; atol = 1e-3)
    end

    @testset "Carr-Madan vs Black-Scholes analytical" begin
        # Define market inputs
        reference_date = Date(2020, 1, 1)
        interest_rate = 0.2
        spot = 100.0
        sigma = 0.4
        market_inputs = BlackScholesInputs(reference_date, interest_rate, spot, sigma)
    
        # Define payoff
        expiry = reference_date + Day(365)
        strike = 100.0
        payoff = VanillaOption(strike, expiry, European(), Call(), Spot())
    
        # Define pricing problem
        prob = PricingProblem(payoff, market_inputs)
    
        # Carr-Madan method
        boundary = 16
        α = 1.0
        carr_madan_method = CarrMadan(α, boundary, LognormalDynamics())
        carr_madan_solution = solve(prob, carr_madan_method)
    
        # Analytical Black-Scholes method
        bs_solution = solve(prob, BlackScholesAnalytic())
    
        @test isapprox(carr_madan_solution.price, bs_solution.price; atol = 1e-6)
    end
    
end