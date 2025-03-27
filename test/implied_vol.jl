using Revise, Hedgehog2, Interpolations, Dates, Test

@testset "Implied Vol Calibration Consistency" begin
    reference_date = Date(2020, 1, 1)
    expiry_date = reference_date + Day(365)
    r = 0.02
    spot = 100.0
    true_vol = 0.65
    strike = 80.0

    # Step 1: Generate market price from known vol
    market_inputs = BlackScholesInputs(reference_date, r, spot, true_vol)
    payoff = VanillaOption(strike, expiry_date, European(), Call(), Spot())
    pricing_problem = PricingProblem(payoff, market_inputs)
    price = solve(pricing_problem, BlackScholesAnalytic()).price

    # Step 2: Calibrate vol to recover implied from price
    market_inputs_new = BlackScholesInputs(reference_date, r, spot, 9000.8) # the vol value is not used
    pricing_problem_new = PricingProblem(payoff, market_inputs_new)
    calib_problem = Hedgehog2.BlackScholesCalibrationProblem(pricing_problem, BlackScholesAnalytic(), price)
    calib_solution = solve(calib_problem)
    implied_vol = calib_solution.u

    @test isapprox(implied_vol, true_vol; atol=1e-8)
end
