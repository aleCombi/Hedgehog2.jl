@testset "Black-Scholes Pricing Tests" begin
    method = BlackScholesAnalytic()
    reference_date = Date(2024, 1, 1)
    expiry_date = reference_date + Day(365)
    r = 0.05
    spot = 100 * exp(-r)
    σ = 0.2
    d1 = 0.5 * σ

    # Common strike and inputs
    strike = spot * exp(r)
    market_inputs = BlackScholesInputs(reference_date, r, spot, σ)

    # Test case 1: European Call Option
    payoff_call = VanillaOption(strike, expiry_date, European(), Call(), Spot())
    prob_call = PricingProblem(payoff_call, market_inputs)
    sol_call = solve(prob_call, method)
    expected_price_call = spot * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ))
    @test isapprox(sol_call.price, expected_price_call, atol = 1e-6)

    # Test case 2: European Put Option
    payoff_put = VanillaOption(strike, expiry_date, European(), Put(), Spot())
    prob_put = PricingProblem(payoff_put, market_inputs)
    sol_put = solve(prob_put, method)
    expected_price_put = spot * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ))
    @test isapprox(sol_put.price, expected_price_put, atol = 1e-6)

    # Test case 3: Zero volatility
    market_inputs_zero_vol = BlackScholesInputs(reference_date, r, spot, 0)
    prob_call_zero_vol = PricingProblem(payoff_call, market_inputs_zero_vol)
    sol_call_zero_vol = solve(prob_call_zero_vol, method)
    @test isapprox(sol_call_zero_vol.price, 0, atol = 1e-9)

    prob_put_zero_vol = PricingProblem(payoff_put, market_inputs_zero_vol)
    sol_put_zero_vol = solve(prob_put_zero_vol, method)
    expected_price_put_zero_vol = 0
    @test isapprox(sol_put_zero_vol.price, expected_price_put_zero_vol, atol = 1e-6)

    # Test case 4: Very short time to expiry
    market_inputs_short_expiry = BlackScholesInputs(reference_date, r, spot, σ)
    payoff_short_expiry =
        VanillaOption(strike, reference_date + Day(1), European(), Call(), Spot())
    prob_short_expiry = PricingProblem(payoff_short_expiry, market_inputs_short_expiry)
    sol_short_expiry = solve(prob_short_expiry, method)
    @test sol_short_expiry.price > 0

    # Test case 5: Deep in-the-money call option
    payoff_itm_call = VanillaOption(50, expiry_date, European(), Call(), Spot())
    prob_itm_call = PricingProblem(payoff_itm_call, market_inputs)
    sol_itm_call = solve(prob_itm_call, method)
    @test sol_itm_call.price > (spot - 50)

    # Test case 6: Deep out-of-the-money put option
    payoff_otm_put = VanillaOption(50, expiry_date, European(), Put(), Spot())
    prob_otm_put = PricingProblem(payoff_otm_put, market_inputs)
    sol_otm_put = solve(prob_otm_put, method)
    @test sol_otm_put.price < 1
end
