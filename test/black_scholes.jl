@testset "Black-Scholes Pricing Tests" begin
    reference_date = Date(2024, 1, 1)
    expiry_date = Date(2024, 12, 31)
    σ = 0.2
    d1 = 0.5 * σ
    r = 0.05
    spot = forward * exp(-r)
    strike = forward

    # Test case 1: European Call Option
    market_inputs = BlackScholesInputs(reference_date, r, spot, σ)
    payoff_call = VanillaOption(strike, expiry_date, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())
    computed_price_call = Pricer(payoff_call, market_inputs, BlackScholesMethod())()

    expected_price_call = exp(-r) * forward * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ))  
    @test isapprox(computed_price_call, expected_price_call, atol=1e-6)

    # Test case 2: European Put Option
    payoff_put = VanillaOption(strike, expiry_date, Hedgehog2.European(), Hedgehog2.Put(), Hedgehog2.Spot())
    computed_price_put = Pricer(payoff_put, market_inputs, BlackScholesMethod())()
    
    expected_price_put = exp(-r) * forward * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ)) 
    @test isapprox(computed_price_put, expected_price_put, atol=1e-6)

    # Test case 3: Zero volatility (should return intrinsic value)
    market_inputs_zero_vol = BlackScholesInputs(reference_date, r, strike + 1, 1E-8)
    computed_price_call_zero_vol = Pricer(payoff_call, market_inputs_zero_vol, BlackScholesMethod())()

    expected_price_call_zero_vol = exp(-r) # discount factor times F - K (1 in our case), as d terms are infinity (ln(F/K) > 0)
    @test isapprox(computed_price_call_zero_vol, expected_price_call_zero_vol, atol=1e-6)

    computed_price_put_zero_vol = Pricer(payoff_put, market_inputs_zero_vol, BlackScholesMethod())()
    expected_price_put_zero_vol = 0 # d terms are minus infinity (ln(F/K) < 0)
    @test isapprox(computed_price_put_zero_vol, expected_price_put_zero_vol, atol=1e-6)

    # Test case 4: Very short time to expiry
    market_inputs_short_expiry = BlackScholesInputs(reference_date, r, forward, σ)
    payoff_short_expiry = VanillaOption(strike, reference_date + Day(1), Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())
    computed_price_short_expiry = Pricer(payoff_short_expiry, market_inputs_short_expiry, BlackScholesMethod())()
    @test computed_price_short_expiry > 0

    # Test case 5: Deep in-the-money call option
    payoff_itm_call = VanillaOption(50, expiry_date, Hedgehog2.European(), Hedgehog2.Call(), Hedgehog2.Spot())
    computed_price_itm_call = Pricer(payoff_itm_call, market_inputs, BlackScholesMethod())()
    @test computed_price_itm_call > (spot - 50)

    # Test case 6: Deep out-of-the-money put option
    payoff_otm_put = VanillaOption(50, expiry_date, Hedgehog2.European(), Hedgehog2.Put(), Hedgehog2.Spot())
    computed_price_otm_put = Pricer(payoff_otm_put, market_inputs, BlackScholesMethod())()
    @test computed_price_otm_put < 1
end
