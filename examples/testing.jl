using Revise, Dates, Hedgehog2, Distributions, Test
method = BlackScholesAnalytic()
reference_date = Date(2024, 1, 1)
expiry_date = reference_date + Day(365)
r = 0.05
spot = 100 * exp(-r)
σ = 0.2
d1 = 0.5 * σ

# Test case 1: European Call Option
strike = spot * exp(r)
market_inputs = BlackScholesInputs(reference_date, r, spot, σ)
payoff_call = VanillaOption(strike, expiry_date, European(), Call(), Spot())
computed_price_call = Pricer(payoff_call, market_inputs, method)()

expected_price_call = spot * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ))
@test isapprox(computed_price_call, expected_price_call, atol = 1e-6)

# Test case 2: European Put Option
payoff_put = VanillaOption(strike, expiry_date, European(), Put(), Spot())
computed_price_put = Pricer(payoff_put, market_inputs, method)()

expected_price_put = spot * (cdf(Normal(), d1) - cdf(Normal(), d1 - σ))
@test isapprox(computed_price_put, expected_price_put, atol = 1e-6)

# Test case 3: Zero volatility (should return intrinsic value)
market_inputs_zero_vol = BlackScholesInputs(reference_date, r, spot, 0)
computed_price_call_zero_vol = Pricer(payoff_call, market_inputs_zero_vol, method)()

# S - exp(-r) K (no volatility, only discounting)
# in this case the d terms are 0/0 forms (ln(F/K) = 1), hence it's a corner case.
@test isapprox(computed_price_call_zero_vol, 0, atol = 1e-9)

computed_price_put_zero_vol = Pricer(payoff_put, market_inputs_zero_vol, method)()
expected_price_put_zero_vol = 0 # d terms are minus infinity (ln(F/K) < 0)
@test isapprox(computed_price_put_zero_vol, expected_price_put_zero_vol, atol = 1e-6)

# Test case 4: Very short time to expiry
market_inputs_short_expiry = BlackScholesInputs(reference_date, r, spot, σ)
payoff_short_expiry =
    VanillaOption(strike, reference_date + Day(1), European(), Call(), Spot())
computed_price_short_expiry =
    Pricer(payoff_short_expiry, market_inputs_short_expiry, method)()
@test computed_price_short_expiry > 0

# Test case 5: Deep in-the-money call option
payoff_itm_call = VanillaOption(50, expiry_date, European(), Call(), Spot())
computed_price_itm_call = Pricer(payoff_itm_call, market_inputs, method)()
@test computed_price_itm_call > (spot - 50)

# Test case 6: Deep out-of-the-money call option
payoff_otm_put = VanillaOption(50, expiry_date, European(), Put(), Spot())
computed_price_otm_put = Pricer(payoff_otm_put, market_inputs, method)()
@test computed_price_otm_put < 1
