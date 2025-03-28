using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.2
expiry = Date(2021, 1, 1)
underlying = Hedgehog2.Forward()

euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

reference_date = Date(2020, 1, 1)
rate = 0.2
spot = 1.0
sigma = 0.4

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# ------------------------------
# Pricing method
# ------------------------------
bs_method = BlackScholesAnalytic()

# ------------------------------
# Define lens to volatility
# ------------------------------
vol_lens = @optic _.market.sigma

# ------------------------------
# Compute vega using ForwardAD
# ------------------------------
gprob = Hedgehog2.GreekProblem(euro_pricing_prob, vol_lens)
vega_ad = solve(gprob, ForwardAD(), bs_method).greek

# ------------------------------
# Compute vega using Finite Difference
# ------------------------------
vega_fd = solve(gprob, FiniteDifference(1e-4), bs_method).greek

# ------------------------------
# Compare and print
# ------------------------------
println("Vega (Forward AD): $vega_ad")
println("Vega (Finite Diff): $vega_fd")
