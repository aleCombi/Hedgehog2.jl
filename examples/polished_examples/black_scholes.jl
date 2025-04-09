using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic

# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2020, 1, 2)
underlying = Hedgehog2.Forward()

euro_payoff = VanillaOption(strike, expiry, European(), Put(), underlying)

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)
euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

# ------------------------------
# Pricing method
# ------------------------------
bs_method = BlackScholesAnalytic()

# ------------------------------
# Define lenses
# ------------------------------
vol_lens = @optic _.market_inputs.sigma
spot_lens = @optic _.market_inputs.spot

# ------------------------------
# Delta (1st order w.r.t. spot)
# ------------------------------
delta_prob = Hedgehog2.GreekProblem(euro_pricing_prob, spot_lens)
fd_method = FiniteDifference(1E-4, Hedgehog2.FDForward())
ad_method = ForwardAD()
analytic_method = AnalyticGreek()
pricing_method = BlackScholesAnalytic()

@btime solve($euro_pricing_prob, $pricing_method)
@btime solve($delta_prob, $fd_method, $pricing_method)
@btime solve($delta_prob, $ad_method, $pricing_method)
@btime solve($delta_prob, $analytic_method, $pricing_method)

rate_greek_prob = GreekProblem(euro_pricing_prob, ZeroRateSpineLens(1))
solve(rate_greek_prob, ad_method, pricing_method)
@btime solve($rate_greek_prob, $ad_method, $pricing_method)
@btime solve($rate_greek_prob, $fd_method, $pricing_method)

vol_greek_prob = GreekProblem(euro_pricing_prob, VolLens(1,1))
@btime solve($vol_greek_prob, $ad_method, $pricing_method)


steps = 80
crr = CoxRossRubinsteinMethod(steps)
solve(euro_pricing_prob, crr)