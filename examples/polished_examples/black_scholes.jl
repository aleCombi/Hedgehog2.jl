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
vol_lens =  VolLens(1,1)
spot_lens = @optic _.market_inputs.spot
rate_lens = ZeroRateSpineLens(1)
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

vol_greek_prob = GreekProblem(euro_pricing_prob, vol_lens)
@btime solve($vol_greek_prob, $ad_method, $pricing_method)

spot_lens = SpotLens()
steps = 80
crr = CoxRossRubinsteinMethod(steps)
solve(euro_pricing_prob, crr)
batch_prob = BatchGreekProblem(euro_pricing_prob, (spot_lens,vol_lens))

solve(batch_prob, ad_method, pricing_method)
# Fast AD-based all-at-once gradient
@btime solve($batch_prob, $ad_method, $pricing_method)

@code_warntype solve(batch_prob, ad_method, pricing_method)

# Fallback to FD or Analytic
solve(batch_prob, FiniteDifference(1e-4), pricing_method)

target=Hedgehog2.BatchGreekTarget(pricing_method, (spot_lens, vol_lens), euro_pricing_prob)
deriv() = ForwardDiff.gradient(target::B, [2.2,2.3])::Vector{Float64}
@code_warntype deriv()