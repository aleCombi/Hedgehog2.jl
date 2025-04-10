using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using Test
using DataFrames

include("run_model_comparison.jl")

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
# Define pricing methods
# ------------------------------
bs_method = BlackScholesAnalytic()
crr_method = CoxRossRubinsteinMethod(800)

# ------------------------------
# Define lenses for Greeks
# ------------------------------
vol_lens = VolLens(1, 1)
spot_lens = @optic _.market_inputs.spot
rate_lens = ZeroRateSpineLens(1)
lenses = (spot_lens, vol_lens, rate_lens)

# ------------------------------
# Run comparison table
# ------------------------------
df = run_model_comparison_table(
    euro_pricing_prob,
    [bs_method, crr_method],
    lenses;
    ad_method = ForwardAD(),
    fd_method = FiniteDifference(1e-4),
    analytic_method = AnalyticGreek(),
)
println(df)